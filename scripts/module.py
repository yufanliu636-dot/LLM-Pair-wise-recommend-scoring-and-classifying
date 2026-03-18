import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


# ===================== Dataset =====================
class PairwiseCSVDataset(Dataset):
    """CSV 文件必须包含列: prompt, response_a, response_b, label"""

    def __init__(self, df, tokenizer, max_len=1024):
        self.df = df.fillna("").astype(str).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label"]
        if label == 0:
            chosen_text = row["prompt"] + " " + row["response_a"]
            rejected_text = row["prompt"] + " " + row["response_b"]
        else:
            chosen_text = row["prompt"] + " " + row["response_b"]
            rejected_text = row["prompt"] + " " + row["response_a"]
        return chosen_text, rejected_text


def collate_fn(batch, tokenizer, max_len=1024):
    chosen_texts, rejected_texts = zip(*batch)
    chosen_tokens = [tokenizer.encode(t, truncation=True, max_length=max_len) for t in chosen_texts]
    rejected_tokens = [tokenizer.encode(t, truncation=True, max_length=max_len) for t in rejected_texts]

    chosen_input_ids = pad_sequence([torch.tensor(x) for x in chosen_tokens], batch_first=True,
                                    padding_value=tokenizer.pad_token_id)
    rejected_input_ids = pad_sequence([torch.tensor(x) for x in rejected_tokens], batch_first=True,
                                      padding_value=tokenizer.pad_token_id)

    chosen_mask = (chosen_input_ids != tokenizer.pad_token_id).long()
    rejected_mask = (rejected_input_ids != tokenizer.pad_token_id).long()

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_mask": chosen_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_mask": rejected_mask
    }


# ===================== LoRA Trainer =====================
class GemmaRewardLoRATrainer:
    def __init__(self, model_path, alpha=0.001, lr=1e-5, save_dir="./result",
                 lora_r=16, lora_alpha=32, lora_dropout=0.05, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 加载本地模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto" if self.device.startswith("cuda") else None
        )

        # LoRA 配置
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.alpha = alpha
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def compute_loss(self, batch):
        rewards_chosen = self.model(
            input_ids=batch['chosen_input_ids'].to(self.device),
            attention_mask=batch['chosen_mask'].to(self.device)
        ).logits.view(-1)

        rewards_rejected = self.model(
            input_ids=batch['rejected_input_ids'].to(self.device),
            attention_mask=batch['rejected_mask'].to(self.device)
        ).logits.view(-1)

        len_chosen = batch['chosen_mask'].sum(dim=-1).float().to(self.device)
        len_rejected = batch['rejected_mask'].sum(dim=-1).float().to(self.device)

        r_diff = rewards_chosen - rewards_rejected
        len_diff = len_chosen - len_rejected
        adjusted_logits = r_diff - self.alpha * len_diff

        loss = -torch.nn.functional.logsigmoid(adjusted_logits).mean()
        reg = 0.001 * (rewards_chosen ** 2 + rewards_rejected ** 2).mean()
        return loss + reg

    def train_step(self, batch, clip_grad=True):
        self.optimizer.zero_grad()
        with torch.no_grad():
            rewards_chosen = self.model(
                input_ids=batch['chosen_input_ids'].to(self.device),
                attention_mask=batch['chosen_mask'].to(self.device)
            ).logits.view(-1)

            rewards_rejected = self.model(
                input_ids=batch['rejected_input_ids'].to(self.device),
                attention_mask=batch['rejected_mask'].to(self.device)
            ).logits.view(-1)

            acc = (rewards_chosen > rewards_rejected).float().mean().item()
            r_diff_mean = (rewards_chosen - rewards_rejected).mean().item()

        loss = self.compute_loss(batch)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), acc, r_diff_mean

    def save_checkpoint(self):
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save_pretrained(self.save_dir)
        self.tokenizer.save_pretrained(self.save_dir)
        print(f"Saved LoRA checkpoint to {self.save_dir}")
