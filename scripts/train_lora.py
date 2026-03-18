import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from trainer_module import GemmaRewardLoRATrainer, PairwiseCSVDataset, collate_fn

# --------- 参数解析 ---------
parser = argparse.ArgumentParser(description="Gemma LoRA Reward Training")
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
args = parser.parse_args()

# --------- 加载 YAML 配置 ---------
with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

torch.manual_seed(cfg["data"]["random_seed"])

trainer = GemmaRewardLoRATrainer(
    model_path=cfg["model"]["model_path"],
    alpha=cfg["training"]["alpha"],
    lr=cfg["training"]["lr"],
    save_dir=cfg["training"]["save_dir"],
    lora_r=cfg["model"]["lora_r"],
    lora_alpha=cfg["model"]["lora_alpha"],
    lora_dropout=cfg["model"]["lora_dropout"]
)

df = pd.read_csv(cfg["data"]["csv_path"])
train_df, val_df = train_test_split(df, test_size=cfg["data"]["test_size"],
                                    random_state=cfg["data"]["random_seed"], shuffle=True)

train_dataset = PairwiseCSVDataset(train_df, trainer.tokenizer, max_len=cfg["model"]["max_len"])
val_dataset = PairwiseCSVDataset(val_df, trainer.tokenizer, max_len=cfg["model"]["max_len"])

train_loader = DataLoader(train_dataset, batch_size=cfg["data"]["batch_size_train"], shuffle=True,
                          collate_fn=lambda x: collate_fn(x, trainer.tokenizer, max_len=cfg["model"]["max_len"]))
val_loader = DataLoader(val_dataset, batch_size=cfg["data"]["batch_size_val"], shuffle=False,
                        collate_fn=lambda x: collate_fn(x, trainer.tokenizer, max_len=cfg["model"]["max_len"]))

for epoch in range(cfg["training"]["epochs"]):
    trainer.model.train()
    train_loss_total = 0
    for step, batch in enumerate(train_loader):
        loss, acc, r_diff_mean = trainer.train_step(batch, clip_grad=cfg["training"]["clip_grad"])
        train_loss_total += loss
        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} | Loss: {loss:.4f} | Acc: {acc:.2f} | Score Diff: {r_diff_mean:.4f}")

    avg_train_loss = train_loss_total / len(train_loader)

    trainer.model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for batch in val_loader:
            val_loss_total += trainer.compute_loss(batch).item()
    avg_val_loss = val_loss_total / len(val_loader)

    print(f"Epoch {epoch} Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

trainer.save_checkpoint()
