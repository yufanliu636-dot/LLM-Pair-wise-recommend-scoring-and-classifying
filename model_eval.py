import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RobustRewardEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"1. 正在加载修复后的模型: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 【核心修复】：必须左截断！
        # 否则 Prompt 一长，右边的 Response 就被切没了，Sa 永远等于 Sb
        self.tokenizer.truncation_side = 'left' 
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print("✅ 模型与分词器（左截断模式）加载成功。")

    def _get_score(self, prompt, response):
        """获取单条文本的原始分值"""
        # 保持与训练一致的拼接格式
        full_text = f"{prompt} {response}"
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048 # 建议与训练时的 max_len 一致
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 拿到打分层的第一个输出（针对 num_labels=1）
            return outputs.logits.view(-1)[0].item()

    def evaluate_pair_calibrated(self, prompt, resp_a, resp_b):
        """对称性验证：抵消位置偏见"""
        # 第一遍：A vs B
        s_a1 = self._get_score(prompt, resp_a)
        s_b1 = self._get_score(prompt, resp_b)
        diff1 = s_a1 - s_b1

        # 第二遍：交换位置输入 (注意传入参数的顺序)
        # 这里必须是 B 在前，A 在后
        s_b2 = self._get_score(prompt, resp_b)
        s_a2 = self._get_score(prompt, resp_a)
        
        # 此时 diff2 应该是 (B 的分 - A 的分)
        diff2 = s_b2 - s_a2

        # 校准分差：(A-B) 应该等于 -(B-A)
        # 所以平均分差是 (diff1 - diff2) / 2
        avg_diff = (diff1 - diff2) / 2
        
        # 你的 Debug 日志显示 Diff 为 0，说明 diff1 肯定等于了 diff2
        # 请确保上面的参数顺序没有传错
        
        return avg_diff, s_a1, s_b1

def run_evaluation(csv_path, model_path):
    evaluator = RobustRewardEvaluator(model_path)
    df = pd.read_csv(csv_path).fillna("Missing Text")
    
    print(f"检测到 {len(df)} 条测试数据，开始评估...")
    
    results = []
    correct_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = str(row['prompt'])
        resp_a = str(row['response_a'])
        resp_b = str(row['response_b'])
        label = int(float(row['label'])) # 0: A更好, 1: B更好

        # 获取校准分差
        avg_diff, s_a, s_b = evaluator.evaluate_pair_calibrated(prompt, resp_a, resp_b)
        
        # 判定逻辑：
        # 如果模型给 A 的分数更高 (avg_diff > 0)，预测标签为 0
        model_pred = 0 if avg_diff > 0 else 1
        
        # 统计正确率
        is_correct = 1 if model_pred == label else 0
        if is_correct:
            correct_count += 1
            
        # 调试：前 5 条数据打印详情，看看 Sa 和 Sb 是否还相等
        if idx < 5:
            print(f"\n[Debug Row {idx}] Sa: {s_a:.4f} | Sb: {s_b:.4f} | Diff: {avg_diff:.4f} | Pred: {model_pred} | Label: {label}")

        results.append({
            "model_pred": model_pred,
            "label": label,
            "avg_diff": avg_diff,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(df)
    
    print("\n" + "="*40)
    print(f"📊 最终评估报告")
    print(f"✅ 准确率 (Accuracy): {accuracy:.2%}")
    # 统计预测分布，看是否还是全为 1
    pred_series = pd.Series([r['model_pred'] for r in results])
    print(f"分布统计: \n{pred_series.value_counts(normalize=True)}")
    print("="*40)

    # 保存结果供回溯
    pd.DataFrame(results).to_csv("final_eval_debug_report.csv", index=False)

if __name__ == "__main__":
    # ！！！请确保路径指向你修复过权重的那个 FIXED_FINAL 文件夹！！！
    FIXED_MODEL_DIR = "/root/autodl-fs/gemma2_reward_model_merged_final2"
    DATA_CSV = "/root/autodl-fs/output.csv"
    
    run_evaluation(DATA_CSV, FIXED_MODEL_DIR)