import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

class RobustRewardEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"1. 正在加载修复后的模型: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.truncation_side = 'left'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        print("✅ 模型与分词器（左截断模式）加载成功。")

    def _get_score(self, prompt, response):
        full_text = f"{prompt} {response}"
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.logits.view(-1)[0].item()

    def evaluate_pair_calibrated(self, prompt, resp_a, resp_b):
        s_a1 = self._get_score(prompt, resp_a)
        s_b1 = self._get_score(prompt, resp_b)
        diff1 = s_a1 - s_b1

        s_b2 = self._get_score(prompt, resp_b)
        s_a2 = self._get_score(prompt, resp_a)
        diff2 = s_b2 - s_a2

        avg_diff = (diff1 - diff2) / 2
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
        label = int(float(row['label']))

        avg_diff, s_a, s_b = evaluator.evaluate_pair_calibrated(prompt, resp_a, resp_b)
        model_pred = 0 if avg_diff > 0 else 1
        is_correct = 1 if model_pred == label else 0
        if is_correct:
            correct_count += 1

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
    pred_series = pd.Series([r['model_pred'] for r in results])
    print(f"分布统计: \n{pred_series.value_counts(normalize=True)}")
    print("="*40)

    pd.DataFrame(results).to_csv("final_eval_debug_report.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Reward Model with Robust Calibration")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file to evaluate")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained/fixed reward model")
    args = parser.parse_args()

    run_evaluation(args.csv_path, args.model_path)
