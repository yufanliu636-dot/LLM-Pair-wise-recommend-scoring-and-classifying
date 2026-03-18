import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file
import argparse

def force_merge(base_path, lora_path, save_path, device="cpu"):
    print(f"1. Loading base model (num_labels=1) from: {base_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    print(f"2. Loading LoRA adapter and performing standard matrix merge...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = model.merge_and_unload()

    print(f"3. Manually calibrating classification head weights (Surgical Alignment)...")
    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        lora_weights = load_file(adapter_file)
        target_key = "base_model.model.score.weight"

        if target_key in lora_weights:
            print(f"Found key: {target_key}, forcibly overwriting...")
            with torch.no_grad():
                merged_model.score.weight.copy_(lora_weights[target_key])
            print("✅ Classification head weights successfully overwritten!")
        else:
            print("⚠️ Target key not found in adapter. Please check the safetensors file.")
    else:
        print(f"❌ Adapter file not found: {adapter_file}")

    merged_model.config.num_labels = 1
    merged_model.config.pad_token_id = tokenizer.pad_token_id
    merged_model.config.architectures = ["Gemma2ForSequenceClassification"]

    print(f"4. Saving the full merged model to: {save_path} ...")
    os.makedirs(save_path, exist_ok=True)
    merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    print("\n" + "="*40)
    print("🚀 All steps completed successfully!")
    print(f"Please use the path: {save_path} for subsequent evaluation.")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force Merge LoRA into Base Model")
    parser.add_argument("--base_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint folder")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run merge on (cpu or cuda)")
    args = parser.parse_args()

    force_merge(args.base_path, args.lora_path, args.save_path, device=args.device)
