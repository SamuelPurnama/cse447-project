#!/usr/bin/env python
"""
Download Qwen/Qwen3-4B-Base with 4-bit quantization and save to work_dir for offline inference.
Uses bitsandbytes for 4-bit quantization to reduce memory usage.
"""
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QWEN_HF_ID = "Qwen/Qwen3-4B-Base"
MODEL_SUBDIR = "qwen3-4b-base"


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-4B-Base with 4-bit quantization")
    parser.add_argument("--work_dir", default="work", help="Directory to save model (e.g. work)")
    parser.add_argument("--no_quantize", action="store_true", help="Download full precision model (not recommended)")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    save_path = os.path.join(work_dir, MODEL_SUBDIR)
    os.makedirs(work_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_path, "config.json")):
        print(f"Qwen3-4B-Base already present at {save_path}, skipping download.")
        return

    print(f"Downloading {QWEN_HF_ID} to {save_path} ...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_HF_ID, trust_remote_code=True)
    
    # Always download and save full-precision model locally
    # We'll apply quantization during inference (not during download)
    print("Downloading full-precision model (will be quantized during inference)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_HF_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Save tokenizer and FULL model weights locally
    print(f"Saving tokenizer and full-precision model to {save_path}...")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)  # Save full weights (~8GB)
    
    # Save a flag indicating we want quantization during inference
    quantize_flag_path = os.path.join(save_path, ".quantize_4bit")
    if not args.no_quantize:
        with open(quantize_flag_path, "w") as f:
            f.write("true\n")
        print(f"Saved full-precision model to {save_path} (~8GB)")
        print("Note: Model will be loaded with 4-bit quantization during inference (saves GPU memory).")
    else:
        if os.path.exists(quantize_flag_path):
            os.remove(quantize_flag_path)
        print(f"Saved full-precision model to {save_path} (~8GB)")
        print("Note: Model will be loaded in full precision during inference.")


if __name__ == "__main__":
    main()
