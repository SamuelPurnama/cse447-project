#!/usr/bin/env python
"""
Download Allen AI Bolmo-1B and save to work_dir for offline inference.
Run once to populate work/bolmo-1b so predict.sh can load from local path.
"""
import os
import argparse

BOLMO_HF_ID = "allenai/Bolmo-1B"
MODEL_SUBDIR = "bolmo-1b"


def main():
    parser = argparse.ArgumentParser(description="Download Bolmo-1B and save to work_dir")
    parser.add_argument("--work_dir", default="work", help="Directory to save model (e.g. work)")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    save_path = os.path.join(work_dir, MODEL_SUBDIR)
    os.makedirs(work_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_path, "config.json")):
        print(f"Bolmo-1B already present at {save_path}, skipping download.")
        return

    print(f"Downloading {BOLMO_HF_ID} to {save_path} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BOLMO_HF_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BOLMO_HF_ID, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Saved Bolmo-1B to {save_path}")


if __name__ == "__main__":
    main()
