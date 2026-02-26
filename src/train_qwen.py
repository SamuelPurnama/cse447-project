#!/usr/bin/env python3
"""
Standalone Qwen fine-tuning logic (full precision, LoRA).
Used by both myprogram.py and finetune_qwen.sh.
"""
import math
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

QWEN_SUBDIR = "qwen3-4b-base"
QWEN_HF_ID = "Qwen/Qwen3-4B-Base"
QWEN_FINETUNE_NON_EN_SUBDIR = "qwen3-4b-nonEnglish-v1"
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _load_lines(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.rstrip("\n").strip()
            if text:
                lines.append(text)
    return lines


def train_qwen(
    work_dir,
    train_data_path,
    lora_r=8,
    lora_alpha=16,
    lora_target_modules=None,
    lora_dropout=0.05,
    max_seq_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=1,
    eval_steps=500,
    save_steps=1000,
    logging_steps=50,
    output_subdir_name=None,
    merge_lora_into_base=True,
    val_ratio=0.05,
    streaming=False,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not os.path.isfile(train_data_path):
        raise ValueError("Qwen training data not found: {}".format(train_data_path))

    os.makedirs(work_dir, exist_ok=True)
    base_path = os.path.join(work_dir, QWEN_SUBDIR)
    model_path = base_path if os.path.exists(os.path.join(base_path, "config.json")) else QWEN_HF_ID
    output_path = os.path.join(work_dir, output_subdir_name or QWEN_FINETUNE_NON_EN_SUBDIR)
    checkpoint_dir = os.path.join(work_dir, "qwen3-4b-finetune-checkpoints")

    if streaming:
        print("Note: --streaming is currently treated as standard loading in train_qwen.py.")
    print("Loading corpus from {} ...".format(train_data_path))
    lines = _load_lines(train_data_path)
    if not lines:
        raise ValueError("No non-empty lines in {}".format(train_data_path))
    raw_ds = Dataset.from_dict({"text": lines})
    use_eval = len(lines) >= 20 and val_ratio > 0.0
    if use_eval:
        split = raw_ds.train_test_split(test_size=val_ratio, seed=42)
        train_ds_raw, eval_ds_raw = split["train"], split["test"]
    else:
        train_ds_raw, eval_ds_raw = raw_ds, None
    print(
        "Dataset ready: train={}, eval={} (val_ratio={}).".format(
            len(train_ds_raw), len(eval_ds_raw) if eval_ds_raw is not None else 0, val_ratio
        )
    )

    print("Loading tokenizer/model from {} (full precision, NOT quantized) ...".format(model_path))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    target_modules = lora_target_modules if lora_target_modules is not None else DEFAULT_LORA_TARGET_MODULES
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    # Required for gradient checkpointing with LoRA: allow gradients to flow through frozen base to LoRA layers
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    max_length = max(256, int(max_seq_length))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    train_ds = train_ds_raw.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds_raw.map(tokenize, batched=True, remove_columns=["text"]) if use_eval else None

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=max(1, min(4, per_device_train_batch_size)),
        per_device_eval_batch_size=max(1, min(4, per_device_train_batch_size)),
        gradient_accumulation_steps=max(1, gradient_accumulation_steps),
        learning_rate=learning_rate,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=eval_steps if use_eval else None,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        report_to=[],
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if use_eval else None,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("Starting Qwen fine-tuning ...")
    trainer.train()
    if use_eval:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        ppl = math.exp(eval_loss) if eval_loss is not None and eval_loss < 50 else float("inf")
        print("Final eval metrics: loss={}, perplexity={}".format(eval_loss, ppl))

    os.makedirs(output_path, exist_ok=True)
    if merge_lora_into_base and hasattr(model, "merge_and_unload"):
        print("Merging LoRA into base weights and saving to {} ...".format(output_path))
        merged = model.merge_and_unload()
        merged.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    else:
        print("Saving LoRA adapters (not merged) to {} ...".format(output_path))
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    print("Qwen fine-tuning done. Output: {}".format(output_path))


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_dir", default="work")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_target_modules", type=str, default=None, help="comma-separated module names")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--streaming", action="store_true", help="Compatibility flag (currently no-op).")
    parser.add_argument("--output_subdir_name", type=str, default=None)
    parser.add_argument("--no_merge_lora_into_base", dest="merge_lora_into_base", action="store_false")
    parser.set_defaults(merge_lora_into_base=True)
    args = parser.parse_args()

    lora_target_modules = None
    if args.lora_target_modules:
        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    train_qwen(
        work_dir=args.work_dir,
        train_data_path=args.train_data,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=lora_target_modules,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        output_subdir_name=args.output_subdir_name,
        merge_lora_into_base=args.merge_lora_into_base,
        val_ratio=args.val_ratio,
        streaming=args.streaming,
    )
