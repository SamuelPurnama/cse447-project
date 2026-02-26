#!/usr/bin/env python3
"""
Standalone Bolmo fine-tuning logic (full precision, LoRA).
Used by both myprogram.py and finetune_bolmo.sh.
"""
import math
import os
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

BOLMO_SUBDIR = "bolmo-1b"
BOLMO_HF_ID = "allenai/Bolmo-1B"
BOLMO_INITIAL_FINETUNE_SUBDIR = "bolmo-InitialFineTune"
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_corpus_and_split(train_data_path, val_ratio=0.05, seed=42):
    from datasets import Dataset

    lines = []
    with open(train_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError("No non-empty lines in {}".format(train_data_path))
    dataset = Dataset.from_dict({"text": lines})
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def _load_corpus_streaming(train_data_path, eval_size=50000):
    from datasets import Dataset, IterableDataset

    eval_lines = []
    if eval_size > 0:
        with open(train_data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= eval_size:
                    break
                line = line.rstrip("\n").strip()
                if line:
                    eval_lines.append(line)
        eval_ds = Dataset.from_dict({"text": eval_lines}) if eval_lines else None
    else:
        eval_ds = None

    def train_gen():
        with open(train_data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < eval_size:
                    continue
                line = line.rstrip("\n").strip()
                if line:
                    yield {"text": line}

    train_ds = IterableDataset.from_generator(train_gen)
    return train_ds, eval_ds


def _count_training_lines_streaming(train_data_path, eval_size=50000):
    count = 0
    with open(train_data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < eval_size:
                continue
            if line.rstrip("\n").strip():
                count += 1
    return count


def train_bolmo(
    work_dir,
    train_data_path,
    lora_r=8,
    lora_alpha=16,
    lora_target_modules=None,
    lora_dropout=0.05,
    max_seq_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=1,
    eval_steps=500,
    save_steps=1000,
    logging_steps=50,
    gradient_checkpointing=True,
    streaming=False,
    streaming_eval_size=50000,
    variant_subdir=None,
    val_ratio=0.05,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTConfig, SFTTrainer

    def causal_lm_loss_from_logits(outputs, labels, num_items_in_batch=None):
        if labels is None:
            raise ValueError("Labels are required for training.")
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    os.makedirs(work_dir, exist_ok=True)
    raw_bolmo_path = os.path.join(work_dir, BOLMO_SUBDIR)
    checkpoint_dir = os.path.join(work_dir, "bolmo-1b-finetune-checkpoints")

    if not os.path.isfile(train_data_path):
        raise ValueError("Bolmo training data not found: {}".format(train_data_path))

    if os.path.exists(os.path.join(raw_bolmo_path, "config.json")):
        model_path = raw_bolmo_path
    else:
        model_path = BOLMO_HF_ID
        print("Bolmo not at {}, loading from {}.".format(raw_bolmo_path, BOLMO_HF_ID))

    print("Loading Bolmo tokenizer/model from {} ...".format(model_path))
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

    if streaming:
        print("Loading corpus in streaming mode from {} (eval = first {} lines) ...".format(
            train_data_path, streaming_eval_size))
        num_train_lines = _count_training_lines_streaming(train_data_path, eval_size=streaming_eval_size)
        train_ds, eval_ds = _load_corpus_streaming(train_data_path, eval_size=streaming_eval_size)
        batch_size = max(1, per_device_train_batch_size * max(1, gradient_accumulation_steps))
        max_steps = max(1, num_train_lines // batch_size)
        print("Streaming dataset ready: train_lines={}, eval_size={}, max_steps={}.".format(
            num_train_lines, len(eval_ds) if eval_ds is not None else 0, max_steps))
    else:
        max_steps = -1
        train_ds, eval_ds = _load_corpus_and_split(train_data_path, val_ratio=val_ratio, seed=42)
        print("Dataset ready: train={}, eval={} (val_ratio={}).".format(len(train_ds), len(eval_ds), val_ratio))

    target_modules = lora_target_modules if lora_target_modules is not None else DEFAULT_LORA_TARGET_MODULES
    print("Applying LoRA with target modules: {}.".format(target_modules))
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    chunk_size = 64
    max_length_aligned = max(chunk_size, (max_seq_length // chunk_size) * chunk_size)
    use_eval = eval_ds is not None and len(eval_ds) > 0
    sft_config_kw = dict(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=max(1, per_device_train_batch_size),
        gradient_accumulation_steps=max(1, gradient_accumulation_steps),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=eval_steps if use_eval else None,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False,
        max_length=max_length_aligned,
        dataset_text_field="text",
        pad_to_multiple_of=chunk_size,
        label_names=["labels"],
    )
    if streaming and max_steps > 0:
        sft_config_kw["max_steps"] = max_steps
        sft_config_kw["num_train_epochs"] = 1

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds if use_eval else None,
        processing_class=tokenizer,
        args=SFTConfig(**sft_config_kw),
        compute_loss_func=causal_lm_loss_from_logits,
    )
    print("Starting Bolmo fine-tuning ...")
    trainer.train()
    if use_eval:
        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        ppl = math.exp(eval_loss) if eval_loss is not None and eval_loss < 50 else float("inf")
        print("Final eval metrics: loss={}, perplexity={}".format(eval_loss, ppl))

    model = model.merge_and_unload()
    bolmo_variant_dir = os.path.join(work_dir, variant_subdir or BOLMO_INITIAL_FINETUNE_SUBDIR)
    os.makedirs(bolmo_variant_dir, exist_ok=True)
    tokenizer.save_pretrained(bolmo_variant_dir)
    model.save_pretrained(bolmo_variant_dir)

    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print("Bolmo fine-tuning done. Output: {}".format(bolmo_variant_dir))


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_dir", default="work")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_target_modules", type=str, default=None, help="comma-separated module names")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--streaming_eval_size", type=int, default=50000)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--variant_subdir", type=str, default=None)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    args = parser.parse_args()

    lora_target_modules = None
    if args.lora_target_modules:
        lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    train_bolmo(
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
        gradient_checkpointing=args.gradient_checkpointing,
        streaming=args.streaming,
        streaming_eval_size=args.streaming_eval_size,
        variant_subdir=args.variant_subdir,
        val_ratio=args.val_ratio,
    )
#!/usr/bin/env python3
"""
Bolmo fine-tuning entrypoint used by myprogram.py.
This module isolates Bolmo training logic from the main model script.
"""
import os
import shutil


BOLMO_SUBDIR = "bolmo-1b"
BOLMO_HF_ID = "allenai/Bolmo-1B"
BOLMO_INITIAL_FINETUNE_SUBDIR = "bolmo-FineTune-v1"
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_corpus_and_split(train_data_path, val_ratio=0.05, seed=42):
    from datasets import Dataset

    lines = []
    with open(train_data_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError("No non-empty lines in {}".format(train_data_path))
    dataset = Dataset.from_dict({"text": lines})
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def _load_corpus_streaming(train_data_path, eval_size=50000):
    from datasets import Dataset, IterableDataset

    eval_lines = []
    if eval_size > 0:
        with open(train_data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= eval_size:
                    break
                line = line.rstrip("\n").strip()
                if line:
                    eval_lines.append(line)
        eval_ds = Dataset.from_dict({"text": eval_lines}) if eval_lines else None
    else:
        eval_ds = None

    def train_gen():
        with open(train_data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < eval_size:
                    continue
                line = line.rstrip("\n").strip()
                if line:
                    yield {"text": line}

    train_ds = IterableDataset.from_generator(train_gen)
    return train_ds, eval_ds


def _count_training_lines_streaming(train_data_path, eval_size=50000):
    count = 0
    with open(train_data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < eval_size:
                continue
            if line.rstrip("\n").strip():
                count += 1
    return count


def train_bolmo(
    work_dir,
    train_data_path,
    lora_r=8,
    lora_alpha=16,
    lora_target_modules=None,
    lora_dropout=0.05,
    max_seq_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=1,
    eval_steps=500,
    save_steps=1000,
    logging_steps=50,
    gradient_checkpointing=True,
    streaming=False,
    streaming_eval_size=50000,
    variant_subdir=None,
):
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTConfig, SFTTrainer

    def causal_lm_loss_from_logits(outputs, labels, num_items_in_batch=None):
        if labels is None:
            raise ValueError("Labels are required for training.")
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

    os.makedirs(work_dir, exist_ok=True)
    raw_bolmo_path = os.path.join(work_dir, BOLMO_SUBDIR)
    checkpoint_dir = os.path.join(work_dir, "bolmo-1b-finetune-checkpoints")

    bolmo_corpus = train_data_path
    if not os.path.isfile(bolmo_corpus):
        raise ValueError("Bolmo training data not found: {}".format(bolmo_corpus))

    if os.path.exists(os.path.join(raw_bolmo_path, "config.json")):
        model_path = raw_bolmo_path
    else:
        model_path = BOLMO_HF_ID
        print("Bolmo not at {}, loading from {}.".format(raw_bolmo_path, BOLMO_HF_ID))

    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

    if streaming:
        print("Loading Bolmo corpus in streaming mode from {} (eval = first {} lines) ...".format(
            bolmo_corpus, streaming_eval_size))
        num_train_lines = _count_training_lines_streaming(bolmo_corpus, eval_size=streaming_eval_size)
        train_ds, eval_ds = _load_corpus_streaming(bolmo_corpus, eval_size=streaming_eval_size)
        batch_size = per_device_train_batch_size * gradient_accumulation_steps
        max_steps = max(1, num_train_lines // batch_size)
    else:
        max_steps = -1
        print("Loading Bolmo corpus from {} and splitting 95% train / 5% val ...".format(bolmo_corpus))
        train_ds, eval_ds = _load_corpus_and_split(bolmo_corpus, val_ratio=0.05)

    target_modules = lora_target_modules if lora_target_modules is not None else DEFAULT_LORA_TARGET_MODULES
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    chunk_size = 64
    max_length_aligned = max(chunk_size, (max_seq_length // chunk_size) * chunk_size)
    use_eval = eval_ds is not None and len(eval_ds) > 0
    sft_config_kw = dict(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=eval_steps if use_eval else None,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=use_eval,
        metric_for_best_model="eval_loss" if use_eval else None,
        greater_is_better=False,
        max_length=max_length_aligned,
        dataset_text_field="text",
        pad_to_multiple_of=chunk_size,
        label_names=["labels"],
    )
    if streaming and max_steps > 0:
        sft_config_kw["max_steps"] = max_steps
        sft_config_kw["num_train_epochs"] = 1

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds if use_eval else None,
        processing_class=tokenizer,
        args=SFTConfig(**sft_config_kw),
        compute_loss_func=causal_lm_loss_from_logits,
    )
    trainer.train()

    model = model.merge_and_unload()
    bolmo_variant_dir = os.path.join(work_dir, variant_subdir or BOLMO_INITIAL_FINETUNE_SUBDIR)
    os.makedirs(bolmo_variant_dir, exist_ok=True)
    tokenizer.save_pretrained(bolmo_variant_dir)
    model.save_pretrained(bolmo_variant_dir)

    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    print("Bolmo fine-tuning done. Saved to {}".format(bolmo_variant_dir))
