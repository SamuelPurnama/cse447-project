#!/usr/bin/env python
"""
CSE447 character prediction: Bolmo-1B byte-level model.
Phase 1: load model from work_dir, run inference only (no training in predict path).
Fine-tuning: optional LoRA + SFTTrainer with --finetune and --train_data; merge into base and save to work_dir/bolmo-1b/.
"""
import os
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Bolmo model subdir under work_dir (same as download_bolmo.py)
BOLMO_SUBDIR = "bolmo-1b"
BOLMO_HF_ID = "allenai/Bolmo-1B"

# Default LoRA target modules (override with --lora_target_modules if your model uses different names)
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_corpus_and_split(train_data_path, val_ratio=0.05, seed=42):
    """Load text corpus from file (one sample per line), split into train/val. Returns (train_ds, eval_ds)."""
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


class MyModel:
    """
    Bolmo-1B byte-level causal LM for next-character prediction.
    Loads from work_dir/bolmo-1b; use download_bolmo.py or train mode to populate.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load_training_data(cls):
        return []

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, encoding="utf-8") as f:
            for line in f:
                inp = line.rstrip("\n")
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, "wt", encoding="utf-8") as f:
            for p in preds:
                f.write("{}\n".format(p))

    def run_train(self, data, work_dir):
        # Phase 1: no training; use train mode to download and save Bolmo to work_dir
        from transformers import AutoModelForCausalLM, AutoTokenizer
        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, BOLMO_SUBDIR)
        if os.path.exists(os.path.join(save_path, "config.json")):
            print("Bolmo-1B already at {}, skipping download.".format(save_path))
            return
        print("Downloading Bolmo-1B to {} ...".format(save_path))
        tokenizer = AutoTokenizer.from_pretrained(BOLMO_HF_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(BOLMO_HF_ID, trust_remote_code=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print("Saved to {}.".format(save_path))

    def run_finetune(
        self,
        train_data_path,
        work_dir,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=None,
        lora_dropout=0.05,
        max_seq_length=2048,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        num_train_epochs=1,
        eval_steps=500,
        save_steps=1000,
        logging_steps=50,
    ):
        """Fine-tune with LoRA and SFTTrainer; 95/5 train/val split, keep best checkpoint, merge into base, save to work_dir/bolmo-1b/."""
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTConfig, SFTTrainer

        def causal_lm_loss_from_logits(outputs, labels, num_items_in_batch=None):
            """Compute causal LM cross-entropy from logits when the model does not return loss (e.g. Bolmo)."""
            if labels is None:
                raise ValueError("Labels are required for training.")
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, BOLMO_SUBDIR)
        checkpoint_dir = os.path.join(work_dir, "bolmo-1b-finetune-checkpoints")

        print("Loading corpus from {} and splitting 95%% train / 5%% val ...".format(train_data_path))
        train_ds, eval_ds = _load_corpus_and_split(train_data_path, val_ratio=0.05)
        print("Train size: {}, Eval size: {}".format(len(train_ds), len(eval_ds)))

        target_modules = lora_target_modules if lora_target_modules is not None else DEFAULT_LORA_TARGET_MODULES
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        print("Applying LoRA (r={}, alpha={}, target_modules={}) ...".format(lora_r, lora_alpha, target_modules))
        self.model = get_peft_model(self.model, lora_config)

        # Bolmo's mLSTM kernel requires sequence length divisible by 64 (CHUNK_SIZE).
        BOLMO_MLSTM_CHUNK_SIZE = 64
        max_length_aligned = max(BOLMO_MLSTM_CHUNK_SIZE, (max_seq_length // BOLMO_MLSTM_CHUNK_SIZE) * BOLMO_MLSTM_CHUNK_SIZE)
        if max_length_aligned != max_seq_length:
            print("Aligning max_length to Bolmo mLSTM chunk size 64: {} -> {}.".format(max_seq_length, max_length_aligned))

        sft_config = SFTConfig(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            max_length=max_length_aligned,
            dataset_text_field="text",
            pad_to_multiple_of=BOLMO_MLSTM_CHUNK_SIZE,
            label_names=["labels"],
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=self.tokenizer,
            args=sft_config,
            compute_loss_func=causal_lm_loss_from_logits,
        )
        print("Training (best checkpoint by eval loss will be merged and saved) ...")
        trainer.train()
        print("Merging LoRA into base and saving to {} ...".format(save_path))
        self.model = self.model.merge_and_unload()
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print("Fine-tuning done. Merged model saved to {}.".format(save_path))

    def run_pred(self, data):
        import torch
        self.model.eval()
        preds = []
        with torch.no_grad():
            for inp in data:
                top3 = self._next_char_top3(inp)
                preds.append(top3)
        return preds

    def _next_char_top3(self, text):
        """Return 3 character guesses for the next character after text (byte-level top-3)."""
        if not text:
            return "eta"  # frequency fallback
        try:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            if input_ids.size(1) == 0:
                return "eta"
            logits = self.model(input_ids).logits
            _, top_ids = logits[0, -1, :].topk(20, dim=-1)
            top_ids = top_ids.cpu().tolist()
            chars = []
            for tid in top_ids:
                if len(chars) >= 3:
                    break
                try:
                    c = self.tokenizer.decode([tid], skip_special_tokens=True)
                    if c:
                        c = c[0]
                        if c not in chars:
                            chars.append(c)
                except Exception:
                    continue
            while len(chars) < 3:
                chars.append("e")
            return "".join(chars[:3])
        except Exception:
            return "eta"

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, BOLMO_SUBDIR)
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)

    @classmethod
    def load(cls, work_dir):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        save_path = os.path.join(work_dir, BOLMO_SUBDIR)
        if os.path.exists(os.path.join(save_path, "config.json")):
            model_path = save_path
        else:
            model_path = BOLMO_HF_ID
            print("Model not found at {}, loading from {} (run download_bolmo.py or train first).".format(
                save_path, BOLMO_HF_ID))
        device = _get_device()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        return cls(model=model, tokenizer=tokenizer, device=device)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="train: download/save Bolmo; test: load and predict")
    parser.add_argument("--work_dir", default="work", help="where to load/save model")
    parser.add_argument("--test_data", default="example/input.txt", help="path to test input")
    parser.add_argument("--test_output", default="pred.txt", help="path to write predictions")
    # Fine-tuning (only used when mode=train and --finetune)
    parser.add_argument("--finetune", action="store_true", help="run LoRA fine-tuning; requires --train_data")
    parser.add_argument("--train_data", default=None, help="path to training corpus (one sample per line, UTF-8); required if --finetune")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (default 8)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default 16)")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="comma-separated LoRA target modules (default: q_proj,v_proj)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default 0.05)")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max sequence length for SFT (default 2048)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="train batch size per device (default 8)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation steps (default 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate (default 1e-5)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="number of train epochs (default 1)")
    parser.add_argument("--eval_steps", type=int, default=500, help="eval every N steps (default 500)")
    parser.add_argument("--save_steps", type=int, default=1000, help="save every N steps (default 1000)")
    parser.add_argument("--logging_steps", type=int, default=50, help="log every N steps (default 50)")
    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
        if args.finetune:
            if not args.train_data or not os.path.isfile(args.train_data):
                raise SystemExit("--finetune requires --train_data pointing to an existing corpus file (e.g. work/data/phase2/corpus_utf8_nfc.txt)")
            print("Loading model for fine-tuning")
            model = MyModel.load(args.work_dir)
            lora_target_modules = None
            if args.lora_target_modules is not None:
                lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
            model.run_finetune(
                args.train_data,
                args.work_dir,
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
            )
        else:
            print("Instantiating model (download-only for Phase 1)")
            model = MyModel.load(args.work_dir)
            print("Loading training data (optional)")
            train_data = MyModel.load_training_data()
            print("Training (download/save only)")
            model.run_train(train_data, args.work_dir)
            print("Saving model")
            model.save(args.work_dir)
    elif args.mode == "test":
        print("Loading model")
        model = MyModel.load(args.work_dir)
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print("Making predictions")
        pred = model.run_pred(test_data)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred))
        MyModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
