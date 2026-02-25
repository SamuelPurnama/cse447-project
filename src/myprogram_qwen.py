#!/usr/bin/env python
"""
CSE447 character prediction: Qwen3-4B-Base subword tokenizer model.
Uses subword tokenization instead of byte-level, which should handle multi-language better.
Predicts next token, then converts token predictions to character predictions.
"""
import os
import shutil
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
import numpy as np

# Qwen model subdir under work_dir
QWEN_SUBDIR = "qwen3-4b-base"
QWEN_HF_ID = "Qwen/Qwen3-4B-Base"

# Default LoRA target modules for Qwen
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _get_device():
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


class QwenModel:
    """
    Qwen3-4B-Base subword tokenizer model for next-character prediction.
    Loads from work_dir/qwen3-4b-base with 4-bit quantization.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # Cache for token-to-first-character mapping
        self._token_to_first_char_cache = {}

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

    def _token_to_first_char(self, token_id):
        """
        Convert a token ID to the first character it decodes to.
        Returns (first_char, num_chars) where num_chars is how many characters the token decodes to.
        Uses caching for efficiency.
        """
        if token_id in self._token_to_first_char_cache:
            return self._token_to_first_char_cache[token_id]
        
        try:
            # Decode the token (skip special tokens to get actual text)
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Check if it's a special token
            if token_id in self.tokenizer.all_special_ids or not token_str:
                result = (None, 0)
            else:
                # Get first character (handle multi-character tokens)
                first_char = token_str[0]
                num_chars = len(token_str)
                
                # Validate character
                if ord(first_char) < 32 or first_char == '\ufffd':
                    result = (None, 0)
                else:
                    result = (first_char, num_chars)
        except Exception:
            result = (None, 0)
        
        self._token_to_first_char_cache[token_id] = result
        return result

    def _get_next_token_logprobs(self, input_ids, top_k=5000):
        """
        Get log-probabilities for the next token.
        Returns log_probs tensor of shape (vocab_size,).
        Only considers top_k tokens for efficiency.
        """
        self.model.eval()
        with torch.no_grad():
            try:
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last position, shape (vocab_size,)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # For efficiency, only consider top_k tokens
                if top_k < logits.size(0):
                    top_k_log_probs, top_k_indices = torch.topk(log_probs, top_k)
                    # Create sparse log_probs tensor with matching dtype
                    sparse_log_probs = torch.full(
                        (logits.size(0),), 
                        float('-inf'), 
                        device=self.device,
                        dtype=log_probs.dtype  # Match the dtype of log_probs
                    )
                    sparse_log_probs[top_k_indices] = top_k_log_probs
                    return sparse_log_probs, top_k_indices
                else:
                    return log_probs, None
            except Exception as e:
                print(f"Error in _get_next_token_logprobs: {e}")
                # Fallback: return uniform distribution over a small set
                vocab_size = len(self.tokenizer)
                # Use float16 if on CUDA, float32 otherwise
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                fallback_log_probs = torch.full(
                    (vocab_size,), 
                    float('-inf'), 
                    device=self.device,
                    dtype=dtype
                )
                # Set some reasonable default tokens to non-zero probability
                for default_token_id in range(min(100, vocab_size)):
                    fallback_log_probs[default_token_id] = -10.0
                return fallback_log_probs, None

    def _next_char_top3(self, text, top_k_tokens=5000, debug=False):
        """
        Predict top-3 next characters using subword tokenizer.
        
        Strategy:
        1. Get next token probabilities
        2. For each candidate token, find the first character it decodes to
        3. Marginalize: sum probabilities of all tokens that decode to the same first character
        4. Return top-3 characters
        """
        if not text:
            return "eta"  # frequency fallback
        
        try:
            # Step 1: Tokenize input text
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(self.device)
            
            if input_ids.size(0) == 0 or input_ids.size(1) == 0:
                return "eta"
            
            # Step 2: Get next token log-probabilities
            log_probs, top_k_indices = self._get_next_token_logprobs(input_ids, top_k=top_k_tokens)
            
            # Step 3: Map tokens to first characters and marginalize
            char_log_probs = {}  # char -> list of log_probs from tokens that decode to it
            
            # Iterate over all tokens (or top_k if we filtered)
            if top_k_indices is not None:
                token_ids_to_check = top_k_indices.cpu().tolist()
            else:
                vocab_size = len(self.tokenizer)
                token_ids_to_check = list(range(vocab_size))
            
            for token_id in token_ids_to_check:
                log_prob = log_probs[token_id].item()
                
                # Skip tokens with very low probability
                if log_prob < -50.0:
                    continue
                
                # Get first character this token decodes to
                first_char, num_chars = self._token_to_first_char(token_id)
                
                if first_char is None:
                    continue
                
                # Skip control characters and replacement character
                if ord(first_char) < 32 or first_char == '\ufffd':
                    continue
                
                # Accumulate log-probability for this character
                if first_char not in char_log_probs:
                    char_log_probs[first_char] = []
                char_log_probs[first_char].append(log_prob)
            
            # Step 4: Marginalize using logsumexp
            char_probs = {}
            for char, log_prob_list in char_log_probs.items():
                if log_prob_list:
                    char_probs[char] = logsumexp(log_prob_list)
            
            # Step 5: Rank and get top-3
            sorted_chars = sorted(char_probs.items(), key=lambda x: x[1], reverse=True)
            
            if debug:
                print(f"\n=== Character probabilities (top 10) ===")
                for i, (char, log_prob) in enumerate(sorted_chars[:10]):
                    prob = np.exp(log_prob)
                    print(f"  {i+1}. '{char}' (U+{ord(char):04X}): log_prob={log_prob:.4f}, prob={prob:.4f}")
            
            top_chars = [c for c, _ in sorted_chars[:3]]
            
            # Step 6: Fill gaps with common characters if needed
            if len(top_chars) < 3:
                # Common characters in order of frequency
                fallback = [" ", ".", "e", "t", "a", "o", "i", "n", "s", "h", "r"]
                for fb_char in fallback:
                    if len(top_chars) >= 3:
                        break
                    if fb_char not in top_chars:
                        top_chars.append(fb_char)
            
            # Ensure exactly 3 characters
            while len(top_chars) < 3:
                top_chars.append("e")
            
            # Final validation: ensure no invalid characters
            result = []
            for char in top_chars[:3]:
                if char and char != '\ufffd' and ord(char) >= 32:
                    result.append(char)
            
            # If we lost characters due to filtering, refill
            while len(result) < 3:
                for fb in [" ", ".", "e", "t", "a", "o", "i"]:
                    if fb not in result:
                        result.append(fb)
                        break
                if len(result) >= 3:
                    break
            
            return "".join(result[:3])
            
        except Exception as e:
            import traceback
            print(f"Error in _next_char_top3 for text '{text[:50]}...': {e}")
            if debug:
                traceback.print_exc()
            return "eta"

    def run_pred(self, data, top_k_tokens=5000):
        """Run prediction on a list of input strings."""
        preds = []
        prediction_times = []
        
        print(f"Running predictions on {len(data)} inputs...")
        for i, inp in enumerate(data):
            start_time = time.time()
            top3 = self._next_char_top3(inp, top_k_tokens=top_k_tokens)
            elapsed_time = time.time() - start_time
            prediction_times.append(elapsed_time)
            preds.append(top3)
            
            # Print progress every 10 items
            if (i + 1) % 10 == 0 or (i + 1) == len(data):
                avg_time = sum(prediction_times) / len(prediction_times)
                print(f"  Processed {i + 1}/{len(data)} inputs (avg: {avg_time*1000:.2f}ms per prediction)")
        
        # Calculate and print statistics
        if prediction_times:
            avg_time = sum(prediction_times) / len(prediction_times)
            min_time = min(prediction_times)
            max_time = max(prediction_times)
            total_time = sum(prediction_times)
            
            print(f"\n=== Prediction Timing Statistics ===")
            print(f"  Total predictions: {len(prediction_times)}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average time per prediction: {avg_time*1000:.2f}ms ({avg_time:.4f}s)")
            print(f"  Min time: {min_time*1000:.2f}ms ({min_time:.4f}s)")
            print(f"  Max time: {max_time*1000:.2f}ms ({max_time:.4f}s)")
            print(f"  Throughput: {len(prediction_times)/total_time:.2f} predictions/second")
        
        return preds

    def run_train(self, data, work_dir):
        """Download and save Qwen model to work_dir."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, QWEN_SUBDIR)
        
        if os.path.exists(os.path.join(save_path, "config.json")):
            print(f"Qwen3-4B-Base already at {save_path}, skipping download.")
            return
        
        print(f"Downloading Qwen3-4B-Base to {save_path} ...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(QWEN_HF_ID, trust_remote_code=True)
        
        # Download and save full-precision model (will be quantized during inference)
        print("Downloading full-precision model (will be quantized during inference)...")
        device = _get_device()
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_HF_ID,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Save tokenizer and FULL model weights locally
        print(f"Saving tokenizer and full-precision model to {save_path}...")
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)  # Save full weights (~8GB)
        
        # Save quantization flag (indicates we want quantization during inference)
        quantize_flag_path = os.path.join(save_path, ".quantize_4bit")
        with open(quantize_flag_path, "w") as f:
            f.write("true\n")
        
        print(f"Saved full-precision model to {save_path} (~8GB)")
        print("Note: Model will be loaded with 4-bit quantization during inference (saves GPU memory).")

    def run_finetune(
        self,
        train_data_path,
        work_dir,
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
        merge_lora_into_base=False,
    ):
        """
        Fine-tune with LoRA and SFTTrainer.

        If merge_lora_into_base is True and the underlying model supports merge_and_unload,
        the merged full-precision model will be saved into work_dir/output_subdir_name.
        Otherwise, LoRA adapters are saved separately under work_dir/QWEN_SUBDIR/lora_adapters.
        """
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTConfig, SFTTrainer

        def causal_lm_loss_from_logits(outputs, labels, num_items_in_batch=None):
            """Compute causal LM cross-entropy from logits."""
            if labels is None:
                raise ValueError("Labels are required for training.")
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, QWEN_SUBDIR)
        checkpoint_dir = os.path.join(work_dir, "qwen3-4b-base-finetune-checkpoints")

        print(f"Loading corpus from {train_data_path} and splitting 95% train / 5% val ...")
        train_ds, eval_ds = _load_corpus_and_split(train_data_path, val_ratio=0.05)
        print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

        target_modules = lora_target_modules if lora_target_modules is not None else DEFAULT_LORA_TARGET_MODULES
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha}, target_modules={target_modules}) ...")
        self.model = get_peft_model(self.model, lora_config)

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
            max_length=max_seq_length,
            dataset_text_field="text",
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
        print("Training (best checkpoint by eval loss will be saved) ...")
        trainer.train()
        
        # If requested and supported, merge LoRA into base and save a full-precision variant.
        if merge_lora_into_base and output_subdir_name is not None:
            variant_dir = os.path.join(work_dir, output_subdir_name)
            try:
                if hasattr(self.model, "merge_and_unload"):
                    print(f"Merging LoRA adapters into base Qwen and saving to {variant_dir} ...")
                    merged_model = self.model.merge_and_unload()
                    os.makedirs(variant_dir, exist_ok=True)
                    merged_model.save_pretrained(variant_dir)
                    self.tokenizer.save_pretrained(variant_dir)
                    print(f"Fine-tuning done. Full-precision Qwen variant saved to {variant_dir}.")
                else:
                    print(
                        "merge_and_unload not available on Qwen model; "
                        "saving LoRA adapters instead."
                    )
                    lora_save_path = os.path.join(save_path, "lora_adapters")
                    from peft import PeftModel
                    if hasattr(trainer.state, "best_model_checkpoint") and trainer.state.best_model_checkpoint:
                        best_checkpoint = trainer.state.best_model_checkpoint
                        print(
                            f"Saving best LoRA adapters from {best_checkpoint} to {lora_save_path}..."
                        )
                        best_model = PeftModel.from_pretrained(self.model, best_checkpoint)
                        best_model.save_pretrained(lora_save_path)
                    else:
                        print(f"Saving LoRA adapters to {lora_save_path}...")
                        self.model.save_pretrained(lora_save_path)
                    self.tokenizer.save_pretrained(save_path)
                    print(
                        "Fine-tuning done. LoRA adapters saved to {} (base model will be "
                        "reloaded with quantization during inference).".format(
                            lora_save_path
                        )
                    )
            finally:
                if os.path.isdir(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
        else:
            # For quantized models, we can't merge LoRA adapters directly
            # Instead, we save the LoRA adapters separately
            lora_save_path = os.path.join(save_path, "lora_adapters")
            
            # Save the best model's LoRA adapters
            if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
                best_checkpoint = trainer.state.best_model_checkpoint
                print(f"Saving best LoRA adapters from {best_checkpoint} to {lora_save_path}...")
                # Load best checkpoint and save adapters
                from peft import PeftModel
                best_model = PeftModel.from_pretrained(self.model, best_checkpoint)
                best_model.save_pretrained(lora_save_path)
            else:
                # Fallback: save current adapters
                print(f"Saving LoRA adapters to {lora_save_path}...")
                self.model.save_pretrained(lora_save_path)
            
            self.tokenizer.save_pretrained(save_path)
            print(f"Fine-tuning done. LoRA adapters saved to {lora_save_path}.")
            print(f"Note: Base model will be reloaded with quantization during inference, then LoRA adapters will be applied.")
            if os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, QWEN_SUBDIR)
        self.tokenizer.save_pretrained(save_path)
        # Quantized models can't be saved directly, so we save config
        self.model.config.save_pretrained(save_path)

    @classmethod
    def load(cls, work_dir, force_quantize=None, variant_subdir=None):
        """
        Load Qwen model from work_dir.
        
        Args:
            work_dir: Directory containing the model
            force_quantize: If True, force quantization. If False, force no quantization.
                           If None, use flag file or default (quantize if CUDA available).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Base Qwen path (raw weights)
        save_path = os.path.join(work_dir, QWEN_SUBDIR)
        
        # If a variant directory is requested and exists, prefer that as the model path.
        variant_path = None
        if variant_subdir is not None:
            candidate = os.path.join(work_dir, variant_subdir)
            if os.path.exists(os.path.join(candidate, "config.json")):
                variant_path = candidate

        # Check if model weights exist locally
        has_local_config = os.path.exists(os.path.join(save_path, "config.json"))
        
        # Check if actual model weights exist (safetensors files)
        has_model_weights = False
        if has_local_config:
            import glob
            safetensors_files = glob.glob(os.path.join(save_path, "*.safetensors"))
            bin_files = glob.glob(os.path.join(save_path, "*.bin"))
            has_model_weights = len(safetensors_files) > 0 or len(bin_files) > 0
        
        # Check quantization preference flag
        quantize_flag_path = os.path.join(save_path, ".quantize_4bit")
        use_quantization_flag = os.path.exists(quantize_flag_path)
        
        # Determine quantization preference
        device = _get_device()
        if force_quantize is not None:
            # Explicit override from command line
            use_quantization = force_quantize and device == "cuda"
        else:
            # Use flag file preference, or default to quantization if CUDA available
            use_quantization = (use_quantization_flag or device == "cuda") and device == "cuda"
        
        # Prefer variant model if provided, else local base model if weights exist, otherwise load from HuggingFace
        if variant_path is not None:
            model_path = variant_path
            print(f"Loading Qwen variant from {variant_path} ...")
        elif has_model_weights:
            # Local full-precision model exists - load from local path
            model_path = save_path
            if use_quantization:
                print(f"Loading model from local path: {save_path} (will apply 4-bit quantization)")
            else:
                print(f"Loading model from local path: {save_path} (full precision)")
        else:
            # No local weights - load from HuggingFace
            model_path = QWEN_HF_ID
            if has_local_config:
                print(f"Model weights not found at {save_path}, loading from {QWEN_HF_ID}...")
            else:
                print(f"Model not found at {save_path}, loading from {QWEN_HF_ID} (run download_qwen.py or train first).")
        
        # Load tokenizer (prefer local if available, otherwise from HF)
        tokenizer_path = save_path if has_local_config else QWEN_HF_ID
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        if use_quantization:
            print(f"Loading Qwen3-4B-Base with 4-bit quantization from {model_path}...")
            print(f"  GPU memory usage: ~2.3GB (quantized)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            print(f"Loading Qwen3-4B-Base in full precision (device={device})...")
            print(f"  GPU memory usage: ~8GB (full precision)")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
        
        # Check for LoRA adapters
        lora_path = os.path.join(save_path, "lora_adapters")
        if os.path.exists(lora_path):
            print(f"Loading LoRA adapters from {lora_path}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
        
        return cls(model=model, tokenizer=tokenizer, device=device)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="train: download/save Qwen; test: load and predict")
    parser.add_argument("--work_dir", default="work", help="where to load/save model")
    parser.add_argument("--test_data", default="example/input.txt", help="path to test input")
    parser.add_argument("--test_output", default="pred.txt", help="path to write predictions")
    # Fine-tuning
    parser.add_argument("--finetune", action="store_true", help="run LoRA fine-tuning; requires --train_data")
    parser.add_argument("--train_data", default="work/data/belle_conversation_20k.txt", help="path to training corpus")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (default 8)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default 16)")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="comma-separated LoRA target modules")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default 0.05)")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max sequence length (default 2048)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="train batch size (default 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation (default 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate (default 1e-5)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="number of epochs (default 1)")
    parser.add_argument("--eval_steps", type=int, default=500, help="eval every N steps (default 500)")
    parser.add_argument("--save_steps", type=int, default=1000, help="save every N steps (default 1000)")
    parser.add_argument("--logging_steps", type=int, default=50, help="log every N steps (default 50)")
    # Prediction parameters
    parser.add_argument("--top_k_tokens", type=int, default=5000, help="top-k tokens to consider (default 5000)")
    parser.add_argument("--debug", action="store_true", help="enable debug output")
    # Quantization control
    parser.add_argument("--no_quantize", action="store_true", help="disable 4-bit quantization, use full precision model (requires more GPU memory)")
    parser.add_argument("--quantize", action="store_true", help="enable 4-bit quantization (default if CUDA available)")
    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print(f"Making working directory {args.work_dir}")
            os.makedirs(args.work_dir)
        if args.finetune:
            if not args.train_data or not os.path.isfile(args.train_data):
                raise SystemExit("--finetune requires --train_data pointing to an existing corpus file")
            # For fine-tuning, determine quantization preference
            if args.no_quantize:
                force_quantize = False
            elif args.quantize:
                force_quantize = True
            else:
                force_quantize = None  # Use default
            print("Loading model for fine-tuning...")
            model = QwenModel.load(args.work_dir, force_quantize=force_quantize)
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
            print("Instantiating model (download-only)")
            # For download, we always download full precision (quantization happens during inference)
            model = QwenModel.load(args.work_dir, force_quantize=False)
            print("Loading training data (optional)")
            train_data = QwenModel.load_training_data()
            print("Training (download/save only)")
            model.run_train(train_data, args.work_dir)
            print("Saving model")
            model.save(args.work_dir)
    elif args.mode == "test":
        # Determine quantization preference from command line
        if args.no_quantize:
            force_quantize = False
            print("Quantization disabled (--no_quantize flag)")
        elif args.quantize:
            force_quantize = True
            print("Quantization enabled (--quantize flag)")
        else:
            force_quantize = None  # Use default (flag file or auto-detect)
        
        print("Loading Qwen3-4B-Base model...")
        model = QwenModel.load(args.work_dir, force_quantize=force_quantize)
        print(f"Loading test data from {args.test_data}")
        test_data = QwenModel.load_test_data(args.test_data)
        print(f"Making predictions with top_k_tokens={args.top_k_tokens}")
        pred = model.run_pred(test_data, top_k_tokens=args.top_k_tokens)
        print(f"Writing predictions to {args.test_output}")
        assert len(pred) == len(test_data), f"Expected {len(test_data)} predictions but got {len(pred)}"
        QwenModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f"Unknown mode {args.mode}")
