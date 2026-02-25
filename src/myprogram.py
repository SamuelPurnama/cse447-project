#!/usr/bin/env python
"""
CSE447 character prediction: hybrid Bolmo + Qwen model.
English inputs use Bolmo-1B byte-level beam search.
Non-English inputs use Qwen full-precision next-token marginalization.
Fine-tuning supports language-focused corpora for both models.
"""
import os
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Bolmo model subdir under work_dir (same as download_bolmo.py)
BOLMO_SUBDIR = "bolmo-1b"
BOLMO_HF_ID = "allenai/Bolmo-1B"

# Qwen for non-English next-character prediction
QWEN_SUBDIR = "qwen3-4b-base"

# Default tagged variant names (flat layout under work_dir)
BOLMO_INITIAL_FINETUNE_SUBDIR = "bolmo-InitialFineTune"
QWEN_FINETUNE_NON_EN_SUBDIR = "qwen3-4b-nonEnglish-v1"

# Default LoRA target modules (override with --lora_target_modules if your model uses different names)
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# ========== UTF-8 helpers ==========

def utf8_expected_continuation_bytes(first_byte):
    """
    Return expected total character length (1-4) from the first byte of a UTF-8 sequence.
    Returns 0 if the byte is invalid as a lead byte.
    """
    if first_byte < 0x80:
        return 1  # ASCII
    elif 0xC2 <= first_byte <= 0xDF:
        return 2  # 2-byte character
    elif 0xE0 <= first_byte <= 0xEF:
        return 3  # 3-byte character
    elif 0xF0 <= first_byte <= 0xF4:
        return 4  # 4-byte character
    else:
        return 0  # Invalid lead byte


def utf8_valid_continuation_byte(prev_bytes, b):
    """
    Check if byte b is a valid continuation given prev_bytes.
    Returns True iff prev_bytes + [b] forms a valid UTF-8 prefix.
    """
    if not prev_bytes:
        return utf8_expected_continuation_bytes(b) > 0
    
    expected_len = utf8_expected_continuation_bytes(prev_bytes[0])
    if expected_len == 0:
        return False
    if len(prev_bytes) >= expected_len:
        return False  # Already complete
    # Continuation bytes must be in range 0x80-0xBF
    if not (0x80 <= b <= 0xBF):
        return False
    
    # Additional validation for specific lead bytes
    if len(prev_bytes) == 1:
        first = prev_bytes[0]
        if first == 0xE0 and b < 0xA0:
            return False  # Overlong encoding
        if first == 0xED and b >= 0xA0:
            return False  # UTF-16 surrogates
        if first == 0xF0 and b < 0x90:
            return False  # Overlong encoding
        if first == 0xF4 and b >= 0x90:
            return False  # Above U+10FFFF
    
    return True


def utf8_decode_single(bytes_list):
    """
    Decode one UTF-8 character from the start of bytes_list.
    Returns (character, num_bytes_consumed) if valid, or (None, 0) if invalid/incomplete.
    Rejects overlong encodings.
    """
    if not bytes_list:
        return (None, 0)
    
    try:
        byte_seq = bytes(bytes_list)
        decoded = byte_seq.decode('utf-8', errors='strict')
        if decoded:
            # Check for overlong by re-encoding and comparing byte length
            char = decoded[0]
            re_encoded = char.encode('utf-8')
            if len(re_encoded) != len(bytes_list):
                # Either incomplete or overlong
                if len(bytes_list) < len(re_encoded):
                    return (None, 0)  # Incomplete
                else:
                    return (None, 0)  # Overlong
            return (char, len(re_encoded))
    except (UnicodeDecodeError, UnicodeError):
        return (None, 0)
    
    return (None, 0)


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


def _load_corpus_streaming(train_data_path, eval_size=50000):
    """
    Load corpus for training on whole dataset without loading file into RAM.
    Returns (train_ds, eval_ds) where train_ds is an IterableDataset (one line at a time)
    and eval_ds is a small in-memory Dataset from the first eval_size lines.
    If eval_size is 0, eval_ds is None (no evaluation during training).
    """
    from datasets import Dataset, IterableDataset

    # Eval: load first eval_size lines into memory (bounded RAM)
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

    # Train: stream the rest of the file (skip first eval_size lines)
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
    """Count non-empty lines after the first eval_size lines (for max_steps when streaming)."""
    count = 0
    with open(train_data_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < eval_size:
                continue
            if line.rstrip("\n").strip():
                count += 1
    return count


def _is_english_text(text):
    """Best-effort English detector with langdetect + ASCII fallback."""
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 3:
        return all(ord(c) < 128 for c in stripped)
    try:
        import langdetect
        return langdetect.detect(stripped) == "en"
    except Exception:
        ascii_chars = sum(1 for c in stripped if ord(c) < 128)
        return (ascii_chars / max(1, len(stripped))) > 0.9


class MyModel:
    """
    Hybrid model:
    - Bolmo-1B byte-level causal LM for English next-character prediction.
    - Qwen for non-English next-character prediction.
    """

    def __init__(self, model, tokenizer, device, qwen_model=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.qwen_model = qwen_model

    def _detect_language(self, text):
        """
        Detect if text is English using langdetect.
        Returns True if English, False otherwise.
        Defaults to True (English/Bolmo) for empty/short/error cases.
        """
        return _is_english_text(text)

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
        # Phase 1: download/save both Bolmo and Qwen to work_dir (raw models only).
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from myprogram_qwen import QwenModel
        os.makedirs(work_dir, exist_ok=True)

        # Bolmo download
        save_path = os.path.join(work_dir, BOLMO_SUBDIR)
        if os.path.exists(os.path.join(save_path, "config.json")):
            print("Bolmo-1B already at {}, skipping download.".format(save_path))
        else:
            print("Downloading Bolmo-1B to {} ...".format(save_path))
            tokenizer = AutoTokenizer.from_pretrained(BOLMO_HF_ID, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(BOLMO_HF_ID, trust_remote_code=True)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
            print("Saved to {}.".format(save_path))

        # Qwen download (full precision saved locally)
        qwen_save_path = os.path.join(work_dir, QWEN_SUBDIR)
        if os.path.exists(os.path.join(qwen_save_path, "config.json")):
            print("Qwen already at {}, skipping download.".format(qwen_save_path))
        else:
            print("Downloading Qwen to {} ...".format(qwen_save_path))
            qwen_model = QwenModel.load(work_dir, force_quantize=False)
            qwen_model.run_train([], work_dir)

        print("Raw Bolmo (en) and Qwen (multi-language) models are available in work_dir.")

    def run_finetune(
        self,
        train_data_path,
        work_dir,
        qwen_train_data_path=None,
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
        bolmo_english_only=True,
        qwen_non_english_only=True,
    ):
        """
        Fine-tune Bolmo (English-focused) and Qwen (non-English-focused).
        Bolmo uses the existing LoRA + SFTTrainer flow.
        Qwen uses the Qwen-specific LoRA flow from myprogram_qwen.py.
        """
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTConfig, SFTTrainer
        from prepare_datasets import build_language_split

        def causal_lm_loss_from_logits(outputs, labels, num_items_in_batch=None):
            """Compute causal LM cross-entropy from logits when the model does not return loss (e.g. Bolmo)."""
            if labels is None:
                raise ValueError("Labels are required for training.")
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        os.makedirs(work_dir, exist_ok=True)
        # Raw Bolmo always lives in BOLMO_SUBDIR; finetuned variants go to sibling subdirs.
        raw_bolmo_path = os.path.join(work_dir, BOLMO_SUBDIR)
        checkpoint_dir = os.path.join(work_dir, "bolmo-1b-finetune-checkpoints")
        split_dir = os.path.join(work_dir, "data", "language_splits")
        os.makedirs(split_dir, exist_ok=True)

        # Build language-focused corpora
        bolmo_corpus = train_data_path
        if bolmo_english_only:
            # Build an English-only corpus for Bolmo from the mixed file.
            bolmo_en_path, _ = build_language_split(
                train_data_path, work_dir, prefix="bolmo"
            )
            if os.path.isfile(bolmo_en_path):
                bolmo_corpus = bolmo_en_path
            else:
                print(
                    "Warning: English corpus not found; using original Bolmo corpus."
                )

        if streaming:
            print("Loading Bolmo corpus in streaming mode from {} (eval = first {} lines) ...".format(
                bolmo_corpus, streaming_eval_size))
            num_train_lines = _count_training_lines_streaming(bolmo_corpus, eval_size=streaming_eval_size)
            train_ds, eval_ds = _load_corpus_streaming(bolmo_corpus, eval_size=streaming_eval_size)
            # Trainer requires max_steps when train_dataset has no __len__ (IterableDataset)
            batch_size = per_device_train_batch_size * gradient_accumulation_steps
            max_steps = max(1, num_train_lines // batch_size)
            print("Train: iterable ({} lines). Eval size: {}. max_steps={}.".format(
                num_train_lines, len(eval_ds) if eval_ds else 0, max_steps))
        else:
            num_train_lines = None
            max_steps = -1
            print("Loading Bolmo corpus from {} and splitting 95%% train / 5%% val ...".format(bolmo_corpus))
            train_ds, eval_ds = _load_corpus_and_split(bolmo_corpus, val_ratio=0.05)
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

        use_eval = eval_ds is not None and len(eval_ds) > 0
        # When streaming, IterableDataset has no __len__ so we must set max_steps
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
            pad_to_multiple_of=BOLMO_MLSTM_CHUNK_SIZE,
            label_names=["labels"],
        )
        if streaming and max_steps > 0:
            sft_config_kw["max_steps"] = max_steps
            sft_config_kw["num_train_epochs"] = 1  # ignored when max_steps set, but avoid confusion
        sft_config = SFTConfig(**sft_config_kw)

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=eval_ds if use_eval else None,
            processing_class=self.tokenizer,
            args=sft_config,
            compute_loss_func=causal_lm_loss_from_logits,
        )
        print("Training (best checkpoint by eval loss will be merged and saved) ...")
        trainer.train()
        # Merge LoRA into base raw Bolmo and save into a tagged variant directory.
        print("Merging LoRA into base Bolmo and saving variant ...")
        self.model = self.model.merge_and_unload()
        # Determine variant path: keep existing initial finetune name as a default target.
        bolmo_variant_dir = os.path.join(work_dir, BOLMO_INITIAL_FINETUNE_SUBDIR)
        os.makedirs(bolmo_variant_dir, exist_ok=True)
        self.tokenizer.save_pretrained(bolmo_variant_dir)
        self.model.save_pretrained(bolmo_variant_dir)
        print(
            "Fine-tuned Bolmo variant saved to {} (raw model remains in {}).".format(
                bolmo_variant_dir, raw_bolmo_path
            )
        )
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        print("Fine-tuning done.")

        # Qwen fine-tune stage (full precision load, non-English-focused)
        if qwen_train_data_path is not None and os.path.isfile(qwen_train_data_path):
            from myprogram_qwen import QwenModel
            qwen_corpus = qwen_train_data_path
            if qwen_non_english_only:
                # Build a non-English-only corpus for Qwen from the mixed file.
                _, qwen_non_en_path = build_language_split(
                    qwen_train_data_path, work_dir, prefix="qwen"
                )
                if os.path.isfile(qwen_non_en_path):
                    qwen_corpus = qwen_non_en_path
                else:
                    print(
                        "Warning: no non-English lines found; falling back to original Qwen corpus."
                    )
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            print("Loading Qwen for fine-tuning (full precision, non-English-focused) ...")
            qwen_model = QwenModel.load(work_dir, force_quantize=False, variant_subdir=None)
            print("Fine-tuning Qwen on {} ...".format(qwen_corpus))
            qwen_model.run_finetune(
                qwen_corpus,
                work_dir,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_target_modules=None,
                lora_dropout=lora_dropout,
                max_seq_length=max(1024, max_seq_length),
                per_device_train_batch_size=max(1, min(4, per_device_train_batch_size)),
                gradient_accumulation_steps=max(1, gradient_accumulation_steps),
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                eval_steps=eval_steps,
                save_steps=save_steps,
                logging_steps=logging_steps,
                output_subdir_name=QWEN_FINETUNE_NON_EN_SUBDIR,
                merge_lora_into_base=True,
            )
            print("Qwen fine-tuning complete.")
        else:
            print("Skipping Qwen fine-tuning (no valid --qwen_train_data provided).")

    def run_pred(self, data, beam_width=8, qwen_top_k_tokens=5000):
        import torch
        self.model.eval()
        if self.qwen_model is not None:
            self.qwen_model.model.eval()
        preds = []
        with torch.no_grad():
            for inp in data:
                # Language detection: English -> Bolmo, non-English -> Qwen
                is_english = self._detect_language(inp)
                if is_english or self.qwen_model is None:
                    # Use Bolmo-1B (causal LM)
                    top3 = self._next_char_top3(inp, beam_width=beam_width)
                else:
                    # Use Qwen tokenizer-based next-character marginalization.
                    top3 = self.qwen_model._next_char_top3(inp, top_k_tokens=qwen_top_k_tokens)
                preds.append(top3)
        return preds

    def _byte_to_token_id(self, byte_val):
        """Convert byte value (0-255) to Bolmo token ID. Assumes Bolmo uses 4-259 for bytes 0x00-0xFF; verify with tokenizer if predictions are wrong."""
        return 4 + byte_val
    
    def _token_id_to_byte(self, token_id):
        """Convert Bolmo token ID to byte value, or None if not a byte token."""
        if 4 <= token_id < 260:
            return token_id - 4
        return None
    
    def _get_next_byte_logprobs(self, input_ids, past_key_values=None, return_cache=False):
        """
        Get log-probabilities for the next byte (as tokens 4-259).
        Returns log_probs over bytes (256 entries), optionally with cache.
        input_ids: shape (1, L)
        past_key_values: KV-cache from previous forward pass  
        return_cache: if True, return (log_probs, cache); else just log_probs
        
        NOTE: Bolmo has issues with single-token sequences, so we enable cache only for longer sequences.
        """
        import torch
        import torch.nn.functional as F
        
        # Enable KV-cache for sequences longer than 2 tokens (Bolmo requirement)
        use_cache_flag = return_cache and input_ids.size(1) >= 2
        
        try:
            outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=use_cache_flag)
            logits = outputs.logits[0, -1, :]  # Last position, shape (vocab_size,)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Extract only byte token log-probs (Bolmo: tokens 4-259 = bytes 0x00-0xFF)
            byte_log_probs = log_probs[4:260]  # Shape (256,)
            
            if return_cache:
                cache = outputs.past_key_values if use_cache_flag and hasattr(outputs, 'past_key_values') else None
                return byte_log_probs, cache
            else:
                return byte_log_probs
                
        except Exception as e:
            # Fallback: run without cache if there's an error
            try:
                outputs = self.model(input_ids, use_cache=False)
                logits = outputs.logits[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                byte_log_probs = log_probs[4:260]  # tokens 4-259 = bytes 0x00-0xFF
                if return_cache:
                    return byte_log_probs, None
                else:
                    return byte_log_probs
            except Exception as e2:
                # If all else fails, return fallback
                import torch
                byte_log_probs = torch.full((256,), -10.0, device=self.device)
                if return_cache:
                    return byte_log_probs, None
                else:
                    return byte_log_probs

    def _next_char_top3(self, text, beam_width=8, debug=False):
        """
        Phase 4: Byte→character with beam search + marginalization.
        Returns 3 character guesses for the next character after text.
        """
        import torch
        from scipy.special import logsumexp
        
        if not text:
            return "eta"  # frequency fallback
        
        try:
            # Encode text
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(self.device)
            if input_ids.size(0) == 0 or input_ids.size(1) == 0:
                return "eta"
            
            # Step 1: Get initial byte log-probs with cache
            log_probs, cache_text = self._get_next_byte_logprobs(input_ids, return_cache=True)
            initial_probs = log_probs.cpu()  # Save for filling gaps later
            
            # Step 2: Build initial beam
            # Hypothesis: (byte_list, log_prob, past_key_values, is_complete)
            hypotheses = []
            for byte_val in range(256):
                expected_len = utf8_expected_continuation_bytes(byte_val)
                if expected_len == 0:
                    continue  # Invalid lead byte
                
                log_prob = log_probs[byte_val].item()
                
                if expected_len == 1:
                    # Single-byte character (complete)
                    hypotheses.append(([byte_val], log_prob, None, True))
                else:
                    # Multi-byte character (incomplete) - will need cache to extend
                    hypotheses.append(([byte_val], log_prob, cache_text, False))
            
            # Sort by log_prob and keep top beam_width
            hypotheses.sort(key=lambda h: h[1], reverse=True)
            hypotheses = hypotheses[:beam_width]
            
            # Separate completed vs incomplete
            completed = [h for h in hypotheses if h[3]]
            beam = [h for h in hypotheses if not h[3]]
            
            # Safety check: if beam is empty (no valid UTF-8 lead bytes in top predictions)
            # jump directly to gap-filling with single-byte characters
            if not hypotheses:
                top_chars = []
                # Use single-byte characters from initial_probs
                single_byte_chars = []
                for byte_val in range(256):
                    if utf8_expected_continuation_bytes(byte_val) == 1:
                        char, _ = utf8_decode_single([byte_val])
                        if char is not None and ord(char) >= 32 and char != '\ufffd':
                            single_byte_chars.append((char, initial_probs[byte_val].item()))
                
                single_byte_chars.sort(key=lambda x: x[1], reverse=True)
                for char, _ in single_byte_chars[:3]:
                    top_chars.append(char)
                
                # Fallback if still not enough
                fallback = [" ", ".", "e", "t", "a"]
                for fb_char in fallback:
                    if len(top_chars) >= 3:
                        break
                    if fb_char not in top_chars:
                        top_chars.append(fb_char)
                
                while len(top_chars) < 3:
                    top_chars.append("e")
                
                return "".join(top_chars[:3])
            
            # Step 3: Beam extension (bytes 2-4) with KV-caching
            # Strategy: Use cache from previous step when possible.
            # For first extension (depth=2), use cache_text from Step 1.
            # For deeper extensions, use the cache from the previous depth.
            max_depth = 4
            for depth in range(2, max_depth + 1):
                if not beam:
                    break
                
                new_beam = []
                
                # Process each hypothesis in the beam
                # Group by prefix to avoid redundant computations
                prefix_results = {}
                
                for byte_list, log_prob, past_kv, _ in beam:
                    prefix_key = tuple(byte_list)
                    
                    # Check if we've already computed log-probs for this prefix
                    if prefix_key not in prefix_results:
                        # When past_kv is None (e.g. single-token input), we must use full sequence
                        # so the model sees full text + byte prefix; incremental would lose context.
                        if past_kv is None:
                            byte_token_ids = [self._byte_to_token_id(b) for b in byte_list]
                            extended_input_ids = torch.cat([
                                input_ids,
                                torch.tensor([byte_token_ids], device=self.device)
                            ], dim=1)
                            next_log_probs, new_cache = self._get_next_byte_logprobs(
                                extended_input_ids,
                                past_key_values=None,
                                return_cache=True
                            )
                        else:
                            # Try incremental extension with KV-cache
                            last_byte = byte_list[-1]
                            token_id = self._byte_to_token_id(last_byte)
                            next_input_ids = torch.tensor([[token_id]], device=self.device)
                            try:
                                next_log_probs, new_cache = self._get_next_byte_logprobs(
                                    next_input_ids,
                                    past_key_values=past_kv,
                                    return_cache=True
                                )
                            except Exception:
                                # Fallback: rebuild full sequence if KV-cache fails
                                byte_token_ids = [self._byte_to_token_id(b) for b in byte_list]
                                extended_input_ids = torch.cat([
                                    input_ids,
                                    torch.tensor([byte_token_ids], device=self.device)
                                ], dim=1)
                                next_log_probs, new_cache = self._get_next_byte_logprobs(
                                    extended_input_ids,
                                    past_key_values=None,
                                    return_cache=True
                                )
                        prefix_results[prefix_key] = (next_log_probs, new_cache)
                    else:
                        next_log_probs, new_cache = prefix_results[prefix_key]
                    
                    # Extend with valid continuation bytes
                    for next_byte in range(256):
                        if not utf8_valid_continuation_byte(byte_list, next_byte):
                            continue
                        
                        new_byte_list = byte_list + [next_byte]
                        new_log_prob = log_prob + next_log_probs[next_byte].item()
                        
                        # Check if complete
                        expected_len = utf8_expected_continuation_bytes(byte_list[0])
                        is_complete = len(new_byte_list) == expected_len
                        
                        if is_complete:
                            # Decode and validate
                            char, consumed = utf8_decode_single(new_byte_list)
                            if char is not None and consumed == len(new_byte_list):
                                completed.append((new_byte_list, new_log_prob, None, True))
                        else:
                            # Incomplete - keep cache for further extension
                            new_beam.append((new_byte_list, new_log_prob, new_cache, False))
                
                # Keep top beam_width incomplete hypotheses
                new_beam.sort(key=lambda h: h[1], reverse=True)
                beam = new_beam[:beam_width]
            
            # Step 4: Marginalization
            # Group completed hypotheses by character
            char_log_probs = {}
            for byte_list, log_prob, _, _ in completed:
                char, consumed = utf8_decode_single(byte_list)
                if char is not None and consumed == len(byte_list):
                    # Skip replacement character and control characters
                    if ord(char) < 32 or char == '\ufffd':
                        continue
                    if char not in char_log_probs:
                        char_log_probs[char] = []
                    char_log_probs[char].append(log_prob)
            
            # Sum probabilities per character using logsumexp
            char_probs = {}
            for char, log_prob_list in char_log_probs.items():
                char_probs[char] = logsumexp(log_prob_list)
            
            # Step 5: Rank and get top-3
            sorted_chars = sorted(char_probs.items(), key=lambda x: x[1], reverse=True)
            
            if debug:
                print(f"\n=== Character probabilities (top 10) ===")
                for i, (char, log_prob) in enumerate(sorted_chars[:10]):
                    prob = torch.exp(torch.tensor(log_prob)).item()
                    print(f"  {i+1}. '{char}' (U+{ord(char):04X}): log_prob={log_prob:.4f}, prob={prob:.4f}")
            
            top_chars = [c for c, _ in sorted_chars[:3]]
            
            # Step 6: Fill gaps with single-byte characters from initial_probs
            if len(top_chars) < 3:
                # Get single-byte characters (ASCII)
                single_byte_chars = []
                for byte_val in range(256):
                    if utf8_expected_continuation_bytes(byte_val) == 1:
                        char, _ = utf8_decode_single([byte_val])
                        if char is not None and char not in top_chars:
                            # Skip control characters and replacement character
                            if ord(char) < 32 or char == '\ufffd':
                                continue
                            single_byte_chars.append((char, initial_probs[byte_val].item()))
                
                single_byte_chars.sort(key=lambda x: x[1], reverse=True)
                for char, _ in single_byte_chars:
                    if len(top_chars) >= 3:
                        break
                    if char not in top_chars:
                        top_chars.append(char)
            
            # Final fallback with common characters
            fallback = [" ", ".", "e", "t", "a"]
            for fb_char in fallback:
                if len(top_chars) >= 3:
                    break
                if fb_char not in top_chars:
                    top_chars.append(fb_char)
            
            # Ensure exactly 3 characters (should never get here, but just in case)
            while len(top_chars) < 3:
                top_chars.append("e")
            
            # Final validation: ensure no invalid characters
            result = []
            for char in top_chars[:3]:
                if char and char != '\ufffd' and ord(char) >= 32:
                    result.append(char)
            
            # If we lost characters due to filtering, refill with fallback
            while len(result) < 3:
                for fb in [" ", ".", "e", "t", "a", "n"]:
                    if fb not in result:
                        result.append(fb)
                        break
                if len(result) >= 3:
                    break
            
            return "".join(result[:3])
            
        except Exception as e:
            # Fallback on any error
            import traceback
            print("Error in _next_char_top3 for text '{}': {}".format(text, e))
            traceback.print_exc()
            return "eta"

    def _next_char_top3_byt5(self, text, beam_width=8, debug=False, greedy=False):
        """
        ByT5-Small encoder-decoder for next-character prediction (non-English).
        If greedy=True: no beam, score each character by greedy byte continuation (pure ByT5).
        Else: beam search + marginalization over UTF-8 bytes.
        Returns 3 character guesses for the next character after text.
        """
        import torch
        import torch.nn.functional as F
        from scipy.special import logsumexp
        
        if not text:
            return "eta"  # frequency fallback
        
        if self.byt5_model is None or self.byt5_tokenizer is None:
            return "eta"  # Fallback if ByT5 not loaded
        
        try:
            # Step 1: Encode input text with ByT5 encoder
            encoder_input = self.byt5_tokenizer(text, return_tensors="pt", add_special_tokens=True)
            encoder_input_ids = encoder_input["input_ids"].to(self.device)
            encoder_attention_mask = encoder_input.get("attention_mask", None)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(self.device)
            
            # Get encoder outputs
            encoder_outputs = self.byt5_model.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask
            )
            
            # Step 2: Get initial byte log-probs from decoder
            # ByT5 decoder_start_token_id (usually 0 for pad_token_id)
            decoder_start_id = self.byt5_model.config.decoder_start_token_id
            if decoder_start_id is None:
                decoder_start_id = self.byt5_tokenizer.pad_token_id
            
            decoder_input_ids = torch.tensor([[decoder_start_id]], device=self.device)
            
            # Forward through decoder
            decoder_outputs = self.byt5_model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=encoder_attention_mask,
            )
            
            # Get logits and log-probs for next byte
            logits = self.byt5_model.lm_head(decoder_outputs.last_hidden_state)
            logits = logits[0, -1, :]  # Last position, shape (vocab_size,)
            log_probs_all = F.log_softmax(logits, dim=-1)
            
            # ByT5: token IDs 0-255 = bytes 0x00-0xFF; 256-258 = special (pad, eos, unk)
            vocab_size = len(self.byt5_tokenizer)
            byte_offset = 0
            byte_end = min(256, vocab_size)
            log_probs = log_probs_all[byte_offset:byte_end]  # Extract byte log-probs
            if log_probs.size(0) < 256:
                pad_size = 256 - log_probs.size(0)
                log_probs = torch.cat([log_probs, torch.full((pad_size,), -100.0, device=self.device)])
            log_probs = log_probs[:256]
            initial_probs = log_probs.cpu()
            
            # Greedy mode: no beam, score each character by greedy continuation, take top-3
            if greedy:
                top3_greedy = self._byt5_greedy_top3(
                    encoder_outputs, encoder_attention_mask,
                    decoder_input_ids, log_probs, byte_offset
                )
                if top3_greedy is not None:
                    return top3_greedy
                # Fall through to beam if greedy path fails
            
            # Step 3: Build initial beam (same as Bolmo)
            hypotheses = []
            for byte_val in range(256):
                expected_len = utf8_expected_continuation_bytes(byte_val)
                if expected_len == 0:
                    continue  # Invalid lead byte
                
                log_prob = log_probs[byte_val].item()
                
                if expected_len == 1:
                    # Single-byte character (complete)
                    hypotheses.append(([byte_val], log_prob, None, True))
                else:
                    # Multi-byte character (incomplete) - need to extend
                    # Store decoder_input_ids for incremental decoding
                    byte_token_id = byte_offset + byte_val
                    next_decoder_input = torch.cat([
                        decoder_input_ids,
                        torch.tensor([[byte_token_id]], device=self.device)
                    ], dim=1)
                    hypotheses.append(([byte_val], log_prob, next_decoder_input, False))
            
            # Sort by log_prob and keep top beam_width
            hypotheses.sort(key=lambda h: h[1], reverse=True)
            hypotheses = hypotheses[:beam_width]
            
            # Separate completed vs incomplete
            completed = [h for h in hypotheses if h[3]]
            beam = [h for h in hypotheses if not h[3]]
            
            # Safety check
            if not hypotheses:
                return self._byt5_fallback_top3(initial_probs, byte_offset)
            
            # Step 4: Beam extension (bytes 2-4) for multi-byte UTF-8
            max_depth = 4
            for depth in range(2, max_depth + 1):
                if not beam:
                    break
                
                new_beam = []
                
                # Cache for same prefix
                prefix_results = {}
                
                for byte_list, log_prob, decoder_input, _ in beam:
                    prefix_key = tuple(byte_list)
                    
                    if prefix_key not in prefix_results:
                        # Get next byte log-probs from decoder
                        try:
                            decoder_outputs = self.byt5_model.decoder(
                                input_ids=decoder_input,
                                encoder_hidden_states=encoder_outputs.last_hidden_state,
                                encoder_attention_mask=encoder_attention_mask,
                            )
                            logits = self.byt5_model.lm_head(decoder_outputs.last_hidden_state)
                            logits = logits[0, -1, :]
                            log_probs_all = F.log_softmax(logits, dim=-1)
                            next_log_probs = log_probs_all[byte_offset:byte_end][:256]
                            if next_log_probs.size(0) < 256:
                                pad_size = 256 - next_log_probs.size(0)
                                next_log_probs = torch.cat([
                                    next_log_probs,
                                    torch.full((pad_size,), -100.0, device=self.device)
                                ])
                            prefix_results[prefix_key] = next_log_probs
                        except Exception:
                            # Fallback: use initial probs
                            next_log_probs = log_probs
                            prefix_results[prefix_key] = next_log_probs
                    else:
                        next_log_probs = prefix_results[prefix_key]
                    
                    # Extend with valid continuation bytes
                    for next_byte in range(256):
                        if not utf8_valid_continuation_byte(byte_list, next_byte):
                            continue
                        
                        new_byte_list = byte_list + [next_byte]
                        new_log_prob = log_prob + next_log_probs[next_byte].item()
                        
                        # Check if complete
                        expected_len = utf8_expected_continuation_bytes(byte_list[0])
                        is_complete = len(new_byte_list) == expected_len
                        
                        if is_complete:
                            # Decode and validate
                            char, consumed = utf8_decode_single(new_byte_list)
                            if char is not None and consumed == len(new_byte_list):
                                completed.append((new_byte_list, new_log_prob, None, True))
                        else:
                            # Incomplete - build new decoder input
                            byte_token_id = byte_offset + next_byte
                            new_decoder_input = torch.cat([
                                decoder_input,
                                torch.tensor([[byte_token_id]], device=self.device)
                            ], dim=1)
                            new_beam.append((new_byte_list, new_log_prob, new_decoder_input, False))
                
                # Keep top beam_width incomplete hypotheses
                new_beam.sort(key=lambda h: h[1], reverse=True)
                beam = new_beam[:beam_width]
            
            # Step 5: Marginalization
            char_log_probs = {}
            for byte_list, log_prob, _, _ in completed:
                char, consumed = utf8_decode_single(byte_list)
                if char is not None and consumed == len(byte_list):
                    if ord(char) < 32 or char == '\ufffd':
                        continue
                    if char not in char_log_probs:
                        char_log_probs[char] = []
                    char_log_probs[char].append(log_prob)
            
            # Sum probabilities per character using logsumexp
            char_probs = {}
            for char, log_prob_list in char_log_probs.items():
                char_probs[char] = logsumexp(log_prob_list)
            
            # Step 6: Rank and get top-3
            sorted_chars = sorted(char_probs.items(), key=lambda x: x[1], reverse=True)
            
            if debug:
                print(f"\n=== ByT5 Character probabilities (top 10) ===")
                for i, (char, log_prob) in enumerate(sorted_chars[:10]):
                    prob = torch.exp(torch.tensor(log_prob)).item()
                    print(f"  {i+1}. '{char}' (U+{ord(char):04X}): log_prob={log_prob:.4f}, prob={prob:.4f}")
            
            top_chars = [c for c, _ in sorted_chars[:3]]
            
            # Step 7: Fill gaps with single-byte characters from initial_probs
            if len(top_chars) < 3:
                single_byte_chars = []
                for byte_val in range(256):
                    if utf8_expected_continuation_bytes(byte_val) == 1:
                        char, _ = utf8_decode_single([byte_val])
                        if char is not None and char not in top_chars:
                            if ord(char) < 32 or char == '\ufffd':
                                continue
                            single_byte_chars.append((char, initial_probs[byte_val].item()))
                
                single_byte_chars.sort(key=lambda x: x[1], reverse=True)
                for char, _ in single_byte_chars:
                    if len(top_chars) >= 3:
                        break
                    if char not in top_chars:
                        top_chars.append(char)
            
            # Final fallback
            fallback = [" ", ".", "e", "a", "o"]
            for fb_char in fallback:
                if len(top_chars) >= 3:
                    break
                if fb_char not in top_chars:
                    top_chars.append(fb_char)
            
            # Ensure exactly 3
            while len(top_chars) < 3:
                top_chars.append("e")
            
            # Final validation
            result = []
            for char in top_chars[:3]:
                if char and char != '\ufffd' and ord(char) >= 32:
                    result.append(char)
            
            while len(result) < 3:
                for fb in [" ", ".", "e", "a", "o", "i"]:
                    if fb not in result:
                        result.append(fb)
                        break
                if len(result) >= 3:
                    break
            
            return "".join(result[:3])
            
        except Exception as e:
            import traceback
            print("Error in _next_char_top3_byt5 for text '{}': {}".format(text, e))
            traceback.print_exc()
            return "eta"

    def _byt5_greedy_top3(self, encoder_outputs, encoder_attention_mask, decoder_input_ids, log_probs, byte_offset):
        """
        Pure ByT5: no beam. Score each character by greedy byte continuation; return top-3.
        """
        import torch
        import torch.nn.functional as F
        
        char_scores = []  # (char, log_prob)
        byte_end = byte_offset + 256
        
        for byte_val in range(256):
            expected_len = utf8_expected_continuation_bytes(byte_val)
            if expected_len == 0:
                continue
            if expected_len == 1:
                char, _ = utf8_decode_single([byte_val])
                if char is not None and ord(char) >= 32 and char != '\ufffd':
                    char_scores.append((char, log_probs[byte_val].item()))
            else:
                # Multi-byte: greedy continuation
                dec_input = decoder_input_ids
                total_lp = log_probs[byte_val].item()
                byte_list = [byte_val]
                for step in range(expected_len - 1):
                    next_byte_token = byte_offset + byte_list[-1]
                    dec_input = torch.cat([
                        dec_input,
                        torch.tensor([[next_byte_token]], device=self.device)
                    ], dim=1)
                    dec_out = self.byt5_model.decoder(
                        input_ids=dec_input,
                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                    logits = self.byt5_model.lm_head(dec_out.last_hidden_state)[0, -1, :]
                    next_lp_all = F.log_softmax(logits, dim=-1)
                    next_lp = next_lp_all[byte_offset:byte_end]
                    if next_lp.size(0) < 256:
                        next_lp = torch.cat([
                            next_lp,
                            torch.full((256 - next_lp.size(0),), -100.0, device=self.device)
                        ])
                    best_cont, best_val = None, -1e9
                    for cb in range(256):
                        if not utf8_valid_continuation_byte(byte_list, cb):
                            continue
                        v = next_lp[cb].item()
                        if v > best_val:
                            best_val = v
                            best_cont = cb
                    if best_cont is None:
                        break
                    byte_list.append(best_cont)
                    total_lp += best_val
                if len(byte_list) == expected_len:
                    char, consumed = utf8_decode_single(byte_list)
                    if char is not None and consumed == len(byte_list) and ord(char) >= 32 and char != '\ufffd':
                        char_scores.append((char, total_lp))
        
        if not char_scores:
            return None
        char_scores.sort(key=lambda x: x[1], reverse=True)
        top_chars = [c for c, _ in char_scores[:3]]
        fallback = [" ", ".", "e", "a", "o"]
        for fb in fallback:
            if len(top_chars) >= 3:
                break
            if fb not in top_chars:
                top_chars.append(fb)
        while len(top_chars) < 3:
            top_chars.append("e")
        return "".join(top_chars[:3])

    def _byt5_fallback_top3(self, initial_probs, byte_offset):
        """Fallback: use single-byte characters from initial_probs."""
        single_byte_chars = []
        for byte_val in range(256):
            if utf8_expected_continuation_bytes(byte_val) == 1:
                char, _ = utf8_decode_single([byte_val])
                if char is not None and ord(char) >= 32 and char != '\ufffd':
                    single_byte_chars.append((char, initial_probs[byte_val].item()))
        single_byte_chars.sort(key=lambda x: x[1], reverse=True)
        top_chars = [c for c, _ in single_byte_chars[:3]]
        fallback = [" ", ".", "e", "a", "o"]
        for fb in fallback:
            if len(top_chars) >= 3:
                break
            if fb not in top_chars:
                top_chars.append(fb)
        while len(top_chars) < 3:
            top_chars.append("e")
        return "".join(top_chars[:3])

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        # Save Bolmo to its current directory (used mainly for raw model).
        save_path = os.path.join(work_dir, BOLMO_SUBDIR)
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)
        # Save Qwen metadata/config if loaded
        if self.qwen_model is not None:
            self.qwen_model.save(work_dir)

    @classmethod
    def load(
        cls,
        work_dir,
        load_qwen=True,
        qwen_force_quantize=False,
        bolmo_variant_subdir=None,
        qwen_variant_subdir=None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from myprogram_qwen import QwenModel
        
        # Load Bolmo-1B
        # Bolmo-1B base (raw) and optional finetuned variant
        base_bolmo_path = os.path.join(work_dir, BOLMO_SUBDIR)
        if bolmo_variant_subdir is not None:
            candidate_variant = os.path.join(work_dir, bolmo_variant_subdir)
        else:
            candidate_variant = os.path.join(work_dir, BOLMO_INITIAL_FINETUNE_SUBDIR)

        if bolmo_variant_subdir is not None and os.path.exists(
            os.path.join(candidate_variant, "config.json")
        ):
            model_path = candidate_variant
            print(
                "Loading Bolmo variant from {} (base raw model remains at {}).".format(
                    candidate_variant, base_bolmo_path
                )
            )
        elif os.path.exists(os.path.join(base_bolmo_path, "config.json")):
            model_path = base_bolmo_path
        else:
            model_path = BOLMO_HF_ID
            print("Model not found at {}, loading from {} (run download_bolmo.py or train first).".format(
                base_bolmo_path, BOLMO_HF_ID))
        device = _get_device()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
        
        # Optionally load Qwen for non-English
        qwen_model = None
        if load_qwen:
            try:
                print(
                    "Loading Qwen for non-English (full precision={} ; variant={}) ...".format(
                        not qwen_force_quantize, qwen_variant_subdir
                    )
                )
                qwen_model = QwenModel.load(
                    work_dir,
                    force_quantize=qwen_force_quantize,
                    variant_subdir=qwen_variant_subdir,
                )
            except Exception as e:
                print("Warning: Failed to load Qwen: {}. Will use Bolmo for all inputs.".format(e))
                qwen_model = None
        
        return cls(model=model, tokenizer=tokenizer, device=device, qwen_model=qwen_model)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="train: download/save Bolmo; test: load and predict")
    parser.add_argument("--work_dir", default="work", help="where to load/save model")
    parser.add_argument("--test_data", default="example/input.txt", help="path to test input")
    parser.add_argument("--test_output", default="pred.txt", help="path to write predictions")
    # Fine-tuning (only used when mode=train and --finetune)
    parser.add_argument("--finetune", action="store_true", help="run LoRA fine-tuning; requires --train_data")
    parser.add_argument("--train_data", default="work/data/opensubtitles_en_hi_full.txt", help="Bolmo training corpus (one sample per line, UTF-8).")
    parser.add_argument("--qwen_train_data", default="work/data/wikipedia_for_qwen.txt", help="Qwen training corpus (one sample per line, UTF-8).")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (default 8)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (default 16)")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="comma-separated LoRA target modules (default: q_proj,v_proj)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default 0.05)")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="max sequence length for SFT (default 1024; lower helps 15GB GPU)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="train batch size per device (default 2 for ~15GB GPU)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="gradient accumulation steps (default 16; effective batch = 2*16=32)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="enable gradient checkpointing to save GPU memory (default: True)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", help="disable gradient checkpointing")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate (default 1e-5)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="number of train epochs (default 1)")
    parser.add_argument("--eval_steps", type=int, default=500, help="eval every N steps (default 500)")
    parser.add_argument("--save_steps", type=int, default=1000, help="save every N steps (default 1000)")
    parser.add_argument("--logging_steps", type=int, default=50, help="log every N steps (default 50)")
    parser.add_argument("--streaming", action="store_true", help="stream corpus from disk (do not load whole file into RAM); safe for full OpenSubtitles")
    parser.add_argument("--streaming_eval_size", type=int, default=50000, help="when --streaming: use first N lines as eval set (default 50000); 0 = no eval")
    # Phase 4: beam search parameter
    parser.add_argument("--beam_width", type=int, default=8, help="beam width for byte→character search (default 8)")
    parser.add_argument("--qwen_top_k_tokens", type=int, default=5000, help="Qwen top-k token candidates for non-English prediction (default 5000).")
    parser.add_argument("--load_qwen", action="store_true", default=True, help="load Qwen for non-English inputs (default: True)")
    parser.add_argument("--no_qwen", dest="load_qwen", action="store_false", help="disable Qwen, use Bolmo for all inputs")
    parser.add_argument("--qwen_quantize", action="store_true", help="use quantized Qwen during test (default: full precision).")
    parser.add_argument("--no_bolmo_english_filter", dest="bolmo_english_only", action="store_false", help="disable English-only filtering for Bolmo fine-tuning.")
    parser.add_argument("--no_qwen_non_english_filter", dest="qwen_non_english_only", action="store_false", help="disable non-English-only filtering for Qwen fine-tuning.")
    parser.set_defaults(bolmo_english_only=True, qwen_non_english_only=True)
    # Dataset / variant selection
    parser.add_argument(
        "--dataset",
        choices=("opensubtitles", "belle", "wikipedia", "custom"),
        default="custom",
        help="opensubtitles: use prepared split corpora from src/prepare_datasets.py; "
             "others: use --train_data/--qwen_train_data as-is.",
    )
    parser.add_argument(
        "--bolmo_variant",
        type=str,
        default=None,
        help="optional Bolmo variant directory name under work_dir (e.g., bolmo-InitialFineTune).",
    )
    parser.add_argument(
        "--qwen_variant",
        type=str,
        default=None,
        help="optional Qwen variant directory name under work_dir (e.g., qwen3-4b-nonEnglish-v1).",
    )
    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
        if args.finetune:
            train_data_path = args.train_data
            qwen_train_data_path = args.qwen_train_data
            # When --dataset opensubtitles: use prepared combined/split corpora.
            if args.dataset == "opensubtitles":
                from prepare_datasets import prepare_opensubtitles_dataset

                multi_path, en_path, non_en_path = prepare_opensubtitles_dataset(
                    args.work_dir, download_if_missing=False
                )
                if multi_path is not None and en_path is not None and non_en_path is not None:
                    # For prepared OpenSubtitles corpora, pass split files directly.
                    train_data_path = en_path
                    qwen_train_data_path = non_en_path
                    args.bolmo_english_only = False
                    args.qwen_non_english_only = False
                    print(
                        "Using prepared OpenSubtitles corpora: "
                        "Bolmo={}, Qwen={} (from {}).".format(
                            train_data_path, qwen_train_data_path, multi_path
                        )
                    )
                else:
                    raise SystemExit(
                        "Dataset 'opensubtitles' is not prepared. Run:\n"
                        "  python3 src/prepare_datasets.py --work_dir work --dataset opensubtitles --download_if_missing\n"
                        "Then re-run train."
                    )
            elif not train_data_path or not os.path.isfile(train_data_path):
                raise SystemExit("--finetune requires --train_data pointing to an existing corpus file (e.g. work/data/opensubtitles_en_hi_20k.txt or work/data/belle_conversation_20k.txt)")
            print("Loading model for fine-tuning (Qwen loaded separately during fine-tune stage)")
            model = MyModel.load(args.work_dir, load_qwen=False)
            lora_target_modules = None
            if args.lora_target_modules is not None:
                lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
            model.run_finetune(
                train_data_path,
                args.work_dir,
                qwen_train_data_path=qwen_train_data_path,
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
                bolmo_english_only=args.bolmo_english_only,
                qwen_non_english_only=args.qwen_non_english_only,
            )
        else:
            print("Instantiating model (download-only for Phase 1)")
            model = MyModel.load(args.work_dir, load_qwen=False)
            print("Loading training data (optional)")
            train_data = MyModel.load_training_data()
            print("Training (download/save only)")
            model.run_train(train_data, args.work_dir)
            print("Saving model")
            model.save(args.work_dir)
    elif args.mode == "test":
        print("Loading model (Bolmo + Qwen={})".format(args.load_qwen))
        model = MyModel.load(
            args.work_dir,
            load_qwen=args.load_qwen,
            qwen_force_quantize=args.qwen_quantize,
            bolmo_variant_subdir=args.bolmo_variant,
            qwen_variant_subdir=args.qwen_variant,
        )
        print("Loading test data from {}".format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        mode_str = "beam_width={}, qwen_top_k_tokens={}".format(args.beam_width, args.qwen_top_k_tokens)
        print("Making predictions with {} (English->Bolmo, non-English->Qwen)".format(mode_str))
        pred = model.run_pred(test_data, beam_width=args.beam_width, qwen_top_k_tokens=args.qwen_top_k_tokens)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred))
        MyModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
