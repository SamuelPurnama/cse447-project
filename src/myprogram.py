#!/usr/bin/env python
"""
CSE447 character prediction: Bolmo-1B byte-level model.
Phase 4: Byte→character with beam search + marginalization for multi-byte UTF-8.
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

    def run_pred(self, data, beam_width=8):
        import torch
        self.model.eval()
        preds = []
        with torch.no_grad():
            for inp in data:
                top3 = self._next_char_top3(inp, beam_width=beam_width)
                preds.append(top3)
        return preds

    def _byte_to_token_id(self, byte_val):
        """Convert byte value (0-255) to Bolmo token ID (4-259)."""
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
            
            # Extract only byte token log-probs (tokens 4-259)
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
                byte_log_probs = log_probs[4:260]
                
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
                        # Try to use KV-cache from previous step
                        # Bolmo bug: single-token inputs with past_key_values crash
                        # Workaround: if single-token fails, rebuild full sequence
                        
                        # Get token ID for the last byte we're extending with
                        last_byte = byte_list[-1]
                        token_id = self._byte_to_token_id(last_byte)
                        next_input_ids = torch.tensor([[token_id]], device=self.device)
                        
                        try:
                            # Try incremental extension with KV-cache
                            next_log_probs, new_cache = self._get_next_byte_logprobs(
                                next_input_ids,
                                past_key_values=past_kv,  # Use cache from previous step
                                return_cache=True
                            )
                        except Exception as e:
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
    # Phase 4: beam search parameter
    parser.add_argument("--beam_width", type=int, default=8, help="beam width for byte→character search (default 8)")
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
        print("Making predictions with beam_width={}".format(args.beam_width))
        pred = model.run_pred(test_data, beam_width=args.beam_width)
        print("Writing predictions to {}".format(args.test_output))
        assert len(pred) == len(test_data), "Expected {} predictions but got {}".format(
            len(test_data), len(pred))
        MyModel.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError("Unknown mode {}".format(args.mode))
