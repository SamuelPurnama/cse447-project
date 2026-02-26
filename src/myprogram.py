#!/usr/bin/env python
"""
CSE447 character prediction: hybrid Bolmo + Qwen model.
English inputs use Bolmo-1B (top-3 bytes). Non-English use Qwen next-token marginalization.
Train mode: download/save base models. Fine-tune via src/finetune_bolmo.sh and src/finetune_qwen.sh.
"""
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Bolmo model subdir under work_dir (same as download_bolmo.py)
BOLMO_SUBDIR = "bolmo-1b"
BOLMO_HF_ID = "allenai/Bolmo-1B"

# Qwen for non-English next-character prediction
QWEN_SUBDIR = "qwen3-4b-base"
QWEN_HF_ID = "Qwen/Qwen3-4B-Base"

# Variant subdirs under work_dir (written by finetune scripts)
BOLMO_INITIAL_FINETUNE_SUBDIR = "bolmo-InitialFineTune"
QWEN_FINETUNE_NON_EN_SUBDIR = "qwen3-4b-nonEnglish-v1"


# ========== Qwen predictor (inline; no myprogram_qwen dependency) ==========


def _load_qwen_predictor(work_dir, force_quantize=False, variant_subdir=None):
    """Load Qwen model and tokenizer for non-English next-character prediction. Returns a small wrapper with .model, .tokenizer, ._next_char_top3()."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.special import logsumexp

    base_path = os.path.join(work_dir, QWEN_SUBDIR)
    variant_path = os.path.join(work_dir, variant_subdir) if variant_subdir else os.path.join(work_dir, QWEN_FINETUNE_NON_EN_SUBDIR)
    if variant_subdir and os.path.exists(os.path.join(variant_path, "config.json")):
        model_path = variant_path
    elif os.path.exists(os.path.join(base_path, "config.json")):
        model_path = base_path
    else:
        model_path = QWEN_HF_ID

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if force_quantize:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto",
        )
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)

    class QwenPredictor:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

        def save(self, work_dir):
            pass  # Qwen already saved under work_dir

        def _next_char_top3(self, text, top_k_tokens=5000):
            """Next-character top-3 via next-token logits; marginalize by first character of each token."""
            import torch.nn.functional as F
            if not text:
                return "\u5929\u6c14\u5feb"  # "天气快" fallback
            try:
                enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=2048)
                input_ids = enc["input_ids"].to(self.device)
                if input_ids.size(1) == 0:
                    return "\u5929\u6c14\u5feb"
                with torch.no_grad():
                    logits = self.model(input_ids).logits[0, -1, :]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                top_k = min(top_k_tokens, log_probs.size(0))
                vals, indices = torch.topk(log_probs, top_k)
                char_log_probs = {}
                for i in range(indices.size(0)):
                    tid = indices[i].item()
                    lp = vals[i].item()
                    decoded = self.tokenizer.decode([tid]).strip()
                    if not decoded:
                        continue
                    first_char = decoded[0]
                    if ord(first_char) < 32 or first_char == "\ufffd":
                        continue
                    if first_char not in char_log_probs:
                        char_log_probs[first_char] = []
                    char_log_probs[first_char].append(lp)
                if not char_log_probs:
                    return "\u5929\u6c14\u5feb"
                aggregated = {c: logsumexp(lps) for c, lps in char_log_probs.items()}
                top3 = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:3]
                result = "".join(c for c, _ in top3)
                fallback = "\u5929\u6c14\u5feb\u4e0d\u9519\u5417"  # 天气快不错吗
                while len(result) < 3:
                    for ch in fallback:
                        if ch not in result:
                            result += ch
                            if len(result) >= 3:
                                break
                return result[:3]
            except Exception:
                return "\u5929\u6c14\u5feb"

    return QwenPredictor(model, tokenizer, device)


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

    @classmethod
    def run_train(cls, data, work_dir):
        """Download and save base Bolmo and Qwen to work_dir. No model instance needed."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
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
            qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_HF_ID, trust_remote_code=True)
            qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_HF_ID, trust_remote_code=True)
            qwen_tokenizer.save_pretrained(qwen_save_path)
            qwen_model.save_pretrained(qwen_save_path)
            print("Saved Qwen raw model to {}.".format(qwen_save_path))

        print("Raw Bolmo (en) and Qwen (multi-language) models are available in work_dir.")

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
        Next-character prediction using top-3 bytes (no beam search).
        For English/Bolmo: one forward pass, take the 3 most likely single-byte (ASCII) next bytes.
        Returns 3 character guesses for the next character after text.
        """
        import torch

        if not text:
            return "eta"  # frequency fallback

        try:
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(self.device)
            if input_ids.size(0) == 0 or input_ids.size(1) == 0:
                return "eta"

            # Single forward: next-byte log-probs (no cache needed)
            log_probs = self._get_next_byte_logprobs(input_ids, return_cache=False)
            log_probs = log_probs.cpu()

            # Only consider bytes that form valid single-byte characters (ASCII)
            single_byte_candidates = []
            for byte_val in range(256):
                if utf8_expected_continuation_bytes(byte_val) != 1:
                    continue
                char, _ = utf8_decode_single([byte_val])
                if char is None or ord(char) < 32 or char == "\ufffd":
                    continue
                single_byte_candidates.append((char, log_probs[byte_val].item()))

            # Sort by log_prob descending and take top 3
            single_byte_candidates.sort(key=lambda x: x[1], reverse=True)
            top_chars = [c for c, _ in single_byte_candidates[:3]]

            if debug:
                print(
                    "Top-3 bytes (single-byte chars):",
                    [(c, lp) for c, lp in single_byte_candidates[:3]],
                )

            # Fill to 3 with next-best single-byte chars if needed
            fallback = [" ", ".", "e", "t", "a"]
            for char, _ in single_byte_candidates[3:]:
                if len(top_chars) >= 3:
                    break
                if char not in top_chars:
                    top_chars.append(char)
            for fb_char in fallback:
                if len(top_chars) >= 3:
                    break
                if fb_char not in top_chars:
                    top_chars.append(fb_char)
            while len(top_chars) < 3:
                top_chars.append("e")

            result = [c for c in top_chars[:3] if c and c != "\ufffd" and ord(c) >= 32]
            while len(result) < 3:
                for fb in [" ", ".", "e", "t", "a", "n"]:
                    if fb not in result:
                        result.append(fb)
                        break
                if len(result) >= 3:
                    break

            return "".join(result[:3])

        except Exception as e:
            import traceback
            print("Error in _next_char_top3 for text '{}': {}".format(text, e))
            traceback.print_exc()
            return "eta"

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
        
        # Optionally load Qwen for non-English (inline; no myprogram_qwen dependency)
        qwen_model = None
        if load_qwen:
            try:
                print(
                    "Loading Qwen for non-English (full precision={} ; variant={}) ...".format(
                        not qwen_force_quantize, qwen_variant_subdir
                    )
                )
                qwen_model = _load_qwen_predictor(
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
    parser.add_argument("--beam_width", type=int, default=8, help="beam width for byte→character search (default 8)")
    parser.add_argument("--qwen_top_k_tokens", type=int, default=5000, help="Qwen top-k token candidates for non-English prediction (default 5000).")
    parser.add_argument("--load_qwen", action="store_true", default=True, help="load Qwen for non-English inputs (default: True)")
    parser.add_argument("--no_qwen", dest="load_qwen", action="store_false", help="disable Qwen, use Bolmo for all inputs")
    parser.add_argument("--qwen_quantize", action="store_true", help="use quantized Qwen during test (default: full precision).")
    # Dataset / variant selection
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
            os.makedirs(args.work_dir)
        train_data = MyModel.load_training_data()
        MyModel.run_train(train_data, args.work_dir)
        print("Done. For fine-tuning, run: bash src/finetune_bolmo.sh or bash src/finetune_qwen.sh")
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
