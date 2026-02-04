#!/usr/bin/env python
"""
CSE447 character prediction: Bolmo-1B byte-level model.
Phase 1: load model from work_dir, run inference only (no training in predict path).
"""
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Bolmo model subdir under work_dir (same as download_bolmo.py)
BOLMO_SUBDIR = "bolmo-1b"
BOLMO_HF_ID = "allenai/Bolmo-1B"


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    args = parser.parse_args()

    if args.mode == "train":
        if not os.path.isdir(args.work_dir):
            print("Making working directory {}".format(args.work_dir))
            os.makedirs(args.work_dir)
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
