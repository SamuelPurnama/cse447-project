#!/usr/bin/env python
"""
Download mC4 (multilingual C4, allenai/c4) and save as one line per sample for Qwen training.

mC4 is a cleaned multilingual web corpus (100+ languages). We sample from multiple
language configs and write one text file (one sample per line) for use with
--train_data in myprogram_qwen.py.

Usage:
  conda activate cse447
  python src/download_mc4_data.py --max_samples 200000
  python src/download_mc4_data.py --languages en es fr de zh ja ko --max_samples 100000
"""
import os
import sys
import argparse
import unicodedata

OUTPUT_DIR_DEFAULT = "work/data"
MAX_SAMPLES_DEFAULT = 200_000

# Default: diverse high-resource languages from allenai/c4 configs
MC4_DEFAULT_LANGUAGES = [
    "en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi", "pt", "ru", "it",
    "th", "vi", "tr", "pl", "id", "nl", "el", "he",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download mC4 (allenai/c4) multilingual samples for Qwen training."
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Output directory",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=MAX_SAMPLES_DEFAULT,
        help="Max total lines to save (default 200000)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Language configs (default: 20 languages). Examples: en, es, de, zh, multilingual.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output filename (default: mc4_multilingual_<N>k.txt)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    lang_list = args.languages or MC4_DEFAULT_LANGUAGES
    max_per_lang = max(1, (args.max_samples + len(lang_list) - 1) // len(lang_list))
    lines = []
    for lang_code in lang_list:
        try:
            print(f"Loading mC4 '{lang_code}' (max {max_per_lang})...", flush=True)
            ds = load_dataset(
                "allenai/c4",
                lang_code,
                split="train",
                trust_remote_code=True,
            )
            n = 0
            for row in ds:
                if n >= max_per_lang:
                    break
                text = (row.get("text") or "").strip()
                if not text or len(text) < 10:
                    continue
                text = unicodedata.normalize("NFC", text)
                text = " ".join(text.split())
                if len(text) > 8000:
                    text = text[:8000]  # cap length for training
                lines.append(text)
                n += 1
            print(f"  Got {n} (total {len(lines)})", flush=True)
        except Exception as e:
            print(f"  Skip {lang_code}: {e}", file=sys.stderr)
            continue
        if len(lines) >= args.max_samples:
            break
    lines = lines[: args.max_samples]
    if not lines:
        raise SystemExit("No mC4 data loaded. Check language configs at https://huggingface.co/datasets/allenai/c4")

    k = len(lines) // 1000
    out_fname = args.output_file or f"mc4_multilingual_{k}k.txt"
    out_path = os.path.join(args.output_dir, out_fname)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Saved {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
