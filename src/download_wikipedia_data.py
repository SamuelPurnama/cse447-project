#!/usr/bin/env python
"""
Download Wikipedia (wikimedia/wikipedia) and save as one line per article for Qwen training.

Uses Hugging Face wikimedia/wikipedia with language configs (e.g. 20231101.en, 20231101.zh).
Each article's "text" is one line (whitespace normalized, length capped). High-quality
encyclopedic text in 300+ languages.

Usage:
  conda activate cse447
  python src/download_wikipedia_data.py --max_samples 200000
  python src/download_wikipedia_data.py --languages en zh ja ko es fr de --max_samples 100000
"""
import os
import argparse
import unicodedata

OUTPUT_DIR_DEFAULT = "work/data"
MAX_SAMPLES_DEFAULT = 200_000
# wikimedia/wikipedia configs: 20231101.<lang>
WIKI_DATE = "20231101"
WIKIPEDIA_DEFAULT_LANGUAGES = [
    "en", "zh", "ja", "ko", "es", "fr", "de", "ar", "hi", "pt", "ru", "it",
    "th", "vi", "tr", "pl", "id", "nl", "el", "he",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia (wikimedia/wikipedia) for Qwen training."
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
        help="Max total lines (articles) to save (default 200000)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Language codes (default: 20 languages). Examples: en, zh, ja, es.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output filename (default: wikipedia_multilingual_<N>k.txt)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    lang_list = args.languages or WIKIPEDIA_DEFAULT_LANGUAGES
    max_per_lang = max(1, (args.max_samples + len(lang_list) - 1) // len(lang_list))
    lines = []
    for lang_code in lang_list:
        config = f"{WIKI_DATE}.{lang_code}"
        try:
            print(f"Loading Wikipedia '{config}' (max {max_per_lang})...", flush=True)
            ds = load_dataset(
                "wikimedia/wikipedia",
                config,
                split="train",
                trust_remote_code=True,
            )
            n = 0
            for row in ds:
                if n >= max_per_lang:
                    break
                text = (row.get("text") or "").strip()
                if not text or len(text) < 20:
                    continue
                text = unicodedata.normalize("NFC", text)
                text = " ".join(text.split())
                if len(text) > 8000:
                    text = text[:8000]
                lines.append(text)
                n += 1
            print(f"  Got {n} (total {len(lines)})", flush=True)
        except Exception as e:
            print(f"  Skip {config}: {e}", file=__import__("sys").stderr)
            continue
        if len(lines) >= args.max_samples:
            break
    lines = lines[: args.max_samples]
    if not lines:
        raise SystemExit(
            "No Wikipedia data loaded. Configs: 20231101.<lang> at https://huggingface.co/datasets/wikimedia/wikipedia"
        )

    k = len(lines) // 1000
    out_fname = args.output_file or f"wikipedia_multilingual_{k}k.txt"
    out_path = os.path.join(args.output_dir, out_fname)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Saved {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
