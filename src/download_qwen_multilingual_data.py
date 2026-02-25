#!/usr/bin/env python
"""
Download a high-quality multilingual corpus for Qwen fine-tuning.

Two sources (use --source):

  hplt (default): HPLT v3 – conversational/interactive web text, 20 languages,
    quality-filtered (register labels). No Hugging Face login. Same pipeline as
    download_phase2_data.py; output is one file with one sample per line.

  culturax: CulturaX – 167 languages, 6.3T tokens, heavily cleaned/deduplicated,
    designed for LLMs. Best quality but gated: run `huggingface-cli login` and
    accept the dataset terms at https://huggingface.co/datasets/uonlp/CulturaX.

Output: work/data/qwen_multilingual_<N>k.txt (or qwen_multilingual_full.txt with --all for CulturaX).

Usage:
  python src/download_qwen_multilingual_data.py --max_samples 200000
  python src/download_qwen_multilingual_data.py --source culturax --max_samples 100000 --languages en es zh fr de
"""
import os
import sys
import argparse
import unicodedata

OUTPUT_DIR_DEFAULT = "work/data"
MAX_SAMPLES_DEFAULT = 200_000

# HPLT: same as phase2 – diverse languages, conversational filter
HPLT_DEFAULT_LANGUAGES = [
    "eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "cmn_Hans", "jpn_Jpan",
    "arb_Arab", "hin_Deva", "por_Latn", "rus_Cyrl", "ita_Latn", "kor_Hang",
    "tha_Thai", "vie_Latn", "tur_Latn", "pol_Latn", "ind_Latn", "nld_Latn",
    "ell_Grek", "heb_Hebr",
]

# CulturaX: language codes for config (subset of 167)
CULTURAX_DEFAULT_LANGUAGES = [
    "en", "es", "zh", "fr", "de", "ja", "pt", "ru", "ar", "hi", "ko", "it",
    "th", "vi", "tr", "pl", "id", "nl", "el", "he",
]


def run_hplt(max_samples, output_path, languages, max_shards_per_lang):
    """Build corpus from HPLT v3 (conversational filter, NFC)."""
    # Allow importing from same package when run as python src/download_qwen_multilingual_data.py
    _src = os.path.dirname(os.path.abspath(__file__))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from download_phase2_data import (
        DEFAULT_LANGUAGES,
        sample_from_language,
    )
    lang_list = languages or DEFAULT_LANGUAGES
    max_per_lang = max(1, (max_samples + len(lang_list) - 1) // len(lang_list))
    all_samples = []
    for lang_code in lang_list:
        print(f"Fetching HPLT {lang_code} (max {max_per_lang} samples)...", flush=True)
        samples = sample_from_language(
            lang_code,
            max_samples=max_per_lang,
            max_shards=max_shards_per_lang,
        )
        for s in samples:
            all_samples.append(unicodedata.normalize("NFC", s))
        print(f"  Got {len(samples)} (total {len(all_samples)})", flush=True)
        if len(all_samples) >= max_samples:
            break
    all_samples = all_samples[:max_samples]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_samples:
            f.write(line + "\n")
    print(f"Saved {len(all_samples)} lines to {output_path}")
    return len(all_samples)


def run_culturax(max_samples, output_path, languages, streaming_cap):
    """Build corpus from CulturaX (Hugging Face). Gated – may require login."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    lang_list = languages or CULTURAX_DEFAULT_LANGUAGES
    max_per_lang = max(1, (max_samples + len(lang_list) - 1) // len(lang_list))
    lines = []
    for lang_code in lang_list:
        try:
            print(f"Loading CulturaX config '{lang_code}' (max {max_per_lang})...", flush=True)
            ds = load_dataset(
                "uonlp/CulturaX",
                lang_code,
                split="train",
                streaming=False,
                trust_remote_code=True,
            )
            n = 0
            for row in ds:
                if n >= max_per_lang:
                    break
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                text = unicodedata.normalize("NFC", text)
                if len(text) < 2:
                    continue
                lines.append(" ".join(text.split()))
                n += 1
            print(f"  Got {n} (total {len(lines)})", flush=True)
        except Exception as e:
            print(f"  Skip {lang_code}: {e}", file=sys.stderr)
            if "gated" in str(e).lower() or "access" in str(e).lower() or "401" in str(e):
                print("  Tip: run 'huggingface-cli login' and accept terms at https://huggingface.co/datasets/uonlp/CulturaX", file=sys.stderr)
            continue
        if len(lines) >= max_samples:
            break
    lines = lines[:max_samples]
    if not lines:
        raise SystemExit("No CulturaX data loaded. If gated, login and accept dataset terms.")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Saved {len(lines)} lines to {output_path}")
    return len(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Download multilingual corpus for Qwen (HPLT or CulturaX)."
    )
    parser.add_argument(
        "--source",
        choices=("hplt", "culturax"),
        default="hplt",
        help="hplt: HPLT v3 (no login). culturax: CulturaX on HF (gated, best quality).",
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
        help="Max lines to save (default 200000)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Language codes (default: 20 languages for HPLT or CulturaX)",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output filename (default: qwen_multilingual_<N>k.txt)",
    )
    parser.add_argument(
        "--max_shards_per_lang",
        type=int,
        default=2,
        help="HPLT only: max shards per language (default 2)",
    )
    args = parser.parse_args()

    k = args.max_samples // 1000
    out_fname = args.output_file or f"qwen_multilingual_{k}k.txt"
    out_path = os.path.join(args.output_dir, out_fname)

    if args.source == "hplt":
        run_hplt(
            args.max_samples,
            out_path,
            args.languages,
            args.max_shards_per_lang,
        )
    else:
        run_culturax(args.max_samples, out_path, args.languages, None)


if __name__ == "__main__":
    main()
