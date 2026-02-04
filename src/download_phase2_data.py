#!/usr/bin/env python
"""
Phase 2: Download conversational/interactive multilingual samples from HPLT v3,
normalize to Unicode NFC, and store as UTF-8.

Produces work/data/phase2/corpus_utf8_nfc.txt with one sample per line (100k by default).
Uses HPLT web-register labels (e.g. ID=Interactive discussion) when available for filtering.
"""
import os
import sys
import unicodedata
import argparse
import json

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import zstandard as zstd
except ImportError:
    print("Install zstandard: pip install zstandard", file=sys.stderr)
    sys.exit(1)


HPLT_BASE = "https://data.hplt-project.org/three/sorted"

# Diverse languages for balanced sampling (language_code from HPLT map names).
# Each will be capped at max_per_lang samples.
DEFAULT_LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "fra_Latn",
    "deu_Latn",
    "cmn_Hans",
    "jpn_Jpan",
    "arb_Arab",
    "hin_Deva",
    "por_Latn",
    "rus_Cyrl",
    "ita_Latn",
    "kor_Hang",
    "tha_Thai",
    "vie_Latn",
    "tur_Latn",
    "pol_Latn",
    "ind_Latn",
    "nld_Latn",
    "ell_Grek",
    "heb_Hebr",
]

# Register keys that indicate conversational / interactive content (Turku annotation).
# ID = Interactive discussion, IP = Informational persuasion, IN = Informational.
CONVERSATIONAL_REGISTER_KEYS = ("ID", "IP", "IN")
REGISTER_THRESHOLD = 0.2  # Keep doc if any of these scores >= threshold


def fetch_map_urls(lang_code: str) -> list[str]:
    """Return list of HPLT shard URLs for a language (from .map file)."""
    url = f"{HPLT_BASE}/{lang_code}.map"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return [u.strip() for u in r.text.strip().split("\n") if u.strip()]


def is_conversational(web_register: dict) -> bool:
    """True if document has non-trivial conversational/interactive register score."""
    if not web_register or not isinstance(web_register, dict):
        return True  # No register info: keep by default
    for key in CONVERSATIONAL_REGISTER_KEYS:
        try:
            if float(web_register.get(key, 0)) >= REGISTER_THRESHOLD:
                return True
        except (TypeError, ValueError):
            continue
    return False


def iter_jsonl_zst_url(url: str, max_docs: int):
    """Download a .jsonl.zst URL, decompress, and yield (text, web_register) for each doc."""
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    dctx = zstd.ZstdDecompressor()
    count = 0
    buf = ""
    with dctx.stream_reader(r.raw) as reader:
        for chunk in iter(lambda: reader.read(65536), b""):
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if count >= max_docs:
                    return
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") if isinstance(obj.get("text"), str) else None
                if not text or not text.strip():
                    continue
                yield text.strip(), obj.get("web-register")
                count += 1


def sample_from_language(
    lang_code: str,
    max_samples: int,
    max_shards: int,
) -> list[str]:
    """Collect up to max_samples NFC-normalized text samples from one language."""
    urls = fetch_map_urls(lang_code)
    if not urls:
        return []
    # Use first max_shards shards (highest quality bin) to limit download size.
    urls = urls[: max_shards]
    samples = []
    for u in urls:
        if len(samples) >= max_samples:
            break
        try:
            for text, web_register in iter_jsonl_zst_url(u, max_samples - len(samples)):
                if not is_conversational(web_register):
                    continue
                normalized = unicodedata.normalize("NFC", text)
                # Skip if empty after normalization or too short
                if not normalized.strip() or len(normalized.strip()) < 2:
                    continue
                samples.append(normalized)
                if len(samples) >= max_samples:
                    break
        except Exception as e:
            print(f"  Warning: {lang_code} shard {u[:60]}... failed: {e}", file=sys.stderr)
            continue
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Download 100k HPLT v3 conversational samples, NFC normalize, save as UTF-8."
    )
    parser.add_argument(
        "--output_dir",
        default="work/data/phase2",
        help="Output directory; corpus will be output_dir/corpus_utf8_nfc.txt",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100_000,
        help="Target number of samples (default 100000)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help=f"Language codes (default: {len(DEFAULT_LANGUAGES)} languages)",
    )
    parser.add_argument(
        "--max_shards_per_lang",
        type=int,
        default=2,
        help="Max HPLT shards to fetch per language (default 2, to limit download)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "corpus_utf8_nfc.txt")

    languages = args.languages if args.languages else DEFAULT_LANGUAGES
    max_per_lang = max(1, (args.max_samples + len(languages) - 1) // len(languages))

    all_samples = []
    for lang_code in languages:
        print(f"Fetching {lang_code} (max {max_per_lang} samples)...", flush=True)
        samples = sample_from_language(
            lang_code,
            max_samples=max_per_lang,
            max_shards=args.max_shards_per_lang,
        )
        all_samples.extend(samples)
        print(f"  Got {len(samples)} samples (total so far: {len(all_samples)})", flush=True)
        if len(all_samples) >= args.max_samples:
            break

    all_samples = all_samples[: args.max_samples]

    print(f"Writing {len(all_samples)} samples to {out_path} ...", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in all_samples:
            f.write(line)
            f.write("\n")
    print(f"Done. Corpus saved to {out_path}")


if __name__ == "__main__":
    main()
