#!/usr/bin/env python
"""
Download OpenSubtitles (OPUS) data and save as one line per sample, plus
two merged split files: all English in one file, all other languages in another.

Each language pair is capped at --max_lines_per_pair (default 5M lines) to avoid
overtuning on huge pairs (e.g. en-es ~1.2GB vs en-hi ~100MB). When over the cap,
we trim from the TOP (drop the first N sentence pairs) so the other language
is prioritized (OPUS order is English file first, then the other language).

Writes to output_dir (default work/data) only the two merged split files:
  - opensubtitles_english.txt    (all English lines from every pair)
  - opensubtitles_non_english.txt (all non-English lines from every pair)
Per-pair files (e.g. opensubtitles_en_es_max5M.txt) are written temporarily and deleted after merging.

Language is determined by the source file in the zip (.en -> English).

Usage:
  conda activate cse447
  python src/download_opensubtitles_with_splits.py --langs hi,ja,zh,es,ko --all
  python src/download_opensubtitles_with_splits.py --langs hi --max_lines 20000
  python src/download_opensubtitles_with_splits.py  # single pair: en, hi, 20k lines
  python src/download_opensubtitles_with_splits.py --no_split  # skip split files (combined only)
  python src/download_opensubtitles_with_splits.py --max_lines_per_pair 3000000  # 3M cap
"""
import os
import sys
import argparse
import zipfile
import tempfile
import urllib.request

OPUS_BASE_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses"
OUTPUT_DIR_DEFAULT = "work/data"
MAX_LINES_DEFAULT = 20_000
# Cap per language pair to avoid overtuning on huge pairs (e.g. en-es ~1.2GB vs en-hi ~100MB).
# When over cap we trim from the TOP so the other language is prioritized (data is en first, then other).
MAX_LINES_PER_PAIR_DEFAULT = 5_000_000


def _normalize(text):
    text = text.decode("utf-8", errors="replace").strip()
    return " ".join(text.split())


def _count_lines(zf, fname):
    """Count non-empty lines in a zip member."""
    n = 0
    with zf.open(fname) as f:
        for line in f:
            if _normalize(line):
                n += 1
    return n


def _download_one_pair(*, output_dir, output_file, max_lines, max_lines_per_pair, l1, l2, no_split, cache_dir, english_file=None, non_english_file=None):
    """Download one OpenSubtitles pair; write combined file and optionally to merged english_file / non_english_file.
    When a pair exceeds max_lines_per_pair, we trim from the TOP (skip first N pairs) so the other language
    is prioritized over English (OPUS order is English file first, then the other language).
    """
    if output_file is not None:
        out_fname = output_file
    else:
        effective_cap = max_lines_per_pair if max_lines == 0 else min(max_lines, max_lines_per_pair)
        if max_lines == 0:
            out_fname = f"opensubtitles_{l1}_{l2}_max{effective_cap // 1_000_000}M.txt"
        else:
            k = effective_cap // 1000
            out_fname = f"opensubtitles_{l1}_{l2}_{k}k.txt"
    out_path = os.path.join(output_dir, out_fname)

    pair = f"{l1}-{l2}"
    url = f"{OPUS_BASE_URL}/{pair}.txt.zip"
    cap = f" (up to {max_lines} lines)" if max_lines else f" (capped at {max_lines_per_pair} lines, trim from top)"
    print(f"Downloading OpenSubtitles {pair} from OPUS{cap} ...")
    print(f"  URL: {url}")

    zip_path = os.path.join(cache_dir or tempfile.gettempdir(), f"OpenSubtitles_{pair}.txt.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("Check that the language pair exists at https://opus.nlpl.eu/OpenSubtitles.php", file=sys.stderr)
        raise SystemExit(1) from e

    base_name = f"OpenSubtitles.{pair}"
    f1_name = f"{base_name}.{l1}"
    f2_name = f"{base_name}.{l2}"

    total = 0
    n_en = 0
    n_non_en = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        to_read = [nm for nm in names if nm.endswith(f".{l1}") or nm.endswith(f".{l2}")]
        if not to_read:
            to_read = [nm for nm in names if nm.endswith(f".{l1}") or nm.endswith(f".{l2}")]
        if not to_read:
            print(f"Expected files like {f1_name} in zip; found: {names[:8]}", file=sys.stderr)
            raise SystemExit(1)

        # Order: same as current behavior — typically .en first then .other (by name)
        to_read = sorted(to_read)

        effective_max = max_lines_per_pair if max_lines == 0 else min(max_lines, max_lines_per_pair)

        # Count lines in BOTH files; use min so we don't skip past the shorter file (e.g. en-es can
        # have a much larger .en file than .es, which would otherwise write 0 lines to non_english).
        counts = [_count_lines(zf, fname) for fname in to_read]
        total_in_corpus = min(counts)
        if len(counts) == 2 and counts[0] != counts[1]:
            print(f"  Line counts: {counts[0]}, {counts[1]} (using min={total_in_corpus} parallel pairs)")
        skip = max(0, total_in_corpus - effective_max)
        if skip > 0:
            print(f"  Pair has {total_in_corpus} parallel pairs; capping at {effective_max} (trimming first {skip} from top to prioritize other language).")

        with open(out_path, "w", encoding="utf-8") as out_f:
            for idx, fname in enumerate(to_read):
                if fname.endswith(f".{l1}"):
                    is_english = l1 == "en"
                elif fname.endswith(f".{l2}"):
                    is_english = l2 == "en"
                else:
                    is_english = fname.endswith(".en")
                with zf.open(fname) as f:
                    written_this_file = 0
                    skipped = 0
                    for line in f:
                        text = _normalize(line)
                        if not text:
                            continue
                        if skip > 0:
                            if skipped < skip:
                                skipped += 1
                                continue
                        if written_this_file >= effective_max:
                            continue
                        out_f.write(text + "\n")
                        written_this_file += 1
                        total += 1
                        if not no_split and english_file is not None and non_english_file is not None:
                            if is_english:
                                english_file.write(text + "\n")
                                n_en += 1
                            else:
                                non_english_file.write(text + "\n")
                                n_non_en += 1

    if no_split:
        print(f"Saved {total} lines to {out_path}")
    else:
        print(f"Merged {total} lines into split files (English {n_en}, non-English {n_non_en})")
        # Keep only opensubtitles_english.txt and opensubtitles_non_english.txt; remove per-pair file.
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
                print(f"  Removed per-pair file {out_fname}")
            except OSError as e:
                print(f"  Warning: could not remove {out_path}: {e}", file=sys.stderr)
    if cache_dir is None and os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenSubtitles (OPUS); save combined + English/non-English split files."
    )
    parser.add_argument(
        "--output_dir",
        default=OUTPUT_DIR_DEFAULT,
        help="Directory to write the corpus file",
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=MAX_LINES_DEFAULT,
        help="Max number of lines to save (default 20k). Use 0 for full pair (then capped by --max_lines_per_pair).",
    )
    parser.add_argument(
        "--max_lines_per_pair",
        type=int,
        default=MAX_LINES_PER_PAIR_DEFAULT,
        help="Hard cap per language pair in lines (default 5M). When exceeded we trim from the top to prioritize the other language.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the full language pair (no line limit). Same as --max_lines 0.",
    )
    parser.add_argument(
        "--lang1",
        default="en",
        help="First language code; usually en. Default: en",
    )
    parser.add_argument(
        "--lang2",
        default="hi",
        help="Second language code when using a single pair. Default: hi",
    )
    parser.add_argument(
        "--langs",
        default=None,
        metavar="L1,L2,...",
        help="Comma-separated non-English language codes to download (en-X for each). e.g. hi,ja,zh,es,ko",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output filename (default: opensubtitles_{lang1}_{lang2}_{k}k.txt)",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Directory to cache the downloaded zip (default: temp dir)",
    )
    parser.add_argument(
        "--no_split",
        action="store_true",
        help="Do not write English/non-English split files (only the combined file).",
    )
    args = parser.parse_args()
    if args.all:
        args.max_lines = 0

    # Which pairs to download: --langs hi,ja,zh -> [(en,hi), (en,ja), (en,zh)]; else [(lang1, lang2)]
    if args.langs:
        other_langs = [x.strip() for x in args.langs.split(",") if x.strip()]
        pairs = [(args.lang1, lang) for lang in other_langs]
    else:
        pairs = [(args.lang1, args.lang2)]

    os.makedirs(args.output_dir, exist_ok=True)

    english_path = os.path.join(args.output_dir, "opensubtitles_english.txt")
    non_english_path = os.path.join(args.output_dir, "opensubtitles_non_english.txt")

    if args.no_split:
        en_f = non_en_f = None
    else:
        en_f = open(english_path, "w", encoding="utf-8")
        non_en_f = open(non_english_path, "w", encoding="utf-8")

    try:
        for l1, l2 in pairs:
            _download_one_pair(
                output_dir=args.output_dir,
                output_file=args.output_file if len(pairs) == 1 else None,
                max_lines=args.max_lines,
                max_lines_per_pair=args.max_lines_per_pair,
                l1=l1,
                l2=l2,
                no_split=args.no_split,
                cache_dir=args.cache_dir,
                english_file=en_f,
                non_english_file=non_en_f,
            )
    finally:
        if en_f is not None:
            en_f.close()
        if non_en_f is not None:
            non_en_f.close()

    if not args.no_split:
        print(f"Merged splits: {english_path}, {non_english_path}")


if __name__ == "__main__":
    main()
