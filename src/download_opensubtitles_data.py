#!/usr/bin/env python
"""
Download OpenSubtitles (OPUS) data directly from OPUS (object.pouta.csc.fi)
and save as one line per sample under work/data for training.

Hugging Face Helsinki-NLP/open_subtitles no longer supports dataset scripts,
so we download the moses .txt.zip for a language pair and extract both sides.

Usage:
  conda activate cse447
  python src/download_opensubtitles_data.py --lang1 en --lang2 hi --max_lines 20000
  python src/download_opensubtitles_data.py  # defaults: en, hi, 20k lines
  python src/download_opensubtitles_data.py --all  # full language pair (no limit; zip can be 100MB–1GB+)
"""
import os
import sys
import argparse
import zipfile
import tempfile
import urllib.request

OPUS_BASE_URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses"
# Inside zip: OpenSubtitles.{lang1}-{lang2}.{lang1} and .{lang2}
OUTPUT_DIR_DEFAULT = "work/data"
MAX_LINES_DEFAULT = 20_000


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenSubtitles (OPUS) from object.pouta.csc.fi; save one line per sample."
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
        help="Max number of lines to save (default 20k). Use 0 for full pair (no limit).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the full language pair (no line limit). Same as --max_lines 0.",
    )
    parser.add_argument(
        "--lang1",
        default="en",
        help="First language code (e.g. en, es, fr). Default: en",
    )
    parser.add_argument(
        "--lang2",
        default="hi",
        help="Second language code (e.g. hi, es, de). Default: hi",
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
    args = parser.parse_args()
    if args.all:
        args.max_lines = 0

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_file is not None:
        out_fname = args.output_file
    else:
        if args.max_lines == 0:
            out_fname = f"opensubtitles_{args.lang1}_{args.lang2}_full.txt"
        else:
            k = args.max_lines // 1000
            out_fname = f"opensubtitles_{args.lang1}_{args.lang2}_{k}k.txt"
    out_path = os.path.join(args.output_dir, out_fname)

    l1, l2 = args.lang1, args.lang2
    pair = f"{l1}-{l2}"
    url = f"{OPUS_BASE_URL}/{pair}.txt.zip"
    cap = f" (up to {args.max_lines} lines)" if args.max_lines else " (full pair)"
    print(f"Downloading OpenSubtitles {pair} from OPUS{cap} ...")
    print(f"  URL: {url}")

    cache_dir = args.cache_dir or tempfile.gettempdir()
    zip_path = os.path.join(cache_dir, f"OpenSubtitles_{pair}.txt.zip")

    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("Check that the language pair exists at https://opus.nlpl.eu/OpenSubtitles.php", file=sys.stderr)
        raise SystemExit(1) from e

    # Expected files inside zip: OpenSubtitles.{pair}.{l1} and OpenSubtitles.{pair}.{l2}
    base_name = f"OpenSubtitles.{pair}"
    f1_name = f"{base_name}.{l1}"
    f2_name = f"{base_name}.{l2}"

    lines = []
    max_lines = args.max_lines  # 0 means no limit

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        to_read = [nm for nm in names if nm.endswith(f".{l1}") or nm.endswith(f".{l2}")]
        if not to_read:
            to_read = [nm for nm in names if nm.endswith(".en") or nm.endswith(".hi")]
        if not to_read:
            print(f"Expected files like {f1_name} in zip; found: {names[:8]}", file=sys.stderr)
            raise SystemExit(1)
        for fname in to_read:
            with zf.open(fname) as f:
                for line in f:
                    if max_lines and len(lines) >= max_lines:
                        break
                    text = line.decode("utf-8", errors="replace").strip()
                    text = " ".join(text.split())
                    if text:
                        lines.append(text)
            if max_lines and len(lines) >= max_lines:
                break

    if max_lines:
        lines = lines[:max_lines]

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Saved {len(lines)} lines to {out_path}")
    if args.cache_dir is None and os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
