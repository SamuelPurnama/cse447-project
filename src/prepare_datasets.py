#!/usr/bin/env python3
"""
Dataset preparation utilities for training:
- Download OpenSubtitles pairs (if needed)
- Concatenate into one multilingual corpus
- Split into English / non-English corpora

This module is intentionally separate from myprogram.py so training/inference
logic stays cleaner.
"""
import os
import sys
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


OPENSUBTITLES_MULTI_PAIRS = [
    ("en", "hi"),  # English + Hindi
    ("ja", "zh"),  # Japanese + Chinese
    ("es", "ko"),  # Spanish + Korean
]
OPENSUBTITLES_MULTI_FILENAME = "opensubtitles_multi_full.txt"


def _default_is_english_text(text):
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


def split_corpus_by_language(
    input_path, english_out_path, non_english_out_path, is_english_detector=None
):
    """
    Split corpus into English and non-English lines.
    Returns (english_count, non_english_count).
    """
    detector = is_english_detector or _default_is_english_text
    english_count = 0
    non_english_count = 0
    with open(input_path, encoding="utf-8") as src, open(
        english_out_path, "w", encoding="utf-8"
    ) as f_en, open(non_english_out_path, "w", encoding="utf-8") as f_non_en:
        for line in src:
            text = line.rstrip("\n").strip()
            if not text:
                continue
            if detector(text):
                f_en.write(text + "\n")
                english_count += 1
            else:
                f_non_en.write(text + "\n")
                non_english_count += 1
    return english_count, non_english_count


def build_language_split(
    base_corpus_path, work_dir, prefix, is_english_detector=None
):
    """
    Build English / non-English corpora from base_corpus_path under:
      work/data/language_splits/{prefix}_english.txt
      work/data/language_splits/{prefix}_non_english.txt
    """
    if not os.path.isfile(base_corpus_path):
        raise ValueError("Base corpus not found at {}".format(base_corpus_path))
    data_dir = os.path.join(work_dir, "data", "language_splits")
    os.makedirs(data_dir, exist_ok=True)
    english_path = os.path.join(data_dir, "{}_english.txt".format(prefix))
    non_english_path = os.path.join(data_dir, "{}_non_english.txt".format(prefix))
    en_count, non_en_count = split_corpus_by_language(
        base_corpus_path,
        english_path,
        non_english_path,
        is_english_detector=is_english_detector,
    )
    print(
        "Language split for {}: English={}, Non-English={}.".format(
            prefix, en_count, non_en_count
        )
    )
    return english_path, non_english_path


def _download_opensubtitles_pair(work_dir, l1, l2):
    """Download one OpenSubtitles pair with --all if missing."""
    data_dir = os.path.join(work_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    pair_path = os.path.join(data_dir, "opensubtitles_{}_{}_full.txt".format(l1, l2))
    if os.path.isfile(pair_path):
        print("OpenSubtitles pair already exists: {}".format(pair_path))
        return pair_path

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    downloader = os.path.join(repo_root, "src", "download_opensubtitles_data.py")
    cmd = [
        sys.executable,
        downloader,
        "--output_dir",
        data_dir,
        "--all",
        "--lang1",
        l1,
        "--lang2",
        l2,
    ]
    print("Downloading OpenSubtitles pair {}-{} ...".format(l1, l2))
    subprocess.run(cmd, check=True)
    return pair_path


def build_opensubtitles_multi_corpus(work_dir, download_if_missing=False):
    """
    Build combined OpenSubtitles corpus from en-hi, ja-zh, es-ko pair files.
    Returns path to work/data/opensubtitles_multi_full.txt, or None if missing.
    """
    data_dir = os.path.join(work_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, OPENSUBTITLES_MULTI_FILENAME)

    paths_to_concat = []
    for l1, l2 in OPENSUBTITLES_MULTI_PAIRS:
        pair_path = os.path.join(data_dir, "opensubtitles_{}_{}_full.txt".format(l1, l2))
        if not os.path.isfile(pair_path):
            if download_if_missing:
                pair_path = _download_opensubtitles_pair(work_dir, l1, l2)
            else:
                print(
                    "OpenSubtitles multi corpus: missing {}. Run: python3 src/download_opensubtitles_data.py "
                    "--output_dir work/data --all --lang1 {} --lang2 {}".format(
                        pair_path, l1, l2
                    )
                )
                return None
        paths_to_concat.append(pair_path)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for p in paths_to_concat:
            with open(p, encoding="utf-8") as in_f:
                for line in in_f:
                    if line.rstrip("\n").strip():
                        out_f.write(line)
    print(
        "OpenSubtitles multi corpus: concatenated {} -> {} ({} files).".format(
            paths_to_concat, out_path, len(paths_to_concat)
        )
    )
    return out_path


def prepare_opensubtitles_dataset(work_dir, download_if_missing=False):
    """
    End-to-end preparation for OpenSubtitles pipeline:
    1) Build multilingual combined corpus
    2) Split to English/non-English corpora
    Returns (multi_path, english_path, non_english_path)
    """
    multi_path = build_opensubtitles_multi_corpus(
        work_dir, download_if_missing=download_if_missing
    )
    if multi_path is None:
        return None, None, None
    english_path, non_english_path = build_language_split(
        multi_path, work_dir, prefix="opensubtitles_multi"
    )
    return multi_path, english_path, non_english_path


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_dir", default="work", help="Working directory root")
    parser.add_argument(
        "--dataset",
        choices=("opensubtitles",),
        default="opensubtitles",
        help="Dataset family to prepare",
    )
    parser.add_argument(
        "--download_if_missing",
        action="store_true",
        help="Automatically download missing OpenSubtitles pair files",
    )
    args = parser.parse_args()

    if args.dataset == "opensubtitles":
        multi_path, en_path, non_en_path = prepare_opensubtitles_dataset(
            args.work_dir, download_if_missing=args.download_if_missing
        )
        if multi_path is None:
            raise SystemExit("Failed to prepare OpenSubtitles dataset (missing pair files).")
        print("Prepared datasets:")
        print("  Multi: {}".format(multi_path))
        print("  English: {}".format(en_path))
        print("  Non-English: {}".format(non_en_path))
