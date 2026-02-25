#!/usr/bin/env python
"""
Download BELLE conversation data (generated_chat_0.4M) and save a subset
as one line per sample under work/data for training (e.g. --train_data).
Supports --offset to download the next N lines (e.g. offset=20000 for 20k–40k).
"""
import os
import argparse

OUTPUT_DIR_DEFAULT = "work/data"
OUTPUT_FNAME = "belle_conversation_20k.txt"
MAX_LINES_DEFAULT = 20_000


def main():
    parser = argparse.ArgumentParser(
        description="Download BELLE conversation data; use --offset for next chunk (e.g. 0, 20000, 40000)."
    )
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT, help="Directory to write the corpus file")
    parser.add_argument("--max_lines", type=int, default=MAX_LINES_DEFAULT, help="Number of lines to save (default 20k)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many samples, then save max_lines (default 0)")
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output filename (default: belle_conversation_20k.txt if offset=0, else belle_conversation_20k_offset{N}.txt)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: pip install datasets")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_file is not None:
        out_fname = args.output_file
    elif args.offset == 0:
        out_fname = OUTPUT_FNAME
    else:
        out_fname = f"belle_conversation_20k_offset{args.offset}.txt"
    out_path = os.path.join(args.output_dir, out_fname)

    n = args.max_lines
    skip = args.offset
    print(f"Loading BelleGroup/generated_chat_0.4M (offset={skip}, taking {n} samples) ...")
    dataset = load_dataset("BelleGroup/generated_chat_0.4M", split="train")

    lines = []
    for i, row in enumerate(dataset):
        if i < skip:
            continue
        if i >= skip + n:
            break
        instruction = (row.get("instruction") or "").strip()
        output = (row.get("output") or "").strip()
        # One line per sample: instruction + output, newlines replaced by space
        text = " ".join((instruction + " " + output).split())
        if text:
            lines.append(text)

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Saved {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
