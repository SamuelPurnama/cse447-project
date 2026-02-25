#!/usr/bin/env python
"""
Download ShareGPT-style conversation data (RyokoAI/ShareGPT52K, ~90k conversations,
multilingual including Japanese) and save a subset as one line per sample under
work/data for training. Supports --offset to download the next N lines (e.g. 20k–40k).

Uses raw JSON + streaming (ijson) because load_dataset() fails on this repo's Arrow conversion.
Requires: pip install ijson huggingface_hub
"""
import os
import argparse

OUTPUT_DIR_DEFAULT = "work/data"
OUTPUT_FNAME = "sharegpt_jp_20k.txt"
MAX_LINES_DEFAULT = 20_000
REPO_ID = "RyokoAI/ShareGPT52K"
# Part1 has ~52k items; part2 has the rest to ~90k
JSON_FILES = ["sg_90k_part1.json", "sg_90k_part2.json"]


def _conversation_to_line(conversations):
    """Flatten a list of {from, value} turns into one line of text."""
    if not conversations:
        return ""
    parts = []
    for turn in conversations:
        val = turn.get("value") if isinstance(turn, dict) else None
        if val and isinstance(val, str):
            parts.append(val.strip())
    return " ".join(" ".join(parts).split())


def _stream_items_from_json(json_path):
    """Yield conversation rows from a ShareGPT JSON file (array of {id, conversations})."""
    try:
        import ijson
    except ImportError:
        raise SystemExit("Install ijson for streaming JSON: pip install ijson")
    with open(json_path, "rb") as f:
        for obj in ijson.items(f, "item"):
            if isinstance(obj, dict):
                yield obj


def main():
    parser = argparse.ArgumentParser(
        description="Download ShareGPT (RyokoAI/ShareGPT52K) data; use --offset for next chunk (e.g. 0, 20000, 40000)."
    )
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT, help="Directory to write the corpus file")
    parser.add_argument("--max_lines", type=int, default=MAX_LINES_DEFAULT, help="Number of lines to save (default 20k)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many samples, then save max_lines (default 0)")
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output filename (default: sharegpt_jp_20k.txt if offset=0, else sharegpt_jp_20k_offset{N}.txt)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_file is not None:
        out_fname = args.output_file
    elif args.offset == 0:
        out_fname = OUTPUT_FNAME
    else:
        out_fname = f"sharegpt_jp_20k_offset{args.offset}.txt"
    out_path = os.path.join(args.output_dir, out_fname)

    n = args.max_lines
    skip = args.offset
    lines = []
    total_skipped = 0
    total_taken = 0

    for filename in JSON_FILES:
        if total_taken >= n:
            break
        print(f"Downloading {REPO_ID} {filename} ...")
        local_path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
        print(f"Streaming (offset={skip}, taking {n} samples) from {filename} ...")
        for row in _stream_items_from_json(local_path):
            if total_skipped < skip:
                total_skipped += 1
                continue
            if total_taken >= n:
                break
            conv = row.get("conversations") or []
            text = _conversation_to_line(conv)
            if text:
                lines.append(text)
                total_taken += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Saved {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
