#!/usr/bin/env bash
# One-shot fine-tune of Bolmo on a corpus. You do *not* need to run fine-tune
# multiple times: a single run trains on the entire dataset for num_train_epochs.
#
# Usage:
#   bash src/finetune_bolmo.sh
#     Uses work/data/opensubtitles_en_hi_20k.txt (or create it first).
#
#   bash src/finetune_bolmo.sh --train_data work/data/opensubtitles_en_hi_full.txt
#     Fine-tune on an existing file (e.g. full OpenSubtitles).
#
#   bash src/finetune_bolmo.sh --download
#     Download full OpenSubtitles en-hi, then fine-tune on it (one run).
#
#   bash src/finetune_bolmo.sh --download --streaming
#     Download full en-hi, then fine-tune with --streaming (safe: no full file in RAM).
#
#   bash src/finetune_bolmo.sh --train_data work/data/opensubtitles_en_hi_full.txt --streaming
#     Fine-tune on existing full file using streaming (recommended for huge files).
#
set -e
set -v

if [ -d /job/src ] && [ -d /job/work ]; then
  cd /job
fi

WORK_DIR="${WORK_DIR:-work}"
DOWNLOAD=""
STREAMING=""
LANG1="en"
LANG2="hi"
TRAIN_DATA=""

# Parse optional --download, --streaming, --train_data (and --lang1/--lang2 for download)
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download)
      DOWNLOAD=1
      shift
      ;;
    --streaming)
      STREAMING="--streaming"
      shift
      ;;
    --train_data)
      TRAIN_DATA="$2"
      shift 2
      ;;
    --lang1)
      LANG1="$2"
      shift 2
      ;;
    --lang2)
      LANG2="$2"
      shift 2
      ;;
    --work_dir)
      WORK_DIR="$2"
      shift 2
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

if [ -n "$DOWNLOAD" ]; then
  echo "Downloading full OpenSubtitles $LANG1-$LANG2 ..."
  python3 src/download_opensubtitles_data.py --all --lang1 "$LANG1" --lang2 "$LANG2" --output_dir "$WORK_DIR/data"
  if [ -z "$TRAIN_DATA" ]; then
    TRAIN_DATA="$WORK_DIR/data/opensubtitles_${LANG1}_${LANG2}_full.txt"
  fi
fi

if [ -z "$TRAIN_DATA" ]; then
  TRAIN_DATA="$WORK_DIR/data/opensubtitles_en_hi_20k.txt"
fi

if [ ! -f "$TRAIN_DATA" ]; then
  echo "Training data not found: $TRAIN_DATA"
  echo "Create it with: python3 src/download_opensubtitles_data.py --max_lines 20000"
  echo "Or run this script with: bash src/finetune_bolmo.sh --download"
  exit 1
fi

echo "Fine-tuning Bolmo on entire dataset: $TRAIN_DATA (one run = whole dataset)"
# Use --streaming when training on full/large file to avoid loading whole corpus into RAM
python3 src/myprogram.py train --work_dir "$WORK_DIR" --finetune --train_data "$TRAIN_DATA" $STREAMING "${EXTRA[@]}"
