#!/usr/bin/env bash
# One-shot fine-tune of Qwen on a corpus. One run = train on the entire dataset
# for num_train_epochs (dataset is loaded into memory; keep file size within RAM).
#
# Usage:
#   bash src/finetune_qwen.sh
#     Uses work/data/wikipedia_for_qwen.txt (create with download_wikipedia_data.py if missing).
#
#   bash src/finetune_qwen.sh --train_data work/data/wikipedia_multilingual_200k.txt
#     Fine-tune on a specific corpus file.
#
#   bash src/finetune_qwen.sh --train_data work/data/belle_conversation_20k.txt --num_train_epochs 2
#     Pass extra args to myprogram_qwen.py.
#
set -e
set -v

if [ -d /job/src ] && [ -d /job/work ]; then
  cd /job
fi

WORK_DIR="${WORK_DIR:-work}"
TRAIN_DATA="${WORK_DIR}/data/wikipedia_for_qwen.txt"
STREAMING=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_data)
      TRAIN_DATA="$2"
      shift 2
      ;;
    --streaming)
      STREAMING="--streaming"
      shift
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

if [ ! -f "$TRAIN_DATA" ]; then
  echo "Training data not found: $TRAIN_DATA"
  echo "Create it with one of:"
  echo "  python3 src/download_wikipedia_data.py --output_dir ${WORK_DIR}/data --output_file wikipedia_for_qwen.txt"
  echo "  python3 src/download_mc4_data.py --max_samples 200000"
  echo "Or run: bash src/finetune_qwen_wikipedia.sh"
  exit 1
fi

echo "Fine-tuning Qwen on entire dataset: $TRAIN_DATA"
# Use --streaming for large files to avoid loading whole corpus into RAM
python3 src/myprogram_qwen.py train --work_dir "$WORK_DIR" --finetune --train_data "$TRAIN_DATA" $STREAMING "${EXTRA[@]}"
