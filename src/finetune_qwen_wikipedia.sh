#!/usr/bin/env bash
# Download Wikipedia (multilingual) and fine-tune Qwen on it.
#
# Usage:
#   bash src/finetune_qwen_wikipedia.sh
#     Downloads 200k Wikipedia lines to work/data/wikipedia_multilingual_200k.txt (if missing),
#     then runs Qwen fine-tune on it.
#
#   bash src/finetune_qwen_wikipedia.sh --max_samples 100000
#     Download 100k then fine-tune.
#
set -e
set -v

if [ -d /job/src ] && [ -d /job/work ]; then
  cd /job
fi

WORK_DIR="${WORK_DIR:-work}"
DATA_DIR="${WORK_DIR}/data"
TRAIN_FILE="${DATA_DIR}/wikipedia_for_qwen.txt"
EXTRA_DOWNLOAD=()
EXTRA_TRAIN=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max_samples)
      EXTRA_DOWNLOAD+=(--max_samples "$2")
      shift 2
      ;;
    --work_dir)
      WORK_DIR="$2"
      DATA_DIR="${WORK_DIR}/data"
      TRAIN_FILE="${DATA_DIR}/wikipedia_for_qwen.txt"
      shift 2
      ;;
    *)
      EXTRA_TRAIN+=("$1")
      shift
      ;;
  esac
done

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Downloading Wikipedia to $TRAIN_FILE ..."
  python3 src/download_wikipedia_data.py --output_dir "$DATA_DIR" --output_file "$(basename "$TRAIN_FILE")" "${EXTRA_DOWNLOAD[@]}"
fi

echo "Fine-tuning Qwen on $TRAIN_FILE ..."
python3 src/myprogram_qwen.py train --work_dir "$WORK_DIR" --finetune --train_data "$TRAIN_FILE" "${EXTRA_TRAIN[@]}"
