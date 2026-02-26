#!/usr/bin/env bash
set -e
set -v

if [ -d /job/src ] && [ -d /job/work ]; then
  cd /job
fi

WORK_DIR="${WORK_DIR:-work}"
TRAIN_DATA=""
DATASET_KEY="${DATASET_KEY:-opensubtitles_non_english}"
LIST_DATASETS=""
EXTRA=()

declare -A DATASET_OPTIONS=(
  [opensubtitles_non_english]="${WORK_DIR}/data/opensubtitles_non_english.txt"
  [wikipedia_qwen]="${WORK_DIR}/data/wikipedia_for_qwen.txt"
  [belle_20k]="${WORK_DIR}/data/belle_conversation_20k.txt"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train_data) TRAIN_DATA="$2"; shift 2 ;;
    --dataset_key) DATASET_KEY="$2"; shift 2 ;;
    --list_datasets) LIST_DATASETS=1; shift ;;
    --work_dir) WORK_DIR="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [ -n "$LIST_DATASETS" ]; then
  echo "Available Qwen datasets:"
  for k in "${!DATASET_OPTIONS[@]}"; do
    echo "  $k -> ${DATASET_OPTIONS[$k]}"
  done
  exit 0
fi

if [ -z "$TRAIN_DATA" ]; then
  TRAIN_DATA="${DATASET_OPTIONS[$DATASET_KEY]}"
  if [ -z "$TRAIN_DATA" ]; then
    echo "Unknown --dataset_key: $DATASET_KEY"
    exit 1
  fi
fi

if [ ! -f "$TRAIN_DATA" ]; then
  echo "Training data not found: $TRAIN_DATA"
  exit 1
fi

echo "Fine-tuning Qwen on dataset: $TRAIN_DATA"
python3 src/train_qwen.py --work_dir "$WORK_DIR" --train_data "$TRAIN_DATA" "${EXTRA[@]}"
