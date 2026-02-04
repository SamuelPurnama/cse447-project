#!/usr/bin/env bash
# Inference only: load model from work_dir and write predictions. No training.
set -e
set -v
# In Docker, working dir is /job; ensure we run from project root so work/ and src/ resolve
if [ -d /job/src ] && [ -d /job/work ]; then
  cd /job
fi
python src/myprogram.py test --work_dir work --test_data "$1" --test_output "$2"
