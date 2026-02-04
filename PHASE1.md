# Phase 1: Bolmo-1B download and deploy

## What was done

- **Model**: Allen AI Bolmo-1B (byte-level causal LM) is used for next-character prediction.
- **Location**: Model is stored under `work/bolmo-1b/`. The `work/` directory is mounted in Docker at `/job/work` during grading.
- **predict.sh**: Runs **inference only** (load from `work/`, no training). Uses `myprogram.py test`.

## How to download and deploy Bolmo-1B

**Option A – one-time download script (recommended)**

```bash
# From repo root, with venv/conda activated and deps installed
pip install -r requirements.txt
python src/download_bolmo.py --work_dir work
```

This downloads `allenai/Bolmo-1B` from Hugging Face and saves it to `work/bolmo-1b/`.

**Option B – use train mode**

```bash
python src/myprogram.py train --work_dir work
```

This loads Bolmo (from HF if `work/bolmo-1b/` is missing), saves it to `work/bolmo-1b/`, and does not run any training.

## Run inference

```bash
# After work/bolmo-1b/ is populated
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt
# Or via the grading interface
bash src/predict.sh example/input.txt pred.txt
```

## Docker

- Build: `docker build -t cse447-proj/demo -f Dockerfile .`
- Run prediction (from host, with `work/` and `src/` present):
  ```bash
  mkdir -p output
  docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output \
    cse447-proj/demo bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
  ```

Ensure `work/bolmo-1b/` exists (via Option A or B) before building/running so the container has the model.

## Dependencies (in requirements.txt)

- `transformers>=4.57.3`
- `xlstm==2.0.4` (Python ≥3.11)
- `huggingface_hub>=0.20.0`
