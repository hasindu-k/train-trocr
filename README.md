# train-trocr

Training utilities for fine-tuning a Sinhala TrOCR model on handwritten samples.

## Current implementation

This repository currently contains:

- `train_handwritten_model.py`: main training pipeline
- `download-dataset.ipynb`: downloads and places dataset files
- `setup-venv.ipynb`: local virtual environment setup
- `train-handwritten-model.ipynb`: notebook version of the training workflow

## What the training script does

`train_handwritten_model.py`:

1. Loads `HF_TOKEN` from environment and logs in to Hugging Face.
2. Loads base checkpoint: `eshangj/TrOCR-Sinhala-finetuned`.
3. Loads metadata CSV and keeps only rows where `is_handwritten` is true.
4. Splits data into train/eval (90/10).
5. Trains with `Seq2SeqTrainer` and computes CER/WER.
6. Saves model + processor artifacts.
7. Writes run metrics to a JSONL log file.
8. Uploads model artifacts and training log to Hugging Face Hub.

## Required environment variables

- `HF_TOKEN` (required)

## Optional environment variables

- `DATA_DIR` (default: `/datasets/train_images`)
- `CSV_PATH` (default: `/datasets/metadata.csv`)
- `SAVE_PATH` (default: `./sinhala_model_v3`)
- `HF_REPO_NAME` (default: `hasindu-k/sinhala-handwritten-notes-v3`)
- `LOG_PATH` (default: `./training_logs/training_metrics.jsonl`)
- `KAGGLE_API_TOKEN` (used by dataset download notebook)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file (based on `.env.example`) and set at least `HF_TOKEN`.

3. Prepare dataset:
   - Use `download-dataset.ipynb`, or
   - Manually place images in `DATA_DIR` and CSV in `CSV_PATH`.

CSV must include these columns:

- `file_name`
- `text`
- `is_handwritten`

## Run training

```bash
python train_handwritten_model.py
```

