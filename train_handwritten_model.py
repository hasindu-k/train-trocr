# =============================
# INSTALL LIBRARIES
# =============================
# pip install transformers datasets jiwer accelerate huggingface_hub matplotlib evaluate python-dotenv

# =============================
# IMPORTS
# =============================
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from huggingface_hub import login, upload_folder
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import evaluate
from datetime import datetime
import json

# =============================
# LOGIN
# =============================
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment. Add it to .env or set environment variable.")

login(hf_token)

# =============================
# LOAD BASE CHECKPOINT
# =============================
model_id = "eshangj/TrOCR-Sinhala-finetuned"

processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on device: {device}")

# =============================
# DATA PATH
# =============================
DATA_DIR = os.getenv("DATA_DIR", "/datasets/train_images")
CSV_PATH = os.getenv("CSV_PATH", "/datasets/metadata.csv")


class SinhalaDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]["file_name"]
        text = self.df.iloc[idx]["text"]

        img_path = os.path.join(DATA_DIR, file_name)
        image = Image.open(img_path).convert("RGB")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# =============================
# LOAD DATA
# =============================
df = pd.read_csv(CSV_PATH)
if "is_handwritten" not in df.columns:
    raise ValueError("CSV is missing required column: is_handwritten")

is_handwritten_map = {
    "true": True, "1": True, "yes": True,
    "false": False, "0": False, "no": False,
}
df["is_handwritten"] = (
    df["is_handwritten"].astype(str).str.strip().str.lower().map(is_handwritten_map)
)
df = df[df["is_handwritten"] == True].reset_index(drop=True)

if df.empty:
    raise ValueError("No handwritten samples found after filtering is_handwritten == true.")

val_df = df.sample(frac=0.1, random_state=42)
train_df = df.drop(val_df.index).reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_dataset = SinhalaDataset(train_df, processor)
eval_dataset = SinhalaDataset(val_df, processor)

# =============================
# METRICS
# =============================
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def postprocess_text(preds, labels):
    return [p.strip() for p in preds], [l.strip() for l in labels]


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"cer": cer, "wer": wer}


# =============================
# RUN LOGGING
# =============================
LOG_PATH = os.getenv("LOG_PATH", "./training_logs/training_metrics.jsonl")


def append_run_log(log_entry):
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def verify_saved_artifacts(save_dir):
    required_files = ["config.json"]
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    processor_files = ["preprocessor_config.json", "processor_config.json"]

    existing_files = set(os.listdir(save_dir))
    missing = [f for f in required_files if f not in existing_files]
    if not any(f in existing_files for f in weight_files):
        missing.append("model weights (model.safetensors or pytorch_model.bin)")
    if not any(f in existing_files for f in processor_files):
        missing.append("processor config (preprocessor_config.json or processor_config.json)")

    if missing:
        raise FileNotFoundError(f"Missing saved artifacts in {save_dir}: {missing}")

    return sorted(existing_files)


def get_best_epoch_metrics(log_history):
    epoch_entries = [e for e in log_history if "eval_loss" in e]
    if not epoch_entries:
        return None
    best = min(epoch_entries, key=lambda e: e["eval_loss"])
    return {
        "epoch": int(best["epoch"]),
        "eval_loss": float(best["eval_loss"]),
        "eval_cer": float(best["eval_cer"]) if "eval_cer" in best else None,
        "eval_wer": float(best["eval_wer"]) if "eval_wer" in best else None,
        "step": int(best["step"]) if "step" in best else None,
    }


# =============================
# TRAINING SETTINGS
# =============================
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    fp16=True,
    save_strategy="epoch",
    logging_steps=20,
    eval_strategy="epoch",
    predict_with_generate=True,
    report_to="none",
)


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_values = []

    def log(self, logs, *args, **kwargs):
        super().log(logs, *args, **kwargs)
        if "loss" in logs:
            self.loss_values.append(logs["loss"])

    def plot_training_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_values)
        plt.xlabel("Steps (Logging Frequency)")
        plt.ylabel("Loss")
        plt.title("Sinhala OCR Training Loss")
        plt.show()


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# =============================
# TRAIN
# =============================
trainer.train()
trainer.plot_training_loss()

# =============================
# SAVE MODEL
# =============================
save_path = os.getenv("SAVE_PATH", "./sinhala_model_v3")
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
processor.save_pretrained(save_path)
saved_files = verify_saved_artifacts(save_path)
print(f"Saved model artifacts: {saved_files}")
print("✅ Continued training completed successfully.")

# =============================
# PUSH TO HUGGING FACE
# =============================
repo_name = os.getenv("HF_REPO_NAME", "hasindu-k/sinhala-handwritten-notes-v3")

print("Computing final evaluation predictions and metrics...")
preds_output = trainer.predict(eval_dataset)
raw_preds = preds_output.predictions
if isinstance(raw_preds, tuple):
    raw_preds = raw_preds[0]

decoded_preds = processor.batch_decode(raw_preds, skip_special_tokens=True)
label_ids = np.where(
    preds_output.label_ids != -100,
    preds_output.label_ids,
    processor.tokenizer.pad_token_id,
)
decoded_labels = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
final_cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
final_wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
print(f"Final CER: {final_cer}")
print(f"Final WER: {final_wer}")

best_epoch_metrics = get_best_epoch_metrics(trainer.state.log_history)
if best_epoch_metrics is not None:
    print(f"Best Epoch Metrics: {best_epoch_metrics}")
else:
    print("Best Epoch Metrics: not available")

run_log = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "model_id": model_id,
    "save_path": save_path,
    "repo_name": repo_name,
    "config": {
        "output_dir": training_args.output_dir,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "fp16": training_args.fp16,
        "save_strategy": (
            training_args.save_strategy.value
            if hasattr(training_args.save_strategy, "value")
            else str(training_args.save_strategy)
        ),
        "logging_steps": training_args.logging_steps,
        "eval_strategy": (
            training_args.eval_strategy.value
            if hasattr(training_args.eval_strategy, "value")
            else str(training_args.eval_strategy)
        ),
        "predict_with_generate": training_args.predict_with_generate,
        "report_to": training_args.report_to,
        "max_length": 64,
        "eval_split_frac": 0.1,
        "random_state": 42,
    },
    "best_epoch_metrics": best_epoch_metrics,
    "final_cer": float(final_cer),
    "final_wer": float(final_wer),
    "logged_loss_count": len(trainer.loss_values),
    "first_loss": float(trainer.loss_values[0]) if trainer.loss_values else None,
    "last_loss": float(trainer.loss_values[-1]) if trainer.loss_values else None,
    "min_loss": float(min(trainer.loss_values)) if trainer.loss_values else None,
    "loss_values": [float(v) for v in trainer.loss_values],
}
append_run_log(run_log)
print(f"Training metrics logged to {LOG_PATH}")

upload_folder(
    repo_id=repo_name,
    folder_path=save_path,
    path_in_repo=".",
    commit_message="Upload trained model artifacts",
)
upload_folder(
    repo_id=repo_name,
    folder_path=os.path.dirname(LOG_PATH) or ".",
    path_in_repo="training_logs",
    allow_patterns=[os.path.basename(LOG_PATH)],
    commit_message="Upload training metrics log",
)

print(f"🚀 Model pushed to https://huggingface.co/{repo_name}")
