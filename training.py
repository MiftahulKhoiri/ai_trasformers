#!/usr/bin/env python3
"""
training.py - Melatih model Transformer dari nol (from scratch) untuk klasifikasi teks.
Konfigurasi terpusat di fungsi get_config().
File data default: data.csv (dengan kolom 'text' dan 'label').
"""

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
import evaluate

# ============================================================
# KONFIGURASI TERPUSAT
# ============================================================
def get_config():
    """
    Mengembalikan dictionary konfigurasi untuk pelatihan.
    Ubah nilai di sini sesuai kebutuhan.
    """
    config = {
        # --- Data ---
        "train_file": "data.csv",          # Nama file CSV data latih
        "valid_file": None,                # (Opsional) file validasi terpisah
        "test_file": None,                 # (Opsional) file uji
        "text_column": "text",             # Nama kolom teks
        "label_column": "label",           # Nama kolom label (integer)
        "val_split_ratio": 0.2,            # Proporsi validasi dari train jika valid_file=None
        "max_length": 512,                 # Panjang maksimum token

        # --- Model (from scratch) ---
        "model_type": "bert-base-uncased", # Arsitektur yang digunakan (konfigurasi saja)
        "num_labels": None,                # None = otomatis dari data

        # --- Training hyperparameters ---
        "output_dir": "./model_scratch",
        "num_train_epochs": 20,            # From scratch butuh banyak epoch
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 64,
        "learning_rate": 1e-4,            # Lebih besar dari fine-tuning
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_steps": 50,
        "eval_steps": 200,
        "save_steps": 200,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "early_stopping_patience": 3,      # 0 untuk nonaktif
        "seed": 42,
    }
    return config

# ============================================================
# FUNGSI BANTU
# ============================================================
def load_data(config):
    """Memuat data CSV sesuai konfigurasi, buat DatasetDict."""
    train_file = config["train_file"]
    valid_file = config["valid_file"]
    test_file = config["test_file"]
    text_col = config["text_column"]
    label_col = config["label_column"]
    val_ratio = config["val_split_ratio"]

    df_train = pd.read_csv(train_file)
    df_train[label_col] = df_train[label_col].astype(int)

    if valid_file:
        df_valid = pd.read_csv(valid_file)
        df_valid[label_col] = df_valid[label_col].astype(int)
        train_dataset = Dataset.from_pandas(df_train[[text_col, label_col]])
        valid_dataset = Dataset.from_pandas(df_valid[[text_col, label_col]])
    else:
        df_tr, df_val = train_test_split(
            df_train,
            test_size=val_ratio,
            stratify=df_train[label_col],
            random_state=config["seed"]
        )
        train_dataset = Dataset.from_pandas(df_tr[[text_col, label_col]])
        valid_dataset = Dataset.from_pandas(df_val[[text_col, label_col]])

    dataset_dict = DatasetDict({"train": train_dataset, "validation": valid_dataset})

    if test_file:
        df_test = pd.read_csv(test_file)
        df_test[label_col] = df_test[label_col].astype(int)
        test_dataset = Dataset.from_pandas(df_test[[text_col, label_col]])
        dataset_dict["test"] = test_dataset

    return dataset_dict

def preprocess_function(examples, tokenizer, text_col):
    """Tokenisasi batch teks."""
    return tokenizer(
        examples[text_col],
        truncation=True,
        padding="max_length",
        max_length=get_config()["max_length"],  # ambil dari config
    )

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

# ============================================================
# FUNGSI UTAMA
# ============================================================
def main():
    # 1. Ambil konfigurasi
    cfg = get_config()

    # 2. Set seed
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # 3. Load data
    print(f"Memuat data dari {cfg['train_file']} ...")
    dataset = load_data(cfg)

    # 4. Tentukan jumlah kelas
    if cfg["num_labels"] is None:
        train_labels = dataset["train"][cfg["label_column"]]
        num_labels = len(set(train_labels))
        print(f"Jumlah kelas terdeteksi: {num_labels}")
    else:
        num_labels = cfg["num_labels"]

    # 5. Tokenizer (tetap pretrained untuk kosakata)
    print(f"Memuat tokenizer {cfg['model_type']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_type"])

    # 6. Model dari nol (konfigurasi saja, bobot acak)
    print(f"Memuat konfigurasi model {cfg['model_type']} dan inisialisasi bobot acak...")
    model_config = AutoConfig.from_pretrained(cfg["model_type"], num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_config(model_config)

    # 7. Tokenisasi dataset
    print("Tokenisasi data...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer, cfg["text_column"]),
        batched=True,
        remove_columns=[cfg["text_column"], cfg["label_column"]],
    )
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # 8. Training arguments dari config
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_dir=os.path.join(cfg["output_dir"], "logs"),
        logging_steps=cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=True,
        report_to="none",
        seed=cfg["seed"],
    )

    # 9. Callbacks
    callbacks = []
    if cfg["early_stopping_patience"] > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])
        )

    # 10. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 11. Training
    print("Mulai pelatihan dari nol...")
    trainer.train()

    # 12. Simpan model terbaik
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Model dan tokenizer tersimpan di {cfg['output_dir']}")

    # 13. Evaluasi test set jika ada
    if "test" in tokenized_datasets:
        print("Evaluasi pada data uji...")
        results = trainer.evaluate(tokenized_datasets["test"])
        print("Hasil Test:", results)

if __name__ == "__main__":
    main()