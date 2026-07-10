#!/usr/bin/env python3
"""
training.py - Melatih model Transformer generatif (Causal LM) dari nol
untuk data reasoning (format tanya jawab dengan langkah).
Konfigurasi terpusat di fungsi get_config().
Data default: data.csv (kolom 'text' saja).
"""

import os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from sklearn.model_selection import train_test_split
import pandas as pd

# ============================================================
# KONFIGURASI TERPUSAT
# ============================================================
def get_config():
    """
    Mengembalikan dictionary konfigurasi untuk pelatihan model generatif.
    Ubah sesuai kebutuhan.
    """
    config = {
        # --- Data ---
        "train_file": "data.csv",           # File CSV dengan satu kolom 'text'
        "text_column": "text",              # Nama kolom teks
        "val_split_ratio": 0.1,             # Proporsi validasi (0.1 = 10%)
        "max_length": 256,                  # Panjang maksimum token (sesuaikan dengan data)

        # --- Model (from scratch) ---
        "model_type": "gpt2",               # Arsitektur GPT-2 kecil (bisa ganti "distilgpt2")
        # Untuk GPT-2, konfigurasi diambil lalu bobot acak

        # --- Training hyperparameters ---
        "output_dir": "./model_reasoning",
        "num_train_epochs": 10,             # From scratch butuh banyak epoch
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "logging_steps": 100,
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",  # Gunakan loss validasi
        "greater_is_better": False,            # Loss makin kecil makin baik
        "seed": 42,
    }
    return config

# ============================================================
# FUNGSI BANTU
# ============================================================
def load_data(config):
    """Memuat data CSV dan membuat DatasetDict (train/validation)."""
    df = pd.read_csv(config["train_file"])
    # Pastikan kolom teks ada
    texts = df[config["text_column"]].tolist()

    # Bagi train / valid
    train_texts, val_texts = train_test_split(
        texts, test_size=config["val_split_ratio"], random_state=config["seed"]
    )

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    return DatasetDict({"train": train_dataset, "validation": val_dataset})

def tokenize_function(examples, tokenizer, max_length):
    """Tokenisasi teks untuk causal LM."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

# ============================================================
# FUNGSI UTAMA
# ============================================================
def main():
    cfg = get_config()

    # Seed
    torch.manual_seed(cfg["seed"])

    # 1. Load data
    print(f"Memuat data dari {cfg['train_file']} ...")
    dataset = load_data(cfg)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")

    # 2. Tokenizer (pretrained untuk vocabulary)
    print(f"Memuat tokenizer {cfg['model_type']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_type"])
    # GPT-2 tidak punya pad_token, atur ke eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model dari nol (bobot acak)
    print(f"Membuat model {cfg['model_type']} dengan bobot acak...")
    model_config = AutoConfig.from_pretrained(cfg["model_type"])
    model = AutoModelForCausalLM.from_config(model_config)

    # 4. Tokenisasi dataset
    print("Tokenisasi dataset...")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, cfg["max_length"]),
        batched=True,
        remove_columns=["text"],
    )

    # 5. Data collator untuk language modeling (tanpa masking, karena causal)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # Causal LM (bukan masked LM)
    )

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        logging_dir=os.path.join(cfg["output_dir"], "logs"),
        logging_steps=cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["save_steps"],       # evaluasi setiap save_steps
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],  # loss -> False
        report_to="none",
        seed=cfg["seed"],
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. Latih
    print("Mulai pelatihan dari nol (causal LM)...")
    trainer.train()

    # 9. Simpan model & tokenizer
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Model generatif tersimpan di {cfg['output_dir']}")

if __name__ == "__main__":
    main()