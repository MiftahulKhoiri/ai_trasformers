#!/usr/bin/env python3
"""
training.py - Melatih model Transformer generatif (Causal LM) dari nol
untuk data reasoning (format tanya jawab dengan langkah).
Dilengkapi pemantauan penggunaan RAM selama training.
Konfigurasi terpusat di fungsi get_config().
Data default: data.csv (kolom 'text' saja).
"""

import os
import time
import torch
import pandas as pd
import psutil  # untuk monitoring RAM
from datasets import Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from sklearn.model_selection import train_test_split

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
        "model_type": "gpt2",               # Arsitektur GPT-2 kecil (bisa "distilgpt2")
        # Bobot acak (from_config)

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
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,         # loss makin kecil makin baik
        "seed": 42,

        # --- Monitoring memori ---
        "log_memory": True,                 # Aktifkan log penggunaan RAM
        "memory_log_steps": 200,            # Cetak info memori setiap N langkah (None = setiap logging step)
        "memory_log_epochs": False,         # Cetak info memori setiap akhir epoch (True/False)
    }
    return config

# ============================================================
# CALLBACK MONITORING MEMORI
# ============================================================
class MemoryLoggingCallback(TrainerCallback):
    """
    Callback untuk mencatat penggunaan RAM sistem.
    """
    def __init__(self, log_steps=None, log_epochs=False):
        self.log_steps = log_steps
        self.log_epochs = log_epochs
        self.process = psutil.Process(os.getpid())  # proses saat ini

    def on_step_end(self, args, state, control, **kwargs):
        if self.log_steps and state.global_step > 0 and state.global_step % self.log_steps == 0:
            self._print_memory("Step", state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.log_epochs:
            self._print_memory("Epoch", state.epoch)

    def _print_memory(self, label, value):
        mem = self.process.memory_info()
        # Konversi ke MB
        rss_mb = mem.rss / (1024 ** 2)  # Resident Set Size (RAM fisik yang digunakan)
        vms_mb = mem.vms / (1024 ** 2)  # Virtual Memory Size
        # Informasi RAM sistem
        system_mem = psutil.virtual_memory()
        total_system = system_mem.total / (1024 ** 2)
        used_system = system_mem.used / (1024 ** 2)
        percent = system_mem.percent
        print(f"[RAM Monitoring] {label} {value} | "
              f"Process: RSS={rss_mb:.1f} MB, VMS={vms_mb:.1f} MB | "
              f"System: Used={used_system:.1f}/{total_system:.1f} MB ({percent:.1f}%)")

# ============================================================
# FUNGSI BANTU
# ============================================================
def load_data(config):
    """Memuat data CSV dan membuat DatasetDict (train/validation)."""
    df = pd.read_csv(config["train_file"])
    texts = df[config["text_column"]].tolist()

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

    # 2. Tokenizer
    print(f"Memuat tokenizer {cfg['model_type']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_type"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Model dari nol
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

    # 5. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6. Training arguments  (diperbaiki: eval_strategy)
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
        eval_strategy="steps",                  # <-- perbaikan di sini
        eval_steps=cfg["save_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        report_to="none",
        seed=cfg["seed"],
    )

    # 7. Callback monitoring memori
    callbacks = []
    if cfg["log_memory"]:
        memory_log_steps = cfg.get("memory_log_steps", None)
        memory_log_epochs = cfg.get("memory_log_epochs", False)
        callbacks.append(MemoryLoggingCallback(log_steps=memory_log_steps, log_epochs=memory_log_epochs))
        print("Monitoring RAM diaktifkan.")

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # 9. Latih
    print("Mulai pelatihan dari nol (causal LM)...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Pelatihan selesai dalam {elapsed/60:.2f} menit.")

    # 10. Simpan model
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Model generatif tersimpan di {cfg['output_dir']}")

if __name__ == "__main__":
    main()