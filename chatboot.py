#!/usr/bin/env python3
"""
chatbot.py - Chatbot tanya jawab menggunakan model klasifikasi teks hasil training.
Model memprediksi intent dari input pengguna, lalu memberikan respons berdasarkan mapping.
Konfigurasi terpusat di fungsi get_chatbot_config().
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# KONFIGURASI TERPUSAT
# ============================================================
def get_chatbot_config():
    """
    Mengatur path model, mapping label, dan respons chatbot.
    Ubah sesuai kebutuhan.
    """
    config = {
        # --- Model ---
        "model_dir": "./model_scratch",          # Folder hasil training (dari training.py)
        "max_length": 512,

        # --- Label mapping (indeks ke nama intent) ---
        "label_map": {
            0: "negatif",
            1: "positif"
            # tambahkan label lain sesuai dataset Anda
        },

        # --- Response mapping (intent -> jawaban chatbot) ---
        "response_map": {
            "negatif": "Maaf Anda merasa tidak puas. Ada yang bisa kami bantu?",
            "positif": "Senang mendengarnya! Terima kasih atas apresiasi Anda.",
            # tambahkan respons untuk intent lain
        },

        # --- Mode ---
        "use_response_map": True,   # True = tampilkan respons, False = tampilkan label & confidence
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    return config

# ============================================================
# FUNGSI BANTU
# ============================================================
def load_model_and_tokenizer(config):
    """Memuat tokenizer dan model dari direktori."""
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_dir"])
    model.to(config["device"])
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model, config):
    """Melakukan prediksi intent dari teks input."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=config["max_length"]
    )
    inputs = {k: v.to(config["device"]) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = np.argmax(prob)
    confidence = prob[pred_idx]
    label_name = config["label_map"].get(pred_idx, str(pred_idx))
    return label_name, confidence

# ============================================================
# UTAMA
# ============================================================
def main():
    cfg = get_chatbot_config()

    print("🔄 Memuat model...")
    tokenizer, model = load_model_and_tokenizer(cfg)
    print("✅ Chatbot siap. Ketik 'keluar' untuk berhenti.\n")

    while True:
        user_input = input("🧑 Anda: ")
        if user_input.lower() in ["keluar", "quit", "exit"]:
            print("👋 Sampai jumpa!")
            break

        if not user_input.strip():
            continue

        label, conf = predict(user_input, tokenizer, model, cfg)

        if cfg["use_response_map"] and label in cfg["response_map"]:
            response = cfg["response_map"][label]
        else:
            response = f"[Intent: {label} | confidence: {conf:.2f}]"

        print(f"🤖 Bot: {response}\n")

if __name__ == "__main__":
    main()