#!/usr/bin/env python3
"""
chatbot.py - Chatbot reasoning menggunakan model generatif (Causal LM).
Model akan menghasilkan jawaban lengkap dengan langkah-langkah.
Cocok untuk model yang dilatih dengan data format:
    "Pengguna: <pertanyaan>\nAI: Langkah 1: ... Jawaban: ..."
Konfigurasi terpusat di fungsi get_chatbot_config().
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# KONFIGURASI TERPUSAT
# ============================================================
def get_chatbot_config():
    """
    Mengatur path model, parameter generasi, dan lain-lain.
    Ubah sesuai kebutuhan.
    """
    config = {
        # --- Model ---
        "model_dir": "./model_reasoning",   # Folder hasil training (training.py)
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # --- Generasi teks ---
        "max_new_tokens": 150,              # Maksimum token baru yang dihasilkan
        "temperature": 0.7,                 # Kontrol kreativitas (0 = deterministik)
        "top_p": 0.9,                       # Nucleus sampling
        "repetition_penalty": 1.2,          # Menghindari pengulangan kata
        "do_sample": True,                  # Sampling (jika False, greedy)

        # --- Format prompt ---
        # Prompt akan dibentuk: "Pengguna: {input}\nAI:"
        "prompt_prefix": "Pengguna: ",
        "prompt_suffix": "\nAI:",
    }
    return config

# ============================================================
# FUNGSI BANTU
# ============================================================
def load_model_and_tokenizer(config):
    """Memuat tokenizer dan model generatif."""
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"])
    model = AutoModelForCausalLM.from_pretrained(config["model_dir"])
    model.to(config["device"])
    model.eval()

    # Set pad_token jika tidak ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

def generate_response(user_input, tokenizer, model, config):
    """
    Membentuk prompt, menghasilkan teks, dan mengembalikan
    hanya bagian jawaban AI (setelah "AI:").
    """
    # Bentuk prompt persis seperti data latih
    prompt = f"{config['prompt_prefix']}{user_input}{config['prompt_suffix']}"

    # Tokenisasi prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(config["device"])
    attention_mask = inputs["attention_mask"].to(config["device"])

    # Panjang token prompt (untuk memotong hasil nanti)
    prompt_len = input_ids.shape[1]

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            repetition_penalty=config["repetition_penalty"],
            do_sample=config["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode hanya token baru (setelah prompt)
    new_tokens = output_ids[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Bersihkan: jika respons mengandung "Pengguna:" lagi, potong di situ
    if "Pengguna:" in response:
        response = response.split("Pengguna:")[0]

    return response.strip()

# ============================================================
# UTAMA
# ============================================================
def main():
    cfg = get_chatbot_config()

    print("🔄 Memuat model generatif...")
    tokenizer, model = load_model_and_tokenizer(cfg)
    print("✅ Chatbot reasoning siap. Ketik 'keluar' untuk berhenti.\n")
    print("Contoh pertanyaan: 'Berapa 5 + 3?', 'Apa warna apel?', 'Suara kucing?'\n")

    while True:
        user_input = input("🧑 Anda: ")
        if user_input.lower() in ["keluar", "quit", "exit"]:
            print("👋 Sampai jumpa!")
            break

        if not user_input.strip():
            continue

        print("🤖 Bot: ", end="", flush=True)
        response = generate_response(user_input, tokenizer, model, cfg)
        print(response)
        print()

if __name__ == "__main__":
    main()