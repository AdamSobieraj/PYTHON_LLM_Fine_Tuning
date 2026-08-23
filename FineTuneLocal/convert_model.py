import os
import torch
import subprocess
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# KONFIGURACJA
# =========================

SOURCE_MODEL = "fine_tuned_local"
OUTPUT_DIR = "converted_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()

print(f"Source model: {os.path.abspath(SOURCE_MODEL)}")
print(f"Output dir:   {os.path.abspath(OUTPUT_DIR)}")


# =========================
# ŁADOWANIE MODELU
# =========================

def load_model():
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE_MODEL,
        dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="cpu",  # CPU dla konwersji
        trust_remote_code=True,
    )
    print("Model loaded!")
    return model, tokenizer


# =========================
# FORMAT 1: SafeTensors (już masz)
# =========================

def save_safetensors(model, tokenizer):
    """Zapisz w formacie SafeTensors (domyślny HuggingFace)"""
    out = os.path.join(OUTPUT_DIR, "safetensors")
    print(f"\n[1/4] Saving SafeTensors → {out}")

    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)

    size = sum(
        f.stat().st_size
        for f in Path(out).rglob("*")
        if f.is_file()
    ) / 1e9

    print(f"      ✅ Done! Size: {size:.2f} GB")
    print(f"      Files: {os.listdir(out)}")


# =========================
# FORMAT 2: PyTorch Pickle (.bin)
# =========================

def save_pytorch_bin(model, tokenizer):
    """Zapisz w formacie PyTorch .bin"""
    out = os.path.join(OUTPUT_DIR, "pytorch_bin")
    print(f"\n[2/4] Saving PyTorch .bin → {out}")

    model.save_pretrained(out, safe_serialization=False)
    tokenizer.save_pretrained(out)

    size = sum(
        f.stat().st_size
        for f in Path(out).rglob("*")
        if f.is_file()
    ) / 1e9

    print(f"Done! Size: {size:.2f} GB")
    print(f"Files: {os.listdir(out)}")


# =========================
# FORMAT 3: GGUF (dla Ollama/LM Studio)
# =========================

def save_gguf(quantization="q4_k_m"):
    """
    Konwertuj do GGUF używając llama.cpp

    Dostępne kwantyzacje:
    - f32    → pełna precyzja (największy)
    - f16    → half precision
    - q8_0   → 8-bit (dobra jakość)
    - q4_k_m → 4-bit (zalecana, dobry balans)
    - q4_k_s → 4-bit (mniejszy)
    - q3_k_m → 3-bit (bardzo mały)
    - q2_k   → 2-bit (najgorsza jakość)
    """
    out_dir = os.path.join(OUTPUT_DIR, "gguf")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"model_{quantization}.gguf")

    print(f"\n[3/4] Converting to GGUF ({quantization}) → {out_file}")
    print("      Requires: pip install llama-cpp-python")
    print("      Or: git clone https://github.com/ggerganov/llama.cpp")

    # Sprawdź czy llama.cpp jest dostępne
    llamacpp_convert = shutil.which("convert_hf_to_gguf.py")

    if llamacpp_convert is None:
        # Spróbuj zainstalować i użyć
        print("      Trying automatic conversion with llama-cpp-python...")
        try:
            # Metoda 1: przez llama-cpp-python
            cmd = [
                "python", "-m", "llama_cpp.llama_chat_format",
                "--model", SOURCE_MODEL,
                "--output", out_file,
            ]
            print(f"      CMD: {' '.join(cmd)}")
            print("      ⚠️  Llama.cpp not found. Manual steps below:")
            print_gguf_manual_steps(out_file, quantization)
            return
        except Exception:
            print_gguf_manual_steps(out_file, quantization)
            return

    # Automatyczna konwersja jeśli llama.cpp jest zainstalowane
    cmd = [
        "python", llamacpp_convert,
        SOURCE_MODEL,
        "--outfile", out_file,
        "--outtype", quantization,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        size = os.path.getsize(out_file) / 1e9
        print(f"Done! Size: {size:.2f} GB")
    else:
        print(f"Error: {result.stderr}")
        print_gguf_manual_steps(out_file, quantization)


def print_gguf_manual_steps(out_file, quantization):
    """Wydrukuj instrukcje ręcznej konwersji do GGUF"""
    print("\n" + "=" * 60)
    print("RĘCZNA KONWERSJA DO GGUF:")
    print("=" * 60)
    print("""
# Krok 1: Sklonuj llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

# Krok 2: Konwertuj model
python convert_hf_to_gguf.py \\
    ../fine_tuned_local \\
    --outfile ../converted_models/gguf/model_f16.gguf \\
    --outtype f16

# Krok 3: Kwantyzuj (opcjonalne, wymaga kompilacji llama.cpp)
./quantize \\
    ../converted_models/gguf/model_f16.gguf \\
    ../converted_models/gguf/model_q4_k_m.gguf \\
    q4_k_m
""")
    print("=" * 60)


# =========================
# FORMAT 4: ONNX
# =========================

def save_onnx():
    """Konwertuj do ONNX"""
    out = os.path.join(OUTPUT_DIR, "onnx")
    print(f"\n[4/4] Converting to ONNX → {out}")
    print("      Requires: pip install optimum[exporters]")

    try:
        from optimum.exporters.onnx import main_export

        main_export(
            model_name_or_path=SOURCE_MODEL,
            output=out,
            task="text-generation",
            device="cpu",
        )

        size = sum(
            f.stat().st_size
            for f in Path(out).rglob("*")
            if f.is_file()
        ) / 1e9

        print(f"      ✅ Done! Size: {size:.2f} GB")

    except ImportError:
        print("      ⚠️  optimum not installed!")
        print("      Run: pip install optimum[exporters]")
        print("\n      Manual command:")
        print(f"      optimum-cli export onnx \\")
        print(f"          --model {SOURCE_MODEL} \\")
        print(f"          --task text-generation \\")
        print(f"          {out}")

    except Exception as e:
        print(f"Error: {e}")


# =========================
# PODSUMOWANIE FORMATÓW
# =========================

def print_summary():
    print("\n" + "=" * 60)
    print("PODSUMOWANIE FORMATÓW")
    print("=" * 60)

    formats = [
        {
            "name": "SafeTensors",
            "path": "converted_models/safetensors/",
            "use_case": "HuggingFace, Python, dalszy trening",
            "load_code": 'AutoModelForCausalLM.from_pretrained("converted_models/safetensors")',
        },
        {
            "name": "PyTorch .bin",
            "path": "converted_models/pytorch_bin/",
            "use_case": "Kompatybilność wsteczna",
            "load_code": 'AutoModelForCausalLM.from_pretrained("converted_models/pytorch_bin")',
        },
        {
            "name": "GGUF q4_k_m",
            "path": "converted_models/gguf/model_q4_k_m.gguf",
            "use_case": "Ollama, LM Studio, CPU inference",
            "load_code": 'ollama create mymodel -f Modelfile',
        },
        {
            "name": "ONNX",
            "path": "converted_models/onnx/",
            "use_case": "Produkcja, mobile, edge devices",
            "load_code": 'ORTModelForCausalLM.from_pretrained("converted_models/onnx")',
        },
    ]

    for fmt in formats:
        exists = os.path.exists(fmt["path"])
        status = "✅" if exists else "⬜"
        print(f"\n{status} {fmt['name']}")
        print(f"   Path:     {fmt['path']}")
        print(f"   Use case: {fmt['use_case']}")
        print(f"   Load:     {fmt['load_code']}")


# =========================
# GŁÓWNA LOGIKA
# =========================

if __name__ == "__main__":
    print("\nDostępne konwersje:")
    print("1 - SafeTensors (HuggingFace)")
    print("2 - PyTorch .bin")
    print("3 - GGUF (Ollama/LM Studio)")
    print("4 - ONNX")
    print("A - Wszystkie")
    print("S - Pokaż podsumowanie")

    choice = input("\nWybierz opcję: ").strip().upper()

    if choice in ("1", "2", "A"):
        model, tokenizer = load_model()

        if choice in ("1", "A"):
            save_safetensors(model, tokenizer)

        if choice in ("2", "A"):
            save_pytorch_bin(model, tokenizer)

    if choice in ("3", "A"):
        save_gguf("q4_k_m")

    if choice in ("4", "A"):
        save_onnx()

    if choice == "S":
        pass

    print_summary()