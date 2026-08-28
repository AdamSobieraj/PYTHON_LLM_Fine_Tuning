# test_model.py
# Uruchom: python test_model.py

import os
from pathlib import Path

# Wymuś single GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

# Opcjonalne wczytanie .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# =========================
# KONFIGURACJA
# =========================

# Główna ścieżka do modelu:
# - może być pełnym modelem,
# - może być merged modelem,
# - może być adapterem PEFT.
MODEL_PATH = os.getenv("MODEL_PATH", "fine_tuned_local")

# Wymagane TYLKO, jeśli MODEL_PATH to adapter.
# Może być:
# - ID z Hugging Face, np. "HuggingFaceTB/SmolLM2-360M-Instruct"
# - albo lokalna ścieżka do bazowego modelu.
BASE_MODEL = os.getenv("BASE_MODEL") or os.getenv("MODEL_ID")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT") or (
    "Classify the article into exactly one category: "
    "business, entertainment, politics, sport, tech. "
    "Reply with only the category word."
)

categories_raw = os.getenv("CATEGORIES") or "business,entertainment,politics,sport,tech"
CATEGORIES = {
    category.strip()
    for category in categories_raw.split(",")
    if category.strip()
}

# Opcjonalnie: merge adaptera w pamięci przed inferencją.
# Domyślnie wyłączony, żeby nie zwiększać zużycia VRAM.
MERGE_FOR_INFERENCE = (
    os.getenv("MERGE_FOR_INFERENCE", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


# =========================
# DETEKCJA SPRZĘTU
# =========================

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
fp16 = cuda and not bf16

dtype = torch.bfloat16 if bf16 else (
    torch.float16 if fp16 else torch.float32
)

device_map = {"": 0} if cuda else "cpu"

print("=" * 50)
print("CUDA available:", cuda)
print("BF16:", bf16)
print("FP16:", fp16)

if cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("Model path:", os.path.abspath(MODEL_PATH))
print("Base model:", BASE_MODEL)
print("=" * 50)


# =========================
# ŁADOWANIE MODELU
# =========================

def load_model_and_tokenizer(model_path: str):
    """
    Ładuje:
    1) pełny model,
    2) merged model,
    3) adapter PEFT + bazowy model.
    """

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono ścieżki modelu: {os.path.abspath(model_path)}"
        )

    print(f"\nLoading tokenizer from: {os.path.abspath(model_path)}")
    tokenizer = AutoTokenizer.from_pretrained(str(path))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_adapter = (
        (path / "adapter_config.json").exists()
        or (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )

    if is_adapter:
        if BASE_MODEL is None:
            raise RuntimeError(
                "Wykryto adapter PEFT w folderze MODEL_PATH.\n"
                "Musisz podać bazowy model, np.:\n"
                "BASE_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct\n"
                "lub:\n"
                "MODEL_ID=HuggingFaceTB/SmolLM2-360M-Instruct"
            )

        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "Wykryto adapter PEFT, ale biblioteka peft nie jest dostępna.\n"
                "Zainstaluj ją:\n"
                "pip install peft"
            )

        print(f"\nDetected PEFT adapter: {os.path.abspath(model_path)}")
        print(f"Loading base model: {BASE_MODEL}")

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        model = PeftModel.from_pretrained(base_model, str(path))

        if MERGE_FOR_INFERENCE:
            try:
                model = model.merge_and_unload()
                print("Adapter merged into base model for inference.")
            except Exception as e:
                print(f"Could not merge adapter for inference: {e}")
                print("Continuing with PeftModel...")
    else:
        print(f"\nLoading full/merged model from: {os.path.abspath(model_path)}")

        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model.eval()
    print("Model loaded successfully!\n")

    return tokenizer, model


tokenizer, model = load_model_and_tokenizer(MODEL_PATH)


# =========================
# FUNKCJA KLASYFIKACJI
# =========================

def classify(text: str, verbose: bool = False) -> str:
    """
    Klasyfikuje tekst do jednej z kategorii:
    business, entertainment, politics, sport, tech
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if verbose:
        print(f"[PROMPT]\n{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt")

    # Bezpiecznie pobierz urządzenie, na którym stoi model.
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Dekoduj tylko nowe tokeny
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Normalizacja odpowiedzi
    clean = raw_answer.lower().strip().rstrip(".").strip()
    final = clean if clean in CATEGORIES else raw_answer

    if verbose:
        print(f"[RAW OUTPUT] {raw_answer!r}")
        print(f"[CLEAN OUTPUT] {clean!r}")

    return final


# =========================
# TESTY PRZYKŁADOWE
# =========================

test_cases = [
    # (tekst, oczekiwana_kategoria)
    ("Apple launches new iPhone with revolutionary AI features", "tech"),
    ("Stock markets fall amid fears of global recession", "business"),
    ("England wins the World Cup after dramatic penalty shootout", "sport"),
    ("Prime Minister announces snap election amid political crisis", "politics"),
    ("Hollywood star wins Oscar for best performance in drama film", "entertainment"),
    ("Tesla reports record quarterly profits", "business"),
    ("New AI model beats human experts at medical diagnosis", "tech"),
    ("Olympic champion breaks world record in 100m sprint", "sport"),
    ("Parliament votes on controversial immigration bill", "politics"),
    ("New blockbuster movie breaks box office records worldwide", "entertainment"),
]

print("=" * 50)
print("RUNNING CLASSIFICATION TESTS")
print("=" * 50)

correct = 0
total = len(test_cases)

for i, (text, expected) in enumerate(test_cases, 1):
    result = classify(text)
    status = "OK" if result == expected else "NOT OK"
    correct += int(result == expected)

    print(f"{status} [{i:2d}] {result:>15s} | expected: {expected:>15s}")
    print(f"       Text: {text[:70]}...")
    print()

print("=" * 50)
print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%")
print("=" * 50)


# =========================
# TRYB INTERAKTYWNY
# =========================

print("\n" + "=" * 50)
print("INTERACTIVE MODE")
print("Type article text, press Enter to classify.")
print("Type 'quit' or 'exit' to stop.")
print("Type 'verbose' to toggle detailed output.")
print("=" * 50 + "\n")

verbose_mode = False

while True:
    try:
        user_input = input("Article text: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "verbose":
            verbose_mode = not verbose_mode
            print(f"Verbose mode: {'ON' if verbose_mode else 'OFF'}\n")
            continue

        result = classify(user_input, verbose=verbose_mode)
        print(f"Category: {result}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
