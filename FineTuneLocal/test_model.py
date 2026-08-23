# test_model.py
# Uruchom: python test_model.py
# Lokalizacja modelu: E:\PROJEKTY\PYTHON_Fine_Tuning\FineTuneLocal\fine_tuned_local\

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# KONFIGURACJA
# =========================

# Ścieżka do wytrenowanego modelu (po merge)
MODEL_PATH = "fine_tuned_local"

# Alternatywnie - sam adapter (wymaga bazowego modelu):
# MODEL_PATH = "lora_adapter"
# BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

SYSTEM_PROMPT = (
    "Classify the article into exactly one category: "
    "business, entertainment, politics, sport, tech. "
    "Reply with only the category word."
)

CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}

# Wymuś single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# =========================
# DETEKCJA SPRZĘTU
# =========================

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()

print("=" * 50)
print("CUDA available:", cuda)
print("BF16:", bf16)

if cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 50)


# =========================
# ŁADOWANIE MODELU
# =========================

print(f"\nLoading model from: {os.path.abspath(MODEL_PATH)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16 if bf16 else torch.float32,
    device_map={"": 0} if cuda else "cpu",
    trust_remote_code=True,
)

model.eval()
print("Model loaded successfully!\n")


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

    if cuda:
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Dekoduj tylko nowe tokeny
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Normalizacja odpowiedzi
    clean = raw_answer.lower().strip().rstrip(".").strip()
    final = clean if clean in CATEGORIES else raw_answer

    if verbose:
        print(f"[RAW OUTPUT] {raw_answer!r}")

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
    correct += result == expected

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