import os
import torch
import functools

# ============================================================
# PATCH 1: Wymuś single GPU
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================
# PATCH 2: Wyłączenie chunked CE w TRL
# ============================================================
def _apply_trl_patch():
    try:
        import trl.trainer.sft_trainer as _sft_mod

        def _noop_patch(target, chunk_size, is_vlm=False):
            pass

        _sft_mod._patch_chunked_ce_lm_head = _noop_patch
        print("[TRL PATCH] Chunked CE disabled successfully")
    except Exception as e:
        print(f"[TRL PATCH] Could not apply: {e}")

_apply_trl_patch()
# ============================================================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# =========================
# DIAGNOSTYKA
# =========================

print("torch version:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}: {torch.cuda.get_device_name(i)}, "
            f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
        )


# =========================
# USTAWIENIA
# =========================

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_FILE = "test.jsonl"
ADAPTER_DIR = "lora_adapter"
MERGED_DIR = "fine_tuned_local"
USE_4BIT = True
MERGE_MODEL = True
MAX_LEN = 512

SYSTEM_PROMPT = (
    "Classify the article into exactly one category: "
    "business, entertainment, politics, sport, tech. "
    "Reply with only the category word."
)

CATEGORIES = {"business", "entertainment", "politics", "sport", "tech"}


# =========================
# DETEKCJA SPRZĘTU
# =========================

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
fp16 = cuda and not torch.cuda.is_bf16_supported()

print("CUDA available:", cuda)
print("BF16:", bf16)
print("FP16:", fp16)

if USE_4BIT and not cuda:
    print("WARNING: QLoRA 4-bit wymaga CUDA. Wyłączam USE_4BIT.")
    USE_4BIT = False


# =========================
# TOKENIZER
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# =========================
# KONFIGURACJA QLoRA
# =========================

bnb_config = None

if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# =========================
# MODEL
# =========================

model_kwargs = {
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
    "dtype": torch.bfloat16 if bf16 else (
        torch.float16 if fp16 else torch.float32
    ),
    "device_map": {"": 0},
}

if bnb_config is not None:
    model_kwargs["quantization_config"] = bnb_config

print("Loading base model:", MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)

if USE_4BIT:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

try:
    model.enable_input_require_grads()
except AttributeError:
    pass


# =========================
# DANE
# =========================

print("Loading data file:", DATA_FILE)
raw = load_dataset("json", data_files=DATA_FILE)
train_raw = raw["train"]


def to_text(example):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in example["messages"]:
        if msg["role"] == "system":
            continue
        messages.append(msg)

    if messages[-1]["role"] != "assistant":
        raise ValueError(
            "Każdy przykład musi kończyć się wiadomością 'assistant'."
        )

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


text_ds = train_raw.map(to_text, remove_columns=train_raw.column_names)

if len(text_ds) >= 20:
    split = text_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    eval_strategy = "epoch"
else:
    train_ds = text_ds
    eval_ds = None
    eval_strategy = "no"

print("Train samples:", len(train_ds))
print("Eval samples:", len(eval_ds) if eval_ds is not None else 0)


# =========================
# LORA CONFIG
# =========================

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)


# =========================
# SFT CONFIG
# =========================

sft_config = SFTConfig(
    output_dir=ADAPTER_DIR,
    dataset_text_field="text",
    max_length=MAX_LEN,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,

    num_train_epochs=3,
    learning_rate=2e-4,

    logging_steps=10,
    save_strategy="epoch",

    eval_strategy=eval_strategy,
    metric_for_best_model="eval_loss" if eval_ds is not None else None,
    greater_is_better=False if eval_ds is not None else None,
    load_best_model_at_end=eval_ds is not None,

    bf16=bf16,
    fp16=fp16,
    report_to=[],
    remove_unused_columns=True,
)


# =========================
# CUSTOM TRAINER - naprawia brak num_valid_tokens
# =========================

class PatchedSFTTrainer(SFTTrainer):
    """
    Naprawia błąd TRL gdzie compute_loss oczekuje num_valid_tokens
    w outputach modelu, ale standardowy model tego nie zwraca.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Wywołaj standardowy forward modelu
        labels = inputs.get("labels")

        outputs = model(**inputs)

        # Standardowa cross-entropy loss
        if labels is not None:
            # Shift logits i labels dla causal LM
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


# =========================
# TRENING
# =========================

print("Creating PatchedSFTTrainer...")
trainer = PatchedSFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=sft_config,
    peft_config=peft_config,
)

print("Starting training...")
trainer.train()

print("\nTraining completed!")
print("Epoch:", trainer.state.epoch)
print("Global step:", trainer.state.global_step)

print("\nRecent log history:")
for entry in trainer.state.log_history[-10:]:
    print(entry)


# =========================
# ZAPIS
# =========================

trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print("\nSaved adapter to:", ADAPTER_DIR)
print("Files:", os.listdir(ADAPTER_DIR))


# =========================
# TEST
# =========================

print("\nTesting model...")
trainer.model.eval()

prompt_msgs = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "A new mobile phone is launched"},
]

prompt = tokenizer.apply_chat_template(
    prompt_msgs,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

with torch.no_grad():
    output_ids = trainer.model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

answer = tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
).strip()

clean = answer.lower().strip().rstrip(".").strip()
if clean in CATEGORIES:
    answer = clean

print("Classification result:", answer)


# =========================
# MERGE
# =========================

if MERGE_MODEL:
    print("\nMerging LoRA into base model...")

    base_full = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16 if bf16 else (
            torch.float16 if fp16 else torch.float32
        ),
        device_map={"": 0},
    )

    merged = PeftModel.from_pretrained(base_full, ADAPTER_DIR)
    final = merged.merge_and_unload()

    final.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    print("Merged model saved to:", MERGED_DIR)
    print("Files:", os.listdir(MERGED_DIR))

else:
    print("\nSkipped merge. Adapter saved to:", ADAPTER_DIR)