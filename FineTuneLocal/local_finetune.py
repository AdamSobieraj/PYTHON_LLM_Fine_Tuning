import os
import torch
from dotenv import load_dotenv

load_dotenv()

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


from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ============================================================
# CUSTOM TRAINER - naprawia brak num_valid_tokens
# ============================================================
class PatchedSFTTrainer(SFTTrainer):
    """
    Naprawia błąd TRL gdzie compute_loss oczekuje num_valid_tokens
    w outputach modelu, ale standardowy model tego nie zwraca.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        outputs = model(**inputs)

        if labels is not None:
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


# ============================================================
# GŁÓWNA KLASA Z PODZIAŁEM NA METODY
# ============================================================
class QLoRASFTWorkflow:
    def __init__(self):
        self.cuda = None
        self.bf16 = None
        self.fp16 = None
        self.use_4bit = bool(os.getenv("USE_4BIT"))

        self.tokenizer = None
        self.bnb_config = None
        self.model = None

        self.train_ds = None
        self.eval_ds = None
        self.eval_strategy = "no"

        self.peft_config = None
        self.sft_config = None
        self.trainer = None

    # ========================================================
    # HELPER: wymagana zmienna środowiskowa
    # ========================================================
    def _require_env(self, name: str) -> str:
        value = os.getenv(name)

        if value is None or not value.strip():
            raise RuntimeError(
                f"Brak wymaganej zmiennej środowiskowej: {name}"
            )

        return value

    # ========================================================
    # DIAGNOSTYKA
    # ========================================================
    def diagnose(self):
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

    # ========================================================
    # DETEKCJA SPRZĘTU
    # ========================================================
    def detect_hardware(self):
        self.cuda = torch.cuda.is_available()
        self.bf16 = self.cuda and torch.cuda.is_bf16_supported()
        self.fp16 = self.cuda and not torch.cuda.is_bf16_supported()

        print("CUDA available:", self.cuda)
        print("BF16:", self.bf16)
        print("FP16:", self.fp16)

        if self.use_4bit and not self.cuda:
            print("WARNING: QLoRA 4-bit wymaga CUDA. Wyłączam USE_4BIT.")
            self.use_4bit = False

    # ========================================================
    # TOKENIZER
    # ========================================================
    def load_tokenizer(self):
        model_id = self._require_env("MODEL_ID")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ========================================================
    # KONFIGURACJA QLoRA
    # ========================================================
    def configure_quantization(self):
        self.bnb_config = None

        if self.use_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

    # ========================================================
    # MODEL
    # ========================================================
    def load_model(self):
        model_id = self._require_env("MODEL_ID")

        dtype = torch.bfloat16 if self.bf16 else (
            torch.float16 if self.fp16 else torch.float32
        )

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
            "device_map": {"": 0} if self.cuda else None,
        }

        if self.bnb_config is not None:
            model_kwargs["quantization_config"] = self.bnb_config

        print("Loading base model:", model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=True,
            )

        try:
            self.model.enable_input_require_grads()
        except AttributeError:
            pass

    # ========================================================
    # DANE
    # ========================================================
    def _to_text(self, example):
        system_prompt = os.getenv("SYSTEM_PROMPT", "")

        messages = [{"role": "system", "content": system_prompt}]

        for msg in example["messages"]:
            if msg["role"] == "system":
                continue
            messages.append(msg)

        if messages[-1]["role"] != "assistant":
            raise ValueError(
                "Każdy przykład musi kończyć się wiadomością 'assistant'."
            )

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    def prepare_data(self):
        data_file = self._require_env("DATA_FILE")
        print("Loading data file:", data_file)

        raw = load_dataset("json", data_files=data_file)
        train_raw = raw["train"]

        text_ds = train_raw.map(
            self._to_text,
            remove_columns=train_raw.column_names,
        )

        if len(text_ds) >= 20:
            split = text_ds.train_test_split(test_size=0.1, seed=42)
            self.train_ds = split["train"]
            self.eval_ds = split["test"]
            self.eval_strategy = "epoch"
        else:
            self.train_ds = text_ds
            self.eval_ds = None
            self.eval_strategy = "no"

        print("Train samples:", len(self.train_ds))
        print("Eval samples:", len(self.eval_ds) if self.eval_ds is not None else 0)

    # ========================================================
    # LORA CONFIG
    # ========================================================
    def configure_lora(self):
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )

    # ========================================================
    # SFT CONFIG
    # ========================================================
    def configure_sft(self):
        adapter_dir = self._require_env("ADAPTER_DIR")
        max_len = int(self._require_env("MAX_LEN"))

        self.sft_config = SFTConfig(
            output_dir=adapter_dir,
            dataset_text_field="text",
            max_length=max_len,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,

            num_train_epochs=3,
            learning_rate=2e-4,

            logging_steps=10,
            save_strategy="epoch",

            eval_strategy=self.eval_strategy,
            metric_for_best_model="eval_loss" if self.eval_ds is not None else None,
            greater_is_better=False if self.eval_ds is not None else None,
            load_best_model_at_end=self.eval_ds is not None,

            bf16=self.bf16,
            fp16=self.fp16,
            report_to=[],
            remove_unused_columns=True,
        )

    # ========================================================
    # CREATION OF TRAINER
    # ========================================================
    def create_trainer(self):
        print("Creating PatchedSFTTrainer...")

        self.trainer = PatchedSFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            args=self.sft_config,
            peft_config=self.peft_config,
        )

    # ========================================================
    # TRENING
    # ========================================================
    def train(self):
        print("Starting training...")
        self.trainer.train()

        print("\nTraining completed!")
        print("Epoch:", self.trainer.state.epoch)
        print("Global step:", self.trainer.state.global_step)

        print("\nRecent log history:")
        for entry in self.trainer.state.log_history[-10:]:
            print(entry)

    # ========================================================
    # ZAPIS
    # ========================================================
    def save(self):
        adapter_dir = self._require_env("ADAPTER_DIR")

        self.trainer.save_model(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

        print("\nSaved adapter to:", adapter_dir)
        print("Files:", os.listdir(adapter_dir))

    # ========================================================
    # TEST
    # ========================================================
    def test(self):
        print("\nTesting model...")
        self.trainer.model.eval()

        system_prompt = os.getenv("SYSTEM_PROMPT", "")

        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "A new mobile phone is launched"},
        ]

        prompt = self.tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")

        device = "cuda:0" if self.cuda else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.trainer.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        answer = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        clean = answer.lower().strip().rstrip(".").strip()

        categories = os.getenv("CATEGORIES") or ""
        if clean in categories:
            answer = clean

        print("Classification result:", answer)

    # ========================================================
    # MERGE
    # ========================================================
    def merge(self):
        if os.getenv("MERGE_MODEL"):
            print("\nMerging LoRA into base model...")

            model_id = self._require_env("MODEL_ID")
            adapter_dir = self._require_env("ADAPTER_DIR")
            merged_dir = self._require_env("MERGED_DIR")

            dtype = torch.bfloat16 if self.bf16 else (
                torch.float16 if self.fp16 else torch.float32
            )

            base_full = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                dtype=dtype,
                device_map={"": 0} if self.cuda else None,
            )

            merged = PeftModel.from_pretrained(base_full, adapter_dir)
            final = merged.merge_and_unload()

            final.save_pretrained(merged_dir)
            self.tokenizer.save_pretrained(merged_dir)

            print("Merged model saved to:", merged_dir)
            print("Files:", os.listdir(merged_dir))

        else:
            adapter_dir = self._require_env("ADAPTER_DIR")
            print("\nSkipped merge. Adapter saved to:", adapter_dir)

    # ========================================================
    # GŁÓWNY PRZEPŁYW
    # ========================================================
    def run(self):
        self.diagnose()
        self.detect_hardware()
        self.load_tokenizer()
        self.configure_quantization()
        self.load_model()
        self.prepare_data()
        self.configure_lora()
        self.configure_sft()
        self.create_trainer()
        self.train()
        self.save()
        self.test()
        self.merge()


def main():
    workflow = QLoRASFTWorkflow()
    workflow.run()


if __name__ == "__main__":
    main()
