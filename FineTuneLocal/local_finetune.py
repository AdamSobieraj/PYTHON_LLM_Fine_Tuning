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


# ============================================================
# Importy wymagane dopiero po patchu TRL
# ============================================================
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

from peft_methods import create_peft_method, PEFTOptions, DoRA


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
        self.use_4bit = False

        self.tokenizer = None
        self.bnb_config = None
        self.model = None

        self.train_ds = None
        self.eval_ds = None
        self.eval_strategy = "no"

        self.peft_config = None
        self.sft_config = None
        self.trainer = None

        self.peft_options = PEFTOptions.from_env()
        self.peft_method = create_peft_method(
            os.getenv("PEFT_METHOD"),
            self.peft_options,
        )

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

        self.use_4bit = self.peft_options.use_4bit

        print("CUDA available:", self.cuda)
        print("BF16:", self.bf16)
        print("FP16:", self.fp16)
        print("PEFT method:", self.peft_method.name)
        print("USE_4BIT:", self.use_4bit)

        if self.peft_method.name == "qlora" and not self.cuda:
            print("WARNING: QLoRA wymaga CUDA. Przełączam na DoRA.")
            self.peft_options.use_4bit = False
            self.peft_method = DoRA(self.peft_options)
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
    # KONFIGURACJA PEFT / QLoRA / DoRA
    # ========================================================
    def configure_peft(self):
        print("Configuring PEFT method:", self.peft_method.name)

        self.bnb_config = self.peft_method.quantization_config()
        self.peft_config = self.peft_method.peft_config()

        if self.peft_config is None:
            print("Full fine-tuning mode: no PEFT adapter will be used.")
        else:
            print("PEFT adapter mode: adapter will be trained.")

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

        self.model = self.peft_method.prepare_model(self.model)
        self.peft_method.enable_input_require_grads(self.model)

    # ========================================================
    # DANE
    # ========================================================
    def _to_prompt_completion(self, example):
        # Rozdzielamy na prompt (system + user) i completion (assistant),
        # zeby loss liczyl sie TYLKO na odpowiedzi (slowo kategorii), a nie na calym tekscie.
        system_prompt = os.getenv("SYSTEM_PROMPT", "")

        user_content = None
        assistant_content = None
        for msg in example["messages"]:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        if user_content is None or assistant_content is None:
            raise ValueError("Przyklad musi zawierac wiadomosc 'user' i 'assistant'.")

        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        completion = [{"role": "assistant", "content": assistant_content}]
        return {"prompt": prompt, "completion": completion}

    def prepare_data(self):
        data_file = self._require_env("DATA_FILE")
        print("Loading data file:", data_file)

        raw = load_dataset("json", data_files=data_file)
        train_raw = raw["train"]

        pc_ds = train_raw.map(
            self._to_prompt_completion,
            remove_columns=train_raw.column_names,
        )

        if len(pc_ds) >= 20:
            split = pc_ds.train_test_split(test_size=0.1, seed=42)
            self.train_ds = split["train"]
            self.eval_ds = split["test"]
            self.eval_strategy = "epoch"
        else:
            self.train_ds = pc_ds
            self.eval_ds = None
            self.eval_strategy = "no"

        print("Train samples:", len(self.train_ds))
        print("Eval samples:", len(self.eval_ds) if self.eval_ds is not None else 0)

    # ========================================================
    # SFT CONFIG
    # ========================================================
    def configure_sft(self):
        adapter_dir = self._require_env("ADAPTER_DIR")
        max_len = int(self._require_env("MAX_LEN"))

        # Domyslna LR zalezy od metody:
        # - full fine-tuning: NISKA LR (za wysoka niszczy wagi pretrained modelu)
        # - LoRA/DoRA/QLoRA: wysoka LR jest OK
        default_lr = 3e-5 if self.peft_method.name == "full" else 2e-4

        lr_env = os.getenv("LEARNING_RATE")
        learning_rate = float(lr_env) if lr_env else default_lr

        batch_size = 4
        grad_accum = 2
        num_epochs = 3

        # TRL 1.x nie przyjmuje parametru 'warmup_ratio' - przeliczamy na 'warmup_steps'
        warmup_env = os.getenv("WARMUP_RATIO")
        warmup_ratio = float(warmup_env) if warmup_env else 0.05

        n_train = len(self.train_ds) if self.train_ds is not None else 0
        total_steps = 0
        if n_train > 0:
            steps_per_epoch = -(-n_train // (batch_size * grad_accum))  # dzielenie z zaokr. w gore
            total_steps = steps_per_epoch * num_epochs
        warmup_steps = max(1, int(round(total_steps * warmup_ratio))) if total_steps else 10

        print(f"PEFT method: {self.peft_method.name}")
        print(f"Learning rate: {learning_rate} (default for this method: {default_lr})")
        print(f"Warmup steps: {warmup_steps} (ratio {warmup_ratio}, total steps ~{total_steps})")

        self.sft_config = SFTConfig(
            output_dir=adapter_dir,
            max_length=max_len,

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=grad_accum,

            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,

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

        trainer_kwargs = {
            "model": self.model,
            "processing_class": self.tokenizer,
            "train_dataset": self.train_ds,
            "eval_dataset": self.eval_ds,
            "args": self.sft_config,
        }

        if self.peft_config is not None:
            trainer_kwargs["peft_config"] = self.peft_config
        else:
            print("Full fine-tuning: peft_config disabled.")

        self.trainer = PatchedSFTTrainer(**trainer_kwargs)

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
        # full fine-tuning: pelny model zapisujemy do MERGED_DIR (sciezka czytana w test_model.py)
        # LoRA/DoRA/QLoRA: adapter zapisujemy do ADAPTER_DIR
        if self.peft_config is None:
            out_dir = self._require_env("MERGED_DIR")
        else:
            out_dir = self._require_env("ADAPTER_DIR")

        self.trainer.save_model(out_dir)
        self.tokenizer.save_pretrained(out_dir)

        if self.peft_config is None:
            print("\nSaved FULL model to:", out_dir)
        else:
            print("\nSaved adapter to:", out_dir)

        print("Files:", os.listdir(out_dir))

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

        # What it means:
        #
        # - self.peft_config is None → you ran full fine-tuning (all model weights were trained directly), not LoRA/QLoRA/DoRA.
        # - Merging is only needed for PEFT adapters — that's the step where a small LoRA adapter is baked into the base model so the final model works without loading the adapter separately.
        # - In full fine-tuning there is no separate adapter — the trained weights already live in the model itself. So there's nothing to merge, and the step is correctly skipped.

        if self.peft_config is None:
            print("\nFull fine-tuning: nothing to merge - full model was already saved to MERGED_DIR.")
            return

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
        self.configure_peft()
        self.load_model()
        self.prepare_data()
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
