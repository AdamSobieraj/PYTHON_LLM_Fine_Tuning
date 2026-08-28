import os
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class PEFTOptions:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"
    use_4bit: bool = False
    double_quant: bool = True
    gradient_checkpointing: bool = True

    @classmethod
    def from_env(cls) -> "PEFTOptions":
        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return default
            return int(raw)

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or not raw.strip():
                return default
            return float(raw)

        target_raw = os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj")
        target_modules = tuple(
            part.strip() for part in target_raw.split(",") if part.strip()
        )

        return cls(
            r=_int("LORA_R", 16),
            lora_alpha=_int("LORA_ALPHA", 32),
            lora_dropout=_float("LORA_DROPOUT", 0.05),
            target_modules=target_modules,
            bias=os.getenv("LORA_BIAS", "none"),
            use_4bit=_env_bool("USE_4BIT", False),
            double_quant=_env_bool("DOUBLE_QUANT", True),
            gradient_checkpointing=_env_bool("GRADIENT_CHECKPOINTING", True),
        )


class PEFTMethod:
    """
    Baza dla QLoRA / DoRA / FullFineTune.
    """

    name: str = "lora"

    def __init__(self, options: Optional[PEFTOptions] = None):
        self.options = options or PEFTOptions()

    def _base_lora_kwargs(self):
        return {
            "task_type": "CAUSAL_LM",
            "r": self.options.r,
            "lora_alpha": self.options.lora_alpha,
            "lora_dropout": self.options.lora_dropout,
            "target_modules": list(self.options.target_modules),
            "bias": self.options.bias,
        }

    def peft_config(self):
        return LoraConfig(**self._base_lora_kwargs())

    def _bnb_config(self):
        if not self.options.use_4bit:
            return None

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.options.double_quant,
        )

    def _prepare_kbit(self, model):
        if not self.options.use_4bit:
            return model

        return prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.options.gradient_checkpointing,
        )

    def quantization_config(self):
        return None

    def prepare_model(self, model):
        return model

    def enable_input_require_grads(self, model):
        try:
            model.enable_input_require_grads()
        except AttributeError:
            pass
        return model


class QLoRA(PEFTMethod):
    """
    QLoRA = LoRA + 4-bit quantization.
    """

    name = "qlora"

    def __init__(self, options: Optional[PEFTOptions] = None):
        options = options or PEFTOptions()
        options.use_4bit = True
        super().__init__(options)

    def quantization_config(self):
        return self._bnb_config()

    def prepare_model(self, model):
        return self._prepare_kbit(model)


class DoRA(PEFTMethod):
    """
    DoRA = LoRA z włączonym use_dora=True.
    Opcjonalnie można też zrobić DoRA + 4-bit, jeśli use_4bit=True.
    """

    name = "dora"

    def peft_config(self):
        kwargs = self._base_lora_kwargs()
        kwargs["use_dora"] = True

        try:
            return LoraConfig(**kwargs)
        except TypeError:
            # Starsze wersje PEFT mogą nie mieć parametru use_dora
            # w sygnaturze LoraConfig.
            kwargs.pop("use_dora", None)
            cfg = LoraConfig(**kwargs)
            cfg.use_dora = True
            return cfg

    def quantization_config(self):
        return self._bnb_config() if self.options.use_4bit else None

    def prepare_model(self, model):
        return self._prepare_kbit(model) if self.options.use_4bit else model


class FullFineTune(PEFTMethod):
    """
    Pełny fine-tuning modelu.
    Brak adaptera PEFT.
    """

    name = "full"

    def __init__(self, options: Optional[PEFTOptions] = None):
        options = options or PEFTOptions()
        options.use_4bit = False
        super().__init__(options)

    def peft_config(self):
        return None

    def quantization_config(self):
        return None

    def prepare_model(self, model):
        # W pełnym fine-tuning uczymy wszystkie parametry.
        for param in model.parameters():
            param.requires_grad = True

        if self.options.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"[FULL FT] Could not enable gradient checkpointing: {e}")

        return model


def create_peft_method(
    method: Optional[str] = None,
    options: Optional[PEFTOptions] = None,
) -> PEFTMethod:
    """
    Wybiera metodę treningu:
    - qlora
    - dora
    - lora
    - full
    """

    method_name = (method or os.getenv("PEFT_METHOD") or "qlora").strip().lower()
    options = options or PEFTOptions.from_env()

    if method_name in {"qlora", "qlofa", "qlora-4bit"}:
        return QLoRA(options)

    if method_name in {"dora", "dora-4bit", "q-dora"}:
        if "4bit" in method_name:
            options.use_4bit = True
        return DoRA(options)

    if method_name in {"lora", "peft"}:
        return PEFTMethod(options)

    if method_name in {"full", "full-finetune", "full_finetune", "ft"}:
        options.use_4bit = False
        return FullFineTune(options)

    raise ValueError(
        f"Nieznana metoda PEFT: {method_name}. "
        "Dozwolone wartości: qlora, dora, lora, full."
    )
