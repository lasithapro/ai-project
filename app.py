"""Simple Question Answering demo with optional QLoRA (4-bit + LoRA) acceleration.

This script will:
 1. Try to load a quantized 4-bit model with LoRA adapters (QLoRA) IF a GPU and bitsandbytes are available.
 2. Fall back automatically to a standard fp32 CPU model if quantization is unsupported (e.g. in CPU-only envs).
 3. Run a single extractive QA inference and print the answer span.

Environment notes:
 - bitsandbytes 4-bit path needs a CUDA GPU. In CPU-only environments it will fall back gracefully.
 - LoRA adapters here are untrained; they illustrate wrapping only. For real fine-tuning you must train them.
"""

from __future__ import annotations

import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BitsAndBytesConfig

_PEFT_AVAILABLE = True
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:  # pragma: no cover - narrow environment
    _PEFT_AVAILABLE = False
    warnings.warn("PEFT not available. Running without LoRA adapters.")


MODEL_NAME = os.environ.get("QA_MODEL", "distilbert-base-uncased-distilled-squad")


def can_use_qlora() -> bool:
    """Return True if we likely can use 4-bit quantization.

    Conditions:
      - CUDA is available
      - bitsandbytes can be imported
    """
    if not torch.cuda.is_available():
        return False
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def build_quant_config() -> BitsAndBytesConfig | None:
    if not can_use_qlora():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model():
    """Load tokenizer + model with optional 4-bit + (dummy) LoRA.

    Returns (model, tokenizer, used_qlora: bool)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    quant_config = build_quant_config()

    kwargs = {}
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = "auto"
    else:
        # Force CPU for deterministic behavior in CPU-only envs.
        kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, **kwargs)

    used_qlora = False
    if quant_config is not None and _PEFT_AVAILABLE:
        try:
            # DistilBERT attention projection layers are named 'q_lin' and 'v_lin'.
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_lin", "v_lin"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.QUESTION_ANS if hasattr(TaskType, "QUESTION_ANS") else TaskType.TOKEN_CLS,
            )
            model = get_peft_model(model, lora_config)
            used_qlora = True
        except Exception as e:  # pragma: no cover
            warnings.warn(f"Failed to wrap model with LoRA adapters: {e}")
    return model, tokenizer, used_qlora


def answer_question(model, tokenizer, question: str, context: str) -> str:
    inputs = tokenizer(question, context, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    start_index = int(torch.argmax(outputs.start_logits))
    end_index = int(torch.argmax(outputs.end_logits))
    if end_index < start_index:
        # Fallback: choose the single best start token
        end_index = start_index
    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return answer.strip()


def main():
    model, tokenizer, used_qlora = load_model()
    device = next(model.parameters()).device
    print(f"Loaded model '{MODEL_NAME}' on device {device}. QLoRA active: {used_qlora}")
    if not can_use_qlora():
        print("(Running without 4-bit quantization — CPU-only fallback.)")

    context = (
        "Sri Lanka is an island country in South Asia, located in the Indian Ocean. "
        "It gained independence from Britain in 1948. Colombo is the commercial capital, "
        "while Sri Jayawardenepura Kotte is the legislative capital."
    )
    question = "When did Sri Lanka gain independence?"
    answer = answer_question(model, tokenizer, question, context)
    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":  # pragma: no cover
    main()