"""
Runs a model against GSM8K and scores it with the pure metric functions
from metrics.py. Same function is called for baseline (no adapter) and
post-training (with LoRA adapter) eval — same code = fair comparison.
"""

import json
import time
from pathlib import Path

from src.eval.metrics import extract_final_answer, numbers_match
from src.data.loader import load_gsm8k

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def _load_model(base_model: str, adapter_path: str | None, load_in_4bit: bool):
    """Loads model + tokenizer, optionally with a LoRA adapter attached."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    return model, tokenizer


def _generate(model, tokenizer, question: str, instruction: str, max_new_tokens: int,
               is_base_model: bool = False) -> str:
    import torch
    from src.tracing.langfuse_client import trace_generation

    with trace_generation("gsm8k_eval_generation", input_text=question) as trace:
        if is_base_model:
            prompt_text = f"Question: {question}\nAnswer: Let's think step by step. "
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        else:
            messages = [{"role": "user", "content": f"{instruction}\n\nProblem: {question}"}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, max_length=None, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[-1]
        output_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        trace.set_output(output_text)

    return output_text


def run_eval(
    base_model: str,
    adapter_path: str | None = None,
    split: str = "test",
    limit: int = 200,
    max_new_tokens: int = 512,
    load_in_4bit: bool = True,
    tag: str = "baseline",
    is_base_model: bool = False,
) -> dict:
    """
    Runs GSM8K eval and writes both detailed predictions and a summary to
    results/. Returns the summary dict (accuracy, n_examples, etc).
    """
    from tqdm import tqdm

    instruction = (
        "Solve the following math problem step by step. "
        "Show your reasoning, then give the final numeric answer on its own "
        "line in the form: #### <answer>"
    )

    ds = load_gsm8k(split=split, limit=limit)
    model, tokenizer = _load_model(base_model, adapter_path, load_in_4bit)

    records, correct = [], 0
    start = time.time()

    for ex in tqdm(ds, desc=f"Evaluating GSM8K [{tag}]"):
        gold = extract_final_answer(ex["answer"])
        raw_output = _generate(model, tokenizer, ex["question"], instruction, max_new_tokens, is_base_model)
        pred = extract_final_answer(raw_output)
        is_correct = numbers_match(pred, gold)
        correct += int(is_correct)
        records.append({
            "question": ex["question"], "gold_answer": gold,
            "predicted_answer": pred, "correct": is_correct, "raw_output": raw_output,
        })

    accuracy = correct / len(ds) if len(ds) else 0.0
    RESULTS_DIR.mkdir(exist_ok=True)

    detail_path = RESULTS_DIR / f"gsm8k_{tag}_predictions.jsonl"
    with open(detail_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "base_model": base_model, "adapter": adapter_path, "tag": tag,
        "n_examples": len(ds), "accuracy": round(accuracy, 4),
        "elapsed_sec": round(time.time() - start, 1),
    }
    summary_path = RESULTS_DIR / f"gsm8k_{tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary