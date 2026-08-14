"""
Pure functions for scoring GSM8K-style outputs. Deliberately has zero
dependency on torch/transformers/datasets — these are just string/number
parsing functions, so they can be unit-tested instantly, anywhere, without
a GPU or even installing ML libraries.
"""

import re


def extract_final_answer(text: str) -> str | None:
    """Extract the number after '####'. Falls back to last number in text."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").rstrip(".")
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").rstrip(".")
    return None


def numbers_match(pred: str | None, gold: str | None) -> bool:
    """Compares two extracted answers, tolerant of float formatting differences."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()
