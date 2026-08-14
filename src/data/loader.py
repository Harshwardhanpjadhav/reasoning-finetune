"""
Dataset loading for GSM8K (train + test).

Kept separate from preprocess.py deliberately: this module only knows how
to FETCH data. Formatting/tokenization logic lives in preprocess.py so you
can swap datasets (e.g. MetaMathQA later) without touching training code.
"""


def load_gsm8k(split: str = "train", limit: int | None = None):
    """
    Load GSM8K from Hugging Face. Requires network access to huggingface.co —
    only run this on Kaggle/Lightning/Colab, not in a network-restricted sandbox.

    Args:
        split: "train" or "test"
        limit: optional cap on number of examples (useful for smoke tests)

    Returns:
        HF Dataset with columns ["question", "answer"]
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_generalization_set(domain: str = "logic_puzzles", limit: int | None = None):
    """
    Placeholder for the held-out generalization test set (Step 6 of the plan).
    Not GSM8K — a different reasoning domain the model wasn't trained on,
    used to check whether fine-tuning taught transferable reasoning or just
    memorized GSM8K-style patterns.

    TODO: wire up an actual dataset (e.g. a logic-puzzle benchmark) once the
    core training + eval loop is validated end-to-end.
    """
    raise NotImplementedError(
        f"Generalization dataset for domain='{domain}' not wired up yet. "
        "This is intentionally deferred until Steps 1-5 are working."
    )