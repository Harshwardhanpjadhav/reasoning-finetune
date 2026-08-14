import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.eval.eval import run_eval


def main():
    # Reverted back to the 'gsm8k' configuration block
    cfg = load_config("eval")["gsm8k"]
    model_cfg = load_config("model")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=model_cfg["base_model"])
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=cfg["limit"])
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    run_eval(
        base_model=args.model,
        adapter_path=args.adapter,
        split=cfg["split"],
        limit=args.limit,
        max_new_tokens=cfg["max_new_tokens"],
        load_in_4bit=model_cfg["load_in_4bit"],
        tag=args.tag,
        is_base_model=model_cfg.get("is_base_model", False),
    )


if __name__ == "__main__":
    main()