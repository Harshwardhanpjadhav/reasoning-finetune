"""
Run (or resume) training. Usage on Lightning Studio:

    python scripts/run_training.py

Safe to re-run after a Studio restart — it auto-detects the latest
checkpoint in configs/training.yaml's output_dir and resumes from there.
Requires: WANDB_API_KEY set as an environment variable (or run `wandb login`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.training.train import train


def main():
    model_cfg = load_config("model")
    training_cfg = load_config("training")
    train(model_cfg, training_cfg)


if __name__ == "__main__":
    main()
