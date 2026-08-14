#!/bin/bash
# Run this once on a fresh Lightning Studio (or after a restart, it's safe to
# re-run — pip install is idempotent).
#
# Usage: bash scripts/setup_lightning.sh

set -e

echo ">>> Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo ">>> Checking for required auth tokens as environment variables..."
echo "    (Set these in Lightning Studio's Settings -> Environment Variables,"
echo "     not hardcoded here, so they aren't committed to the repo.)"
echo ""

check_var() {
    if [ -z "${!1}" ]; then
        echo "  [MISSING] $1 — $2"
    else
        echo "  [OK]      $1"
    fi
}

check_var "HF_TOKEN" "needed to download models/datasets from Hugging Face"
check_var "WANDB_API_KEY" "needed for experiment tracking"
check_var "LANGFUSE_PUBLIC_KEY" "optional — tracing no-ops if unset"
check_var "LANGFUSE_SECRET_KEY" "optional — tracing no-ops if unset"

echo ""
echo ">>> Logging into Hugging Face (if HF_TOKEN is set)..."
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

echo ""
echo ">>> Logging into W&B (if WANDB_API_KEY is set)..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi

echo ""
echo ">>> Setup complete. Next steps:"
echo "    1. Smoke test:  python scripts/run_eval.py --tag smoke --model unsloth/Qwen2.5-0.5B-Instruct --limit 5"
echo "    2. Baseline:    python scripts/run_eval.py --tag baseline"
echo "    3. Train:       python scripts/run_training.py"
