# Reasoning Fine-Tune: CoT Distillation on Math Reasoning

Fine-tune Qwen2.5-7B-Instruct on chain-of-thought reasoning traces (GSM8K),
with a modular pipeline: config-driven, checkpoint-resumable, tracked in
W&B, traced in Langfuse.

## Architecture

```
reasoning-finetune/
├── configs/                 # every knob lives here, not hardcoded
│   ├── model.yaml            # base model, LoRA settings
│   ├── training.yaml         # hyperparams, checkpoint/resume, W&B project
│   └── eval.yaml              # benchmark settings
│
├── src/
│   ├── config.py              # YAML -> dict loader, used everywhere
│   ├── data/
│   │   ├── loader.py           # fetches GSM8K from HF
│   │   └── preprocess.py       # formats examples into chat-template training text
│   ├── training/
│   │   └── train.py            # Unsloth + trl SFTTrainer, auto checkpoint-resume
│   ├── eval/
│   │   ├── metrics.py          # pure answer-extraction/scoring (no deps, no GPU)
│   │   └── gsm8k_eval.py       # inference + scoring loop, used for before/after
│   └── tracing/
│       └── langfuse_client.py  # no-ops if Langfuse env vars aren't set
│
├── scripts/                  # what you actually run
│   ├── setup_lightning.sh     # one-time env setup on a Lightning Studio
│   ├── run_eval.py             # baseline AND post-training eval (same code)
│   └── run_training.py         # train, safe to re-run after a restart
│
├── tests/
│   └── test_pipeline.py       # 16 tests, zero GPU/network needed, run anywhere
│
└── results/                   # eval outputs land here (gitignored large files)
```

## Where things run

This repo is developed and unit-tested in a network-restricted sandbox
(no GPU, no huggingface.co access) — that's intentional. Anything that
doesn't need a model or dataset download is tested here first.

**Actual model/data work (training, eval) runs on Lightning AI (T4 Studio).**

## Setup (on Lightning Studio)

```bash
git clone <this-repo>
cd reasoning-finetune
bash scripts/setup_lightning.sh
```

Set these as environment variables in Lightning's Studio settings (not in code):
- `HF_TOKEN` — Hugging Face access token
- `WANDB_API_KEY` — Weights & Biases
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — optional, tracing no-ops without them

## Run order

```bash
# 1. Smoke test — tiny model, 5 examples, confirms the pipeline works end-to-end
python scripts/run_eval.py --tag smoke --model unsloth/Qwen2.5-0.5B-Instruct --limit 5

# 2. Baseline eval — the real "before" number
python scripts/run_eval.py --tag baseline

# 3. Train — safe to re-run after a 4hr Studio restart, auto-resumes
python scripts/run_training.py

# 4. Post-training eval — the "after" number, same benchmark as step 2
python scripts/run_eval.py --tag post-finetune --adapter results/checkpoints/gsm8k-lora/final_adapter
```

## Status
- [x] Modular repo scaffold
- [x] Config layer (model/training/eval YAML)
- [x] Data loader + preprocessing (unit tested)
- [x] Eval metrics + harness (unit tested)
- [x] Training module with checkpoint/resume (unit tested)
- [x] Langfuse tracing wrapper
- [ ] Run smoke test on Lightning
- [ ] Run baseline eval on Lightning
- [ ] Run real training
- [ ] Post-training eval
- [ ] Error analysis + generalization test (held-out domain, DeepEval)
- [ ] Write-up
