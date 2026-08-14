"""
Fine-tuning via Unsloth + trl's SFTTrainer.

Built specifically to survive Lightning AI's 4-hour Studio restarts (and
Kaggle's 9-hour session limits): if output_dir already contains a checkpoint
when this runs, we resume from it automatically instead of starting over.
"""

from pathlib import Path


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Returns the path to the latest checkpoint-N dir, or None if none exist."""
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(
        out.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def train(model_config: dict, training_config: dict):
    """
    Runs (or resumes) a LoRA fine-tune.
    """
    from unsloth import FastLanguageModel, get_chat_template
    from trl import SFTTrainer, SFTConfig
    import wandb

    from src.data.loader import load_gsm8k
    from src.data.preprocess import preprocess_dataset

    t_cfg = training_config["training"]
    w_cfg = training_config["wandb"]
    d_cfg = training_config["dataset"]

    resume_ckpt = None
    if t_cfg.get("resume_from_checkpoint") == "auto":
        resume_ckpt = find_latest_checkpoint(t_cfg["output_dir"])
        if resume_ckpt:
            print(f"Found existing checkpoint, resuming from: {resume_ckpt}")
        else:
            print("No existing checkpoint found, starting fresh.")

    wandb.init(project=w_cfg["project"], name=w_cfg["run_name"], tags=w_cfg.get("tags", []))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config["base_model"],
        max_seq_length=d_cfg["max_seq_length"],
        load_in_4bit=model_config["load_in_4bit"],
        dtype=None,
    )

    # Assign the Gemma chat template to the tokenizer so apply_chat_template works
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    lora_cfg = model_config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    raw_ds = load_gsm8k(split=d_cfg["split"])
    train_ds = preprocess_dataset(raw_ds, tokenizer)

    args = SFTConfig(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        warmup_ratio=t_cfg["warmup_ratio"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        report_to="wandb",
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_path = f"{t_cfg['output_dir']}/final_adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final adapter saved to: {final_path}")
    wandb.finish()
    return final_path