Load the base model — Qwen2.5-1.5B (base, no instruction-tuning)
Load the dataset — GSM8K, but this splits into two separate parts:
Test split (~1,300 examples, we use 200) → used for eval only, never for training
Train split (~7,500 examples) → used for training only, never for eval
Run the base model on the test split, predict answers, save results → this is your baseline (--tag baseline)
Fine-tune the model using the train split (a different set of examples than what you evaluated on)
Run the fine-tuned model on the same test split from step 3, predict answers, save results → this is your post-finetune eval