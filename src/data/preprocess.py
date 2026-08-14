"""
Turns raw GSM8K examples into the chat-formatted training text the model
actually trains on. Isolated from loader.py so prompt format changes don't
require touching data-fetching code.
"""

COT_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then give the final numeric answer on its own "
    "line in the form: #### <answer>"
)


def format_for_training(example: dict) -> dict:
    """
    Formats a single GSM8K example (question + gold CoT answer) into the
    chat-template structure expected by trl's SFTTrainer.

    GSM8K's own 'answer' field already contains step-by-step reasoning
    ending in '#### <number>', so it doubles as a CoT training target.
    """
    return {
        "messages": [
            {"role": "user", "content": f"{COT_INSTRUCTION}\n\nProblem: {example['question']}"},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def preprocess_dataset(dataset, tokenizer):
    """
    Applies format_for_training to every example, then renders each through
    the tokenizer's chat template so the dataset is ready for SFTTrainer.
    """
    def _map_fn(example):
        formatted = format_for_training(example)
        text = tokenizer.apply_chat_template(
            formatted["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return dataset.map(_map_fn, remove_columns=dataset.column_names)
