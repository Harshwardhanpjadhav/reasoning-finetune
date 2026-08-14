"""
Turns raw GSM8K examples into the chat-formatted training text.
"""

COT_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then give the final numeric answer on its own "
    "line in the form: #### <answer>"
)


def format_for_training(example: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"{COT_INSTRUCTION}\n\nProblem: {example['question']}"},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def preprocess_dataset(dataset, tokenizer):
    def _map_fn(example):
        formatted = format_for_training(example)
        text = tokenizer.apply_chat_template(
            formatted["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    return dataset.map(_map_fn, remove_columns=dataset.column_names)