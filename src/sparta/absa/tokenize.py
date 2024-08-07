from typing import Any, Callable, Dict

from transformers import AutoTokenizer, BatchEncoding


def get_tokenize_function(tokenizer_name: str, max_length: int = 100) -> Callable[[Dict[str, Any]], BatchEncoding]:
    """Get tokenizer function for a dataset.

    Args:
        tokenizer (str): Name or path of a pre-trained tokenizer
        max_length (int, optional): Maximum sequence length. Defaults to 100.

    Returns:
        BatchEncoding: Tokenized dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples: Dict[str, Any]) -> BatchEncoding:
        # We know that 100 tokens is enough to encode items in this dataset using the intended model
        return tokenizer(examples["text"], padding="max_length", max_length=max_length, truncation=True)

    return tokenize_function
