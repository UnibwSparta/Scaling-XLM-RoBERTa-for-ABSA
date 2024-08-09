from typing import Any, Callable, Dict

from transformers import AutoTokenizer, BatchEncoding


def get_tokenize_function(
    tokenizer_name: str,
    max_length: int = 100,
    padding_max_length: bool = True,
    text_column: str = "text",
    pad_token_is_eos_token: bool = False,
) -> Callable[[Dict[str, Any]], BatchEncoding]:
    """Get tokenizer function for a dataset.

    Args:
        tokenizer (str): Name or path of a pre-trained tokenizer
        max_length (int, optional): Maximum sequence length. Defaults to 100.
        padding_max_length (bool, optional): Whether to pad to max_length. Defaults to True.
        text_column (str, optional): Name of the column containing text. Defaults to "text".
        pad_token_is_eos_token (bool, optional): Whether the pad token is the same as the eos. Defaults to False.

    Returns:
        BatchEncoding: Tokenized dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token if pad_token_is_eos_token else tokenizer.pad_token

    def tokenize_function(examples: Dict[str, Any]) -> BatchEncoding:
        # We know that 100 tokens is enough to encode items in this dataset using the intended model
        if padding_max_length:
            return tokenizer(examples[text_column], padding="max_length", max_length=max_length, truncation=True)
        else:
            return tokenizer(examples[text_column], max_length=max_length, truncation=True)

    return tokenize_function
