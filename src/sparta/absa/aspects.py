from string import Template
from typing import Any, Dict, List, Set, Tuple

from datasets import Dataset
from transformers import BatchEncoding

from sparta.absa.tokenize import get_tokenize_function


def prepare_dataset_for_absa_laptop_2014(ds: Dataset, tokenizer_name: str) -> Dataset:
    """Tokenize dataset and find aspect masks.

    Args:
        dataset (Dataset): Full original dataset
        tokenizer_name (str): Name of a pre-trained tokenizer to use

    Returns:
        Dataset: Tokenized dataset
    """
    tokenize = get_tokenize_function(tokenizer_name, max_length=100)
    tokenized_as_batch_encoding = tokenize(ds)

    # Get aspect masks for all examples in the train and test datasets
    _, aspect_masks = get_all_aspect_masks(
        batch_encoding=tokenized_as_batch_encoding,
        start_positions=ds["start"],
        end_positions=ds["end"],
    )

    # Add tokenized inputs and aspect masks to the original dataset
    ds = ds.add_column("input_ids", tokenized_as_batch_encoding["input_ids"])
    ds = ds.add_column("attention_mask", tokenized_as_batch_encoding["attention_mask"])
    ds = ds.add_column("aspect_mask", aspect_masks)

    # Rename the label column to labels to match the model's requirements
    ds = ds.rename_column("label", "labels")

    # Remap the labels to the fit the model's requirements
    ds = ds.map(remap_labels_for_absa_laptop_2014)

    return ds


def remap_labels_for_absa_laptop_2014(example: Dict[str, Any]) -> Dict[str, Any]:
    """Remap the labels to the range [0, num_labels - 1].

    The model expects labels in the range [0, num_labels - 1],
    but we have them in the range [-1, num_labels - 2] for this special case of this dataset.

    Args:
        example (Dict[str, Any]): Example from the dataset

    Returns:
        Dict[str, Any]: Example with remapped labels
    """
    example["labels"] = example["labels"] + 1
    return example


def get_all_aspect_masks(batch_encoding: BatchEncoding, start_positions: List[int], end_positions: List[int]) -> Tuple[List[Set[int]], List[List[bool]]]:
    """Return tokens correponding to a spans in the original texts of a tokenized dataset.

    Args:
        batch_encoding (BatchEncoding): BatchEncoding object after preprocessing of an item
        start_positions (List[int]): Start characters of aspect spans in the original texts
        end_positions (List[int]): End characters of aspect spans in the original texts

    Returns:
        Set[int]: List of token indexes
        List[bool]: Mask for this tokens
    """
    aspect_indices = []
    aspect_masks = []
    for item_encoding, start, end in zip(batch_encoding[:], start_positions, end_positions):
        max_len_tokens = len(item_encoding.ids)
        indices, mask = get_aspect_mask(item_encoding, start, end, max_len_tokens)
        aspect_indices.append(indices)
        aspect_masks.append(mask)

    return aspect_indices, aspect_masks


def get_aspect_mask(
    item_encoding: BatchEncoding,
    start: int,
    end: int,
    max_len_tokens: int,
) -> Tuple[Set[int], List[bool]]:
    """Return tokens correponding to a span in the original text.

    Args:
        batch_encoding (BatchEncoding): BatchEncoding object after preprocessing of an item
        start (int): Start character of aspect span in the original text
        end (int): End character of aspect span in the original text

    Returns:
        Set[int]: List of token indexes
        List[bool]: Mask for this tokens
    """
    token_indices = set()
    for i in range(start, end):
        token = item_encoding.char_to_token(i)
        if token is not None:
            token_indices.add(token)

    tokens_mask = [i in token_indices for i in range(max_len_tokens)]

    if len(token_indices) < 1:
        raise ValueError("Empty aspect mask")

    return token_indices, tokens_mask


def prepare_dataset_for_absa_laptop_2014_llama(
    ds: Dataset,
    tokenizer_name: str,
    aspect_suffix: str,
) -> Dataset:
    """Prepare dataset for aspect-based sentiment analysis with a Llama-like model for sequence classification.

    This function adds an aspect suffix to the text, renames the label column to labels, remaps the labels, and tokenizes the dataset.
    LlamaForSequenceClassification expects the labels to be in the range [0, num_labels - 1], but the dataset has them in the range [-1, num_labels - 2].
    The aspect suffix is added to the text to help the model understand the aspect of the review.
    LlamaForSequenceClassification will use the last token for final classification.

    Args:
        ds (Dataset): Dataset
        tokenizer_name (str): Name of the tokenizer to use
        aspect_suffix (str): Surfix to add to the text before the additional aspect (e.g., "This is a review about")

    Returns:
        Dataset: Prepared dataset
    """
    # Get tokenizer function
    tokenize = get_tokenize_function(
        tokenizer_name,
        max_length=100,
        padding_max_length=True,
        pad_token_is_eos_token=True,
    )

    # Add aspect suffix to the dataset
    suffix = Template(aspect_suffix + " $aspect")

    def add_aspect_suffix_to_example(example: Dict[str, Any]) -> Dict[str, Any]:
        example["text"] = example["text"] + suffix.substitute(aspect=example["aspect"])
        return example

    ds = ds.map(add_aspect_suffix_to_example)

    # Rename the label column to labels to match the model's requirements
    ds = ds.rename_column("label", "labels")

    # Remap the labels to the fit the model's requirements
    ds = ds.map(remap_labels_for_absa_laptop_2014)

    # Tokenize dataset
    ds = ds.map(tokenize)

    return ds
