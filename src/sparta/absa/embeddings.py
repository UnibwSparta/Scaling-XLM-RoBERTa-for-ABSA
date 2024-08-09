from string import Template
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import Dataset
from transformers import BatchEncoding

from sparta.absa.aspects import remap_labels_for_absa_laptop_2014
from sparta.absa.tokenize import get_tokenize_function


def prepare_embeddings_dataset_for_absa_laptop_2014(
    ds: Dataset,
    tokenizer_name: str,
    pre_aspect_suffix: str,
) -> Dataset:
    """Add aspect suffix, tokenize dataset, and find multiple aspect masks.

    Args:
        dataset (Dataset): Full original dataset
        tokenizer_name (str): Name of a pre-trained tokenizer to use
        pre_aspect_suffix (str): Suffix to add to the text before the additional aspect (e.g., "This is a review about")

    Returns:
        Dataset: Tokenized dataset
    """
    # Add aspect suffix to the dataset
    pre_aspect_suffix_t = Template(pre_aspect_suffix + " $aspect")

    def add_aspect_suffix_to_example(example: Dict[str, Any]) -> Dict[str, Any]:
        example["text_with_suffix"] = example["text"] + pre_aspect_suffix_t.substitute(aspect=example["aspect"])
        example["end_suffix"] = len(example["text_with_suffix"])
        example["start_suffix"] = example["end_suffix"] - len(example["aspect"])
        return example

    ds = ds.map(add_aspect_suffix_to_example)

    # Tokenize dataset
    tokenize = get_tokenize_function(tokenizer_name, max_length=100, padding_max_length=False, text_column="text_with_suffix")
    tokenized_as_batch_encoding = tokenize(ds)

    # Get original aspect masks for all examples in the train and test datasets
    aspect_indices, aspect_masks = get_all_aspect_masks_by_token_to_chars(
        dataset=ds,
        batch_encoding=tokenized_as_batch_encoding,
        start_positions=ds["start"],
        end_positions=ds["end"],
        text_column="text",
    )

    # Get suffix-added aspect masks for all examples in the train and test datasets
    suffix_aspect_indices, suffix_aspect_masks = get_all_aspect_masks_by_token_to_chars(
        dataset=ds,
        batch_encoding=tokenized_as_batch_encoding,
        start_positions=ds["start_suffix"],
        end_positions=ds["end_suffix"],
        text_column="text_with_suffix",
    )

    # Combine the original and suffix-added aspect masks
    aspect_masks_all: List[Optional[List[bool]]] = [None] * len(aspect_masks)
    aspect_indices_all: List[Optional[Set[int]]] = [None] * len(aspect_indices)
    for i in range(len(aspect_masks)):
        aspect_masks_all[i] = [any(x) for x in zip(aspect_masks[i], suffix_aspect_masks[i])]
        aspect_indices_all[i] = aspect_indices[i].union(suffix_aspect_indices[i])

    # Add tokenized inputs and aspect masks to the original dataset
    ds = ds.add_column("input_ids", tokenized_as_batch_encoding["input_ids"])
    ds = ds.add_column("attention_mask", tokenized_as_batch_encoding["attention_mask"])
    ds = ds.add_column("aspect_mask", aspect_masks)
    ds = ds.add_column("aspect_mask_suffix", suffix_aspect_masks)
    ds = ds.add_column("aspect_mask_all", aspect_masks_all)

    # Also add aspect indices to the dataset
    ds = ds.add_column("aspect_indices", aspect_indices)
    ds = ds.add_column("aspect_indices_suffix", suffix_aspect_indices)
    ds = ds.add_column("aspect_indices_all", aspect_indices_all)

    # Rename the label column to labels to match the model's requirements
    ds = ds.rename_column("label", "labels")

    # Remap the labels to the fit the model's requirements
    ds = ds.map(remap_labels_for_absa_laptop_2014)

    return ds


def get_all_aspect_masks_by_token_to_chars(
    dataset: Dataset,
    batch_encoding: BatchEncoding,
    start_positions: List[int],
    end_positions: List[int],
    text_column: str,
) -> Tuple[List[Set[int]], List[List[bool]]]:
    """Return tokens correponding to a spans in the original texts of a tokenized dataset using the reverse approach with token_to_chars().

    Note:
        PreTrainedTokenizerFast with Llama models seems to have a bug in char_to_token() method.
        Therefore, we use token_to_chars() method instead.
        Although this method also has some issues, because it only returns the correct start character of the token span.

    Args:
        dataset (Dataset): Full original dataset
        batch_encoding (BatchEncoding): BatchEncoding object after preprocessing of an item
        start_positions (List[int]): Start characters of aspect spans in the original texts
        end_positions (List[int]): End characters of aspect spans in the original texts
        text_column (str): Name of the column containing text

    Returns:
        Set[int]: List of token indexes
        List[bool]: Mask for this tokens
    """
    aspect_indices = []
    aspect_masks = []
    num_items = len(batch_encoding["input_ids"])
    for item, input_ids, batch_index, start, end in zip(dataset, batch_encoding["input_ids"], range(num_items), start_positions, end_positions):
        max_len_tokens = len(input_ids)
        indices, mask = get_aspect_mask_by_token_to_chars(
            item=item,
            batch_encoding=batch_encoding,
            batch_index=batch_index,
            start=start,
            end=end,
            max_len_tokens=max_len_tokens,
            text_column=text_column,
        )
        aspect_indices.append(indices)
        aspect_masks.append(mask)

    return aspect_indices, aspect_masks


def get_aspect_mask_by_token_to_chars(
    item: Dict[str, Any],
    batch_encoding: BatchEncoding,
    batch_index: int,
    start: int,
    end: int,
    max_len_tokens: int,
    text_column: str,
) -> Tuple[Set[int], List[bool]]:
    """Return tokens correponding to a span in the original text.

    Args:
        item (Dict[str, Any]): Item from the dataset
        batch_encoding (BatchEncoding): BatchEncoding object after preprocessing of an item
        batch_index (int): Index of the item in the batch
        start (int): Start character of aspect span in the original text
        end (int): End character of aspect span in the original text
        max_len_tokens (int): Maximum number of tokens in the item
        text_column (str): Name of the column containing text

    Returns:
        Set[int]: List of token indexes
        List[bool]: Mask for this tokens
    """
    # Get tokens to chars mapping
    char_to_token = {}
    for token_i in range(len(batch_encoding["input_ids"][batch_index])):
        char_span = batch_encoding.token_to_chars(batch_index, token_i)
        if char_span is not None:
            char_to_token[char_span.start] = token_i

    # Make mapping for all characters
    char_to_token_list = []
    last_token = None
    for i in range(len(item[text_column])):
        if i in char_to_token:
            last_token = char_to_token[i]
        char_to_token_list.append(last_token)

    # Collect tokens for the aspect
    token_indices = set()
    for i in range(start, end):
        if i >= len(char_to_token_list):
            break
        token = char_to_token_list[i]
        if token is not None:
            token_indices.add(token)

    tokens_mask = [i in token_indices for i in range(max_len_tokens)]

    if len(token_indices) < 1:
        raise ValueError("Empty aspect mask")

    return token_indices, tokens_mask
