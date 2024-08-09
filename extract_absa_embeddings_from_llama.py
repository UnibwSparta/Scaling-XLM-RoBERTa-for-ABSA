#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""extract_llama_embeddings.py: Extract embeddings from a Llama-like model at aspect positions and at the end.

We use the Laptop domain dataset from SemEval 2014 for aspect-based sentiment analysis.
To distinguish items with different aspects, we add a suffix to each item and a repeatition of aspect at the very end.
Then we extract embeddings for the aspect and the aspect at the end.
"""

import warnings
from typing import Any, Callable, Dict

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, pipeline

from sparta.absa.embeddings import prepare_embeddings_dataset_for_absa_laptop_2014

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# TODO 1: Define the path to the model
# model_path = "/mnt/exafs/models/llama3/Meta-Llama-3-8B-Instruct/"
# model_path = "/mnt/exafs/models/llama3/Meta-Llama-3-8B/"
model_path = "/mnt/exafs/models/llama31/Meta-Llama-3.1-8B/"
# model_path = "gpt2"

# TODO 2: Define the device to use
device = "cuda:3"

# TODO 3: Customize the aspect suffix. Note that the target aspect will be added after this suffix.
suffix = "–This is a review about"


def get_aspect_features_with_suffix(
    examples: Dict[str, Any],
    pipe: Callable[[str], torch.Tensor],
) -> Dict[str, Any]:
    """Get aspect features for a dataset.

    Args:
        example (Dict[str, Any]): Dataset example
        pipe (Callable[[str], torch.Tensor]): Feature extraction pipeline

    Returns:
        Dict[str, Any]: Dataset with aspect features
    """
    aspect_features = pipe(examples["text_with_suffix"])

    zipped_examples = zip(
        aspect_features,
        examples["aspect_indices"],
        examples["aspect_indices_suffix"]
    )
    all_embeddings_aspect = []
    all_embeddings_aspect_suffix = []
    for features, aspect_indices, aspect_indices_suffix in zipped_examples:
        # Get embeddings for the original aspect
        embeddings_aspect = []
        for token_idx in sorted(aspect_indices):
            embeddings_aspect.append(features[0][token_idx])

        # Get embeddings for the suffix-added aspect
        embeddings_aspect_suffix = []
        for token_idx in sorted(aspect_indices_suffix):
            embeddings_aspect_suffix.append(features[0][token_idx])

        # Max pooling for the original aspect
        embeddings_aspect_stacked = torch.stack(embeddings_aspect)
        all_embeddings_aspect.append(torch.max(embeddings_aspect_stacked, dim=0).values.tolist())

        # Max pooling for the suffix-added aspect
        embeddings_aspect_suffix_stacked = torch.stack(embeddings_aspect_suffix)
        all_embeddings_aspect_suffix.append(torch.max(embeddings_aspect_suffix_stacked, dim=0).values.tolist())

    examples["embeddings_aspect"] = all_embeddings_aspect
    examples["embeddings_aspect_suffix"] = all_embeddings_aspect_suffix

    return examples


if __name__ == "__main__":
    # Define the path to the model
    if model_path[-1] == "/":
        dataset_subfolder = model_path.split("/")[-2]
    else:
        dataset_subfolder = model_path.split("/")[-1]
    dataset_subfolder

    # Load aspect-based sentiment analysis dataset for laptop domain
    dataset = load_dataset("yqzheng/semeval2014_laptops")

    # Prepare dataset splits
    train_ds = prepare_embeddings_dataset_for_absa_laptop_2014(
        dataset["train"],
        model_path,
        pre_aspect_suffix=suffix,
    )
    test_ds = prepare_embeddings_dataset_for_absa_laptop_2014(
        dataset["test"],
        model_path,
        pre_aspect_suffix=suffix,
    )

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    # Create a feature extraction pipeline
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=device, return_tensors=True)

    # Get aspect features for the train and test datasets
    train_ds = train_ds.map(lambda x: get_aspect_features_with_suffix(x, pipe), batched=True, batch_size=16)
    test_ds = test_ds.map(lambda x: get_aspect_features_with_suffix(x, pipe), batched=True, batch_size=16)

    # Save the prepared datasets to disk
    train_ds.save_to_disk(f"prepared_datasets/{dataset_subfolder}/train_ds")
    test_ds.save_to_disk(f"prepared_datasets/{dataset_subfolder}/test_ds")
