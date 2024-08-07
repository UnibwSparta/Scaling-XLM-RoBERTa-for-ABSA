#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""use_absa.py: Use a model to predict sentiments for aspect-based sentiment analysis using FSDP.

Before running complete TODO 1 and TODO 2 tasks in the script by uncommenting the model you want to use.

To run this script use accelerate within poetry environment:

    - For RoBERTA-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/roberta.yaml use_absa.py

    - For XLM-RoBERTA-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_base_large.yaml use_absa.py

    - For XLM-RoBERTA-XL model:
        poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_xl.yaml use_absa.py

    - For XLM-RoBERTA-XXL model:
        poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_xxl.yaml use_absa.py

   - For (m)DeBerta-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/deberta.yaml use_absa.py

   - For ELECTRA-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/electra.yaml use_absa.py

   - For ERNIE-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/ernie.yaml use_absa.py
"""

from typing import Dict, List
import warnings

import torch
from accelerate import Accelerator
from datasets import load_dataset

from sparta.absa.aspects import prepare_dataset_for_absa_laptop_2014


# TODO 1: Uncomment the model you want to use:
# - RobertaForABSA is the roberta-base or -large model
# - XLMRobertaForABSA is the xlm-roberta-base or -large model
# - XLMRobertaXLForABSA is the xlm-roberta-xl or -xxl model
# - DebertaForABSA is the (m)deberta-v3-base or -large model
# - ElectraForABSA is the electra-base or -large model
# - ErnieForABSA is the ernie-2.0-base or -large model
from sparta.absa.models import RobertaForABSA as ModelForABSA
# from sparta.absa.models import XLMRobertaForABSA as ModelForABSA
# from sparta.absa.models import XLMRobertaXLForABSA as ModelForABSA
# from sparta.absa.models import DebertaForABSA as ModelForABSA
# from sparta.absa.models import ElectraForABSA as ModelForABSA
# from sparta.absa.models import ErnieForABSA as ModelForABSA

# TODO 2: Ucomment the model you want to use. You need to train the model first. See: finetune_absa.py
model_path = "absa_models/roberta-base"
# model_path = "absa_models/roberta-large"
# model_path = "absa_models/xlm-roberta-base"
# model_path = "absa_models/xlm-roberta-large"
# model_path = "absa_models/facebook/xlm-roberta-xl"
# model_path = "absa_models/facebook/xlm-roberta-xxl"
# model_path = "absa_models/facebook/mdeberta-v3-base"
# model_path = "absa_models/facebook/deberta-v3-base"
# model_path = "absa_models/facebook/deberta-v3-large"
# model_path = "absa_models/google/electra-base-discriminator"
# model_path = "absa_models/google/electra-large-discriminator"
# model_path = "absa_models/nghuyong/ernie-2.0-base-en"
# model_path = "absa_models/nghuyong/ernie-2.0-large-en"


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


def predict_batch(
    accelerator: Accelerator,
    model: ModelForABSA,
    batch: Dict[str, torch.Tensor],
) -> List[str]:
    """Predict the class labels for a batch of examples for aspect-based sentiment analysis.

    Args:
        accelerator (Accelerator): Accelerator object for distributed execution
        model (XLMRobertaForABSA): XLM-RoBERTa model for aspect-based sentiment analysis
        batch (Dict[str, torch.Tensor]): Batch as a dictionary of tensors

    Returns:
        List[str]: List of class labels
    """
    batch_per_device = accelerator.prepare(batch)
    with torch.no_grad():
        ouput_per_device = model(
            input_ids=torch.tensor(batch_per_device["input_ids"]),
            attention_mask=torch.tensor(batch_per_device["attention_mask"]),
            aspect_mask=torch.tensor(batch_per_device["aspect_mask"]),
        )
        logits_per_device = ouput_per_device[0]

    # Gather logits over all processes
    logits = accelerator.gather(logits_per_device)

    # Apply softmax to get probabilities
    sm = torch.nn.Softmax(dim=1)
    probs = sm(logits)

    # Apply argmax to get the predicted classes
    predicted_class_ids = probs.argmax(dim=1).tolist()

    # Get the class labels
    predicted_class_labels = [model.config.id2label[class_id] for class_id in predicted_class_ids]

    # Clean up GPU
    del batch_per_device["input_ids"]
    del batch_per_device["attention_mask"]
    del batch_per_device["aspect_mask"]
    del batch_per_device
    torch.cuda.empty_cache()

    return predicted_class_labels


if __name__ == "__main__":

    # Load aspect-based sentiment analysis dataset for laptop domain
    dataset = load_dataset("yqzheng/semeval2014_laptops")

    # Prepare the test dataset split
    data = prepare_dataset_for_absa_laptop_2014(dataset["test"], model_path)

    # Get the first N items from the dataset to keep it simple
    # IMPORTANT: N should be a multiple of the number of GPUs used;
    #            otherwise, accelerator.prepare() on batch will fill up
    #            the missing items with the first items in the batch,
    #            and you will need to manually remove them from predictions afterwards.
    batch_size = 16
    batch = data[:batch_size]

    # Load the model
    model = ModelForABSA.from_pretrained(model_path)

    # Prepare the model for distributed usage
    accelerator = Accelerator()
    model = accelerator.prepare(model)

    # Get predictions
    pred_labels = predict_batch(accelerator, model, batch)

    # Truncate the predictions to the original batch size in case that N was not multiple of the number of GPUs
    pred_labels = pred_labels[:batch_size]

    # Print your predictions by the main process
    for text, aspect, true_label, pred_label in zip(batch["text"], batch["aspect"], batch["labels"], pred_labels):
        accelerator.print(f"Text: {text}")
        accelerator.print(f"True label: {model.config.id2label[true_label]}")
        accelerator.print(f"Predicted label: {pred_label}")
        accelerator.print()
