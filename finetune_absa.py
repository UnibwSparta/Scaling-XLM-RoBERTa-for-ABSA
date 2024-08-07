#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""finetune_absa.py: Fine-tune a model for aspect-based sentiment analysis using FSDP.

Before running complete TODO 1 and TODO 2 tasks in the script by uncommenting the model you want to use.

To run this script use accelerate within poetry environment:

    - For RoBERTA-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/roberta.yaml finetune_absa.py

    - For XLM-RoBERTA-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_base_large.yaml finetune_absa.py

    - For XLM-RoBERTA-XL model:
        poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_xl.yaml finetune_absa.py

    - For XLM-RoBERTA-XXL model:
        poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_xxl.yaml finetune_absa.py

   - For (m)DeBerta-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/deberta.yaml finetune_absa.py

   - For ELECTRA-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/electra.yaml finetune_absa.py

   - For ERNIE-base or -large model:
        poetry run accelerate launch --config_file accelerate_configs/ernie.yaml finetune_absa.py
"""

import warnings

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from sparta.absa.aspects import prepare_dataset_for_absa_laptop_2014
from sparta.absa.metrics import get_metrics_function

# TODO 1: Uncomment the model you want to use:
# - RobertaForABSA is the roberta-base or -large model
# - XLMRobertaForABSA is the xlm-roberta-base or -large model
# - XLMRobertaXLForABSA is the xlm-roberta-xl or -xxl model
# - DebertaForABSA is the (m)deberta-v3-base or -large model
# - ElectraForABSA is the electra-base or -large model
# - ErnieForABSA is the ernie-2.0-base or -large model
# from sparta.absa.models import RobertaForABSA as ModelForABSA
# from sparta.absa.models import XLMRobertaForABSA as ModelForABSA
# from sparta.absa.models import XLMRobertaXLForABSA as ModelForABSA
from sparta.absa.models import DebertaForABSA as ModelForABSA
# from sparta.absa.models import ElectraForABSA as ModelForABSA
# from sparta.absa.models import ErnieForABSA as ModelForABSA

# TODO 2: Ucomment the model you want to use
model_name = "roberta-base"
# model_name = "roberta-large"
# model_name = "xlm-roberta-base"
# model_name = "xlm-roberta-large"
# model_name = "facebook/xlm-roberta-xl"
# model_name = "facebook/xlm-roberta-xxl"
# model_name = "microsoft/mdeberta-v3-base"
# model_name = "microsoft/deberta-v3-base"
# model_name = "microsoft/deberta-v3-large"
# model_name = "microsoft/deberta-v2-xlarge"
model_name = "microsoft/deberta-v2-xxlarge"
# model_name =  "google/electra-base-discriminator"
# model_name =  "google/electra-large-discriminator"
# model_name =  "nghuyong/ernie-2.0-base-en"
# model_name =  "nghuyong/ernie-2.0-large-en"

# NOTE: It makes sense to first download the large models to a local path and then use it
# model_name = "/path_to_local/xlm-roberta-xl/"
# model_name = "/path_to_local/xlm-roberta-xxl/"


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


if __name__ == "__main__":

    # Load aspect-based sentiment analysis dataset for laptop domain
    dataset = load_dataset("yqzheng/semeval2014_laptops")

    # Prepare dataset splits
    train_ds = prepare_dataset_for_absa_laptop_2014(dataset["train"], model_name)
    test_ds = prepare_dataset_for_absa_laptop_2014(dataset["test"], model_name)

    # Update the model's configuration with the id2label mapping
    config = AutoConfig.from_pretrained(model_name)

    # Create an id2label mapping for this specific task
    config.id2label = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    # Load the model with the new configuration
    model = ModelForABSA.from_pretrained(model_name, config=config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="absa_checkpoints",

        # Training hyperparameters
        # Number of epoch can be high since we have early stopping in Trainer
        num_train_epochs=16,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,

        # Optimizer/scheduler parameters
        # learning_rate=5e-5,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=100,

        # Logging/saving parameters
        # For best model selection and early stopping you need to store at least one checkpoint
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        # Best model selection
        # We can use eval_loss, eval_accuracy, or eval_f1 for best model selection and as early stopping metric
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # metric_for_best_model="eval_accuracy",
        # metric_for_best_model="eval_f1",
    )

    # Init a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=get_metrics_function(),

        # Early stopping
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Do train
    trainer.train()

    # OPTIONAL: Besides automatic storing of model checkpoints by Trainer, you can also store the model manually at the end.
    #           Checkpoint contains model weights, optimizer and scheduler states, but not a tokenizer.
    #           Checkpoints are useful to resume training or to fine-tune the model later.
    #           For production you need only weights and optionally a tokenizer.
    #           You can still use the original tokenizer, of you don't store it, but it makes things easier later.

    # Save model weights w/o optimizer and scheduler states and tokenizer
    store_path = f"absa_models/{model_name}"
    trainer.save_model(store_path)

    # Additionally save the tokenizer at the same location
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(store_path)
