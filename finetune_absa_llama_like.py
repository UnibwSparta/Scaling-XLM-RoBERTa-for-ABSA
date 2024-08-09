#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""finetune_absa_llama.py: Fine-tune a Llama-like model for aspect-based sentiment analysis using FSDP.

Before running complete TODO 1 and TODO 2 tasks in the script by uncommenting the model you want to use.

To run this script use accelerate within poetry environment:

    poetry run accelerate launch --config_file accelerate_configs/llama.yaml finetune_absa_llama_like.py
"""

import warnings

from datasets import load_dataset
from transformers import EarlyStoppingCallback, LlamaForSequenceClassification, Trainer, TrainingArguments

from sparta.absa.aspects import prepare_dataset_for_absa_laptop_2014_llama
from sparta.absa.metrics import get_metrics_function

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# TODO 1: Define the path to the model
model_path = "meta-llama/Meta-Llama-3.1-8B"

# TODO 2: Customize the aspect suffix. Note that the target aspect will be added after this suffix.
aspect_suffix = "–This is a review about"


if __name__ == "__main__":

    # Load aspect-based sentiment analysis dataset for laptop domain
    dataset = load_dataset("yqzheng/semeval2014_laptops")

    max_length = 128
    train_ds = prepare_dataset_for_absa_laptop_2014_llama(
        ds=dataset["train"],
        tokenizer_name=model_path,
        aspect_suffix=aspect_suffix,
        max_length=max_length,
    )
    test_ds = prepare_dataset_for_absa_laptop_2014_llama(
        dataset["test"],
        tokenizer_name=model_path,
        aspect_suffix=aspect_suffix,
        max_length=max_length,
    )

    # Create the model
    model = LlamaForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model.config.pad_token_id = model.config.eos_token_id

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
        # learning_rate=1e-4,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Do train
    trainer.train()
