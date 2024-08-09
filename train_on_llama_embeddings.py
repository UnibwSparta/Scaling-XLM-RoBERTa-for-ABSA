#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""train_on_llama_embeddings.py: Fine-tune a simple classifier using embeddings extracted from a Llama-like model."""

import warnings
from typing import Optional, Tuple, Union

import torch
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from sparta.absa.metrics import get_metrics_function

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# Create a HuggingFace models from scratch
# The model is a simple feedforward neural network
# Its forward pass is defined in the forward method
# The forward pass takes the embeddings of the aspect and the aspect with suffix as input and returns the sentiment prediction
class SentimentPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super(SentimentPredictor, self).__init__()
        self.num_labels = 3
        self.fc1 = torch.nn.Linear(8192, 4096)
        self.fc2 = torch.nn.Linear(4096, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 3)

    def forward(
        self,
        embeddings_aspect: torch.Tensor,
        embeddings_aspect_suffix: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.cat([embeddings_aspect, embeddings_aspect_suffix], dim=1)
        x = self.fc1(x)
        # Add a dropout layer
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.1)
        x = self.fc3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, p=0.1)
        x = self.fc4(x)
        x = torch.nn.functional.relu(x)
        logits = x

        loss = None
        if labels is not None:
            # Move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Return the final output of the model logits (in prediction mode) or logits and the loss (in training mode)
        output = (logits,)
        if loss is not None:
            return (loss,) + output

        return output


if __name__ == "__main__":
    model_short_name = "Meta-Llama-3.1-8B"

    # Load the prepared datasets from disk
    train_ds_loaded = load_from_disk(f"prepared_datasets/{model_short_name}/train_ds")
    test_ds_loaded = load_from_disk(f"prepared_datasets/{model_short_name}/test_ds")

    # Create the model
    model_head = SentimentPredictor()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="absa_checkpoints",

        # Training hyperparameters
        # Number of epoch can be high since we have early stopping in Trainer
        num_train_epochs=100,
        per_device_train_batch_size=160,
        per_device_eval_batch_size=160,
        gradient_accumulation_steps=1,

        # Optimizer/scheduler parameters
        learning_rate=5e-5,
        # learning_rate=1e-5,
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
        model=model_head,
        args=training_args,
        train_dataset=train_ds_loaded,
        eval_dataset=test_ds_loaded,
        compute_metrics=get_metrics_function(),

        # Early stopping
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Do train
    trainer.train()
