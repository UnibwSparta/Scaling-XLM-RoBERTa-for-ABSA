from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
    XLMRobertaXLForSequenceClassification,
    DebertaV2ForSequenceClassification,
    ElectraForSequenceClassification,
)


class RobertaForABSA(RobertaForSequenceClassification):
    """Aspect-based sentiment analysis model based on XLM-RoBERTa-base/large for sequence classification."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        aspect_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return absa_forward(self, "roberta", input_ids, attention_mask, aspect_mask, labels)


class XLMRobertaForABSA(XLMRobertaForSequenceClassification):
    """Aspect-based sentiment analysis model based on XLM-RoBERTa-base/large for sequence classification."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        aspect_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return absa_forward(self, "roberta", input_ids, attention_mask, aspect_mask, labels)

class XLMRobertaXLForABSA(XLMRobertaXLForSequenceClassification):
    """Aspect-based sentiment analysis model based on XLM-RoBERTa-XL/XXL for sequence classification."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        aspect_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return absa_forward(self, "roberta", input_ids, attention_mask, aspect_mask, labels)


class DebertaForABSA(DebertaV2ForSequenceClassification):
    """Aspect-based sentiment analysis model based on Deberta-base/large for sequence classification."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        aspect_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return absa_forward(self, "deberta", input_ids, attention_mask, aspect_mask, labels)


class ElectraForABSA(ElectraForSequenceClassification):
    """Aspect-based sentiment analysis model based on electra-base/large for sequence classification."""

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        aspect_mask: torch.BoolTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return absa_forward(self, "electra", input_ids, attention_mask, aspect_mask, labels)


def absa_forward(
    model: Union[XLMRobertaForABSA, XLMRobertaXLForABSA],
    encoder_module_name: str,
    input_ids: torch.LongTensor,
    attention_mask: torch.FloatTensor,
    aspect_mask: torch.BoolTensor,
    labels: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Conduct a forward pass of the model for aspect-based sentiment analysis.

    This is an adaptation from:
        https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/finetune.py#L160-L168

    Args:
        model (Union[XLMRobertaForABSA, XLMRobertaXLForABSA]): XLM RoBERTa model for aspect-based sentiment analysis (base, large, XL, or XXL)
        encoder_module_name (str): Name of the encoder module in the model
        input_ids (torch.LongTensor): Input IDs
        attention_mask (torch.FloatTensor): Attention mask
        aspect_mask (torch.BoolTensor): Aspect mask, indicating which tokens correspond to the aspect
        labels (Optional[torch.LongTensor], optional): Labels. Defaults to None.

    Returns:
        Tuple[torch.Tensor]: Logits (in prediction mode) or loss and logits (in training mode)
    """
    # Get the encoder module from the model
    encoder = getattr(model, encoder_module_name)

    # Call the RoBERTa encoder to get contextualized embeddings
    encoder_output = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    )
    intermediate_output = encoder_output[0]

    # Get max pooling of tokens for all aspects
    fill_mask = aspect_mask.unsqueeze(-1).eq(0)  # bsz x max_len x 1
    intermediate_for_aspect = intermediate_output.masked_fill(fill_mask, -10000.0)
    preds, _ = intermediate_for_aspect.max(dim=1)

    # ALTERNATIVE: Get mean pooling of tokens for all aspects
    # fill_mask = aspect_mask.unsqueeze(-1).eq(0)
    # intermediate_for_aspect = intermediate_output.masked_fill(fill_mask, 0)
    # intermediate_for_aspect = intermediate_for_aspect.sum(dim=1)
    # div_mask = aspect_mask.sum(dim=1, keepdims=True).float()  # type: ignore
    # preds = intermediate_for_aspect / div_mask

    # Add one dimension, because original XLMRobertaClassificationHead expects it to extract only embeddings for the CLS token.
    # In this case it is only mocked, since we provide a different embedding in conjunction with an aspect mask.
    # Transformation here (bsz x 768) -> (bsz x 1 x 768).
    # XLMRobertaClassificationHead will take only first token from each item [:, 0, :] and thus reduce dimensions to (bsz x 768) again.
    preds_reshaped = preds.unsqueeze(1)

    # Apply classifier
    logits = model.classifier(preds_reshaped)

    # Compute classification loss in case of labels (in training mode)
    loss = None
    if labels is not None:
        # Move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))

    # Return the final output of the model logits (in prediction mode) or logits and the loss (in training mode)
    output = (logits,)
    if loss is not None:
        return (loss,) + output

    return output
