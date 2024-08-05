## Fine-tuning XLM-RoBERTa Models of Different Sizes on One or Multiple GPUs for Aspect-based Sentiment Analysis (ABSA)

This article provides a detailed tutorial on how to fine-tune differently sized [XLM-RoBERTa](https://arxiv.org/abs/1911.02116v2)-based language models for the specific task of aspect-based sentiment analysis (ABSA) using the approach proposed by [Gao et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8864964) originally for the [BERT](https://arxiv.org/abs/1810.04805?amp=1) model. The implementation uses the HuggingFace libraries [transformers](https://huggingface.co/docs/transformers) and [accelerate](https://huggingface.co/docs/accelerate/index). We provide a full executable code example for [fine-tuning an ABSA model](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/finetune_absa.py) and subsequent [evaluation](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/use_absa.py) of the trained model. Please install necesary [requirements](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/requirements.txt) for this tutorial first.

```
transformers
tokenizers
datasets
evaluate
numpy
protobuf==3.20.2
sentencepiece==0.1.99
```

### Model Architecture

The model architecture introduced by Gao et al. in the paper *[Target-Dependent Sentiment Classification With BERT](https://ieeexplore.ieee.org/abstract/document/8864964)* is shown in the figure below. For classification, the model embeddings coresponding to an aspect term, e.g. *battery timer*, are extracted from the models output. Max or mean pooling is then used to summarize multiple embeddings to one vector. Upon this vector a fully connected layer conducts the final classification into three classes.

[![Model architecture according to Gao et al. (2019)](td-bert-gao-2019.png "Model architecture according to Gao et al. (2019)")](https://ieeexplore.ieee.org/abstract/document/8864964)

### Preparing a Dataset

As a working example, we take the [Laptop 2014 dataset](https://huggingface.co/datasets/yqzheng/semeval2014_laptops) from the [SemEval 2014 Task 4](https://paperswithcode.com/dataset/semeval-2014-task-4-sub-task-2) for aspect-based sentiment analysis. This dataset provides texts with aspect positions, aspects and polarities per aspect. Similarly, a dataset for multi-target stance detection can be used, e.g. [stance towards German politicians and political parties](https://github.com/UnibwSparta/German-Elections-NRW22-Stance-Dataset).

```python
from datasets import load_dataset

dataset = load_dataset("yqzheng/semeval2014_laptops")
ds_train = dataset["train"]
ds_test = dataset["test"]
```

To be able extracting correct embeddings from the model, we need to map aspect words to model tokens. This can be achieved by HuggingFace's function [BatchEncoding.char_to_token()](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.char_to_token) that will return coresponding token positions for every character of the original input sentence. First, we use model's tokenizer to create `BatchEncoding` objects for all dataset items. In this case, maximal length of 100 model tokens is enough to fit any text from the given dataset.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
batch_encoding = tokenizer(ds_train["text"], padding="max_length", max_length=100, truncation=True)
```

Second, we use [BatchEncoding.char_to_token()](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.char_to_token) to create a function `get_aspect_mask()` for extracting all aspect tokens using start and end positions for the aspect withing original text. Using token positions we build boolean aspect mask for model input. Boolean masks are used later in the model to extract aspect embeddings.

```python
from typing import List, Set, Tuple
from transformers import BatchEncoding

def get_aspect_mask(
    item_encoding: BatchEncoding,
    start: int,
    end: int,
    max_len_tokens: int,
) -> Tuple[Set[int], List[bool]]:
    token_indices = set()
    for i in range(start, end):
        token = item_encoding.char_to_token(i)
        if token is not None:
            token_indices.add(token)

    tokens_mask = [i in token_indices for i in range(max_len_tokens)]

    if len(token_indices) < 1:
        raise ValueError("Empty aspect mask")

    return token_indices, tokens_mask
```

In aftermath, an additional function `get_all_aspect_masks()` extract aspect masks from the complete dataset. Please refer to the [full code for this function](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/src/sparta/absa/aspects.py#L55-L106).

```python
_, aspect_masks = get_all_aspect_masks(
    batch_encoding=tokenized_as_batch_encoding,
    start_positions=ds["start"],
    end_positions=ds["end"]
)
```

Tokenized dataset and aspect masks need to be added to the original dataset object. Also, we rename the colung `label` to `labels` to fit the input of our model later.

```python
# Add tokenized inputs and aspect masks to the original dataset
ds_train = ds_train.add_column("input_ids", batch_encoding["input_ids"])
ds_train = ds_train.add_column("attention_mask", batch_encoding["attention_mask"])
ds_train = ds_train.add_column("aspect_mask", aspect_masks)

# Rename the label column to labels to match the model's requirements
ds_train = ds_train.rename_column("label", "labels")
```

Finally, we need to fix the labels of Laptop 2014 dataset to fit classifier output expectations. Labels in the dataset are `[-1, 0, 1]` and need to be `[0, 1, 2]`. Thus, need to be incremented.

```python
ds_train = ds_train.map(lambda example: example["labels"] = example["labels"] + 1)
```

We repeat the steps conducted for `ds_train` dataset split for `ds_test` as well. Please refer to the [complete code example for dataset split preparation](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/src/sparta/absa/aspects.py#L9-L36).

### Create a Model

Unfortunately, the HuggingFace library does not provide a dedicated model for aspect-based sentiment analysis. Thus, we need to derive a class for an XLM-RoBERTa-based model and adapt it:

* For `xlm-roberta-base` and `-large` - [XLMRobertaForSequenceClassification](https://huggingface.co/docs/transformers/v4.43.4/en/model_doc/xlm-roberta#transformers.XLMRobertaForSequenceClassification)
* For `facebook/xlm-roberta-xl` and `-xxl` - [XLMRobertaXLForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForSequenceClassification)

Derived model need to provide a `forward()` method that takes all necessary parameters from our dataset: input IDs, attention mask, aspect mask, and labels.

```python
from typing import Optional
import torch

# For base/large
class XLMRobertaForABSA(XLMRobertaForSequenceClassification):

# For xl/xxl
# class XLMRobertaForABSA(XLMRobertaXLForSequenceClassification):

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        aspect_mask: torch.BoolTensor,
        labels: Optional[torch.Tensor] = None,
    ):
        ...
```

The procedure is:

1. to call the underlying XLM-RoBERTa encoder and take the complete intermediate output,
2. to extract aspect embeddings and to max pool them (originally from [Dai et al. (2021)](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/finetune.py#L166-L168)),
3. to apply fully connected classifier,
4. to compute cross entropy loss, if labels are provided.

```python
# (1) Call the RoBERTa encoder to get contextualized embeddings
output_roberta = self.roberta(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=False,
)
intermediate_output = output_roberta[0]

# (2) Get max pooling of tokens for the aspect(s)
fill_mask = aspect_mask.unsqueeze(-1).eq(0)
intermediate_for_aspect = intermediate_output.masked_fill(fill_mask, -10000.0)
preds, _ = intermediate_for_aspect.max(dim=1)

# (3) Apply classifier
preds_reshaped = preds.unsqueeze(1)
logits = self.classifier(preds_reshaped)

# (4) Compute classification loss in case of labels (in training mode)
loss = None
if labels is not None:
    # Move labels to correct device to enable model parallelism
    labels = labels.to(logits.device)
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
```

The `forward()` method will return model predictions as logits and additionally the loss if labels are provided.

```python
output = (logits,)
if loss is not None:
    return (loss,) + output

return output
```

Now, we can instantiate our new model.

```python
model_name = "xlm-roberta-base"
# model_name = "xlm-roberta-large"
# model_name = "facebook/xlm-roberta-xl"
# model_name = "facebook/xlm-roberta-xxl"

# Create an id2label mapping for this specific task
config.id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Load the model with the new configuration
model = XLMRobertaForABSA.from_pretrained(model_name, config=config)
```


### Model Fine-tuning

Thanks to the HuggingsFace's [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class, the fine-tuning procedure is straight forward to implement. The Trainer supports training on single or multiple GPUs. For our use case, we focus on multi-GPU setup and train models with [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) paradigma. To conduct this, the training script need to be started with the [accelerate](https://huggingface.co/docs/accelerate/index) tool. The first step is to create a configuration file:

```bash
accelerate config --config_file accelerate.cfg
```

For FSDP, the most important parameters are number of GPUs to use and the model's transformer layer name. Knowledge about the correct transformer layer is crucial for model splitting and distribution over the GPUs:

* For `xlm-roberta-base` and `-large` - [XLMRobertaLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L384)
* For `facebook/xlm-roberta-xl` and `-xxl` - [XLMRobertaXLLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L368)

A complete FSDP configuration for `accelerate` looks like:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: true
  fsdp_offload_params: true
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: XLMRobertaLayer
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Please refer to the full examles for [base/large](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/accelerate_configs/roberta_base_large.yaml) (1 GPU), [xl](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/accelerate_configs/roberta_xl.yaml) (2 GPUs), and [xxl](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/accelerate_configs/roberta_xxl.yaml) (4 GPUs). The training script need to be started in the following way:

```bash
accelerate launch --config_file accelerate.cfg finetune_absa.py
```

Within this script, we first create training arguments.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="absa_checkpoints",

    # Training hyperparameters
    # Number of epoch can be high since we have early stopping in Trainer
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,

    # Optimizer/scheduler parameters
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
```

The `metric_for_best_model` need to correspond to a computed metric. For this, we create an additional function that can return accuracy and macro F1 score metrics. Please also refer to the [full code](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/src/sparta/absa/metrics.py).

```python
from typing import Dict
import evaluate
import numpy as np
from transformers.trainer_utils import EvalPrediction

metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics_function(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred

    # Take argmax to get the predicted class
    predictions = np.argmax(logits, axis=-1)

    # Compute accuracy and macro F1 score
    metrics = metric_accuracy.compute(predictions=predictions, references=labels)
    metrics.update(metric_f1.compute(predictions=predictions, references=labels, average="macro"))
    return metrics
```

This metrics function is then passed to the `Trainer` class. Additionally, we provide an early stopping criteria that will use metric given in `metric_for_best_model`.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    compute_metrics=compute_metrics_function,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
```

### Fine-tuning Results

We used the [above code](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/finetune_absa.py) to train differently sized XLM-RoBERTa models without dedicated hyper parameter search. Training results for the [Laptop 2014 dataset](https://huggingface.co/datasets/yqzheng/semeval2014_laptops) with the ABSA model are shown in the table below. We conducted three runs for every setup evaluating on the test set and provide result ranges.

| **model** | **macro F1** | **accuracy** | **min\_gpus** | **min\_gpu\_size** |
| :-- | :-- | :-- | :-- | :-- |
| [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 73-76% | 76-80% | 1 | 8 |
| [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) | 77-78% | 80-82% | 1 | 16 |
| [xlm-roberta-xl](https://huggingface.co/facebook/xlm-roberta-xl) | 79-80% | 81-83% | 2 | 32 |
| [xlm-roberta-xxl](https://huggingface.co/facebook/xlm-roberta-xxl) | 81-82% | 83-84% | 4 | 40 |

In comparison, the most recent ensemble approach by [Yang and Li (2024)](https://arxiv.org/pdf/2110.08604) named *LSAE-X-DeBERTa* achieve macro F1 scores over 84% and accuracy over 86%. Provided [PyABSA](https://github.com/yangheng95/PyABSA) package is worth looking at.

### Using a Fine-tuned Model