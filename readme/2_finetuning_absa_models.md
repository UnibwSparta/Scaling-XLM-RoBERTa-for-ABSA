## Fine-tuning BERT-like Models of Different Sizes on Multiple GPUs for Aspect-based Sentiment Analysis (ABSA)

This article provides a detailed tutorial on how to fine-tune differently sized BERT-like language models, such as [XLM-RoBERTa](https://arxiv.org/abs/1911.02116v2) or [(m)DebertaV3](https://arxiv.org/pdf/2111.09543v4), for the specific task of aspect-based sentiment analysis (ABSA) using the approach proposed by [Gao et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8864964) originally for the [BERT](https://arxiv.org/abs/1810.04805?amp=1) model. The implementation uses the HuggingFace's frameworks [transformers](https://huggingface.co/docs/transformers) and [accelerate](https://huggingface.co/docs/accelerate/index). We provide full executable code examples for [fine-tuning an ABSA model](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/finetune_absa_bert_like.py) and subsequent [evaluation](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/use_absa_bert_like.py) of the trained model. Please install the necessary [requirements](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/requirements.txt) for this tutorial first.

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

The model architecture introduced by Gao et al. in the paper *[Target-Dependent Sentiment Classification With BERT](https://ieeexplore.ieee.org/abstract/document/8864964)* is shown in the figure below. For classification, the model embeddings corresponding to an aspect term, e.g., *battery timer*, are extracted from the model's output. Max or mean pooling is then used to summarize multiple embeddings to a single vector. A fully connected layer performs the final classification into three classes based on this vector. We can substitute the BERT model by any BERT-like model, e.g., [XLM-RoBERTA](https://arxiv.org/abs/1911.02116v2), [Deberta](https://arxiv.org/pdf/2111.09543v4), [ELECTRA](https://arxiv.org/pdf/2003.10555v1), [ERNIE](https://arxiv.org/abs/2107.02137), etc.

[![Model architecture according to Gao et al. (2019)](td-bert-gao-2019.png "Model architecture according to Gao et al. (2019)")](https://ieeexplore.ieee.org/abstract/document/8864964)

*Image source: [Gao et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8864964)*

### Preparing a Dataset

As a working example, we take the [Laptop 2014 dataset](https://huggingface.co/datasets/yqzheng/semeval2014_laptops) from the [SemEval 2014 Task 4](https://paperswithcode.com/dataset/semeval-2014-task-4-sub-task-2) for aspect-based sentiment analysis. This dataset provides texts with aspect positions, aspects, and polarities for each aspect. Similarly, a dataset for multi-target stance detection copuld be used here instead, such as the [stance towards German politicians and political parties](https://github.com/UnibwSparta/German-Elections-NRW22-Stance-Dataset).

```python
from datasets import load_dataset

dataset = load_dataset("yqzheng/semeval2014_laptops")
ds_train = dataset["train"]
ds_test = dataset["test"]
```

To correctly extract embeddings from the model, we need to map aspect words to model tokens. This can be achieved using HuggingFace's function [BatchEncoding.char_to_token()](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.char_to_token), which returns corresponding token positions for each character of the original input text. First, we use model's tokenizer to create `BatchEncoding` objects for all dataset items. In this case, a maximum length of 100 model tokens is sufficient to accomodate any text from the given dataset.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
batch_encoding = tokenizer(ds_train["text"], padding="max_length", max_length=100, truncation=True)
```

Second, we use [BatchEncoding.char_to_token()](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.char_to_token) to create a function `get_aspect_mask()` for extracting all aspect tokens using the start and end positions for the aspect within the original text. From resulting token positions, we build a boolean aspect mask for model input. Boolean masks are later used in the model to extract aspect embeddings.

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

Subsequently, an additional function `get_all_aspect_masks()` extracts aspect masks from the complete dataset. Please refer to the [full code for this function](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/src/sparta/absa/aspects.py#L60-L80).

```python
_, aspect_masks = get_all_aspect_masks(
    batch_encoding=tokenized_as_batch_encoding,
    start_positions=ds["start"],
    end_positions=ds["end"]
)
```

Tokenized items (input IDs, attention masks, and aspect masks) need to be added to the original dataset object. Also, we rename the column `label` to `labels` for fitting the input structure requirements of our model later.

```python
# Add tokenized inputs and aspect masks to the original dataset
ds_train = ds_train.add_column("input_ids", batch_encoding["input_ids"])
ds_train = ds_train.add_column("attention_mask", batch_encoding["attention_mask"])
ds_train = ds_train.add_column("aspect_mask", aspect_masks)

# Rename the label column to labels to match the model's requirements
ds_train = ds_train.rename_column("label", "labels")
```

Finally, we need to adjust the labels in the [Laptop 2014 dataset](https://huggingface.co/datasets/yqzheng/semeval2014_laptops) to match classifier's output expectations. Labels in the dataset are `[-1, 0, 1]` and need to be `[0, 1, 2]`. Thus, a simple incrementation fixes this issue.

```python
ds_train = ds_train.map(lambda example: example["labels"] = example["labels"] + 1)
```

After completing these steps, items in the training dataset look as follows:

```python
{
    'text': 'I charge it at night and skip taking the cord with me because of the good battery life.',
    'aspect': 'cord',
    'start': 41,
    'end': 45,
    'labels': 1,
    'input_ids': [0, ...],
    'attention_mask': [1, ...],
    'aspect_mask': [False, ...]
}
```

We repeat the steps performed for the `ds_train` dataset split for the `ds_test` as well. Please refer to the [complete code example for dataset split preparation](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/src/sparta/absa/aspects.py#L10-L36).

### Create a Model

Unfortunately, the HuggingFace framework does not provide a dedicated model for aspect-based sentiment analysis. Therefore, we need to derive from an existing class for sequence classification model and adapt it:

* For `roberta-base` and `-large` - [RobertaForSequenceClassification](https://huggingface.co/docs/transformers/v4.43.4/en/model_doc/roberta#transformers.RobertaForSequenceClassification)
* For `xlm-roberta-base` and `-large` - [XLMRobertaForSequenceClassification](https://huggingface.co/docs/transformers/v4.43.4/en/model_doc/xlm-roberta#transformers.XLMRobertaForSequenceClassification)
* For `facebook/xlm-roberta-xl` and `-xxl` - [XLMRobertaXLForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/xlm-roberta-xl#transformers.XLMRobertaXLForSequenceClassification)
* For `microsoft/(m)deberta-v3-base` and `-large` - [DebertaV2ForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/deberta-v2#transformers.DebertaV2ForSequenceClassification)

The derived model needs to implement the `forward()` method, which takes all necessary parameters from our dataset, i.e., input IDs, attention masks, aspect masks, and labels.

```python
from typing import Optional

import torch
from transformers import (
    RobertaForSequenceClassification,
    # XLMRobertaForSequenceClassification,
    # XLMRobertaXLForSequenceClassification,
    # DebertaV2ForSequenceClassification,
)

# For roberta-base/large
class ModelForABSA(RobertaForSequenceClassification):

# For xlm-roberta-base/large
# class ModelForABSA(XLMRobertaForSequenceClassification):

# For xlm-roebrta-xl/xxl
# class ModelForABSA(XLMRobertaXLForSequenceClassification):

# For (m)deberta-v3-base/large
# class ModelForABSA(DebertaV2ForSequenceClassification):

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        aspect_mask: torch.BoolTensor,
        labels: Optional[torch.Tensor] = None,
    ):
        ...
```

The procedure of the forward pass is as follows:

1. Call the underlying encoder and obtain the complete intermediate output,
2. Extract aspect embeddings and apply max-pooling (see code lines originally from [Dai et al. (2021)](https://github.com/ROGERDJQ/RoBERTaABSA/blob/main/Train/finetune.py#L166-L168)),
3. Apply fully connected classifier,
4. Compute cross entropy loss, if labels are provided.

```python
# (1) Call the RoBERTa encoder to get contextualized embeddings
encoder_module_name = "roberta"    # for RoBERTa models
# encoder_module_name = "deberta"  # for Deberta models

encoder = getattr(self, encoder_module_name)
encoder_output = encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=False,
)
intermediate_output = encoder_output[0]

# (2) Get max pooling of the aspect embeddings
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

The `forward()` method will return model predictions as logits and, if labels are provided, also return the loss.

```python
output = (logits,)
if loss is not None:
    return (loss,) + output

return output
```

Now, we can instantiate our new model. Select a model that matches previously selected architecture.

```python
model_name = "roberta-base"
# model_name = "roberta-large"
# model_name = "xlm-roberta-base"
# model_name = "xlm-roberta-large"
# model_name = "facebook/xlm-roberta-xl"
# model_name = "facebook/xlm-roberta-xxl"
# model_name = "microsoft/mdeberta-v3-base"
# model_name = "microsoft/deberta-v3-base"
# model_name = "microsoft/deberta-v3-large"

# Create an id2label mapping for this specific task
config.id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Load the model with the new configuration
model = ModelForABSA.from_pretrained(model_name, config=config)
```


### Model Fine-tuning

Thanks to the HuggingsFace's [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class, the fine-tuning procedure is straightforward to implement. The Trainer supports training on single or multiple GPUs. For our use case, we focus on multi-GPU setup and train models using the [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) paradigm. In this case, the training script must be started with the [accelerate](https://huggingface.co/docs/accelerate/index) tool. The first step is to create a configuration file by running a command:

```bash
accelerate config --config_file accelerate.cfg
```

For our FSDP configuration, the most important parameters are the number of GPUs to use (`num_processes`) and the model's transformer layer name (`fsdp_transformer_layer_cls_to_wrap`). Knowledge about the correct transformer layer is crucial for model splitting and distribution over the GPUs:

* For `roberta-base` and `-large` - [RobertaLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L383)
* For `xlm-roberta-base` and `-large` - [XLMRobertaLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L384)
* For `facebook/xlm-roberta-xl` and `-xxl` - [XLMRobertaXLLayer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L368)
* For `microsoft/(m)deberta-base` and `-large` - [DebertaV2Layer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py#L368)

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
  fsdp_transformer_layer_cls_to_wrap: RobertaLayer
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

Please refer to the full [configuration examles for different models](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/accelerate_configs/). The training script must be started in the following way:

```bash
accelerate launch --config_file accelerate.cfg finetune_absa_bert_like.py
```

Within the `finetune_absa_bert_like.py` script, we first create training arguments for the Trainer. These include different hyperparameters and settings for Trainer behavior, such as how many model checkpoints to save and how to evaluate the best model.

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

The `metric_for_best_model` corresponds to the computed metric of model performance during training. Evauation loss `eval_loss` is returned directly by the model. To compute other metrics, such as `eval_accuracy` and `eval_f1`, we write an additional function that returns accuracy and macro F1 score metrics. Please refer also to the [full code](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/src/sparta/absa/metrics.py).

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

The metrics function is then passed to the Trainer class as an argument. Additionally, we provide an early stopping criteria that will use metric name given in `metric_for_best_model` to stop the training procedure. The number of epochs—passed in training arguments—can be set high, because early stopping will terminate the training when appropriate.

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

The Trainer will save the model checkpoints during the training to the `absa_checkpoints` directory. These include model weights, optimizer, and scheduler states, but not the tokenizer. Checkpoints are good for re-initializing or continuing the training procedure. For production use, only model weights and tokenizer are important. It makes sense to store these two artefacts in the same location for later use after the training has finished:

```python
from transformers import AutoTokenizer

store_path = f"absa_models/{model_name}"
trainer.save_model(store_path)

# Additionally save a cpoy of the original tokenizer at the same location
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(store_path)
```

Please refer to the [full code of `finetune_absa_bert_like.py` script](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/finetune_absa_bert_like.py).

### Fine-tuning Results

We used the [above code](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/finetune_absa_bert_like.py) to train differently sized (XLM-)RoBERTa and (m)Deberta v2/v3 models without dedicated hyperparameter search. The training results for the [Laptop 2014 dataset](https://huggingface.co/datasets/yqzheng/semeval2014_laptops) with the described ABSA model are shown in the table below. We conducted three runs for each setup, evaluating on the test set, and provide rounded result ranges to examplarily showcase the effects of scaling, multilinguality, and model architecture.

| **roberta model** | **language** | **dimension** | **macro F1** | **accuracy** | **min\_gpus** | **min\_gpu\_size** |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [roberta-base](https://huggingface.co/FacebookAI/roberta-base) | English | base | 78-80% | 81-83% | 1 | 8 |
| [roberta-large](https://huggingface.co/FacebookAI/roberta-large) | English | large | 80-81% | 83-84% | 1 | 16 |
| [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | multilingual | base | 73-76% | 76-80% | 1 | 8 |
| [xlm-roberta-large](https://huggingface.co/FacebookAI/xlm-roberta-large) | multilingual | large | 77-78% | 80-82% | 1 | 16 |
| [xlm-roberta-xl](https://huggingface.co/facebook/xlm-roberta-xl) | multilingual | xl | 79-80% | 81-83% | 2 | 32 |
| [xlm-roberta-xxl](https://huggingface.co/facebook/xlm-roberta-xxl) | multilingual | xxl | 81-82% | 83-84% | 4 | 40 |

| **deberta model** |  **language** | **dimension** | **macro F1** | **accuracy** | **min\_gpus** | **min\_gpu\_size** |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | English | base | 77-80% | 81-83% | 1 | 8 |
| [deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) | English | large | 81-82% | 84-85% | 1 | 16 |
| [deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge) | English | xl | 81-83% | 84-85% | 1 | 24 |
| [deberta-v2-xxlarge](https://huggingface.co/microsoft/deberta-v2-xxlarge) | English | xxl | 81-83% | 85-86% | 1 | 40 |
| [mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base) |  multilingual | base | 72-76% | 77-80% | 1 | 8 |


The results show that:

* **Scaling improves the performance on this task slowly**, e.g., compare the results for `xlm-roberta-base`, `-large`, `-xl`, and `-xxl`. Note that `deberta-v2-xlarge` and `deberta-v2-xxlarge` are a way smaller compared to similarly named `xlm-roberta` models. This difference is evident in the GPU requirements.
* **Monolingual English models generally outperform multilingual models**, e.g., compare `roberta-base` vs. `xlm-roberta-base`, or `deberta-v3-base` vs. `mdeberta-v-3-base`. However, multilingual models are sometimes the only choice for other languages. It is disappointing that multilingual very large `xlm-roberta-xxl` model achieves results similar to the much smaller English model `roberta-large`. Nonetheless, a better hyperparameter search might improve the performance gap.
* **The `deberta-v3-large` model performs slightly better than equally sized `roberta-large` on this dataset**. In general, [Deberta models are known to outperform RoBERTa models](https://github.com/microsoft/DeBERTa?tab=readme-ov-file#fine-tuning-on-nlu-tasks) by 2-3 percent points of macro F1 score. A more recent ensemble approach of three `deberta-v3-large`-based models by [Yang and Li (2021)](https://arxiv.org/pdf/2110.08604), named *LSA<sub>E</sub>-X-DeBERTa*, achieves macro F1 scores over 84% and accuracy over 86% on the same dataset. Provided [PyABSA](https://github.com/yangheng95/PyABSA) package is worth exploring.

### Using the Fine-tuned Model

In contrast to the training procedure, the [Evaluator](https://huggingface.co/docs/evaluate/package_reference/evaluator_classes#evaluator) class from the HuggingFace's `transformers` framework cannot be used in the same way we used the Trainer class. The Trainer class handled the FSDP appoach for model distribution over multiple GPUs automatically. For evaluation, this needs to be implemented manually using the functionality of `accelerate` framework. Therefore, we create a script `use_absa_bert_like.py` that we will run later with the `accelerate` tool.

```
accelerate launch --config_file accelerate.cfg use_absa_bert_like.py
```

In the script, we first load and prepare our fine-tuned model using the `Accelerator` object.

```python
from accelerate import Accelerator

# Load the model
model = ModelForABSA.from_pretrained(model_path)

# Prepare the model for distributed usage
accelerator = Accelerator()
model = accelerator.prepare(model)
```

For model evaluation, we use the test set from the [Laptop 2014 dataset](https://huggingface.co/datasets/yqzheng/semeval2014_laptops). First, the test dataset needs to be prepared in the same way we did for training. We use the [preparation function](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/src/sparta/absa/aspects.py#L10-L36) `prepare_dataset_for_absa_laptop_2014` elaborated in the previous section of this tutorial. Only a small part of dataset is needed for a simple test, e.g., 16 items–a batch. However, be aware that the number of items should be a multiple of the number of GPUs used. Otherwise, the batch preparation procedure will fill up the missing items with the first items in the batch, and you will need to manually remove them from predictions afterwards.

```python
from datasets import load_dataset

# Load aspect-based sentiment analysis dataset for laptop domain
dataset = load_dataset("yqzheng/semeval2014_laptops")

# Prepare the test dataset split
data = prepare_dataset_for_absa_laptop_2014(dataset["test"], model_path)

# Get the first N items from the dataset to keep it simple
batch_size = 16
batch = data[:batch_size]
```

Next, the batch is prepared for distributed usage on multiple GPUs. It is split into equal parts—one part per GPU. The prediction is then conducted in different processes on multiple GPUs. Afterward, we need to collect the predictions from all GPUs to comibine them into a single vector.

```python
# Prepare the batch for distributed prediction
batch_per_device = accelerator.prepare(batch)

# Run the predictions on different GPUs
with torch.no_grad():
    ouput_per_device = model(
        input_ids=torch.tensor(batch_per_device["input_ids"]),
        attention_mask=torch.tensor(batch_per_device["attention_mask"]),
        aspect_mask=torch.tensor(batch_per_device["aspect_mask"]),
    )
    logits_per_device = ouput_per_device[0]

# Gather logits over all processes
logits = accelerator.gather(logits_per_device)
```

After gathering logits from all GPUs, all parallel processes will have the same state. From the collected logits, we derive probabilities and predict classes.

```python
# Apply softmax to get probabilities
sm = torch.nn.Softmax(dim=1)
probs = sm(logits)

# Apply argmax to get the predicted classes
predicted_class_ids = probs.argmax(dim=1).tolist()

# Get the class labels
predicted_class_labels = [model.config.id2label[class_id] for class_id in predicted_class_ids]
```

Finally, we should clean up the GPUs by removing data tensors.

```python
del batch_per_device["input_ids"]
del batch_per_device["attention_mask"]
del batch_per_device["aspect_mask"]
del batch_per_device
torch.cuda.empty_cache()
```

Now, you may print the prediction results from the main process.

```python
for text, aspect, true_label, pred_label in zip(batch["text"], batch["aspect"], batch["labels"], pred_class_labels):
    accelerator.print(f"Text: {text}")
    accelerator.print(f"True label: {model.config.id2label[true_label]}")
    accelerator.print(f"Predicted label: {pred_label}")
    accelerator.print()
```

Please refer to the [full code example of `use_absa_bert_like.py` script](https://github.com/UnibwSparta/Scaling-XLM-RoBERTa-for-ABSA/blob/main/use_absa_bert_like.py).