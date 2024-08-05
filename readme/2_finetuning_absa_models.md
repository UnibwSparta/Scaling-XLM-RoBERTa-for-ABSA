## Fine-tuning XLM-RoBERTa Models of Different Sizes on One or Multiple GPUs for Aspect-based Sentiment Analysis (ABSA)

This article provides a detailed tutorial on how to fine-tune an [XLM-RoBERTa](https://arxiv.org/abs/1911.02116v2)-based language model for the specific task of aspect-based sentiment analysis (ABSA) using the approach proposed by [Gao et al. (2019)](https://ieeexplore.ieee.org/abstract/document/8864964) originally for the [BERT](https://arxiv.org/abs/1810.04805?amp=1) model.

### Model Architecture for ABSA

The model architecture introduced by Gao et al. in the paper *[Target-Dependent Sentiment Classification With BERT](https://ieeexplore.ieee.org/abstract/document/8864964)* is shown in the figure below. For classification, the model embeddings coresponding to an aspect term, e.g. *battery timer*, are extracted from the models output. Max or mean pooling is then used to summarize multiple embeddings to one vector. Upon this vector a fully connected layer conducts the final classification into three classes.

[![Model architecture according to Gao et al. (2019)](td-bert-gao-2019.png "Model architecture according to Gao et al. (2019)")](https://ieeexplore.ieee.org/abstract/document/8864964)

### Preparing a Dataset

As a working example, we take the [Laptop 2014 dataset](https://huggingface.co/datasets/yqzheng/semeval2014_laptops) from the [SemEval 2014 Task 4](https://paperswithcode.com/dataset/semeval-2014-task-4-sub-task-2) for aspect-based sentiment analysis. This dataset provides texts with aspect positions, aspects and polarities per aspect.

```python
from datasets import load_dataset

dataset = load_dataset("yqzheng/semeval2014_laptops")
train_ds = dataset["train"]
test_ds = dataset["test"]
```

To be able extracting correct embeddings from the model, we need to map aspect words to model tokens. This can be achieved by HuggingFace's function [BatchEncoding.char_to_token()](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.BatchEncoding.char_to_token) that will return coresponding tokens for every character of the input sentence. First, we use model's tokenizer to create `BatchEncoding` objects for all dataset items. In this case, max. length of 100 model tokens is enough to fit any text from given dataset.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
batch_encoding = tokenizer(train_ds["text"], padding="max_length", max_length=100, truncation=True)
```

### Create an ABSA Model

### Fine-tuning Results

Training results for Laptop 2014 task with ABSA:

* xlm-roberta-base 73-76% macro F1, accuracy 76-80%
* xlm-roberta-large 77-78% macro F1, accuracy 80-82%
* xlm-roberta-xl 79-80% macro F1, accuracy 81-83%
* xlm-roberta-xxl % 81-82% macro F1, accuracy 83-84%


### Using a Fine-tuned Model