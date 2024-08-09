# Fine-tune and Use BERT-like Models on Multiple GPUs for Sentiment Analysis or Stance Detection

## Installation

With [Poetry](https://python-poetry.org/docs/) (recommended)

```bash
# Install poetry first
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies and this package
poetry install
```

With [pip](https://pip.pypa.io/en/stable/installation/)

```bash
pip install -r requirements.txt
```

## Readme

* [Hardware Requirements for Training of Language Models](./readme/0_model_sizes.md)
* [Using (Large) Language Models for Text Classification](./readme/1_using_models.md)
* [Fine-tuning BERT-like Models of Different Sizes on Multiple GPUs for Aspect-based Sentiment Analysis (ABSA)](./readme/2_finetuning_absa_models.md)

## Examples

### Fine-tune an Aspect-based Sentiment Analysis (ABSA) model

First, adapt the script `finetune_absa_bert_like.py` by completing **TODO 1** and **2** to select the base model for fine-tuning.

```python
# TODO 1: Uncomment the model you want to use:
from sparta.absa.models import RobertaForABSA as ModelForABSA
# from sparta.absa.models import XLMRobertaForABSA as ModelForABSA
# from sparta.absa.models import XLMRobertaXLForABSA as ModelForABSA
...

# TODO 2: Ucomment the model you want to use
model_name = "roberta-base"
# model_name = "roberta-large"
# model_name = "xlm-roberta-base"
# model_name = "xlm-roberta-large"
...
```

Second, select the corresponding YAML accelerate configuration and adapt the number of available GPUs `num_processes`. Please check the [minimal requirements for differently sized models](./readme/0_model_sizes.md). There are [provided configurations](./accelerate_configs/) for:
- [RoBERTa-base and -large](./accelerate_configs/roberta.yaml)
- [XLM-RoBERTa-base and -large](./accelerate_configs/xlm_roberta_base_large.yaml)
- [XLM-RoBERTa-XL](./accelerate_configs/xlm_roberta_xl.yaml)
- [XLM-RoBERTa-XXL](./accelerate_configs/xlm_roberta_xxl.yaml)
- [Deberta-base and -large](./accelerate_configs/deberta.yaml)
- [ELECTRA-base and -large](./accelerate_configs/electra.yaml)
- [ERNIE-base and -large](./accelerate_configs/ernie.yaml)

Finally, run the training ussing `accelerate` and the configuration for a particular model:

- For RoBERTA-base or -large model:
    ```
    poetry run accelerate launch --config_file accelerate_configs/roberta.yaml finetune_absa_bert_like.py
    ```

- For XLM-RoBERTA-base or -large model:
    ```
    poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_base_large.yaml finetune_absa_bert_like.py
    ```

- For XLM-RoBERTA-XL model:
    ```
    poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_xl.yaml finetune_absa_bert_like.py
    ```

- For XLM-RoBERTA-XXL model:
    ```
    poetry run accelerate launch --config_file accelerate_configs/xlm_roberta_xxl.yaml finetune_absa_bert_like.py
    ```

- For (m)DeBerta-base or -large model:
    ```
    poetry run accelerate launch --config_file accelerate_configs/deberta.yaml finetune_absa_bert_like.py
    ```

- For ELECTRA-base or -large model:
    ```
    poetry run accelerate launch --config_file accelerate_configs/electra.yaml finetune_absa_bert_like.py
    ```

- For ERNIE-base or -large model:
    ```
    poetry run accelerate launch --config_file accelerate_configs/ernie.yaml finetune_absa_bert_like.py
    ```

The model checkpoints will be located in `absa_checkpoints` and the final best model in `absa_models`.


### Use the Fine-tuned Model

Complete the same TODOs as for fine-tuning the model but within the script `use_absa_bert_like.py`. Then run the script with the corresponding configuration as show in the previos section. Just replace `finetune_absa_bert_like.py` by `use_absa_bert_like.py` in the command line.