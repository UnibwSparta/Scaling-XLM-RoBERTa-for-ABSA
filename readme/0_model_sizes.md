## Hardware Requirements for Training of Language Models

This article describes the hardware requirements for training various sizes of language models on GPUs, assuming the use of [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) or [AMD ROCm](https://www.amd.com/en/products/software/rocm.html) software stacks.  We do not cover some alternative [AI-specific hardware architectures](#alternatives-to-gpus) here. As an example, we will focus on a particular [Transformer-based](https://en.wikipedia.org/wiki/Transformer_%28deep_learning_architecture%29) [encoder](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder) architecture, namely [XLM-RoBERTa](https://arxiv.org/abs/1911.02116), and consider differently sized models: [Base](https://huggingface.co/FacebookAI/xlm-roberta-base), [Large](https://huggingface.co/FacebookAI/xlm-roberta-large), [XL](https://huggingface.co/facebook/xlm-roberta-xl) and [XXL](https://huggingface.co/facebook/xlm-roberta-xl). We explicitly provide examples of GPU counts and sizes for the inference and fine-tuning of these models for the task of 3-way classification of tweets/posts, such as [sentiment analysis](https://paperswithcode.com/task/sentiment-analysis). Additionally, we present hardware requirements for further pre-training.


### Memory Footprint of a Language Model

GPUs, originally created for graphical processing, accelerate the execution of artificial neural models compared to CPUs because they execute many instructions in parallel. However, GPUs are manufactured with a fixed memory size that cannot be dynamically extended, e.g., by memory bars, which limits the execution of neural models. The [HuggingFace](https://huggingface.co/) plattform provides a [model size estimator](https://huggingface.co/docs/accelerate/main/en/usage_guides/model_size_estimator) that simply loads an appropriate model and reports the memory requirements for inference and fine-tuning as if the model were running on a single GPU. For larger models that do not fit on a single GPU, such an [estimation can be more sophisticated](https://ieeexplore.ieee.org/abstract/document/9355301?figureId=fig1#fig1), even if the model size (number of its parameters) is known. Specifically for the Transformer-based language models, the required number of GPUs and their size (amount of memory) depend on a variety of different parameters:

- **Model Architecture**: Transformer-based models are typically classified as [decoder, encoder, or encoder-decoder](https://magazine.sebastianraschka.com/p/understanding-encoder-and-decoder). For instance, decoder models, such as the famous [GPT architecture](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), usually require less memory than encoder models of comparable size, such as [BERT](https://arxiv.org/abs/1810.04805).
- **Model Parameters**: The number of parameters (or weights) is typically given in millions (M), billions (B) or even trillions (T). They often refer to model size, although the number of layers or the maximum sequence length also describe model dimensions.
- **Number Format (Computational Precision)**: At the hardware level, there are several types of number formats with different precision at which the model can be operated: [FP32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) - single precision (4 bytes), [NVIDIA's TF32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) (19 bits < 3 bytes), [FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) or [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) - half precision (2 bytes), [INT 8](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html) (1 byte), and [INT 4](https://arxiv.org/abs/2306.11987) (a half byte). There is also a [mixed precision](https://arxiv.org/abs/1710.03740) method that uses a mixture of number formats (2 to 4 bytes). INT-based methods are called quantization and cannot be used in every type of model training. Each model parameter requires a corresponding number of bytes on a GPU. For example, the parameters of a 3B model at half precision would take 3·10<sup>9</sup> × 2 bytes = 6 GB of GPU memory. In this way, one can roughly estimate the minimum memory footprint of a model when used during inference.
- **Type of Model Use**: Model inference and training have different requirements. During training, the model occupies approximately three times more GPU memory than during inference, since it needs to store not only the model parameters but also the optimizer states and gradients.
- **Maximum Sequence Length**: The maximum sequence length can be understood as the maximum input length. For decoder models, it is the combination of input (prompt) and output. Language models are pre-trained with a fixed maximum sequence length, e.g., [512](https://huggingface.co/FacebookAI/xlm-roberta-base), [1024](https://huggingface.co/openai-community/gpt2), [32k](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), [1M](https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf) tokens. However, they can also be used with a shorter length, which has a positive effect on memory consumption and execution time.
- **Vocabulary and Embeddings Sizes:** Vocabulary sizes vary depending on the number of supported languages and input formats. For multilingual XLM-RoBERTa-based models, the token vocabulary is about 250k tokens. Furthermore, differently sized models use different vector dimensions to represent tokens as embeddings, e.g., [Base](https://huggingface.co/FacebookAI/xlm-roberta-base) - 768, [Large](https://huggingface.co/FacebookAI/xlm-roberta-large) - 1024, [XL](https://huggingface.co/facebook/xlm-roberta-xl) - 2560, [XXL](https://huggingface.co/facebook/xlm-roberta-xl) - 4096. Multiplying both numbers by the amount of bytes required for the number format gives the necessary amount of memory for the input/output layer of the model.
- **Batch Size**: The number of elements (e.g., texts/tweets/posts/etc.) sent to the GPU at one time. A higher batch size during training usually has a positive influence on the learning process. For inference, a higher batch size means a shorter average inference time. A technique called [gradient accumulation](https://kozodoi.me/blog/20210219/gradient-accumulation) can be used to increase the effective batch size during training. However, it requires additional memory depending on the implementation and usage.
- **Type of Training**: (Full parameter) fine-tuning for a specific downstream task usually needs less memory than pre-training. The difference between these two types comes from the different model heads (last model layers) used. For pre-training, the model head must be able to predict all vocabulary tokens at each position of the model's token sequence. In fine-tuning for a 3-way classification task, the head predicts only 3 tokens at a single position. Parameter efficient fine-tuning (PEFT) is even more efficient than full fine-tuning. There are many [PEFT techniques](https://github.com/synbol/Awesome-Parameter-Efficient-Transfer-Learning) with different memory requirements, which can be further combined with quantization techniques.
- **Optimizer**: When training models, different optimizers can be used. The [AdamW](https://paperswithcode.com/method/adamw) optimizer uses 8 bytes for each model parameter, [Adafactor](https://paperswithcode.com/method/adafactor) - 4 bytes, and [8-bit Adam](https://arxiv.org/abs/2110.02861) - 2 bytes. For instance, for a 3B model, AdamW will allocate 3·10<sup>9</sup> × 8 bytes = 24 GB of memory for optimizer states.
- **Type of Distributed Training**: Large language models demand more GPUs and [other distributed approaches](https://huggingface.co/docs/transformers/v4.37.2/en/perf_train_gpu_many) for both training and inference. While the [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/beginner/ddp_series_theory.html) approach can only run models in parallel that would also fit on a single GPU, methods such as [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) or [ZeRO](https://www.deepspeed.ai/tutorials/zero/) help to distribute layers of a large model accross multiple GPUs.


### <a name="common-gpu-sizes"></a> Common GPU Sizes

As mentioned above, the required hardware capacities differ for production and development use cases, i.e., model inference vs. model training. Therefore, it is advisable to use cloud computing services with higher resources for temporary development and to acquire own GPU infrastructure only for permanent use with lower resources. All major cloud computing services offer suitable services for deep learning:

- [AWS services](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [Google Cloud Services](https://cloud.google.com/compute/vm-instance-pricing#accelerator-optimized)
- [Microsoft Azure](https://azure.microsoft.com/en-us/pricing/details/machine-learning/#pricing) (see also [VM sizes](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu))

Below is a list of some GPUs with typical sizes that are offered by these services or can be purchased if needed:

- 8 GB - [AMD Radeon™ PRO V520](https://www.techpowerup.com/gpu-specs/radeon-pro-v520.c3755)
- 16 GB - [NVIDIA Tesla P100](https://www.techpowerup.com/gpu-specs/tesla-p100-pcie-16-gb.c2888)
- 24 GB - [NVIDIA Tesla K80](https://www.techpowerup.com/gpu-specs/tesla-k80.c2616)
- 32 GB - [NVIDIA V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184)
- 40 GB - [NVIDIA A100](https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623)
- 80 GB - [NVIDIA A100 80 GB](https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821)
- 80 GB - [NVIDIA H100 80 GB](https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899)


### Requirements of Different XLM-RoBERTa Models for Sentiment Analysis

In this section, we consider the task of 3-classes sentiment analysis on tweets/posts. We provide model GPU requirements, training times, and additional CPU memory for four differently sized XLM-RoBERTa models: [Base](https://huggingface.co/FacebookAI/xlm-roberta-base), [Large](https://huggingface.co/FacebookAI/xlm-roberta-large), [XL](https://huggingface.co/facebook/xlm-roberta-xl), and [XXL](https://huggingface.co/facebook/xlm-roberta-xl). We use the following general settings for both fine-tuning and pre-training:

- The maximum sequence length is set to 128 tokens. This sequence length will fit 99.99% of tweets/posts with a maximum length of 280 characters.
- AdamW optimizer.
- [DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html) and [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) approaches for XL and XXL models.
- [CPU offloading](https://www.deepspeed.ai/tutorials/zero-offload/) when training with FSDP, which saves some GPU memory.
- We use PyTorch 2, but do not [compile the models](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). Using compiled models can improve training time [up to 30%](https://huggingface.co/docs/transformers/main/perf_torch_compile#benefits-of-torchcompile).

For a specific type of training, we distinguish the following settings:

- For fine-tuning, we use a labeled dataset with 10 thousand tweets and stick to the full parameter fine-tuning.
- For pre-training, we use an unlabeled dataset with one million tweets. Again, we train the parameters of all layers.

#### Fine-Tuning Requirements

The following table shows the training settings, model requirements, and test results in terms of training times for fine-tuning on a dataset of 10 thousand tweets for one epoch. We used the same batch size and no gradient accumulation for experiments with all model dimensions. We ran all experiments on the same hardware with 40 GB GPUs. However, we report the maximum used GPU memory using [common GPU dimensions](#common-gpu-sizes). As can be seen, the XXL model completes an epoch pass in half the time of the XL model. This is due to the double number of GPUs used. Although it has about three times as many model parameters, it can be executed efficiently due to parallelism and, on average, takes the same time as the XL model to process a single batch.

| Model Dimension | Training Type | Num. Parameters | Num. GPUs | Max. per GPU Size Usage | Batch Size per GPU | Gradient Accumulation | Global Batch Size | GPU Type | Time for One Epoch (hours) | Additional CPU Memory (GB) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Base | Single | 279M | 1 | 8 | 16 | 1 | 16 | NVIDIA A100 | 1 | 8 |
| Large | Single | 561M | 1 | 16 | 16 | 1 | 16 | NVIDIA A100 | 3 | 12 |
| XL | DDP | 3.5B | 2 | 32 | 16 | 1 | 32 | NVIDIA A100 | 50 | 96 |
| XXL | FSDP | 10.7B | 4 | 40 | 16 | 1 | 64 | NVIDIA A100 | 25 | 200 |


#### Pre-Training Requirements

The following table shows the training settings, model requirements, and test resutls in terms of training times for unsupervised pre-training on a dataset of one million tweets for one epoch. For the Base, Large, and XL models we ran multiple experiments with different numbers of GPUs and batch sizes, trying to keep the global batch size similar as much as possible on our hardware. We used the same hardware for all but one experiment. For the pre-training of the XXL model, we employed NVIDIA H100 hardware with more GPUs and memory.


| **Pre-Training** | Training Type | Num. Parameters | Num. GPUs | Max. per GPU Size Usage | Batch Size per GPU | Gradient Accumulation | Global Batch Size | GPU Type | Time for One Epoch (hours) | Additional CPU Memory (GB) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Base | Single | 279M | 1 | 16 | 16 | 512 | 8192 | NVIDIA A100 | 5.1 | 8 |
|     |     |     | 1 | 24 | 26 | 316 | 8216 | NVIDIA A100 | 3.3 | 8 |
|     |     |     | 1 | 40 | 52 | 158 | 8216 | NVIDIA A100 | 2.3 | 8 |
| Large | Single | 561M | 1 | 16 | 8 | 1024 | 8192 | NVIDIA A100 | 15.3 | 12 |
|     |     |     | 1 | 24 | 16 | 512 | 8192 | NVIDIA A100 | 8.4 | 12 |
|     |     |     | 1 | 40 | 36 | 228 | 8208 | NVIDIA A100 | 4.7 | 12 |
| XL | FSDP | 3.5B | 2 | 40 | 8 | 256 | 4096 | NVIDIA A100 | 33.2 | 96 |
|     | FSDP |     | 4 | 40 | 8 | 256 | 8192 | NVIDIA A100 | 18.3 | 120 |
| XXL | FSDP | 10.7B | 8 | 80 | 4 | 256 | 8192 | **NVIDIA H100** | 13.6 | 200 |



### <a name="alternatives-to-gpus"></a> Specialized Hardware as Alternatives to GPUs

There are some interesting alternatives to traditional GPUs. Be aware that many of them employ propietary software stacks and may not be supported by popular frameworks such as [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

- [AMD Ryzen AI NPU](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html#tabs-74833e1024-item-833270fb2a-tab)
- [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) (only for inference)
- [Cerebras WSE](https://www.cerebras.net/product-chip/)
- [Furiosa NPU](https://www.furiosa.ai/)
- [Google TPU](https://cloud.google.com/tpu/?hl=en)
- [Graphcore IPU](https://www.graphcore.ai/ipus-in-the-cloud)
- [Groq LPU](https://wow.groq.com/lpu-inference-engine/) (only for inference)
- [Habana Gaudi Accelerator (HPU)](https://habana.ai/products/gaudi/)
- [Qualcomm Cloud AI 100 Accelerator](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence/cloud-ai-100)
