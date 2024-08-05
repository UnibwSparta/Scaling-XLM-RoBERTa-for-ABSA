## Using (Large) Language Models for Text Classification

This article provides a brief tutorial on how to use some common Transformer-based language models for text classification. Here we focus on the [sentiment analysis](https://paperswithcode.com/task/sentiment-analysis) task and showcase differences to the [stance detection](https://doi.org/10.1016/j.ipm.2021.102597) task. In this tutorial we use the HuggingFace [plattform](https://huggingface.co/) and [library](https://github.com/huggingface/transformers) to access and run pre-trained language models.

### Sentiment Analysis Models

On the HuggingFace plattform you can find very different models that have been pre-trained and fine-tuned specifically to solve the task of sentiment analysis. This task can have very different characteristics depending on:

- number of sentiment classes, e.g.
    - [one-way (binary regression)](https://en.wikipedia.org/wiki/Binary_regression) - a score from 0 to 100
    - [two-way classification](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary) - negative vs. positive
    - [three-way classification](https://ieee-dataport.org/open-access/coronavirus-covid-19-tweets-dataset) - also considering neutral class
    - [fine-grained five-star classification](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained)
- target of classification, e.g.
    - [whole text](https://aclanthology.org/S17-2088/) - no specific target or implicit target
    - [target-dependent](https://ieeexplore.ieee.org/document/8448158) - single or multiple targets
    - [aspect-based](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval-5) - multiple aspects of a single target
    - [targeted aspect-based](https://arxiv.org/pdf/1906.06945.pdf) - multiple aspects of different targets
- language (monolingual, e.g. German, or multilingual)
- domain, e.g. sentiment of [movie reviews](https://paperswithcode.com/sota/sentiment-analysis-on-imdb) or [product revews](https://paperswithcode.com/dataset/amazon-review)
- text quality, e.g. considering the language used in news paper articles of very high quality and the language used on social media networks of low quality (noisy)
- text length, e.g. tweets are very short while news articles can be very long, thus, even exceeding the model's maximum input sequence length

**TODO** Definition of the Sentiment Analysis task and specifics!

Here are some example models for sentiment analysis from HuggingFace:

| model | number of sentiment classes | target of classification | language | domain | text length/quality |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | two-way | whole text | English | movie reviews | sentence |
| [Elron/deberta-v3-large-sentiment](https://huggingface.co/Elron/deberta-v3-large-sentiment) | three-way | whole text | English | tweets | tweet |
| [nlptown/bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) | five-star | whole text | English | product reviews | review |
| [yangheng/deberta-v3-large-absa-v1.1](https://huggingface.co/yangheng/deberta-v3-large-absa-v1.1) | three-way | aspect-based | English | laptop and restaurant reviews | review |
| [kevinscaria/instructabsa](https://github.com/kevinscaria/instructabsa) | three-way | aspect-based | English | laptop and restaurant reviews | review |
| [oliverguhr/german-sentiment-bert](https://huggingface.co/oliverguhr/german-sentiment-bert) | three-way | whole text | German | different reviews and social media comments | review/comment |
| [ssary/XLM-RoBERTa-German-sentiment](https://huggingface.co/ssary/XLM-RoBERTa-German-sentiment) | three-way | whole text | German | different reviews and social media comments | review/comment |
| [mox/gBERt\_base\_twitter\_sentiment\_politicians](https://huggingface.co/mox/gBERt_base_twitter_sentiment_politicians) | three-way | whole text | German | tweets of politicians | tweet |
| [aari1995/German\_Sentiment](https://huggingface.co/aari1995/German_Sentiment) | three-way | whole text | German | tweets | tweet |
| [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) | three-way | whole text | multi-lingual | tweets | tweet |
| [cardiffnlp/xlm-twitter-politics-sentiment](https://huggingface.co/cardiffnlp/xlm-twitter-politics-sentiment) | three-way | whole text | multi-lingual | tweets of members of parliament | tweet |
| [clampert/multilingual-sentiment-covid19](https://huggingface.co/clampert/multilingual-sentiment-covid19) | two-way | whole text | multi-lingual | tweets about COVID-19 | tweet |
<!-- | [UHH-CI/GermanPolitical-Gelectra-base](https://huggingface.co/UHH-CI/GermanPolitical-Gelectra-base) | one-way | whole text | German | statements regarding political issues | statement | -->


### Stance Detection Models

**TODO** Difference of stance detection to the Sentiment Analysis task.

- number of classes, e.g.
    - [two-way](https://aclanthology.org/2021.findings-acl.208.pdf)
    - [three-way](https://huggingface.co/datasets/mlburnham/PoliStance_Affect)
    - [four-way](https://dl.acm.org/doi/pdf/10.1145/3161603)
    - etc. - number and semantic of classes can be very different and depending on the specific case
- target of classification
    - [whole text](https://journalqd.org/article/view/5896)
    - [entity-/target-dependent](https://aclanthology.org/S16-1003.pdf)
    - [pair of targets](https://aclanthology.org/E17-2088.pdf)
    - [multi-target](https://link.springer.com/chapter/10.1007/978-3-031-15086-9_9)
    - [claim-based](https://ceur-ws.org/Vol-2624/paper9.pdf)

Here are some example models for stance detection from HuggingFace. As can be seen, data domains used for stance detection are much more specific compared to sentiment analysis and frequently focus on dedicated topics or issues:

| stance models | number of stance classes | target of classification | language | domain | text length/quality |
| :-- | :-- | :-- | :-- | :-- | :-- |
| [kornosk/bert-election2020-twitter-stance-biden-KE-MLM](https://huggingface.co/kornosk/bert-election2020-twitter-stance-biden-KE-MLM) | three-way | whole text | English | tweets about Biden as presidential candidate | tweet |
| [kornosk/bert-election2020-twitter-stance-trump-KE-MLM](https://huggingface.co/kornosk/bert-election2020-twitter-stance-trump-KE-MLM) | three-way | whole text | English | tweets about Trump as presidential candidate | tweet |
| [cardiffnlp/bertweet-base-stance-abortion](https://huggingface.co/cardiffnlp/bertweet-base-stance-abortion) | three-way | whole text | English | tweets about abortion | tweet |
| [seantw/covid-19-vaccination-tweet-stance](https://huggingface.co/seantw/covid-19-vaccination-tweet-stance) | three-way | whole text | English | tweets about vaccination | tweet |
| [krishnagarg09/stance-detection-semeval2016](https://huggingface.co/krishnagarg09/stance-detection-semeval2016) | three-way | target-dependent | English | tweets about abortion, atheism, feminism, and Hillary Clinton ([SemEval 2016 Task 6](https://huggingface.co/datasets/krishnagarg09/SemEval2016Task6)) | tweet |
| [eevvgg/StanceBERTa](https://huggingface.co/eevvgg/StanceBERTa) | three-way | whole text | English | tweets and Reddit posts | tweet/post |
| [pacoreyes/StanceFit](https://huggingface.co/pacoreyes/StanceFit) | two-way | whole text | English | mixed | statement |
| [MattiaSangermano/bert-political-leaning-it](https://huggingface.co/MattiaSangermano/bert-political-leaning-it) | four-way | whole text | Italian | political statements | statement |
| [cristinae/roberta\_politicalStanceClassifier](https://huggingface.co/cristinae/roberta_politicalStanceClassifier) | two-way | whole text | English, German, Spanish | news paper articles | article |
| [mlburnham/deberta-v3-large-polistance-affect-v1.1](https://huggingface.co/mlburnham/deberta-v3-large-polistance-affect-v1.1) | three-way | target-dependent | English | political tweets about U.S. politicians and political groups | tweet |
| [juliamendelsohn/social-movement-stance](https://huggingface.co/juliamendelsohn/social-movement-stance) | three-way | whole text | English | tweets on guns, LGBTQ rights, and immigration | tweet |
| [ningkko/drug-stance-bert](https://huggingface.co/ningkko/drug-stance-bert) | three-way | whole text | English | tweets about the use of drugs | tweet |


### Trivial Example of Usage

The majority of models for text classification that published of the HuggingFace plattform are usually easy to use. One just need an installed [Python distribution](https://www.python.org/about/gettingstarted/), some basic skills in using a [command line](https://www.freecodecamp.org/news/command-line-for-beginners/#shell) in a shell, and access to at least one GPU (or many CPUs). It is recommended to use a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) dedicated to one project, instead of installing necessary Python packages system-wide. Most convinient choice is frequently [Anaconda](https://docs.anaconda.com/anaconda/install/). For more advanced users, [Poetry](https://python-poetry.org/) is a good choice as well.

* Install packages into your new Python virtual environment, e.g. for AnaConda:

    ```
    # Go to your project folder
    cd YOUR_PROJECT_FOLDER

    # Initialize a virtual environment
    conda create -n venv

    # Activate the virtual environment
    conda activate venv

    # Install HuggingFace Transformers library
    conda install conda-forge::transformers
    ```

    or for Poetry:

    ```bash
    # Go to your project folder
    cd YOUR_PROJECT_FOLDER

    # Initialize a virtual environment
    poetry init

    # Install HuggingFace Transformers library
    poetry add transformers
    ```

* Create a simple script `my_script.py` to load and run a HuggingFace model:

    ```python
    import torch
    from transformers import pipeline

    # Select GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load one of the sentiment models for text classification as a pipeline
    pipe = pipeline(
        task="text-classification",
        model="yangheng/deberta-v3-large-absa-v1.1",
        device=device,
    )

    # Run the model on an example
    result = pipe("This restaurant has a very bad food. It will won't a hit!")

    # Print your result
    print(result)
    ```

* Run your script either within
    - AnaConda `python my_script.py`, or
    - Poetry `poetry run python my_script.py`

    and it will output:

    ```
    [{'label': 'Negative', 'score': 0.988707423210144}]
    ```

This trivial example shows how simple it is to use a fine-tuned language model that needs only one text as input. You may also pass a list of texts to the pipeline to get all the results at once.

If necessary, it is posssible to fine-tune (train) a base model on a specific dataset, e.g. for sentiment analysis or stance detection. As long as you classify the whole input text, you can rely on [widely provided tutorials](https://huggingface.co/learn/nlp-course/chapter3/3) to achieve this.

On a separate page, we provide a more complex tutorial on [how to run and fine-tune aspect-based sentiment analysis models for different model sizes on one or multiple GPUs](./2_finetuning_absa_models.md).