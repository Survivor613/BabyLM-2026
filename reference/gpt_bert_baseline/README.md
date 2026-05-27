---
license: apache-2.0
language:
- en
tags:
- babylm-baseline
- strict-small
- babylm-2025
---

# Model Card for GPT-BERT Mixed

<!-- Provide a quick summary of what the model is/does. [Optional] -->
A 120M model trained on 1B (100M unique words) able to do both causal and masked inference.


#  Table of Contents

- [Model Card for GPT-BERT Small Causal Focus](#model-card-for--model_id-)
- [Table of Contents](#table-of-contents)
- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Hyperparameters](#hyperparameters)
  - [Training Procedure](#training-procedure)
    - [Size and Checkpoints](#size-and-checkpoints)
- [Evaluation](#evaluation)
  - [Testing Data & Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Metrics](#metrics)
    - [Hyperparameters](#hyperparameters)
  - [Results](#results)
- [Technical Specifications](#technical-specifications-optional)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Hardware](#hardware)
    - [Software](#software)
    - [Training Time](#training-time)
- [Citation](#citation)
- [Model Card Authors](#model-card-authors-optional)
- [Bibliography](#bibliography)


# Model Details

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
This one of the three GPT-BERT baselines for the strict-small track of the 2025 BabyLM challenge.
This specific model is trained with a equal number of examples being causal and masked.

- **Developed by:** Lucas Georges Gabriel Charpentier
- **Model type:** Language model (Causal and Masked)
- **Language(s) (NLP):** eng
- **License:** apache-2.0
- **Resources for more information:**
    - [GitHub Repo](https://github.com/ltgoslo/gpt-bert)


# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
This is a pre-trained language model.
It can be used to evaluate tasks zero-shot in both a causal and masked setting.
It can also be fine-tuned by adding a new head and dropping the language modeling head.
It can be used for language generation but given its small size and low number of words trained on, do not expect LLM-level performance.
It can also be used for mask infilling.

# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

We used the BabyLM 100M (Strict) dataset to train the model. It is composed in the following way:

| Source | Weight | Domain | Citation | Website | License |
| --- | --- | --- | --- | --- | --- |
| BNC | 8% | Dialogue | BNC Consortium (2007) | [link](http://www.natcorp.ox.ac.uk/) | [link](http://www.natcorp.ox.ac.uk/docs/licence.html) <sup>1</sup> |
| CHILDES | 29% | Dialogue, Child-Directed | MacWhinney (2000) | | [link](https://talkbank.org/share/rules.html) |
| Project Gutenberg | 26% | Fiction, Nonfiction | Gerlach & Font-Clos (2020) | [link](https://github.com/pgcorpus/gutenberg) | [link](https://www.gutenberg.org/policy/license.html) |
| OpenSubtitles | 20% | Dialogue, Scripted | Lison & Tiedermann (2016) | [link](https://opus.nlpl.eu/OpenSubtitles-v2018.php) | Open source |
| Simple English Wikipedia | 15% | Nonfiction | -- | [link](https://dumps.wikimedia.org/simplewiki/20221201/) | [link](https://dumps.wikimedia.org/legal.html) |
| Switchboard | 1% | Dialogue | Godfrey et al. (1992), Stolcke et al., (2000) | [link](http://compprag.christopherpotts.net/swda.html) | [link](http://compprag.christopherpotts.net/swda.html) |

<sup>1</sup> Our distribution of part of the BNC Texts is permitted under the fair dealings provision of copyright law (see term (2g) in the BNC license).

## Hyperparameters

| Hyperparameter | Value |
| --- | --- |
| % Causal Objective | 50.00% |
| % Masked Objective | 50.00% |
| Sequence Length | 128 &rarr; 512 |
| Batch Size (in tokens) | 131 072 |
| Learning Rate | 0.007 |
| Number of Steps | 12 330 |
| Warmup Ratio | 1.6% |
| Cooldown Ratio | 1.6% |
| Mask Ratio | 0.3 &rarr; 0.15 |
| Random Ratio | 0.1 |
| Keep Ratio | 0.1 |
| Weight Decay | 0.1 |
| Optimizer | LAMB |
| Optimizer Epsilon | 10<sup>-8</sup> |
| Optimizer Beta_1 | 0.9 |
| Optimizer Beta_2 | 0.98 |
| Grdient Clipping | 2.0 |
| Z-Loss weight | 0.0001 |


## Training Procedure

During training we vary both the mask token percentage (linear decay from 30% to 15%), and the sequence length.
For the sequence length we make sure to keep the total tokens per batch the same by reducing the batch size proportionally to the sequence length.
We have three steps in the sequence length:
- We start with a sequence length of 128 for 60% of the training.
- The next 20% has a sequence length of 256.
- The final 20% has a sequence length of 512.
We use a Warmup-Cosine-Cooldown scheduler for the training with the percentages reported in the [Hyperparameters](#hyperparameters)

### Size and checkpoints

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

The model has 120M parameters.
In total we train on around 1B words (or ten repetitions of the training set).
We provide multiple checkpoints from the training.
Specifically we provode:
- Checkpoints every 1M words of pretraining for the first 10M words (or every 12.33 steps)
- Checkpoints every 10M words of pretraining for the first 100M words (or every 123.3 steps)
- Checkpoints every 100M words of pretraining for the first 1B words (or every 1233 steps)
 
# Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

This model is evaluated in three different fashions:
1. We provide a validation loss calculated on 1M words from the development set of the BabyLM data (same source as those found in [Training Data](#training-data)).
2. We do zero-shot evaluation on 7 tasks.
3. We do fine-tuning on a subset of the (Super)GLUE tasks (Wang et al., ICLR 2019; Wang et al., NeurIPS 2019) .

## Testing Data & Metrics

### Testing Data

<!-- This should link to a Data Card if possible. -->

For the BLiMP, BLiMP supplement, and EWoK tasks, we use a filtered version of the dataset to only include examples with words found in the BabyLM dataset.
For the Finetuning task, we both filter and sample down to a maximum 10 000 train examples.

*Validation Data*

1M words from the developement split of BabyLM.
The evaluation is done using the Masked Next Token Prediction objective.

*Zero-shot Tasks*

- **BLiMP**: The Benchmark of Linguistic Minimal Pairs evaluates the model's linguistic ability by seeing if it can recognize the grammatically correct sentence from a pair of minimally different sentences. It tests various grammatical phenomena.(Warstadt et al., TACL 2020)
- **BLiMP Supplement**: A supplement to BLiMP introduced in the first edition of the BabyLM challenge. More focused on dialogue and questions. (Warstadt et al., CoNLL-BabyLM 2023)
- **EWoK**: Works similarly to BLiMP but looks the model's internal world knowledge. Looking at both whter a model has physical and social knowledge. (Ivanova et al., 2024)
- **Eye Tracking and Self-paced Reading**: Looks at whether the model can mimick the eye tracking and reading time of a human but using surprisal of a word as a proxy for time spent reading a word. (de Varda et al., BRM 2024)
- **Entity Tracking**: Checks whether a model can keep track of the changes to the states of entities as text/dialogue unfolds. (Kim & Schuster, ACL 2023)
- **WUGs**: Tests morphological generalization in LMs through an adjective nominalization task. (Hofmann et al., 2024)

*Finetuning Tasks*

- **BoolQ**: A yes/no QA dataset with unprompted and unconstrained questions. (Clark et al., NAACL 2019)
- **MNLI**: The Multi-Genre Natural Language Inference corpus tests the language understanding of a model by seeing wehther it can recognize textual entailment. (Williams et al., NAACL 2018)
- **MRPC**: The Microsoft Research Paraphrase Corpus contains pairs of sentences that are either paraphrases/semntically equivalent to each other or unrelated.(Dolan & Brockett, IJCNLP 2005)
- **QQP**<sup>2</sup>: Similarly to MRPC, the Quora Question Pairs corpus tests the models ability to determine whether a pair of questions are sematically similar to each other. These questions are sourced from Quora.
- **MultiRC**: The Multi-Sentence Reading Comprehension corpus is a QA task that evaluates the model's ability to the correct answer from a list of answers given a question and context paragraph. In this version the data is changed to a binary classification judging whether the answer to a question, context pair is correct. (Khashabi et al., NAACL 2018)
- **RTE**: Similar the Recognizing Text Entailement tests the model's ability to recognize text entailement. (Dagan et al., Springer 2006; Bar et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., TAC 2009)
- **WSC**: The Winograd Schema Challenge tests the models ability to do coreference resolution on sentences with a pronoun and a list of noun phrases found in the sentence. This version edits it to be a binary classification on examples consisting of a pronoun and noun phrase.(Levesque et al., PKRR 2012)

<sup>2</sup> https://www.quora.com/profile/Ricky-Riche-2/First-Quora-Dataset-Release-Question-Pairs

### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The metrics used to evaluate the model are the following:
- Validation Data
  - Cross-entropy loss on the masked tokens
- Zero-shot
  - Accuracy on predicting the correct completion/sentence for BLiMP, BLiMP Supplement, EWoK, Entity Tracking, and WUGs
  - Change in R^2 prediction from baseline for Eye Tracking (with no spillover) and Self-paced Reading (1-word spillover)
- Finetuning
  - 3 class Accuracy for MNLI
  - Binary Accuracy for BoolQ, MultiRC, and WSC
  - F1-score for MRPC and QQP

The metrics were chosen based on the advice of the papers the tasks come from.

### Hyperparameters

| Hyperparameter | MNLI, RTE, QQP, MRPC | BoolQ, MultiRC | WSC |
| --- | --- | --- | --- |
| Learning Rate | 3\*10<sup>-5</sup> | 3\*10<sup>-5</sup> | 3\*10<sup>-5</sup> |
| Batch Size | 32 | 16 | 32 |
| Epochs | 10 | 10 | 30 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| Optimizer | AdamW | AdamW | AdamW |
| Scheduler | cosine | cosine | cosine |
| Warmup percentage | 6% | 6% | 6% |
| Dropout | 0.1 | 0.1 | 0.1 |

## Results 

*Validation (Loss)*

- 2.10

*Zero-shot*

| Task | Metric | Causal Score | MNTP Score |
| --- | --- | --- | --- |
| BLiMP | Acc | 78.37 | 80.50 |
| BLiMP Supplement | Acc | 69.23 | 73.04 |
| EWoK | Acc | 51.79 | 52.40 |
| Eye Tracking | change in R^2 | 8.74 | 9.15 |
| Self-paced Reading | change in R^2 | 3.59 | 3.43 |
| Entity Tracking | Acc | 33.09 | 39.77 |
| WUGs | Acc | 39.50 | 37.00 |

*Finetuning*

| Task | Metric | Score |
| --- | --- | --- |
| BoolQ | Acc | 73.99 |
| MNLI | Acc | 63.37 |
| MRPC | F1 | 90.10 |
| QQP | F1 | 74.34 |
| MultiRC | Acc | 69.76 |
| RTE | Acc | 58.99 |
| WSC | Acc | 63.46 |

# Technical Specifications

## Model Architecture and Objective

The model architecture used is based on the GPT-BERT (Charpentier & Samuel, CoNLL-BabyLM 2024) architecture (based off the LTG-BERT (Samuel et al., Findings 2023) architecture).
We train on two objectives Masked Next Token Prediction and Causal Language Modeling. 
During the training we had as many examples with the causal objective as the MNTP objective.

## Compute Infrastructure

We use the LUMI supercomputer to train this model.

We acknowledge Norway for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through Sigma2.

The computations were performed on resources provided by
Sigma2 - the National Infrastructure for High-Performance Computing and
Data Storage in Norway

### Hardware

- 8 AMD MI250X GPUs (each are split into two compute units, functionally working as 16 GPUs)

### Software

PyTorch

### Training Time

The model took 3 hours to train (which equates to 48 GPU-hours).

# Citation

```latex
@misc{charpentier2025babylmturns3papers,
      title={BabyLM Turns 3: Call for papers for the 2025 BabyLM workshop}, 
      author={Lucas Charpentier and Leshem Choshen and Ryan Cotterell and Mustafa Omer Gul and Michael Hu and Jaap Jumelet and Tal Linzen and Jing Liu and Aaron Mueller and Candace Ross and Raj Sanjay Shah and Alex Warstadt and Ethan Wilcox and Adina Williams},
      year={2025},
      eprint={2502.10645},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.10645}, 
}
```

# Model Card Authors

Lucas Georges Gabriel Charpentier

# Bibliography

[BERT or GPT: why not both?](https://aclanthology.org/2024.conll-babylm.24/) (Charpentier & Samuel, CoNLL-BabyLM 2024)

[Trained on 100 million words and still in shape: BERT meets British National Corpus](https://aclanthology.org/2023.findings-eacl.146/) (Samuel et al., Findings 2023)

[GLUE: A multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7) (Wang et al., ICLR 2019)

[SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems](https://proceedings.neurips.cc/paper_files/paper/2019/file/4496bf24afe7fab6f046bf4923da8de6-Paper.pdf) (Wang et al., NeurIPS 2019)

[BLiMP: The Benchmark of Linguistic Minimal Pairs for English](https://aclanthology.org/2020.tacl-1.25/) (Warstadt et al., TACL 2020)

[Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora](https://aclanthology.org/2023.conll-babylm.1/) (Warstadt et al., CoNLL-BabyLM 2023)

[🌏 Elements of World Knowledge (EWoK): A cognition-inspired framework for evaluating basic world knowledge in language models](https://arxiv.org/pdf/2405.09605v1) (Ivanova et al., 2024)

[Cloze probability, predictability ratings, and computational estimates for 205 English sentences, aligned with existing EEG and reading time data](https://link.springer.com/article/10.3758/s13428-023-02261-8) (de Varda et al., BRM 2024)

[Entity Tracking in Language Models](https://aclanthology.org/2023.acl-long.213/) (Kim & Schuster, ACL 2023)

[Derivational Morphology Reveals Analogical Generalization in Large Language Models](https://arxiv.org/pdf/2411.07990) (Hofmann et al., 2024)

[Automatically Constructing a Corpus of Sentential Paraphrases](https://aclanthology.org/I05-5002/) (Dolan & Brockett, IJCNLP 2005)

[A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://aclanthology.org/N18-1101/) (Williams et al., NAACL 2018)

[The Winograd Schema Challenge]( http://dl.acm.org/citation.cfm?id=3031843.3031909) (Levesque et al., PKRR 2012)

[The PASCAL Recognising Textual Entailment Challenge](https://link.springer.com/chapter/10.1007/11736790_9) (Dagan et al., Springer 2006)

[The Second PASCAL Recognising Textual Entailment Challenge]() (Bar et al., 2006)

[The Third PASCAL Recognizing Textual Entailment Challenge](https://aclanthology.org/W07-1401/) (Giampiccolo et al., 2007)

[The Fifth PASCAL Recognizing Textual Entailment Challenge](https://tac.nist.gov/publications/2009/additional.papers/RTE5_overview.proceedings.pdf) (Bentivogli et al., TAC 2009)

[BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://aclanthology.org/N19-1300/) (Clark et al., NAACL 2019)

[Looking Beyond the Surface: A Challenge Set for Reading Comprehension over Multiple Sentences](https://aclanthology.org/N18-1023/) (Khashabi et al., NAACL 2018)
