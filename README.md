# FlowerTune LLM on Medical Dataset

### Evaluation in the three baseline datasets:

|        | PubMedQA | MedMCQA | MedQA |  Avg  |
| :-----: | :------: | :-----: | :---: | :---: |
| Acc (%) |   71.20  |  58.11  | 61.98 | 63.76 |

#### Communication budget: used 1040.31 MB (last round)

### Evaluation of the baseline model proposed 

|        | PubMedQA | MedMCQA | MedQA |  Avg  |
| :-----: | :------: | :-----: | :---: | :---: |
| Acc (%) |   59.00  |  23.69  | 27.10 | 36.60 |  


### Introduction

This directory conducts federated instruction tuning with a pretrained [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.3) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.


## Changes from baseline

* Following the advances obtained with the approach presented by the [Gachon Cognitive Computing Lab](https://github.com/gachon-CCLab/GCCL-Medical-LLM-FlowerTune), we have used as a base model the [ContactDoctor/Bio-Medical-Llama-3-8B](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) fine tuned model.
* We train the model during 5 rounds, `num-server-rounds = 5`.
* We train the model locally during 5 epochs: `train.training-arguments.num-train-epochs = 5`
* We use the [FedAvgOpt](https://arxiv.org/abs/2501.15949) aggregation function.


## Methodology

This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedAvg strategy.
This provides a baseline performance for the leaderboard of Medical challenge.


## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

## Experimental setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `5` rounds.
All settings are defined in `pyproject.toml`.


## Running the experiment

First, login in huggingface:
```bash
huggingface-cli login
```

Then, run the experiment:

```bash
flwr run .
```

Evaluation in the three baseline datasets:

```bash
python eval.py --base-model-name-path="ContactDoctor/Bio-Medical-Llama-3-8B" --peft-path="peft_5" --batch-size=16 --quantization=4 --datasets=pubmedqa,medmcqa,medqa
```
