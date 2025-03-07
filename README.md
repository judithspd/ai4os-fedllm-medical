# FlowerTune LLM on Medical Dataset

### Introduction

This directory conducts federated instruction tuning with a pretrained [ContactDoctor/Bio-Medical-Llama-3-8B](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

### Evaluation in the three baseline datasets with the proposed approach:

|        | PubMedQA | MedMCQA | MedQA |  Avg  |
| :-----: | :------: | :-----: | :---: | :---: |
| Acc (%) |   70.80  |  58.04  | 62.84 | 63.89 |

#### Communication budget: 2080.62 MB

## Changes from baseline

* Following the advances obtained with the approach presented by the [Gachon Cognitive Computing Lab](https://github.com/gachon-CCLab/GCCL-Medical-LLM-FlowerTune), we have used as a base model the [ContactDoctor/Bio-Medical-Llama-3-8B](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) fine tuned model.
* We train the model during 10 rounds, `num-server-rounds = 10`, see [peft_10](https://github.com/judithspd/ai4os-fedllm-medical/tree/main/flowertune-eval-medical/peft_10).
* We train the model locally during 2 epochs: `train.training-arguments.num-train-epochs = 2`.
* We use the [FedAvgOpt](https://arxiv.org/abs/2501.15949) aggregation function.


## Aditional setting

We include this new scenario in case any user wants to reproduce it.
During the search for the model that would give us the best performance, we found that the setting in which we use the [ContactDoctor/Bio-Medical-Llama-3-8B](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) model, the [FedAvgOpt](https://arxiv.org/abs/2501.15949) aggregation function and we train for 5 epochs in each client and 20 rounds, gives the following results for the checkpoint of round 5:

|        | PubMedQA | MedMCQA | MedQA |  Avg  |
| :-----: | :------: | :-----: | :---: | :---: |
| Acc (%) |   72.60  |  58.64  | 63.39 | 64.88 |


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
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `10` rounds.
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
python eval.py --base-model-name-path="ContactDoctor/Bio-Medical-Llama-3-8B" --peft-path="peft_10" --batch-size=16 --quantization=4 --datasets=pubmedqa,medmcqa,medqa
```


