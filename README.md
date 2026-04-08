# Law LLM

A serious-minded LLM experimentation scaffold for distributed training, local-data fine-tuning, and checkpoint-based generation.

## Why This Repo Exists

`law_llm` is built like the early skeleton of a domain-specialized language model stack. It is structured for iteration: train, fine-tune, merge, monitor, checkpoint, and generate.

## What It Does

- runs distributed training workflows
- supports local text corpus training through `local_data/`
- exposes checkpoint-based inference and model-merging utilities
- includes monitoring and training support modules

## Project Structure

```text
law_llm/
|-- train.py
|-- trainwithlocaldata.py
|-- generate.py
|-- model.py
|-- data_management.py
|-- distributed_training.py
|-- monitoring.py
|-- requirements.txt
|-- local_data/
`-- README.md
```

## Requirements

- Python 3.8+
- NVIDIA GPU recommended
- DeepSpeed-compatible CUDA and PyTorch setup

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

Distributed training:

```bash
deepspeed --hostfile hostfile train.py
```

Train from local text files:

```bash
python trainwithlocaldata.py
```

Generate from a checkpoint:

```bash
python generate.py --checkpoint_path ./checkpoints/<checkpoint>.pt
```

## How It Works

- `train.py` is the distributed training entry point.
- `trainwithlocaldata.py` focuses on local text-based corpora.
- `data_management.py` and `distributed_training.py` handle data and execution setup.
- `model.py` and related modules define the model and optimization flow.
- `generate.py` turns saved checkpoints into usable inference output.
