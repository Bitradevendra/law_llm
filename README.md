# law_llm

`law_llm` is a Python project for distributed language-model training and local-data fine-tuning.

## Overview

The repository provides training, checkpointing, monitoring, and inference utilities for domain-focused LLM experimentation.

## Project Structure

```text
law_llm/
|-- train.py
|-- trainwithlocaldata.py
|-- generate.py
|-- model.py
|-- distributed_training.py
|-- data_management.py
|-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.8+
- NVIDIA GPU recommended
- DeepSpeed-compatible CUDA and PyTorch environment

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running The Project

Distributed training:

```bash
deepspeed --hostfile hostfile train.py
```

Training on local text files:

```bash
python trainwithlocaldata.py
```

Generation from a checkpoint:

```bash
python generate.py --checkpoint_path ./checkpoints/<checkpoint>.pt
```

## How It Works

- `train.py` is the main distributed training entry point
- `trainwithlocaldata.py` targets local text-based datasets
- `data_management.py` and `distributed_training.py` handle loaders and cluster setup
- `generate.py` runs inference against saved checkpoints
