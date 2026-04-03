# law_llm

`law_llm` is a Python project for distributed language-model training and local-data fine-tuning.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Use

Distributed training:

```bash
deepspeed --hostfile hostfile train.py
```

Local-data training:

```bash
python trainwithlocaldata.py
```

Generation:

```bash
python generate.py --checkpoint_path ./checkpoints/<checkpoint>.pt
```

## How It Works

- `train.py` is the main training entry point
- `data_management.py` and `distributed_training.py` handle distributed execution
- `generate.py` runs inference from saved checkpoints
