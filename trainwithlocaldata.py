import os
import glob
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from full_transformer import FullTransformerLM
import deepspeed
from omegaconf import OmegaConf
from logging_config import setup_logging
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from torch.profiler import profile, record_function, ProfilerActivity

# Configurations (can be extended or loaded from YAML)
MODEL_NAME = 'gpt2'  # Use a small open tokenizer for demo; change as needed
DATA_DIR = 'local_data'
CHECKPOINT_DIR = 'checkpoints_local'
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 8
MAX_SEQ_LEN = 512
VOCAB_SIZE = 50257  # GPT-2 default; will be set by tokenizer

class TextFileDataset(Dataset):
    def __init__(self, filepaths, tokenizer, max_length):
        self.samples = []
        for path in filepaths:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = tokenizer(text, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
                self.samples.append(tokens['input_ids'].squeeze(0))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx][:-1], self.samples[idx][1:]  # input, target

def auto_batch_size(num_samples, vram_gb):
    if num_samples < 1000:
        return min(2, DEFAULT_BATCH_SIZE)
    elif vram_gb >= 24:
        return min(32, DEFAULT_BATCH_SIZE * 4)
    elif vram_gb >= 12:
        return DEFAULT_BATCH_SIZE
    else:
        return 2

def main():
    assert torch.cuda.is_available(), "CUDA GPU is required for training!"
    logger = setup_logging()
    logger.info('Starting local data training script.')

    # 1. Find all .txt files
    filepaths = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    assert filepaths, f"No .txt files found in {DATA_DIR}."
    logger.info(f"Found {len(filepaths)} text files for training.")

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_size = len(tokenizer)

    # 3. Build dataset
    dataset = TextFileDataset(filepaths, tokenizer, MAX_SEQ_LEN)
    num_samples = len(dataset)
    logger.info(f"Dataset size: {num_samples} samples.")

    # 4. Auto batch size/epochs
    vram_gb = torch.cuda.get_device_properties(0).total_memory // 2**30
    batch_size = auto_batch_size(num_samples, vram_gb)
    epochs = DEFAULT_EPOCHS if num_samples > 1000 else max(1, DEFAULT_EPOCHS // 2)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 5. Model
    model = FullTransformerLM(vocab_size=vocab_size, dim=768, num_heads=12, num_layers=6, max_seq_len=MAX_SEQ_LEN)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 7. DeepSpeed config
    ds_config = {
        "train_batch_size": batch_size * torch.cuda.device_count(),
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 2},
        "fp16": {"enabled": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 5e-5}}
    }

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 8. DeepSpeed init
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        config=ds_config
    )

    use_wandb = WANDB_AVAILABLE and os.environ.get('WANDB_API_KEY')
    if use_wandb:
        wandb.init(project='llm-local-training', config=ds_config)
        logger.info('WandB logging enabled.')
    else:
        logger.info('WandB not enabled. Set WANDB_API_KEY to enable.')

    # 9. Training loop
    for epoch in range(epochs):
        model_engine.train()
        logger.info(f"Epoch {epoch+1}/{epochs} started.")
        if epoch == 0:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True, profile_memory=True) as prof:
                for step, (inputs, targets) in enumerate(dataloader):
                    with record_function("train_step"):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model_engine(inputs)
                        loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                        model_engine.backward(loss)
                        model_engine.step()
                        if step % 10 == 0 and torch.distributed.get_rank() == 0:
                            logger.info(f"Epoch {epoch+1}/{epochs} Step {step} Loss {loss.item():.4f}")
                            if use_wandb:
                                wandb.log({"loss": loss.item(), "epoch": epoch+1, "step": step})
                    if step > 50:
                        break
            prof.export_chrome_trace(os.path.join(CHECKPOINT_DIR, 'profile_epoch1.json'))
            logger.info('PyTorch profiler trace saved.')
        else:
            for step, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_engine(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                model_engine.backward(loss)
                model_engine.step()
                if step % 10 == 0 and torch.distributed.get_rank() == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} Step {step} Loss {loss.item():.4f}")
                    if use_wandb:
                        wandb.log({"loss": loss.item(), "epoch": epoch+1, "step": step})
        # Save checkpoint after each epoch
        if torch.distributed.get_rank() == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
            torch.save(model_engine.module.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main() 