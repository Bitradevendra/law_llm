import torch
import torch.nn as nn
from model import CheckpointedTransformerBlock

class FullTransformerLM(nn.Module):
    """A full transformer language model with embeddings, stacked transformer blocks, and output head."""
    def __init__(self, vocab_size, dim=768, num_heads=12, num_layers=12, max_seq_len=1024, mlp_ratio=4.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([
            CheckpointedTransformerBlock(dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        bsz, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len, "Input sequence length exceeds model's max_seq_len."
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits 