import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

class MemoryEfficientAttention(nn.Module):
    """Implements a memory-efficient attention mechanism using chunking."""
    def __init__(self, dim, num_heads, chunk_size=1024):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.chunked_attention(q, k, v)

    def chunked_attention(self, q, k, v):
        """Process attention in chunks to reduce memory usage."""
        batch_size, seq_len, _ = q.shape
        output_chunks = []

        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, i:end_idx]

            # Compute attention for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)

            attn_weights = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn_weights, v)

            output_chunks.append(chunk_output)

        concatenated_output = torch.cat(output_chunks, dim=1)
        return self.out_proj(concatenated_output)

class CheckpointedTransformerBlock(nn.Module):
    """A transformer block that uses gradient checkpointing to save memory."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.attention = MemoryEfficientAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Custom forward function for checkpointing that handles the custom attention arguments
        def custom_attention_forward(inp):
            return self.attention(inp)

        def custom_mlp_forward(inp):
            return self.mlp(inp)

        # Use gradient checkpointing for memory efficiency during training
        if self.training:
            x = x + checkpoint(custom_attention_forward, self.norm1(x))
            x = x + checkpoint(custom_mlp_forward, self.norm2(x))
        else:
            x = x + self.attention(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x
