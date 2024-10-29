import torch
from torch import nn, einsum


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim**-0.5
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        return self.emb(pos) * self.scale


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        self.register_buffer("inv_freq", theta**-freq_seq, persistent=False)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale
