from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum, nn

from models.pos_emb import AbsolutePositionalEmbedding, ScaledSinusoidalEmbedding
from models.rel_attn import (
    RelativeAttention,
    RelativeStructureAttention,
    SimpleRelativeAttention,
    SimpleRelativeStructureAttention,
)
from models.hook import ForHook


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def always_zero(*args, **kwargs):
    return 0


@dataclass
class LayerIntermediates:
    cached_kvs: Optional[List[Tensor]] = None
    hidden_states: Optional[List[Tensor]] = None


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ff_inner_dim, bias=True),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_inner_dim, cfg.dim, bias=True),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads

        self.to_q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_k = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.to_v = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.attn_hook = ForHook()
        self.normattn_hook = ForHook()
        self.to_out = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def create_causal_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def forward(self, x, return_cached_kv=False, cache=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, query length (=1 when generation), key length)
        d - head dimension (dim // heads)
        """
        h = self.heads

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.to_k(x), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.to_v(x), "b n (h d) -> b h n d", h=h)

        if cache is not None:
            ck, cv = cache
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        if return_cached_kv:
            cached_kv = (k, v)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * (q.shape[-1] ** -0.5)

        if q.shape[-2] != 1:
            causal_mask = self.create_causal_mask(
                dots.shape[-2], dots.shape[-1], device=q.device
            )
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = F.softmax(dots, dim=-1)
        _ = self.attn_hook(attn=attn)
        _ = self.normattn_hook(attn=attn, value=v)
        
        attn = self.dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        if not return_cached_kv:
            return out, None

        return out, cached_kv


class DecTransformerLayer(nn.Module):
    def __init__(self, cfg, max_seq_len=1024):
        super().__init__()
        self.norm1 = LayerNorm(cfg.dim)
        self.rel_music = False
        srel = cfg.simple_rel if hasattr(cfg, "simple_rel") else False
        if cfg.rel_time or cfg.rel_pitch:
            if srel:
                self.attn = SimpleRelativeStructureAttention(
                    cfg=cfg, max_seq_len=max_seq_len
                )
            else:
                self.attn = RelativeStructureAttention(cfg=cfg, max_seq_len=max_seq_len)
            self.rel_music = True
        elif cfg.rel_idx:
            if srel:
                self.attn = SimpleRelativeAttention(cfg=cfg, max_seq_len=max_seq_len)
            else:
                self.attn = RelativeAttention(cfg=cfg, max_seq_len=max_seq_len)
        else:
            self.attn = Attention(cfg=cfg)
        self.attn_only = cfg.attn_only if hasattr(cfg, "attn_only") else False
        if not self.attn_only:
            self.norm2 = LayerNorm(cfg.dim)
            self.ff = FeedForward(cfg=cfg)

    def forward(self, x, rel_structure_matrix={}, cache=None, return_cached_kv=False):
        attn_kwargs = {"cache": cache, "return_cached_kv": return_cached_kv}
        if self.rel_music:
            attn_kwargs["rel_structure_matrix"] = rel_structure_matrix

        x_new, attn_cache = self.attn(self.norm1(x), **attn_kwargs)
        x = x + x_new
        if self.attn_only:
            return x, attn_cache

        x_new = self.ff(self.norm2(x))
        x = x + x_new
        return x, attn_cache


class DecTransformer(nn.Module):
    def __init__(self, cfg, pad_value=0, max_seq_len=1024, num_tokens=1):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, cfg.dim, padding_idx=pad_value)
        if cfg.pos_enc == "learnable":
            self.pos_emb = AbsolutePositionalEmbedding(cfg.dim, max_seq_len)
        elif cfg.pos_enc == "sinusoidal":
            self.pos_emb = ScaledSinusoidalEmbedding(cfg.dim)
        elif cfg.pos_enc == "None":
            self.pos_emb = always_zero
        else:
            raise ValueError("Invalid positional encoding type")

        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList(
            [
                DecTransformerLayer(cfg=cfg, max_seq_len=max_seq_len)
                for _ in range(cfg.depth)
            ]
        )
        self.final_norm = LayerNorm(cfg.dim)
        self.to_logits = nn.Linear(cfg.dim, num_tokens)

    def forward(
        self, x, rel_structure_matrix=None, return_intermediates=False, cache=None
    ):
        # absolute positional embedding
        x = self.token_emb(x) + self.pos_emb(x)
        x = self.emb_dropout(x)

        # cached key / values
        cached_kvs = []
        if cache is not None:
            x = x[:, -1:]
            cached_kvs = cache.cached_kvs
        iter_attn_cache = iter(cached_kvs)

        cached_kvs = []
        hidden_states = []
        for layer in self.layers:
            if return_intermediates:
                hidden_states.append(x)

            x, cached_kv = layer(
                x,
                rel_structure_matrix=rel_structure_matrix,
                cache=next(iter_attn_cache, None),
                return_cached_kv=return_intermediates,
            )

            if return_intermediates:
                cached_kvs.append(cached_kv)

        if return_intermediates:
            hidden_states.append(x)

        x = self.final_norm(x)
        logits = self.to_logits(x)

        if return_intermediates:
            new_caches = LayerIntermediates(
                cached_kvs=cached_kvs, hidden_states=hidden_states
            )
            return logits, new_caches
        return logits, None


if __name__ == "__main__":
    pass
