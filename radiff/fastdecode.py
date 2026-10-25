"""Inference-only decoder that reuses the image memory's cross-attention K/V.

Numerically equivalent to nn.TransformerDecoder(nn.TransformerDecoderLayer(..., norm_first=True))
and loads the SAME state_dict, so no retraining or checkpoint conversion is needed.

Why: the visual memory is 2 x 1369 = 2738 tokens and is constant for a whole study, but its
per-layer W_k/W_v projections were being recomputed on every one of the 96 denoiser calls per
report (8 blocks x 12 steps). Projecting it once per study removes ~39 GFLOP/call of pure
recomputation.

EVAL ONLY: dropout is omitted (identity at inference). Never use this for training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _split_heads(x: torch.Tensor, nhead: int) -> torch.Tensor:
    b, l, d = x.shape
    return x.view(b, l, nhead, d // nhead).transpose(1, 2)  # [B, H, L, hd]


class CachedDecoderLayer(nn.Module):
    """Mirrors nn.TransformerDecoderLayer(norm_first=True) at inference."""

    def __init__(self, layer: nn.TransformerDecoderLayer, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.self_attn = layer.self_attn
        self.multihead_attn = layer.multihead_attn
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.norm1, self.norm2, self.norm3 = layer.norm1, layer.norm2, layer.norm3

    def memory_kv(self, memory: torch.Tensor):
        """Project the (constant) memory into cross-attention K/V once. [B,H,S,hd] each."""
        w, bias = self.multihead_attn.in_proj_weight, self.multihead_attn.in_proj_bias
        d = memory.shape[-1]
        k = F.linear(memory, w[d:2 * d], bias[d:2 * d] if bias is not None else None)
        v = F.linear(memory, w[2 * d:], bias[2 * d:] if bias is not None else None)
        return _split_heads(k, self.nhead), _split_heads(v, self.nhead)

    def _attn(self, q, k, v, attn_mask, key_padding_mask, out_proj):
        # q/k/v: [B,H,L,hd] / [B,H,S,hd]; attn_mask bool True=blocked, [B*H,L,S] or [L,S]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if attn_mask is not None:
            m = attn_mask
            if m.dim() == 2:
                m = m[None, None]
            else:
                m = m.view(q.shape[0], self.nhead, m.shape[-2], m.shape[-1])
            scores = scores.masked_fill(m, float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ v  # [B,H,L,hd]
        b, h, l, hd = out.shape
        out = out.transpose(1, 2).reshape(b, l, h * hd)
        return F.linear(out, out_proj.weight, out_proj.bias)

    def forward(self, x, mem_k, mem_v, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        h = self.norm1(x)
        w, bias = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        d = h.shape[-1]
        q = _split_heads(F.linear(h, w[:d], bias[:d] if bias is not None else None), self.nhead)
        k = _split_heads(F.linear(h, w[d:2 * d], bias[d:2 * d] if bias is not None else None), self.nhead)
        v = _split_heads(F.linear(h, w[2 * d:], bias[2 * d:] if bias is not None else None), self.nhead)
        x = x + self._attn(q, k, v, tgt_mask, tgt_key_padding_mask, self.self_attn.out_proj)

        h = self.norm2(x)
        wq, bq = self.multihead_attn.in_proj_weight[:d], self.multihead_attn.in_proj_bias
        q = _split_heads(F.linear(h, wq, bq[:d] if bq is not None else None), self.nhead)
        x = x + self._attn(q, mem_k, mem_v, None, memory_key_padding_mask,
                           self.multihead_attn.out_proj)

        h = self.norm3(x)
        return x + self.linear2(F.gelu(self.linear1(h)))


class CachedDecoder(nn.Module):
    """Drop-in for nn.TransformerDecoder at inference, with per-study memory K/V reuse."""

    def __init__(self, decoder: nn.TransformerDecoder, nhead: int):
        super().__init__()
        self.layers = nn.ModuleList([CachedDecoderLayer(l, nhead) for l in decoder.layers])

    def build_memory_kv(self, memory: torch.Tensor):
        """Call ONCE per study; pass the result to every forward for that study."""
        return [l.memory_kv(memory) for l in self.layers]

    def forward(self, x, memory_kv, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        for layer, (mk, mv) in zip(self.layers, memory_kv):
            x = layer(x, mk, mv, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return x
