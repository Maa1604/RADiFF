"""Visual encoder (frozen rad-dino) + shared text decoder. See SPEC.md §2.

All variants (AR, fixed-block, anatomical diffusion) share this module; they
differ only in attention masks, block boundaries, and whether t-conditioning
is used (AR runs with t fixed to 0).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 518  # rad-dino: resize shortest edge + center crop (source JPEGs are 512)
MAX_VIEWS = 2
D_MODEL = 768
N_HEAD = 8
N_LAYER = 6
D_FFN = 3072
MAX_LEN = 384
DROPOUT = 0.1
# microsoft/rad-dino snapshot the released weights were trained and evaluated with.
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"


class SinusoidalTimeEmbed(nn.Module):
    """Continuous noise-level embedding t in [0, 1] -> d_model."""

    def __init__(self, dim: int = 256, out_dim: int = D_MODEL):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class VisualEncoder(nn.Module):
    """Frozen microsoft/rad-dino. Returns projected patch tokens per view.

    forward(images, view_mask) -> (tokens [B, V*P, d_model], memory_key_padding_mask [B, V*P])
    images: [B, V, 1, IMG_SIZE, IMG_SIZE] float in [0, 1]; V == MAX_VIEWS.
    view_mask: [B, V] bool, True = padded (non-existent) view.
    """

    def __init__(self, d_model: int = D_MODEL, max_views: int = MAX_VIEWS,
                 dtype: torch.dtype = torch.bfloat16, unfreeze: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.max_views = max_views
        # A trainable ViT is kept in fp32: this codebase uses no autocast/GradScaler, and
        # AdamW on bf16 parameters without master weights loses small updates entirely.
        self.unfreeze = unfreeze
        self.dtype = torch.float32 if unfreeze else dtype
        vit = AutoModel.from_pretrained("microsoft/rad-dino", revision=RAD_DINO_REVISION)
        vit.requires_grad_(unfreeze)
        if not unfreeze:
            vit.eval()
        self.vit = vit
        d_v = vit.config.hidden_size
        self.proj = nn.Linear(d_v, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.view_embed = nn.Embedding(max_views, d_model)

    @staticmethod
    def normalize(images: torch.Tensor) -> torch.Tensor:
        """[B,V,1,H,W] in [0,1] -> [B*V,3,H,W] with rad-dino grayscale stats."""
        mean = torch.tensor([0.5307] * 3, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2583] * 3, device=images.device).view(1, 3, 1, 1)
        x = images.repeat(1, 1, 3, 1, 1)
        x = (x - mean) / std
        return x.reshape(-1, 3, images.shape[-2], images.shape[-1])

    def encode(self, images: torch.Tensor, view_mask: torch.Tensor):
        """ViT forward. images [B,V,1,H,W], view_mask [B,V] (True=pad).

        Grad flows only when built with unfreeze=True AND the caller has not already
        disabled it. Every generation path runs under torch.no_grad, so inference keeps
        the previous no-grad behaviour whether or not the encoder is trainable.
        """
        b, v = images.shape[:2]
        with torch.set_grad_enabled(self.unfreeze and torch.is_grad_enabled()):
            return self._encode(images)

    def _encode(self, images: torch.Tensor):
        b, v = images.shape[:2]
        x = self.normalize(images).to(self.dtype)
        feats = self.vit(pixel_values=x).last_hidden_state  # [B*V, P(+1), d_v]
        feats = feats[:, 1:, :]  # drop CLS; keep patch tokens (SPEC §2)
        return feats.to(torch.float32).view(b, v, feats.shape[1], feats.shape[2])

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor):
        feats = self.encode(images, view_mask)  # [B, V, P, d_v]
        tok = self.proj(feats)  # [B, V, P, d_model]
        tok = tok + self.view_embed.weight[None, :, None, :]
        tok = self.norm(tok)
        b, v, p, d = tok.shape
        tokens = tok.reshape(b, v * p, d)
        mem_pad = view_mask[:, :, None].expand(b, v, p).reshape(b, v * p)  # True=pad
        return tokens, mem_pad


class ContextEncoder(nn.Module):
    """Embeds the clinical context into the SAME cross-attention memory as the visual tokens.

    Why memory and not the decoder's input sequence: memory is built once per study and its
    per-layer K/V is cached (build_memory_kv), so context costs one projection per report rather
    than lengthening self-attention on all 24 denoiser calls. Latency stays essentially unchanged,
    which matters because speed is the headline claim.

    `tok_embed` is shared with the decoder so a word means the same vector whether it arrives as
    context or is generated as output; only the position table, a single learned "this is text,
    not pixels" bias, and a LayerNorm are new (~0.1M params).
    """

    def __init__(self, tok_embed: nn.Embedding, d_model: int, max_len: int):
        super().__init__()
        self.tok_embed = tok_embed
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.type_embed = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        b, l = ctx_ids.shape
        pos = torch.arange(l, device=ctx_ids.device)[None, :].expand(b, l)
        x = self.tok_embed(ctx_ids) + self.pos_embed(pos) + self.type_embed
        return self.norm(x)


class TextDecoder(nn.Module):
    """Pre-norm transformer decoder with t-conditioning and image cross-attention."""

    def __init__(self, vocab_size: int = 8192, d_model: int = D_MODEL,
                 nhead: int = N_HEAD, num_layers: int = N_LAYER,
                 dim_ffn: int = D_FFN, max_len: int = MAX_LEN, dropout: float = DROPOUT):
        super().__init__()
        self.num_heads = nhead
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.time_embed = SinusoidalTimeEmbed(dim=256, out_dim=d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self._fast = None  # lazily-built inference-only cached decoder (src/fastdecode.py)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def build_memory_kv(self, memory: torch.Tensor):
        """Project the constant visual memory into per-layer cross-attention K/V ONCE.

        Inference only. The result is valid for one study and can be reused across every
        block and denoising step; see src/fastdecode.py for why this matters.
        """
        if self._fast is None:
            from .fastdecode import CachedDecoder
            self._fast = CachedDecoder(self.decoder, self.num_heads)
        return self._fast.build_memory_kv(memory)

    def forward(self, tgt_ids: torch.Tensor, attn_allow: torch.Tensor, t: torch.Tensor,
                memory: torch.Tensor, tgt_key_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor, memory_kv=None) -> torch.Tensor:
        """
        tgt_ids: [B, L] long.
        attn_allow: [B, L, L] bool, True = attention allowed (self-attention).
        t: [B] float in [0, 1] noise level.
        memory: [B, S, d_model] visual tokens.
        tgt_key_padding_mask / memory_key_padding_mask: [B, L] / [B, S] bool, True = pad.
        returns logits [B, L, vocab_size].
        """
        b, l = tgt_ids.shape
        pos = torch.arange(l, device=tgt_ids.device)[None, :].expand(b, l)
        x = self.tok_embed(tgt_ids) + self.pos_embed(pos) + self.time_embed(t)[:, None, :]
        # nn.MultiheadAttention bool mask: True = blocked; 3D mask must be
        # [B*num_heads, L, L] with identical mask per head.
        attn_mask = (~attn_allow).repeat_interleave(self.num_heads, dim=0)
        if memory_kv is not None:
            h = self._fast(x, memory_kv, attn_mask, tgt_key_padding_mask,
                           memory_key_padding_mask)
        else:
            h = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.lm_head(self.final_norm(h))


class RRGModel(nn.Module):
    """Full model: frozen visual encoder + shared text decoder."""

    def __init__(self, vocab_size: int = 8192, unfreeze_encoder: bool = False,
                 use_context: bool = False, max_context_len: int = 96, **dec_kwargs):
        super().__init__()
        d_model = dec_kwargs.get("d_model", D_MODEL)
        self.visual = VisualEncoder(d_model=d_model, unfreeze=unfreeze_encoder)
        self.decoder = TextDecoder(vocab_size=vocab_size, **dec_kwargs)
        self.use_context = use_context
        # Constructed only when requested, so an image-only model has no extra parameters and
        # existing checkpoints load with strict=True.
        self.context = (ContextEncoder(self.decoder.tok_embed, d_model, max_context_len)
                        if use_context else None)

    def encode_images(self, images: torch.Tensor, view_mask: torch.Tensor,
                      ctx_ids: torch.Tensor | None = None,
                      ctx_pad: torch.Tensor | None = None):
        """Visual tokens for a batch of studies: (memory, memory_key_padding_mask).

        The encoder is frozen and its output does not depend on the text being denoised, so
        generation should compute this ONCE per study and reuse it for every block and every
        denoising step - otherwise rad-dino is re-run 32 times per report (8 blocks x 4 steps)
        over ~2,738 visual tokens, which dominates inference cost and makes latency independent
        of batch size (measured: 0.598 / 0.678 / 0.748 s per study at batch 1 / 8 / 32).
        """
        memory, mem_pad = self.visual(images, view_mask)
        if self.context is not None and ctx_ids is not None:
            ctx = self.context(ctx_ids)
            memory = torch.cat([memory, ctx], dim=1)
            mem_pad = torch.cat([mem_pad, ctx_pad], dim=1)
        return memory, mem_pad

    def build_memory_kv(self, memory: torch.Tensor):
        """Per-study cross-attention K/V cache; pass to forward()/generation as memory_kv."""
        return self.decoder.build_memory_kv(memory)

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor,
                tgt_ids: torch.Tensor, attn_allow: torch.Tensor, t: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor, memory=None, mem_pad=None,
                memory_kv=None, ctx_ids=None, ctx_pad=None) -> torch.Tensor:
        """`memory`/`mem_pad` may be supplied from a previous encode_images() call to skip the
        visual encoder; `images` is then ignored. Training passes neither (one forward per
        batch, so there is nothing to reuse)."""
        if memory is None:
            memory, mem_pad = self.encode_images(images, view_mask, ctx_ids, ctx_pad)
        return self.decoder(tgt_ids, attn_allow, t, memory,
                            tgt_key_padding_mask, mem_pad, memory_kv=memory_kv)


def causal_allow(length: int, device) -> torch.Tensor:
    """[L, L] bool, query i attends to keys j <= i."""
    return torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))


def full_allow(length: int, device) -> torch.Tensor:
    return torch.ones(length, length, dtype=torch.bool, device=device)
