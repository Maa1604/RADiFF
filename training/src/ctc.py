"""Alignment-aware (CTC) anatomy-block denoising, fluency path G (2026-10-10).

Why: with plain masked denoising each slot must predict the token at an exact position, so two
adjacent masked slots both hedge over where the phrase sits ("No pleural effusion" vs "No large
pleural effusion") and both commit "effusion"; separators get committed early and cut sentences.
CTC (Graves et al. 2006) lets every slot emit a blank, and merges adjacent identical emissions, so the
position of each word inside the block is a latent alignment rather than a target. Iterative
refinement follows the Imputer (Saharia et al. 2020): during training a random fraction of slots is
revealed with the model's own best (Viterbi) alignment, and the CTC loss is computed over alignments
consistent with the revealed slots; at inference slots are committed by confidence, then collapsed.

Blank = <pad> (never a real content token). Empty sections keep the original targets
(<empty> at slot 0, <eob> fillers), so the per-category emission thresholds work unchanged.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from . import data as D
from .generate import _decode_tokens, gen_capacity

BLANK = D.PAD_ID
NEG = -1e9


@torch.no_grad()
def ctc_viterbi(lp: torch.Tensor, tgt: torch.Tensor, tgt_len: torch.Tensor) -> torch.Tensor:
    """Best CTC alignment. lp [B,T,V] log-probs, tgt [B,N] (padded), tgt_len [B].
    Returns [B,T] symbols (BLANK or token). Rows with no valid alignment get all BLANK."""
    b, t_len, _ = lp.shape
    n = tgt.shape[1]
    s_len = 2 * n + 1
    dev = lp.device
    ext = torch.full((b, s_len), BLANK, dtype=torch.long, device=dev)
    ext[:, 1::2] = tgt
    valid_s = torch.arange(s_len, device=dev)[None, :] < (2 * tgt_len[:, None] + 1)
    emit = lp.gather(2, ext[:, None, :].expand(b, t_len, s_len))          # [B,T,S]
    emit = emit.masked_fill(~valid_s[:, None, :], NEG)
    skip_ok = torch.zeros(b, s_len, dtype=torch.bool, device=dev)
    skip_ok[:, 2:] = (ext[:, 2:] != BLANK) & (ext[:, 2:] != ext[:, :-2])
    dp = torch.full((b, s_len), NEG, device=dev)
    dp[:, 0] = emit[:, 0, 0]
    dp[:, 1] = emit[:, 0, 1]
    back = torch.zeros(b, t_len, s_len, dtype=torch.long, device=dev)
    for t in range(1, t_len):
        stay = dp
        step = torch.cat([torch.full((b, 1), NEG, device=dev), dp[:, :-1]], dim=1)
        skip = torch.cat([torch.full((b, 2), NEG, device=dev), dp[:, :-2]], dim=1)
        skip = skip.masked_fill(~skip_ok, NEG)
        cand = torch.stack([stay, step, skip], dim=-1)                    # [B,S,3]
        best, arg = cand.max(dim=-1)
        dp = best + emit[:, t]
        back[:, t] = arg
    last = 2 * tgt_len                                                     # final blank state
    ar = torch.arange(b, device=dev)
    end_blank = dp[ar, last]
    end_tok = dp[ar, (last - 1).clamp(min=0)]
    s = torch.where(end_tok > end_blank, last - 1, last)
    ok = torch.maximum(end_blank, end_tok) > NEG / 2
    path = torch.zeros(b, t_len, dtype=torch.long, device=dev)
    for t in range(t_len - 1, -1, -1):
        path[:, t] = ext[ar, s.clamp(0, s_len - 1)]
        if t > 0:
            s = s - back[ar, t, s.clamp(0, s_len - 1)]
    path[~ok] = BLANK
    return path


def collapse(symbols: list[int]) -> list[int]:
    """CTC collapse: merge adjacent repeats, then drop blanks."""
    out, prev = [], None
    for s in symbols:
        if s != prev and s != BLANK:
            out.append(s)
        prev = s
    return out


def ctc_block_loss(model, images, view_mask, ids, region, filler, allow, pad, ctx_ids, ctx_pad,
                   t, filler_weight: float):
    """Loss for one active block in the fixed layout (shapes as in train.compute_loss).
    Present sections: Imputer-style CTC over the block slots. Empty sections: original CE."""
    b, l = ids.shape
    dev = ids.device
    slot_idx = region.nonzero(as_tuple=False)                              # rows sorted by (b, pos)
    t_len = int(region.sum(dim=1).max())
    pos = slot_idx[:, 1].view(b, t_len)                                    # every row has T slots
    tok = ids.gather(1, pos)                                               # [B,T] reference layout
    is_empty = tok[:, 0] == D.EMPTY_ID
    content = (tok != D.EOB_ID) & ~filler.gather(1, pos) & ~is_empty[:, None]
    tgt_len = content.sum(dim=1)
    n_max = max(1, int(tgt_len.max()))
    tgt = torch.full((b, n_max), BLANK, dtype=torch.long, device=dev)
    ar_n = torch.arange(n_max, device=dev)[None, :]
    tgt[ar_n < tgt_len[:, None]] = tok[content]

    # 1) model's own best alignment from the all-masked state (no grad)
    with torch.no_grad():
        full = ids.masked_fill(region, D.MASK_ID)
        lg0 = model(images, view_mask, full, allow, torch.ones(b, device=dev), pad,
                    ctx_ids=ctx_ids, ctx_pad=ctx_pad).float()
        lp0 = F.log_softmax(lg0.gather(1, pos[:, :, None].expand(b, t_len, lg0.shape[-1])), dim=-1)
        align = ctc_viterbi(lp0, tgt, tgt_len)                             # [B,T]
    # 2) reveal a random (1 - t) fraction of slots with that alignment
    masked = torch.rand(b, t_len, device=dev) < t[:, None]
    slot_in = torch.where(masked, torch.full_like(align, D.MASK_ID), align)
    slot_in = torch.where(is_empty[:, None], torch.where(masked, torch.full_like(tok, D.MASK_ID), tok),
                          slot_in)
    inp = ids.clone()
    inp.scatter_(1, pos, slot_in)
    tt = masked.float().mean(dim=1)
    logits = model(images, view_mask, inp, allow, tt, pad, ctx_ids=ctx_ids, ctx_pad=ctx_pad).float()
    lg = logits.gather(1, pos[:, :, None].expand(b, t_len, logits.shape[-1]))
    lg[..., D.MASK_ID] = NEG
    lp = F.log_softmax(lg, dim=-1)

    losses = []
    pres = ~is_empty & (tgt_len > 0)
    if pres.any():
        # revealed slots are fixed to their aligned symbol (log-prob 0, everything else -inf)
        onehot = torch.full_like(lp, NEG).scatter_(2, align[:, :, None], 0.0)
        lp_c = torch.where(masked[:, :, None], lp, onehot)[pres]
        ctc = F.ctc_loss(lp_c.transpose(0, 1), tgt[pres], torch.full((int(pres.sum()),), t_len,
                         device=dev, dtype=torch.long), tgt_len[pres], blank=BLANK,
                         reduction="none", zero_infinity=True)
        ctc = ctc / tgt_len[pres].float()
        # emission: a present section must not look empty at slot 0
        p_e = lp[pres, 0, D.EMPTY_ID].exp() + lp[pres, 0, D.EOB_ID].exp()
        emit = -torch.log1p(-p_e.clamp(max=1 - 1e-6)) * masked[pres, 0].float()
        losses.append(ctc + emit)
    if is_empty.any():
        ce = F.cross_entropy(lg[is_empty].reshape(-1, lg.shape[-1]), tok[is_empty].reshape(-1),
                             reduction="none").view(-1, t_len)
        w = masked[is_empty].float() * torch.where(filler.gather(1, pos)[is_empty], filler_weight, 1.0)
        losses.append((ce * w).sum(dim=1) / w.sum(dim=1).clamp(min=1))
    return torch.cat(losses).mean()


@torch.no_grad()
def generate_anatomical_ctc(model, tokenizer, images, view_mask, steps: int = 3,
                            empty_threshold=None, use_kv_cache: bool = True,
                            collect_p_empty: bool = False) -> list[dict]:
    """Sequential over the 8 blocks (as generate_anatomical); inside a block: emptiness by
    threshold at slot 0 (t=1), then `steps` confidence-ranked commits over blank+content symbols,
    then CTC collapse. The collapsed block is re-laid canonically (content, <eob> fill) as context."""
    dev = images.device
    b = images.shape[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    memory, mem_pad = model.encode_images(images, view_mask)
    memory_kv = model.build_memory_kv(memory) if use_kv_cache else None
    prefix = torch.zeros(b, 0, dtype=torch.long, device=dev)
    sections = [dict() for _ in range(b)]
    diags = [{"missing_terminator": 0, "empty": 0, "unexpected_special": 0, "denoiser_calls": 0}
             for _ in range(b)]
    n_special = len(D.SPECIAL_IDS)
    for k, cat in enumerate(D.CATEGORIES):
        cap = gen_capacity(cat)
        p = prefix.shape[1]
        blk = torch.full((b, cap), D.MASK_ID, dtype=torch.long, device=dev)
        ids = torch.cat([prefix, torch.full((b, 1), D.CAT_IDS[k], device=dev, dtype=torch.long), blk], 1)
        l = ids.shape[1]
        allow = torch.ones(b, l, l, dtype=torch.bool, device=dev)
        pad = torch.zeros(b, l, dtype=torch.bool, device=dev)

        def call(tt):
            return model(images, view_mask, ids, allow, tt, pad, memory, mem_pad,
                         memory_kv=memory_kv)[:, p + 1:, :].float()

        thr = empty_threshold[k] if isinstance(empty_threshold, (list, tuple)) else empty_threshold
        is_empty = torch.zeros(b, dtype=torch.bool, device=dev)
        pe = None
        if thr is not None or collect_p_empty:
            pr0 = F.softmax(call(torch.ones(b, device=dev))[:, 0], dim=-1)
            pe = pr0[:, D.EMPTY_ID] + pr0[:, D.EOB_ID]
            if thr is not None:
                is_empty = pe >= thr
        masked = ~is_empty[:, None].expand(b, cap).clone()
        for step in range(steps):
            if not masked.any():
                break
            lg = call(masked.float().mean(dim=1))
            blank_logit = lg[..., BLANK].clone()
            lg[..., :n_special] = NEG                     # only blank + content symbols
            lg[..., BLANK] = blank_logit
            pr = F.softmax(lg, dim=-1)
            conf, pred = pr.max(dim=-1)
            conf = conf.masked_fill(~masked, -1.0)
            n_un = (masked.sum(dim=1).float() / (steps - step)).ceil().long()
            ranks = conf.argsort(dim=1, descending=True).argsort(dim=1)
            commit = masked & (ranks < n_un[:, None])
            blk_ids = ids[:, p + 1:]
            ids[:, p + 1:] = torch.where(commit, pred, blk_ids)
            masked &= ~commit
        rows = ids[:, p + 1:].tolist()
        new_blk = torch.full((b, cap + 1), D.EOB_ID, dtype=torch.long, device=dev)
        new_blk[:, 0] = D.CAT_IDS[k]
        for i in range(b):
            diags[i]["denoiser_calls"] += steps
            if pe is not None:
                diags[i].setdefault("p_empty", {})[cat] = float(pe[i])
            if is_empty[i]:
                sections[i][cat] = ""
                diags[i]["empty"] += 1
                new_blk[i, 1] = D.EMPTY_ID
                continue
            toks = collapse(rows[i])[:cap - 1]
            if not toks:
                sections[i][cat] = ""
                new_blk[i, 1] = D.EMPTY_ID
                continue
            sections[i][cat] = _decode_tokens(tokenizer, toks)
            new_blk[i, 1:1 + len(toks)] = torch.tensor(toks, device=dev)
        prefix = torch.cat([prefix, new_blk], dim=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    lat = (time.time() - t0) / b
    out = []
    for i in range(b):
        diags[i]["latency_s"] = lat
        out.append({"text": D.serialize_findings(sections[i]), "diagnostics": diags[i]})
    return out
