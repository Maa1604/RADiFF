"""Full-report generation for all training variants. SPEC.md Sections 4-5.

Shared grammar for every variant's flat token sequence: [cat_k] content... [eob]
repeated over the 8 categories in canonical order, optionally trailing [eor].
`anatomical`/`anatomical_nocontext` generate block-by-block using
`diffusion.denoise_block` (exact block boundaries known). `ar` and `fixed`
produce a single flat token sequence (causal decode / positional-chunk
diffusion respectively) which is parsed back into per-category content by
scanning for <cat_k> markers, tolerant of missing terminators.
"""

from __future__ import annotations

import math
import re
import time

import torch
import torch.nn.functional as F

from . import data as D
from .diffusion import (N_SPECIAL, commit_no_repeat, denoise_block, extract_block_content,
                        make_attention_mask, pick_slots)
from .model import causal_allow

FIXED_CHUNK = 32
# Matches the anatomical variant's theoretical max serialized length: 8 * (1 cat + 1 eob)
# + sum(DEFAULT_CAPACITIES) + 1 eor = 16 + 345 + 1 = 362. Used so `fixed` gets the same
# total token budget as `anatomical` for a controlled comparison (SPEC.md recommendation).
FIXED_TOTAL_LEN = 362


_ALPHA_IDS: dict = {}


# --bullet_conf for the sequential sampler (2026-10-11): diagnostics["bullets"][cat] = [[text, mean, min], ...] with
# the probability each token was committed at (same format as src/parallel.py, for prune_bullets.py)
BULLET_CONF = False
NL_TOK = 212


def word_token_ids(tokenizer) -> frozenset:
    """Ids whose surface form contains a letter: the tokens --no_repeat treats as words."""
    key = id(tokenizer)
    if key not in _ALPHA_IDS:
        _ALPHA_IDS[key] = frozenset(i for i in range(tokenizer.get_vocab_size())
                                    if re.search(r"[A-Za-z]", tokenizer.decode([i]) or ""))
    return _ALPHA_IDS[key]


def gen_capacity(cat: str) -> int:
    """denoise_block's `capacity` arg counts content slots INCLUDING the eob slot;
    data.DEFAULT_CAPACITIES is a content-token-ONLY cap (matches tokenize_report's
    truncation, which never counts eob against the cap). So the generation slot
    budget per category is the content cap plus one slot for eob.

    Verified against tokenize_report: a block's real span length is
    1 (cat) + min(n, cap) + 1 (eob) <= 1 + (cap + 1), i.e. exactly the slot count
    denoise_block needs (1 clean cat prompt + `cap+1` generated slots).
    """
    return D.DEFAULT_CAPACITIES[cat] + 1


_REPEATED_PUNCT = re.compile(r"([.;:,])\1{1,}")
_MERGED_FINDINGS = re.compile(r"([.;:])(?=[A-Z])")


def _decode_tokens(tokenizer, ids: list[int]) -> str:
    """Decode content tokens and normalize the surface form.

    The model sometimes omits the newline token that separates findings inside a block, so two
    findings arrive as "No focal consolidation.No pneumothorax." A sentence-ending punctuation
    mark directly followed by a capital letter is unambiguously a bullet boundary here (the
    references never contain ".X" without a space), so the newline is restored; runs of repeated
    punctuation ("effusion..") are collapsed. Both are surface normalizations of the same kind
    as detokenization - they never add, drop or reorder content.
    """
    ids = [t for t in ids if t >= len(D.SPECIAL_IDS)]  # drop any special ids defensively
    if not ids:
        return ""
    text = tokenizer.decode(ids)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return _MERGED_FINDINGS.sub(r"\1\n", text)


def parse_generated_ids(tokenizer, ids: list[int]) -> tuple[dict, dict]:
    """Parse a flat generated token sequence into {category: content_text}.

    Tolerant of missing <eob>/<eor>, garbage before the first <cat_k> marker, and
    duplicate <cat_k> markers (first occurrence wins, matching canonical-order
    training). Categories with no marker at all are left absent (empty section).
    """
    cat_positions = [(i, D.CATEGORIES[D.CAT_IDS.index(tok)]) for i, tok in enumerate(ids) if tok in D.CAT_IDS]
    sections: dict[str, str] = {}
    diag = {
        "missing_terminator": 0, "empty": 0, "unexpected_special": 0,
        "unrecognized_prefix_tokens": cat_positions[0][0] if cat_positions else len(ids),
    }
    for idx, (pos, cat) in enumerate(cat_positions):
        end = cat_positions[idx + 1][0] if idx + 1 < len(cat_positions) else len(ids)
        content_ids = ids[pos + 1:end]
        if D.EOR_ID in content_ids:
            content_ids = content_ids[:content_ids.index(D.EOR_ID)]
        terminated = D.EOB_ID in content_ids
        if terminated:
            content_ids = content_ids[:content_ids.index(D.EOB_ID)]
        else:
            diag["missing_terminator"] += 1
        if cat in sections:
            continue  # keep first occurrence only
        if content_ids and content_ids[0] == D.EMPTY_ID:
            sections[cat] = ""
            diag["empty"] += 1
            continue
        diag["unexpected_special"] += sum(1 for t in content_ids if t < len(D.SPECIAL_IDS))
        sections[cat] = _decode_tokens(tokenizer, content_ids)
    return sections, diag


@torch.no_grad()
def generate_ar(model, images, view_mask, max_len: int = FIXED_TOTAL_LEN, temperature: float = 0.0,
                cat_bias=None) -> torch.Tensor:
    """Greedy (or sampled if temperature>0) autoregressive decode. t is fixed at 0
    throughout per SPEC.md Section 2. Starts from <cat_0> as the start signal (no BOS)."""
    b = images.shape[0]
    device = images.device
    ids = torch.full((b, 1), D.CAT_IDS[0], dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    memory, mem_pad = model.encode_images(images, view_mask)  # constant across decoding steps
    for _ in range(max_len - 1):
        l = ids.shape[1]
        allow = causal_allow(l, device)[None].expand(b, l, l)
        t = torch.zeros(b, device=device)
        pad = torch.zeros(b, l, dtype=torch.bool, device=device)
        logits = model(images, view_mask, ids, allow, t, pad, memory, mem_pad)
        next_logits = logits[:, -1, :]
        if cat_bias is not None:
            # Per-category emission control: a category is present iff its <cat_k> marker is
            # emitted (see parse_generated_ids). Biasing those logits is the AR/fixed analogue
            # of anatomical's per-category empty threshold.
            for ci, cid in enumerate(D.CAT_IDS):
                next_logits[:, cid] = next_logits[:, cid] + cat_bias[ci]
        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
        else:
            nxt = next_logits.argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, D.PAD_ID), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        finished = finished | (nxt == D.EOR_ID)
        if bool(finished.all()):
            break
    return ids


@torch.no_grad()
def generate_fixed(model, images, view_mask, steps: int = 12, sample: bool = False,
                   total_len: int = FIXED_TOTAL_LEN, chunk: int = FIXED_CHUNK,
                   cat_bias=None, no_repeat: bool = False,
                   spread: bool = False) -> tuple[torch.Tensor, dict]:
    """Positional-chunk diffusion generation. Position 0 is the deterministic
    <cat_0> start signal (clean prompt, matching training's position-0 token);
    every later chunk is generated fully masked, conditioned bidirectionally on
    all previously generated (clean) positions. SPEC.md: block boundaries are
    positional, not semantic; same diffusion machinery as anatomical."""
    b = images.shape[0]
    device = images.device
    ids = torch.full((b, 1), D.CAT_IDS[0], dtype=torch.long, device=device)
    memory, mem_pad = model.encode_images(images, view_mask)
    n_chunks = math.ceil((total_len - 1) / chunk)
    denoiser_calls = 0
    for _ in range(n_chunks):
        remaining = total_len - ids.shape[1]
        w = min(chunk, remaining)
        if w <= 0:
            break
        ids = torch.cat([ids, torch.full((b, w), D.MASK_ID, dtype=torch.long, device=device)], dim=1)
        p = ids.shape[1] - w
        l = ids.shape[1]
        allow = torch.ones(b, l, l, dtype=torch.bool, device=device)  # full prefix+active bidirectional
        pad = torch.zeros(b, l, dtype=torch.bool, device=device)
        masked = torch.ones(b, w, dtype=torch.bool, device=device)
        for step in range(steps):
            tt = masked.float().mean(dim=1)  # actual masked fraction, as in training
            logits = model(images, view_mask, ids, allow, tt, pad, memory, mem_pad)[:, p:, :]
            logits[..., D.MASK_ID] = float("-inf")
            if cat_bias is not None:
                for ci, cid in enumerate(D.CAT_IDS):
                    logits[..., cid] = logits[..., cid] + cat_bias[ci]
            probs = F.softmax(logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)
            # See src/diffusion.py's denoise_block for why this must be a LOW sentinel
            # (excludes already-revealed slots from the ranking) not a high one.
            conf = conf.masked_fill(~masked, -1.0)
            if sample:
                g = torch.distributions.Categorical(probs).sample()
                pred = torch.where(masked, g, pred)
            n_unmask = (masked.sum(dim=1).float() / max(steps - step, 1)).ceil().long().clamp(min=1)
            if not (no_repeat or spread):
                rank = conf.argsort(dim=1, descending=True)
                for i in range(b):
                    k = int(n_unmask[i])
                    chosen = rank[i, :k]
                    ids[i, p + chosen] = pred[i, chosen]
                    masked[i, chosen] = False
            else:
                top = probs.topk(8, dim=-1).indices.tolist() if no_repeat else None
                pred_l = pred.tolist()
                for i in range(b):
                    k = min(int(n_unmask[i]), int(masked[i].sum()))
                    if k == 0:
                        continue
                    chosen = pick_slots(conf[i], masked[i], k, spread)
                    if no_repeat:
                        row = ids[i, p:].tolist()
                        toks = commit_no_repeat(row, chosen, pred_l[i], top[i], N_SPECIAL, no_repeat)
                    else:
                        toks = [pred_l[i][j] for j in chosen]
                    ci = torch.tensor(chosen, device=device)
                    ids[i, p + ci] = torch.tensor(toks, device=device)
                    masked[i, ci] = False
            denoiser_calls += 1
        if masked.any():
            tt = torch.zeros(b, device=device)
            logits = model(images, view_mask, ids, allow, tt, pad, memory, mem_pad)[:, p:, :]
            logits[..., D.MASK_ID] = float("-inf")
            pred = logits.argmax(dim=-1)
            leftover = masked.nonzero(as_tuple=True)
            ids[leftover[0], p + leftover[1]] = pred[leftover]
    return ids, {"denoiser_calls": denoiser_calls}


@torch.no_grad()
def canonicalize_block(block_ids: torch.Tensor) -> torch.Tensor:
    """[B, 1 + slots] generated block (leading <cat>) -> the training layout: every slot
    after the first <eob> becomes <eob> filler; an <empty> block becomes
    [cat, <empty>, <eob>, ...]; a block with no <eob> gets one in its last slot (a
    capacity-truncated block in training ends exactly that way)."""
    out = block_ids.clone()
    slots = out[:, 1:]
    n = slots.shape[1]
    is_eob = slots == D.EOB_ID
    last = torch.full((slots.shape[0],), n - 1, dtype=torch.long, device=slots.device)
    first = torch.where(is_eob.any(dim=1), is_eob.long().argmax(dim=1), last)
    first = torch.where(slots[:, 0] == D.EMPTY_ID, torch.ones_like(first), first)
    after = torch.arange(n, device=slots.device)[None, :] >= first[:, None]
    slots[after] = D.EOB_ID
    return out


@torch.no_grad()
def generate_anatomical(model, tokenizer, images, view_mask, variant: str,
                        steps: int = 12, sample: bool = False, chunk: int | None = None,
                        empty_threshold: float | list[float] | None = None,
                        eob_threshold: float | None = None,
                        collect_p_empty: bool = False,
                        use_kv_cache: bool = False,
                        remask_threshold: float | None = None,
                        remask_reserve: int = 2,
                        ctx_ids=None, ctx_pad=None,
                        no_repeat: bool = False, spread: bool = False,
                        revise: bool = False, bigram=None, bigram_lambda: float = 0.0) -> list[dict]:
    """variant in {"anatomical", "anatomical_nocontext"}.

    Blocks are generated into the fixed-capacity layout used in training
    (data.fixed_layout): block k always sits at the same absolute positions. Each finished
    block is canonicalized (everything after its first <eob> becomes <eob> filler) before
    it becomes context, so later blocks condition on exactly the kind of prefix they were
    trained on. `anatomical_nocontext` also receives the prefix, but only to place block k
    at its trained position - its queries are not allowed to attend to it.
    """
    b = images.shape[0]
    device = images.device
    # The frozen encoder's output is the same for every block and every denoising step, so it is
    # computed once per study here instead of ~32 times inside model.forward.
    # ctx_ids/ctx_pad are the clinical context (indication/technique/comparison); they join the
    # cross-attention memory here, so everything downstream - including the K/V cache - is
    # unchanged. Pass None for an image-only model.
    memory, mem_pad = model.encode_images(images, view_mask, ctx_ids, ctx_pad)
    # The visual memory is constant for the study, so its per-layer cross-attention K/V can be
    # projected once instead of on each of the ~96 denoiser calls. Output-identical; see
    # src/fastdecode.py.
    memory_kv = model.build_memory_kv(memory) if use_kv_cache else None
    use_context = variant != "anatomical_nocontext"
    prefix = torch.zeros(b, 0, dtype=torch.long, device=device)
    sections_per_ex: list[dict] = [dict() for _ in range(b)]
    diag_per_ex = [{"missing_terminator": 0, "empty": 0, "unexpected_special": 0, "denoiser_calls": 0}
                  for _ in range(b)]
    total_latency = 0.0
    for k, cat in enumerate(D.CATEGORIES):
        cap = gen_capacity(cat)
        block_prefix = prefix if prefix.shape[1] else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = denoise_block(model, images, view_mask, block_prefix, cat_id=D.CAT_IDS[k],
                            capacity=cap, steps=steps, sample=sample, attend_prefix=use_context,
                            chunk=chunk,
                            empty_threshold=(empty_threshold[k]
                                             if isinstance(empty_threshold, (list, tuple))
                                             else empty_threshold),
                            eob_threshold=eob_threshold, collect_p_empty=collect_p_empty,
                            memory=memory, mem_pad=mem_pad, memory_kv=memory_kv,
                            remask_threshold=remask_threshold,
                            remask_reserve=remask_reserve,
                            no_repeat=no_repeat, spread=spread,
                            revise=revise, bigram=bigram, bigram_lambda=bigram_lambda)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_latency += time.time() - t0
        if out.get("p_empty") is not None:
            for i in range(b):
                diag_per_ex[i].setdefault("p_empty", {})[cat] = float(out["p_empty"][i])
        block_ids = out["ids"][:, out["block_start"]:out["block_end"]]
        ext = extract_block_content(block_ids, D.EOB_ID, D.EMPTY_ID)
        for i in range(b):
            content = ext["content"][i]
            diag_per_ex[i]["denoiser_calls"] += steps
            if ext["empty"][i]:
                sections_per_ex[i][cat] = ""
                diag_per_ex[i]["empty"] += 1
            else:
                toks = content[:-1] if (ext["terminated"][i] and content and content[-1] == D.EOB_ID) else content
                diag_per_ex[i]["unexpected_special"] += sum(1 for t in toks if t < len(D.SPECIAL_IDS))
                sections_per_ex[i][cat] = _decode_tokens(tokenizer, toks)
                if BULLET_CONF:
                    cf = out["cconf"][i, :len(toks)].tolist()   # content starts right after <cat>
                    bl, cur, cc = [], [], []
                    for t, c_ in list(zip(toks, cf)) + [(NL_TOK, None)]:
                        if t == NL_TOK:
                            txt = _decode_tokens(tokenizer, cur) if cur else ""
                            if txt.strip():
                                bl.append([txt, sum(cc) / len(cc), min(cc)])
                            cur, cc = [], []
                        else:
                            cur.append(t)
                            cc.append(c_)
                    diag_per_ex[i].setdefault("bullets", {})[cat] = bl
            if not ext["terminated"][i]:
                diag_per_ex[i]["missing_terminator"] += 1
        prefix = torch.cat([prefix, canonicalize_block(block_ids)], dim=1)
    results = []
    for i in range(b):
        text = D.serialize_findings(sections_per_ex[i])
        diag_per_ex[i]["latency_s"] = total_latency / b
        results.append({"text": text, "diagnostics": diag_per_ex[i]})
    return results


@torch.no_grad()
def generate_batch(model, tokenizer, images, view_mask, variant: str,
                   steps: int = 12, sample: bool = False, chunk: int | None = None,
                   empty_threshold: float | list[float] | None = None,
                   eob_threshold: float | None = None,
                   collect_p_empty: bool = False,
                   use_kv_cache: bool = False,
                   cat_bias=None,
                   remask_threshold: float | None = None,
                   remask_reserve: int = 2,
                   ctx_ids=None, ctx_pad=None,
                   no_repeat: bool = False, spread: bool = False,
                   revise: bool = False, bigram=None, bigram_lambda: float = 0.0) -> list[dict]:
    """Top-level entry point. Returns [{"text": findings_section_str, "diagnostics": {...}}, ...]."""
    no_repeat = word_token_ids(tokenizer) if no_repeat else None
    if variant == "ar":
        t0 = time.time()
        ids = generate_ar(model, images, view_mask, cat_bias=cat_bias)
        latency = (time.time() - t0) / images.shape[0]
        results = []
        for row in ids.tolist():
            if D.EOR_ID in row:
                row = row[:row.index(D.EOR_ID)]
            sections, diag = parse_generated_ids(tokenizer, row)
            diag["latency_s"] = latency
            diag["denoiser_calls"] = 0  # AR has no denoiser steps
            results.append({"text": D.serialize_findings(sections), "diagnostics": diag})
        return results
    if variant == "fixed":
        t0 = time.time()
        ids, gdiag = generate_fixed(model, images, view_mask, steps=steps, sample=sample,
                                    cat_bias=cat_bias, no_repeat=no_repeat, spread=spread)
        latency = (time.time() - t0) / images.shape[0]
        results = []
        for row in ids.tolist():
            if D.EOR_ID in row:
                row = row[:row.index(D.EOR_ID)]
            sections, diag = parse_generated_ids(tokenizer, row)
            diag.update(gdiag)
            diag["latency_s"] = latency
            results.append({"text": D.serialize_findings(sections), "diagnostics": diag})
        return results
    if variant in ("anatomical", "anatomical_nocontext"):
        return generate_anatomical(model, tokenizer, images, view_mask, variant, steps=steps,
                                   sample=sample, chunk=chunk,
                                   empty_threshold=empty_threshold, eob_threshold=eob_threshold,
                                   collect_p_empty=collect_p_empty, use_kv_cache=use_kv_cache,
                                   remask_threshold=remask_threshold,
                                   remask_reserve=remask_reserve,
                                   ctx_ids=ctx_ids, ctx_pad=ctx_pad,
                                   no_repeat=no_repeat, spread=spread,
                                   revise=revise, bigram=bigram, bigram_lambda=bigram_lambda)
    raise ValueError(f"unknown variant: {variant!r}")
