"""Parallel-section decoding for the no-context anatomy-block model (fluency path L, 2026-10-10).

In `anatomical_nocontext` a block's queries attend only to that block (earlier blocks fix absolute
positions and nothing else), so the 8 sections of a study are independent given the image and can
be denoised together: each section becomes one row of a [B*8, W] batch, cropped to its own slots,
with `pos_offset` restoring its absolute positions in the fixed layout. One model call then advances
all 8 sections, so S steps cost 1 + S calls instead of 8 * (1 + S).

Per row the schedule is denoise_block's default path (chunk=None, no remask/no_repeat/spread):
emptiness decided once by threshold on P(<empty>) + P(<eob>) at slot 0, then confidence-ranked
commits of ceil(remaining / steps_left) slots per step, weak <eob> replaced by the best real token.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from . import data as D
from .diffusion import defer_boundary, extract_block_content
from .generate import _decode_tokens, gen_capacity, word_token_ids

MASK_ID, EMPTY_ID, EOB_ID = D.MASK_ID, D.EMPTY_ID, D.EOB_ID
# Adjacent-repeat ban for the packed decoder (fluency objective, 2026-10-11; test.py --no_adjacent): a masked
# slot may not predict the word token already committed in a neighbouring slot, and of two neighbouring slots
# committing the same word token in one step only the more confident one is kept (the other stays masked).
NO_ADJ = False
# Bullet grammar for the packed decoder (2026-10-11; test.py --struct). Token ids of the project tokenizer:
# "." 27, "\n" 212, "-" 26. After "." only "\n", <eob> or a digit continuation ("4" "." "5"); after "\n" only
# "-"; after "-" only a space-initial content token; before "-" only "\n"; before "." only content. Applied to
# masked slots next to committed tokens, and to neighbouring slots committed in the same step (weaker dropped).
STRUCT = False
# --bullet_conf: report each bullet with the mean/min confidence its tokens were committed at
# (diagnostics["bullets"]), for offline low-confidence bullet pruning (prune_bullets.py)
BULLET_CONF = False
# NO_ADJ = 2 (--no_adjacent2): words are compared case/space-insensitively (" Cardiomediastinal" vs
# " cardiomediastinal" are one word) and also across a period ("contours. contours")
DOT_ID = 27
# COMPACT (--compact): after the emptiness decision, sections decided empty are dropped from every later denoiser
# call (block-diagonal attention: nothing attends to them and their outputs are unused). Same math on ~half the
# positions; the logits are scattered back to the fixed layout so the schedule below is unchanged.
COMPACT = False
_VARIANTS = {}


def _word_groups(tokenizer, V, device):
    """gid [V]: equal for tokens that are the same word up to case/leading space (unique for non-words);
    var [V, G]: all ids of a token's group, padded with the token itself."""
    key = (id(tokenizer), V, str(device))
    if key not in _VARIANTS:
        words = word_token_ids(tokenizer)
        groups, gid = {}, list(range(V, 2 * V))
        for i in range(min(V, tokenizer.get_vocab_size())):
            if i in words and i >= len(D.SPECIAL_IDS):
                k = (tokenizer.decode([i]) or "").strip().lower()
                gid[i] = groups.setdefault(k, (len(groups), []))[0]
                groups[k][1].append(i)
        G = max(len(v) for _, v in groups.values())
        var = torch.arange(V)[:, None].repeat(1, G)
        for _, members in groups.values():
            for i in members:
                var[i, :len(members)] = torch.tensor(members)
        _VARIANTS[key] = (torch.tensor(gid, device=device), var.to(device))
    return _VARIANTS[key]
NL_ID = 212
DOT, NL, DASH = 27, 212, 26


def _struct_masks(tokenizer, V, device):
    n_sp = len(D.SPECIAL_IDS)
    dec = [tokenizer.decode([i]) or "" for i in range(V)]
    content = torch.tensor([i >= n_sp and i not in (DOT, NL, DASH) for i in range(V)], device=device)
    after_dot = torch.tensor([i in (NL, EOB_ID) or (i >= n_sp and dec[i].isdigit()) for i in range(V)], device=device)
    after_nl = torch.zeros(V, dtype=torch.bool, device=device); after_nl[DASH] = True
    after_dash = content & torch.tensor([d.startswith(" ") for d in dec], device=device)
    before_dash = torch.zeros(V, dtype=torch.bool, device=device); before_dash[NL] = True
    return content, after_dot, after_nl, after_dash, before_dash


def _pair_ok(a, b, M):
    """a, b: token tensors (a left of b) -> bool tensor, True if the bigram is allowed by the grammar."""
    content, after_dot, after_nl, after_dash, before_dash = M
    ok = torch.ones_like(a, dtype=torch.bool)
    ok &= (a != DOT) | after_dot[b]
    ok &= (a != NL) | after_nl[b]
    ok &= (a != DASH) | after_dash[b]
    ok &= (b != DASH) | before_dash[a]
    ok &= (b != DOT) | content[a]
    return ok


@torch.no_grad()
def generate_parallel_rows(model, tokenizer, images, view_mask, steps: int = 12,
                      empty_threshold=None, eob_threshold: float | None = None,
                      use_kv_cache: bool = True, collect_p_empty: bool = False) -> list[dict]:
    device = images.device
    b, K = images.shape[0], len(D.CATEGORIES)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    caps = [gen_capacity(c) for c in D.CATEGORIES]          # content slots incl. eob
    starts = [s for s, _ in D.fixed_block_spans()]
    W = 1 + max(caps)
    R = b * K
    kk = torch.arange(R, device=device) % K
    cap_r = torch.tensor(caps, device=device)[kk]            # [R]
    col = torch.arange(W, device=device)[None, :]
    real = col <= cap_r[:, None]                             # cat + cap slots
    content = real & (col >= 1)
    ids = torch.full((R, W), D.PAD_ID, dtype=torch.long, device=device)
    ids[:, 0] = torch.tensor(D.CAT_IDS, device=device)[kk]
    ids[content] = MASK_ID
    eye = torch.eye(W, dtype=torch.bool, device=device)[None]
    allow = (real[:, :, None] & real[:, None, :]) | eye      # pads attend only to themselves
    pad = torch.zeros(R, W, dtype=torch.bool, device=device)
    pos_offset = torch.tensor(starts, device=device)[kk]

    memory, mem_pad = model.encode_images(images, view_mask)
    memory = memory.repeat_interleave(K, dim=0)
    mem_pad = mem_pad.repeat_interleave(K, dim=0)
    memory_kv = model.build_memory_kv(memory) if use_kv_cache else None

    def call(tt):
        return model(images, view_mask, ids, allow, tt, pad, memory, mem_pad,
                     memory_kv=memory_kv, pos_offset=pos_offset).float()

    masked = content.clone()
    n_slots = content.sum(dim=1).float()
    thr = None
    if empty_threshold is not None:
        thr = (torch.tensor(empty_threshold, device=device, dtype=torch.float)[kk]
               if isinstance(empty_threshold, (list, tuple))
               else torch.full((R,), float(empty_threshold), device=device))
    p_empty = None
    if thr is not None or collect_p_empty:
        probs0 = F.softmax(call(torch.ones(R, device=device))[:, 1, :], dim=-1)
        p_empty = probs0[:, EMPTY_ID] + probs0[:, EOB_ID]
    if thr is not None:
        is_empty = p_empty >= thr
        ids[is_empty, 1] = EMPTY_ID
        fill = is_empty[:, None] & content & (col >= 2)
        ids[fill] = EOB_ID
        masked[is_empty] = False
    for step in range(steps):
        if not masked.any():
            break
        tt = masked.sum(dim=1).float() / n_slots
        logits = call(tt)
        logits[..., MASK_ID] = float("-inf")
        if thr is not None:
            logits[:, 1, EMPTY_ID] = float("-inf")
            logits[:, 1, EOB_ID] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        if eob_threshold is not None:
            alt = probs.clone()
            alt[..., EOB_ID] = 0.0
            alt_conf, alt_pred = alt.max(dim=-1)
            weak = (pred == EOB_ID) & (conf < eob_threshold)
            pred = torch.where(weak, alt_pred, pred)
            conf = torch.where(weak, alt_conf, conf)
        conf = defer_boundary(conf, pred, step, steps)
        conf = conf.masked_fill(~masked, -1.0)
        n_unmask = (masked.sum(dim=1).float() / (steps - step)).ceil().long()
        ranks = conf.argsort(dim=1, descending=True).argsort(dim=1)
        commit = masked & (ranks < n_unmask[:, None])
        ids = torch.where(commit, pred, ids)
        masked = masked & ~commit
    if masked.any():
        logits = call(torch.zeros(R, device=device))
        logits[..., MASK_ID] = float("-inf")
        ids = torch.where(masked, logits.argmax(dim=-1), ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - t0) / b

    results = []
    rows = ids.tolist()
    for i in range(b):
        sections, diag = {}, {"missing_terminator": 0, "empty": 0, "unexpected_special": 0,
                              "denoiser_calls": 1 + steps, "latency_s": latency}
        for k, cat in enumerate(D.CATEGORIES):
            r = i * K + k
            block = torch.tensor([rows[r][:1 + caps[k]]])
            ext = extract_block_content(block, EOB_ID, EMPTY_ID)
            c = ext["content"][0]
            if ext["empty"][0]:
                sections[cat] = ""
                diag["empty"] += 1
            else:
                toks = c[:-1] if (ext["terminated"][0] and c and c[-1] == EOB_ID) else c
                diag["unexpected_special"] += sum(1 for t in toks if t < len(D.SPECIAL_IDS))
                sections[cat] = _decode_tokens(tokenizer, toks)
            if not ext["terminated"][0]:
                diag["missing_terminator"] += 1
            if p_empty is not None:
                diag.setdefault("p_empty", {})[cat] = float(p_empty[r])
        results.append({"text": D.serialize_findings(sections), "diagnostics": diag})
    return results


@torch.no_grad()
def generate_parallel(model, tokenizer, images, view_mask, steps: int = 12,
                      empty_threshold=None, eob_threshold: float | None = None,
                      use_kv_cache: bool = True, collect_p_empty: bool = False) -> list[dict]:
    """Packed version of generate_parallel_rows: the 8 sections stay in ONE row per study in the fixed
    layout, with a block-diagonal self-attention mask (block k sees only block k, exactly as in the
    no-context sequential sampler). No padding to the widest section and no 8x copy of the image
    memory, so a step costs one full-length call. Same per-block schedule as generate_parallel_rows."""
    device = images.device
    b, K = images.shape[0], len(D.CATEGORIES)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    spans = D.fixed_block_spans()
    L = spans[-1][1]
    blk = torch.empty(L, dtype=torch.long, device=device)
    for k, (s0, e0) in enumerate(spans):
        assert e0 - s0 == 1 + gen_capacity(D.CATEGORIES[k])
        blk[s0:e0] = k
    starts = torch.tensor([s0 for s0, _ in spans], device=device)
    is_cat = torch.zeros(L, dtype=torch.bool, device=device)
    is_cat[starts] = True
    first = torch.zeros(L, dtype=torch.bool, device=device)      # slot right after <cat_k>
    first[starts + 1] = True
    content = (~is_cat)[None, :].expand(b, L)
    ids = torch.full((b, L), MASK_ID, dtype=torch.long, device=device)
    ids[:, starts] = torch.tensor(D.CAT_IDS, device=device)
    allow = (blk[:, None] == blk[None, :])[None].expand(b, L, L)
    pad = torch.zeros(b, L, dtype=torch.bool, device=device)
    onehot = F.one_hot(blk, K).bool()                             # [L, K]

    memory, mem_pad = model.encode_images(images, view_mask)
    memory_kv = model.build_memory_kv(memory) if use_kv_cache else None

    cidx = None

    def call(tt_blk):                                            # tt_blk: [b, K] -> per-position t
        if cidx is None:
            return model(images, view_mask, ids, allow, tt_blk[:, blk], pad, memory, mem_pad,
                         memory_kv=memory_kv).float()
        lg = model(images, view_mask, ids.gather(1, cidx), allow_c, tt_blk.gather(1, blk_c.clamp(max=K - 1)),
                   pad_c, memory, mem_pad, memory_kv=memory_kv, pos_offset=pos_c).float()
        out = lg.new_zeros(b, L, lg.shape[-1])
        return out.scatter_(1, cidx[..., None].expand(-1, -1, lg.shape[-1]), lg)

    def per_block(x):                                            # [b, L] bool -> [b, K] counts
        return (x[:, :, None] & onehot[None]).sum(dim=1).float()

    masked = content.clone()
    n_slots = per_block(content)
    cconf = torch.zeros(b, L, device=device)
    if STRUCT:
        M = _struct_masks(tokenizer, model.decoder.tok_embed.num_embeddings, device)
        content_m, after_dot, after_nl, after_dash, before_dash = M
    if NO_ADJ:
        wtok = torch.zeros(model.decoder.tok_embed.num_embeddings, dtype=torch.bool, device=device)
        wtok[list(word_token_ids(tokenizer))] = True
        wtok[:len(D.SPECIAL_IDS)] = False
        if NO_ADJ >= 2:
            gid, var = _word_groups(tokenizer, wtok.shape[0], device)
    p_empty = None
    thr = None
    if empty_threshold is not None:
        thr = (torch.tensor(empty_threshold, device=device, dtype=torch.float)
               if isinstance(empty_threshold, (list, tuple))
               else torch.full((K,), float(empty_threshold), device=device))
    if thr is not None or collect_p_empty:
        probs0 = F.softmax(call(torch.ones(b, K, device=device))[:, starts + 1, :], dim=-1)
        p_empty = probs0[..., EMPTY_ID] + probs0[..., EOB_ID]   # [b, K]
    if thr is not None:
        is_empty = p_empty >= thr[None, :]                       # [b, K]
        e_pos = is_empty[:, blk] & content                       # [b, L]
        ids = torch.where(e_pos & first[None], torch.full_like(ids, EMPTY_ID), ids)
        ids = torch.where(e_pos & ~first[None], torch.full_like(ids, EOB_ID), ids)
        masked = masked & ~e_pos
        if COMPACT:
            keep = ~is_empty[:, blk]                             # [b, L]: slots of non-empty sections
            n_keep = keep.sum(dim=1)
            Lk = int(n_keep.max())
            col = torch.arange(Lk, device=device)[None, :]
            cidx = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)[:, :Lk]   # kept slots first, in order
            valid = col < n_keep[:, None]
            blk_c = torch.where(valid, blk[cidx], K + col)       # padding columns: own block, attend to self
            allow_c = blk_c[:, :, None] == blk_c[:, None, :]
            pad_c = torch.zeros(b, Lk, dtype=torch.bool, device=device)
            pos_c = cidx - col
    for step in range(steps):
        if not masked.any():
            break
        tt = per_block(masked) / n_slots
        logits = call(tt)
        logits[..., MASK_ID] = float("-inf")
        if thr is not None:
            logits[:, starts + 1, EMPTY_ID] = float("-inf")
            logits[:, starts + 1, EOB_ID] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        if NO_ADJ:
            nbs = [torch.roll(ids, 1, dims=1), torch.roll(ids, -1, dims=1)]
            if NO_ADJ >= 2:
                nbs += [torch.where(nbs[0] == DOT_ID, torch.roll(ids, 2, dims=1), MASK_ID),
                        torch.where(nbs[1] == DOT_ID, torch.roll(ids, -2, dims=1), MASK_ID)]
            for nb in nbs:
                ban = (masked & wtok[nb])[..., None]
                idx = var[nb] if NO_ADJ >= 2 else nb[..., None]
                g = probs.gather(-1, idx)
                probs.scatter_(-1, idx, torch.where(ban, torch.zeros_like(g), g))
        if STRUCT:
            left, right = torch.roll(ids, 1, dims=1), torch.roll(ids, -1, dims=1)
            allowed = torch.ones_like(probs, dtype=torch.bool)
            for cond, m in (((left == DOT), after_dot), ((left == NL), after_nl), ((left == DASH), after_dash),
                            ((right == DASH), before_dash), ((right == DOT), content_m)):
                c = cond & masked
                allowed &= ~c[..., None] | m[None, None, :]
            constrained = probs * allowed
            has_mass = constrained.sum(dim=-1, keepdim=True) > 0
            probs = torch.where(has_mass, constrained, probs)
        conf, pred = probs.max(dim=-1)
        if eob_threshold is not None:
            alt = probs.clone()
            alt[..., EOB_ID] = 0.0
            alt_conf, alt_pred = alt.max(dim=-1)
            weak = (pred == EOB_ID) & (conf < eob_threshold)
            pred = torch.where(weak, alt_pred, pred)
            conf = torch.where(weak, alt_conf, conf)
        conf0 = conf
        conf = defer_boundary(conf, pred, step, steps)
        conf = conf.masked_fill(~masked, -1.0)
        n_unmask = (per_block(masked) / (steps - step)).ceil().long()   # [b, K]
        commit = torch.zeros_like(masked)
        for k, (s0, e0) in enumerate(spans):
            ranks = conf[:, s0:e0].argsort(dim=1, descending=True).argsort(dim=1)
            commit[:, s0:e0] = masked[:, s0:e0] & (ranks < n_unmask[:, k:k + 1])
        if NO_ADJ:
            same = (gid[pred[:, :-1]] == gid[pred[:, 1:]]) if NO_ADJ >= 2 else (pred[:, :-1] == pred[:, 1:])
            clash = commit[:, :-1] & commit[:, 1:] & same & wtok[pred[:, :-1]]
            right_weaker = conf[:, 1:] <= conf[:, :-1]
            commit[:, 1:] &= ~(clash & right_weaker)
            commit[:, :-1] &= ~(clash & ~right_weaker)
            if NO_ADJ >= 2:                                      # w . w committed in the same step
                new = torch.where(commit, pred, ids)
                c2 = (commit[:, :-2] & commit[:, 2:] & (gid[pred[:, :-2]] == gid[pred[:, 2:]])
                      & wtok[pred[:, :-2]] & (new[:, 1:-1] == DOT_ID))
                rw = conf[:, 2:] <= conf[:, :-2]
                commit[:, 2:] &= ~(c2 & rw)
                commit[:, :-2] &= ~(c2 & ~rw)
        if STRUCT:
            bad = commit[:, :-1] & commit[:, 1:] & ~_pair_ok(pred[:, :-1], pred[:, 1:], M)
            right_weaker = conf[:, 1:] <= conf[:, :-1]
            commit[:, 1:] &= ~(bad & right_weaker)
            commit[:, :-1] &= ~(bad & ~right_weaker)
        ids = torch.where(commit, pred, ids)
        cconf = torch.where(commit, conf0, cconf)
        masked = masked & ~commit
    if masked.any():
        logits = call(torch.zeros(b, K, device=device))
        logits[..., MASK_ID] = float("-inf")
        lc, lp = F.softmax(logits, dim=-1).max(dim=-1)
        ids = torch.where(masked, lp, ids)
        cconf = torch.where(masked, lc, cconf)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = (time.time() - t0) / b
    crow = cconf.tolist() if BULLET_CONF else None

    results = []
    rows = ids.tolist()
    for i in range(b):
        sections, diag = {}, {"missing_terminator": 0, "empty": 0, "unexpected_special": 0,
                              "denoiser_calls": 1 + steps, "latency_s": latency}
        for k, cat in enumerate(D.CATEGORIES):
            s0, e0 = spans[k]
            ext = extract_block_content(torch.tensor([rows[i][s0:e0]]), EOB_ID, EMPTY_ID)
            c = ext["content"][0]
            if ext["empty"][0]:
                sections[cat] = ""
                diag["empty"] += 1
            else:
                toks = c[:-1] if (ext["terminated"][0] and c and c[-1] == EOB_ID) else c
                diag["unexpected_special"] += sum(1 for t in toks if t < len(D.SPECIAL_IDS))
                sections[cat] = _decode_tokens(tokenizer, toks)
                if BULLET_CONF:
                    # content starts at s0 + 1 and is contiguous; split on newline tokens
                    cf = crow[i][s0 + 1:s0 + 1 + len(toks)]
                    out, cur, cc = [], [], []
                    for t, c_ in list(zip(toks, cf)) + [(NL_ID, None)]:
                        if t == NL_ID:
                            txt = _decode_tokens(tokenizer, cur) if cur else ""
                            if txt.strip():
                                out.append([txt, sum(cc) / len(cc), min(cc)])
                            cur, cc = [], []
                        else:
                            cur.append(t)
                            cc.append(c_)
                    diag.setdefault("bullets", {})[cat] = out
            if not ext["terminated"][0]:
                diag["missing_terminator"] += 1
            if p_empty is not None:
                diag.setdefault("p_empty", {})[cat] = float(p_empty[i, k])
        results.append({"text": D.serialize_findings(sections), "diagnostics": diag})
    return results
