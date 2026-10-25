"""Masked-token discrete diffusion: corruption, block masks, loss, sampler. SPEC.md §3-4.

Objective: absorbing-state masked diffusion. Sample t ~ U[0,1]; each token of the
active block is replaced by <mask> independently with probability t (t=1 allowed:
fully masked block). Loss = token cross-entropy at corrupted positions only,
per-sample mean over corrupted positions, then mean over the batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import EMPTY_ID, EOB_ID, MASK_ID, PAD_ID, SPECIAL_IDS
from .model import causal_allow

# Adjacent-repeat ban for the sequential block sampler (2026-10-11; test.py --no_adjacent2 without
# --parallel_blocks): (wtok [V] bool, gid [V], var [V, G]) from src.parallel._word_groups, or None = off.
# Same rule as parallel.NO_ADJ = 2: a masked slot may not predict the word (case/space-insensitive) committed
# next to it or across a period, and of two neighbours committing the same word in one step the weaker waits.
ADJ_BAN = None
DOT_TOK = 27
# Bullet grammar for the sequential sampler (test.py --struct without --parallel_blocks): the masks of
# src.parallel._struct_masks, or None = off. Same rule as parallel.STRUCT.
STRUCT_M = None
# Commit schedule (test.py --sched a,b,...): fraction of the still-masked slots committed at each step (last step
# commits the rest); None = the linear ceil(n / steps_left) schedule.
SCHED = None

VARIANTS = ("ar", "anatomical", "fixed", "anatomical_nocontext")
N_SPECIAL = len(SPECIAL_IDS)  # ids below this are special tokens


def sample_t(b: int, device) -> torch.Tensor:
    """[B] float, U[0,1]."""
    return torch.rand(b, device=device)


def corrupt(ids: torch.Tensor, block_spans, active: torch.Tensor, t: torch.Tensor):
    """Mask tokens inside the active block.

    ids: [B, L] long. block_spans: list of (start, end) tuples per sample (len K).
    active: [B] long block index. t: [B] float mask probability.
    Returns (corrupted_ids [B, L], corr_mask [B, L] bool True=corrupted).
    Special tokens inside the block (cat_k, eob) are maskable: the model must
    learn terminator and empty-section placement.
    """
    b, l = ids.shape
    device = ids.device
    corr_mask = torch.zeros(b, l, dtype=torch.bool, device=device)
    for i in range(b):
        s, e = block_spans[i][int(active[i])]
        probs = torch.rand(e - s, device=device)
        corr_mask[i, s:e] = probs < t[i]
    corrupted = ids.clone()
    corrupted[corr_mask] = MASK_ID
    return corrupted, corr_mask


def make_attention_mask(variant: str, block_spans, active: torch.Tensor,
                        length: int) -> torch.Tensor:
    """[B, L, L] bool self-attention allow-mask (True = allowed).

    anatomical: queries attend keys in blocks <= active (clean prefix visible,
    future hidden). nocontext: only the active block (plus nothing else).
    fixed: same rule over positional chunks. ar: causal (handled by caller via
    causal_allow; this function returns causal for completeness).
    """
    b = active.shape[0]
    device = active.device
    if variant == "ar":
        return causal_allow(length, device)[None].expand(b, length, length).clone()

    allow = torch.zeros(b, length, length, dtype=torch.bool, device=device)
    for i in range(b):
        a = int(active[i])
        if variant == "anatomical_nocontext":
            vis_lo, vis_hi = a, a + 1
        else:  # anatomical / fixed: prefix + active visible
            vis_lo, vis_hi = 0, a + 1
        key_idx = []
        for j in range(vis_lo, vis_hi):
            s, e = block_spans[i][j]
            key_idx.append(torch.arange(s, e, device=device))
        keys = torch.cat(key_idx)
        allow[i][:, keys] = True
    return allow


def denoising_loss(logits: torch.Tensor, target_ids: torch.Tensor,
                   corr_mask: torch.Tensor) -> torch.Tensor:
    """Mean CE over corrupted positions: per-sample mean, then batch mean."""
    loss_flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction="none",
        ignore_index=PAD_ID,
    ).reshape(logits.shape[:2])
    per_sample = (loss_flat * corr_mask).sum(dim=1) / corr_mask.sum(dim=1).clamp(min=1)
    return per_sample.mean()


def pick_slots(conf_row: torch.Tensor, cand_row: torch.Tensor, k: int, spread: bool) -> list[int]:
    """Choose which k candidate slots to commit this step, by confidence.

    spread=False is the original rule (top-k by confidence). spread=True skips a slot whose
    neighbour was already picked this step, so adjacent slots are never committed in the same
    parallel step and each sees the other's token before deciding; if too few non-adjacent
    candidates exist, the skipped ones fill the remainder in confidence order, so the number of
    commits per step (and hence the schedule) is unchanged.
    """
    rank = conf_row.argsort(descending=True).tolist()
    if not spread:
        return rank[:k]
    cand = cand_row.tolist()
    picked, skipped, blocked = [], [], set()
    for j in rank:
        if not cand[j]:
            continue
        if j in blocked:
            skipped.append(j)
            continue
        picked.append(j)
        blocked.update((j - 1, j + 1))
        if len(picked) == k:
            break
    if len(picked) < k:
        picked += skipped[:k - len(picked)]
    return picked


def commit_no_repeat(row: list[int], chosen: list[int], pred_row: list[int],
                     top_row: list[list[int]], n_special: int, alpha_ids,
                     window: int = 2) -> list[int]:
    """Tokens to write at `chosen` (processed in the given confidence order) with no local
    repetition. A word token (one in `alpha_ids`) may not equal any token within `window` slots
    on either side, which also catches the punctuation-interleaved form "X . X"; any other
    content token (punctuation, newline, bullet) may not equal an adjacent token. `row` is the
    block's current content slots and is updated in place, so slots committed earlier in the same
    step count as neighbours. A colliding prediction is replaced by the model's best-ranked
    content token that does not collide. Special tokens (<eob> filler, <empty>, <mask>) are
    never constrained or used as replacements. window=2 keeps legitimate phrasing such as
    "right atrium and right ventricle" (distance 3) allowed."""
    n = len(row)

    def collides(j, t):
        if t < n_special:
            return False
        w = window if t in alpha_ids else 1
        return any(row[x] == t for x in range(max(0, j - w), min(n, j + w + 1)) if x != j)

    out = []
    for j in chosen:
        t = pred_row[j]
        if collides(j, t):
            for c in top_row[j]:
                if c >= n_special and not collides(j, c):
                    t = c
                    break
        row[j] = t
        out.append(t)
    return out

def viterbi_tokens(chosen: torch.Tensor, cand: torch.Tensor, logu: torch.Tensor, row: torch.Tensor,
                   cat_tok: int, bigram: torch.Tensor, lam: float) -> torch.Tensor:
    """Option B: jointly choose the tokens for the slots committed in one step.

    chosen: [n] slot indices, sorted. cand/logu: [C, K] each slot's K best tokens and their log-probs
    (after the <eob> threshold rule). row: [C] the block's current content slots (chosen slots are
    still <mask>). bigram: [V, V] log P(next | prev) from training text. Maximises
        sum_j logu_j(y_j) + lam * sum log P(y_{j+1} | y_j)
    over adjacent pairs among the chosen slots plus the boundary pairs with already-committed
    neighbours (slot 0's left neighbour is the block's <cat_k> token). Non-adjacent chosen slots are
    independent. lam = 0 reduces to per-slot argmax, i.e. the original commit."""
    n, C = chosen.numel(), row.numel()
    c = cand[chosen]                                   # [n, K]
    u = logu[chosen].float().clone()                   # [n, K]
    prev = chosen - 1
    lt = torch.where(prev >= 0, row[prev.clamp(min=0)], torch.full_like(prev, cat_tok))
    lval = (prev < 0) | (lt != MASK_ID)
    u += lam * torch.where(lval[:, None], bigram[lt[:, None], c].float(), torch.zeros_like(u))
    nxt = chosen + 1
    rt = row[nxt.clamp(max=C - 1)]
    rval = (nxt < C) & (rt != MASK_ID)
    u += lam * torch.where(rval[:, None], bigram[c, rt[:, None]].float(), torch.zeros_like(u))
    if n > 1:
        pair = lam * bigram[c[:-1].unsqueeze(2), c[1:].unsqueeze(1)].float()   # [n-1, K, K]
        pair = pair * (chosen[1:] == chosen[:-1] + 1).float()[:, None, None]
        pair = pair.cpu().numpy()
    uu = u.cpu().numpy()
    score, back = uu[0], []
    for t in range(1, n):
        m = score[:, None] + pair[t - 1]
        back.append(m.argmax(axis=0))
        score = m.max(axis=0) + uu[t]
    best = int(score.argmax()); path = [best]
    for bp in reversed(back):
        best = int(bp[best]); path.append(best)
    path.reverse()
    return c[torch.arange(n, device=c.device), torch.tensor(path, device=c.device)]


# Boundary deferral (fluency objective, 2026-10-10): before the last step, the commit confidence of a
# "." or newline prediction is multiplied by DEFER_ALPHA, so frequent sentence-boundary tokens stop
# winning the early confidence ranking and cutting phrases short. None = off (original sampler).
DEFER_ALPHA = None
DEFER_IDS = (27, 212)  # "." and "\n" in the project tokenizer


def defer_boundary(conf, pred, step, steps):
    if DEFER_ALPHA is None or step >= steps - 1:
        return conf
    hit = (pred == DEFER_IDS[0]) | (pred == DEFER_IDS[1])
    return torch.where(hit, conf * DEFER_ALPHA, conf)


@torch.no_grad()
def denoise_block(model, images, view_mask, prefix_ids: torch.Tensor,
                  cat_id: int, capacity: int, steps: int,
                  sample: bool = False, attend_prefix: bool = True,
                  chunk: int | None = None, empty_threshold: float | None = None,
                  eob_threshold: float | None = None, collect_p_empty: bool = False,
                  memory=None, mem_pad=None, memory_kv=None,
                  remask_threshold: float | None = None,
                  remask_reserve: int = 2,
                  no_repeat=None, spread: bool = False,
                  revise: bool = False, bigram=None, bigram_lambda: float = 0.0) -> dict:
    """Iteratively denoise one block. SPEC §4.

    The block occupies `capacity` slots after prefix_ids: [cat_id] slot0 slot1 ...
    slot_{capacity-2}; slot0..capacity-2 are all <mask> initially; the leading
    cat_id is a clean prompt. Slots = 1 (cat) + capacity - 1 content/eob slots;
    capacity counts content tokens incl. eob (see data.capacities).

    Unmasking: confidence-ranked parallel unmasking; t condition linear 1 -> 0.
    no_repeat: None/empty, or the set of word-token ids (see commit_no_repeat).
    Returns dict with ids for the block content (list of token ids up to and
    including first EOB, or truncated), plus stats.
    """
    device = images.device
    b = images.shape[0]
    content_len = capacity  # content slots incl. eob position
    block = torch.full((b, content_len), MASK_ID, dtype=torch.long, device=device)
    cat_col = torch.full((b, 1), cat_id, dtype=torch.long, device=device)
    block_full = torch.cat([cat_col, block], dim=1)  # [B, 1+content_len]

    ids = torch.cat([prefix_ids, block_full], dim=1) if prefix_ids is not None else block_full
    l = ids.shape[1]
    # self-attention: active block bidirectional + prefix visible; future none (none here)
    allow = torch.ones(b, l, l, dtype=torch.bool, device=device)
    # prefix positions: bidirectional among themselves
    p = prefix_ids.shape[1] if prefix_ids is not None else 0
    if p > 0:
        allow[:, :p, :p] = True
    # block positions: attend prefix + whole block (bidirectional within block)
    allow[:, p:, :p] = attend_prefix  # False: prefix only fixes absolute positions (nocontext)
    allow[:, p:, p:] = True
    pad_mask = torch.zeros(b, l, dtype=torch.bool, device=device)

    masked = torch.ones(b, content_len, dtype=torch.bool, device=device)  # content slots
    cconf = torch.zeros(b, content_len, device=device)  # probability each slot was committed at
    # chunk=None: all slots compete in one pool for `steps` steps (original sampler).
    # chunk=C: semi-autoregressive - slots are denoised in left-to-right windows of C, each
    # for `steps` steps, later windows staying masked (but attended) meanwhile. With
    # <eob>-padded training, one pool lets confident <eob> fillers claim early positions and
    # truncate content (the LLaDA instruct-model failure). Once a window holds an <eob>, the
    # rest of the block is filler by construction and is written directly, no model calls.
    if chunk is None:
        windows = [(0, content_len)]
    else:
        windows = [(s, min(s + chunk, content_len)) for s in range(0, content_len, chunk)]
    done = torch.zeros(b, dtype=torch.bool, device=device)
    # Section emptiness decided once, by threshold rather than argmax. Most categories are
    # absent in 48-90% of reports, so argmax collapses to <empty> and coverage (Category
    # recall) craters; a threshold makes that trade-off an explicit, validation-tunable knob.
    p_empty = None
    if empty_threshold is not None or collect_p_empty:
        probs0 = F.softmax(model(images, view_mask, ids, allow,
                                 torch.ones(b, device=device), pad_mask,
                                 memory, mem_pad, memory_kv=memory_kv)[:, p + 1, :].float(), dim=-1)
        # A section ends up absent either via <empty> or via an immediate <eob> at slot 0, so
        # the quantity that controls coverage is their combined probability - thresholding
        # <empty> alone just pushes the model into the <eob> route (seen in the e1 sweep:
        # "zero empty blocks" yet Category recall still ~47-68%).
        p_empty = probs0[:, EMPTY_ID] + probs0[:, EOB_ID]
    if empty_threshold is not None:
        is_empty = p_empty >= empty_threshold
        if is_empty.any():
            ids[is_empty, p + 1] = EMPTY_ID
            ids[is_empty, p + 2:] = EOB_ID
            masked[is_empty] = False
            done = done | is_empty
    for ws, we in windows:
        if done.any():
            ids[done, p + 1 + ws:p + 1 + we] = EOB_ID
            masked[done, ws:we] = False
        eligible = torch.zeros(content_len, dtype=torch.bool, device=device)
        eligible[ws:we] = True
        for step in range(steps):
            cand = masked & eligible[None, :]
            if not cand.any():
                break
            # Condition on the block's actual masked fraction: in training t is the per-token
            # mask probability, so at the first step (all slots masked) the right value is
            # 1.0, not the 1 - 1/steps a step-count schedule would give.
            tt = masked.float().mean(dim=1)
            logits = model(images, view_mask, ids, allow, tt, pad_mask, memory, mem_pad,
                           memory_kv=memory_kv)
            logits = logits[:, p + 1:, :]  # content slots only
            logits[..., MASK_ID] = float("-inf")  # never re-predict <mask> (reference bugfix)
            if empty_threshold is not None:
                # emptiness already decided above: a block kept non-empty must start with
                # real content, so neither route to "no section" is available at slot 0
                logits[:, 0, EMPTY_ID] = float("-inf")
                logits[:, 0, EOB_ID] = float("-inf")
            probs = F.softmax(logits.float(), dim=-1)
            if ADJ_BAN is not None:
                wtok, gid, var = ADJ_BAN
                cont = ids[:, p + 1:]
                lft = torch.cat([ids[:, p:p + 1], cont[:, :-1]], dim=1)        # slot 0: left is <cat>
                rgt = torch.cat([cont[:, 1:], torch.full_like(cont[:, :1], MASK_ID)], dim=1)
                lft2 = torch.cat([torch.full_like(cont[:, :2], MASK_ID), cont[:, :-2]], dim=1)
                rgt2 = torch.cat([cont[:, 2:], torch.full_like(cont[:, :2], MASK_ID)], dim=1)
                nbs = [lft, rgt, torch.where(lft == DOT_TOK, lft2, MASK_ID), torch.where(rgt == DOT_TOK, rgt2, MASK_ID)]
                for nb in nbs:
                    ban = (masked & wtok[nb])[..., None]
                    idx = var[nb]
                    g = probs.gather(-1, idx)
                    probs.scatter_(-1, idx, torch.where(ban, torch.zeros_like(g), g))
            if STRUCT_M is not None:
                from .parallel import DOT, NL, DASH
                content_m, after_dot, after_nl, after_dash, before_dash = STRUCT_M
                cont = ids[:, p + 1:]
                lft = torch.cat([ids[:, p:p + 1], cont[:, :-1]], dim=1)
                rgt = torch.cat([cont[:, 1:], torch.full_like(cont[:, :1], MASK_ID)], dim=1)
                allowed = torch.ones_like(probs, dtype=torch.bool)
                for cond, m in (((lft == DOT), after_dot), ((lft == NL), after_nl), ((lft == DASH), after_dash),
                                ((rgt == DASH), before_dash), ((rgt == DOT), content_m)):
                    c = cond & masked
                    allowed &= ~c[..., None] | m[None, None, :]
                constrained = probs * allowed
                has_mass = constrained.sum(dim=-1, keepdim=True) > 0
                probs = torch.where(has_mass, constrained, probs)
            conf, pred = probs.max(dim=-1)  # [B, content_len]
            if eob_threshold is not None:
                # Only end the block where <eob> is confidently right; otherwise take the best
                # real token. Argmax alone truncates mid-phrase ("- No lung") because short
                # sections are the population mode. Higher threshold = longer sections.
                alt = probs.clone()
                alt[..., EOB_ID] = 0.0
                alt_conf, alt_pred = alt.max(dim=-1)
                weak_eob = (pred == EOB_ID) & (conf < eob_threshold)
                pred = torch.where(weak_eob, alt_pred, pred)
                conf = torch.where(weak_eob, alt_conf, conf)
            # Ineligible and already-unmasked slots must rank LAST (below any real
            # probability), not first. A high sentinel here previously made revealed slots
            # dominate the ranking, starving real progress and dumping over a third of each
            # block into a final near-context-free greedy pass (diag_sampler.py).
            conf0, was_masked = conf, masked.clone()
            conf = defer_boundary(conf, pred, step, steps)
            conf = conf.masked_fill(~cand, -1.0)
            if sample:
                g = torch.distributions.Categorical(probs).sample()
                pred = torch.where(cand, g, pred)
            n_unmask = (cand.sum(dim=1).float() / (steps - step)).ceil().long()
            if SCHED is not None and step < steps - 1:
                n_unmask = (cand.sum(dim=1).float() * SCHED[min(step, len(SCHED) - 1)]).ceil().long().clamp(min=1)
            if revise or (bigram is not None and bigram_lambda > 0):
                # Consensus commits (Options A/B). Slot choice is unchanged (same confidence rank and
                # count); only the tokens written to the chosen slots differ.
                rank = conf.argsort(dim=1, descending=True)
                use_bg = bigram is not None and bigram_lambda > 0
                if use_bg:
                    padj = probs.clone()
                    if eob_threshold is not None:
                        pe = padj[..., EOB_ID]
                        padj[..., EOB_ID] = torch.where(pe < eob_threshold, torch.zeros_like(pe), pe)
                    topv, topi = padj.topk(8, dim=-1)
                    logu = topv.clamp_min(1e-12).log()
                redo = torch.zeros_like(masked)
                for i in range(b):
                    k = int(n_unmask[i])
                    if k == 0:
                        continue
                    chosen = rank[i, :k].sort().values
                    if use_bg:
                        toks = viterbi_tokens(chosen, topi[i], logu[i], ids[i, p + 1:], int(ids[i, p]),
                                              bigram, bigram_lambda)
                    else:
                        toks = pred[i, chosen]
                    ids[i, p + 1 + chosen] = toks
                    masked[i, chosen] = False
                    if revise:
                        redo[i, chosen[chosen % 2 == 0]] = True
                if revise and redo.any():
                    # Option A, checkerboard revision: re-mask the even-indexed slots committed this
                    # step and re-predict them with the odd-indexed drafts visible, so every slot
                    # re-predicted sees its neighbours' tokens. (The model's output at a visible slot
                    # is untrained, so the slot being revised must itself be masked.)
                    content = ids[:, p + 1:]
                    content[redo] = MASK_ID
                    tt2 = (masked | redo).float().mean(dim=1)
                    lg = model(images, view_mask, ids, allow, tt2, pad_mask, memory, mem_pad,
                               memory_kv=memory_kv)[:, p + 1:, :]
                    lg[..., MASK_ID] = float("-inf")
                    if empty_threshold is not None:
                        lg[:, 0, EMPTY_ID] = float("-inf")
                        lg[:, 0, EOB_ID] = float("-inf")
                    pr2 = F.softmax(lg.float(), dim=-1)
                    c2, p2 = pr2.max(dim=-1)
                    if eob_threshold is not None:
                        a2 = pr2.clone(); a2[..., EOB_ID] = 0.0
                        _, ap2 = a2.max(dim=-1)
                        p2 = torch.where((p2 == EOB_ID) & (c2 < eob_threshold), ap2, p2)
                    content[redo] = p2[redo]
            elif ADJ_BAN is not None or STRUCT_M is not None or SCHED is not None:
                ranks = conf.argsort(dim=1, descending=True).argsort(dim=1)
                commit = cand & (ranks < n_unmask[:, None])
                if ADJ_BAN is not None:
                    wtok, gid, var = ADJ_BAN
                    same = gid[pred[:, :-1]] == gid[pred[:, 1:]]
                    clash = commit[:, :-1] & commit[:, 1:] & same & wtok[pred[:, :-1]]
                    rw = conf[:, 1:] <= conf[:, :-1]
                    commit[:, 1:] &= ~(clash & rw)
                    commit[:, :-1] &= ~(clash & ~rw)
                    new_c = torch.where(commit, pred, ids[:, p + 1:])
                    c2 = (commit[:, :-2] & commit[:, 2:] & (gid[pred[:, :-2]] == gid[pred[:, 2:]])
                          & wtok[pred[:, :-2]] & (new_c[:, 1:-1] == DOT_TOK))
                    rw2 = conf[:, 2:] <= conf[:, :-2]
                    commit[:, 2:] &= ~(c2 & rw2)
                    commit[:, :-2] &= ~(c2 & ~rw2)
                if STRUCT_M is not None:
                    from .parallel import _pair_ok
                    bad = commit[:, :-1] & commit[:, 1:] & ~_pair_ok(pred[:, :-1], pred[:, 1:], STRUCT_M)
                    rwb = conf[:, 1:] <= conf[:, :-1]
                    commit[:, 1:] &= ~(bad & rwb)
                    commit[:, :-1] &= ~(bad & ~rwb)
                ids[:, p + 1:] = torch.where(commit, pred, ids[:, p + 1:])
                masked = masked & ~commit
            elif not (no_repeat or spread):
                rank = conf.argsort(dim=1, descending=True)
                for i in range(b):
                    k = int(n_unmask[i])
                    if k == 0:
                        continue
                    chosen = rank[i, :k]
                    ids[i, p + 1 + chosen] = pred[i, chosen]
                    masked[i, chosen] = False
            else:
                top = probs.topk(8, dim=-1).indices.tolist() if no_repeat else None
                pred_l = pred.tolist()
                for i in range(b):
                    k = int(n_unmask[i])
                    if k == 0:
                        continue
                    chosen = pick_slots(conf[i], cand[i], k, spread)
                    if no_repeat:
                        row = ids[i, p + 1:].tolist()
                        toks = commit_no_repeat(row, chosen, pred_l[i], top[i], N_SPECIAL, no_repeat)
                    else:
                        toks = [pred_l[i][j] for j in chosen]
                    ci = torch.tensor(chosen, device=device)
                    ids[i, p + 1 + ci] = torch.tensor(toks, device=device)
                    masked[i, ci] = False
            # Probability the model - under the CURRENT, fuller context - still assigns to
            # the token written at each slot. This is what remasking thresholds on: "does the
            # model still believe what it wrote here?"
            #
            # This MUST be gathered AFTER the commit loop above. Gathered before it, every slot
            # still masked holds <mask>, and logits[..., MASK_ID] is -inf, so p_written was
            # exactly 0.0 for every slot committed in this same step. Those zeros then sorted
            # first among the "weak" set and filled the cap on every single firing, so remasking
            # degenerated to "commit n slots, immediately un-commit n-1 of them" and could never
            # revisit an older decision. Verified: over 20k remask events, pw_pos was 0 and the
            # cap bound 297/297 times, which is also why thresholds 0.3/0.5/0.7 gave bit-identical
            # metrics.
            cconf = torch.where(was_masked & ~masked, conf0, cconf)
            p_written = probs.gather(-1, ids[:, p + 1:].unsqueeze(-1)).squeeze(-1)
            if remask_threshold is not None and step < steps - remask_reserve:
                # Return already-committed slots the model no longer supports to the masked
                # pool. Two guards keep the schedule terminating: never remask more than
                # (committed this step - 1), so net progress is >= 1 slot per step; and never
                # remask in the last `remask_reserve` steps, so the tail always converges.
                for i in range(b):
                    cap = int(n_unmask[i]) - 1
                    if cap <= 0:
                        continue
                    committed = (~masked[i]) & eligible
                    weak = committed & (p_written[i] < remask_threshold)
                    idx = weak.nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        continue
                    if idx.numel() > cap:  # remask the least-supported first
                        order = p_written[i, idx].argsort()
                        idx = idx[order[:cap]]
                    masked[i, idx] = True
                    ids[i, p + 1 + idx] = MASK_ID
        window = ids[:, p + 1 + ws:p + 1 + we]
        done = done | (window == EOB_ID).any(dim=1) | (ids[:, p + 1] == EMPTY_ID)
    # final: unmask any leftover by argmax
    if masked.any():
        tt = torch.zeros(b, device=device)
        logits = model(images, view_mask, ids, allow, tt, pad_mask, memory, mem_pad,
                       memory_kv=memory_kv)[:, p + 1:, :]
        logits[..., MASK_ID] = float("-inf")
        lconf, pred = F.softmax(logits.float(), dim=-1).max(dim=-1)
        leftover = masked.nonzero(as_tuple=True)
        cconf[leftover] = lconf[leftover]
        ids[leftover[0], p + 1 + leftover[1]] = pred[leftover]
        masked[leftover] = False

    return {"ids": ids, "prefix_len": p, "content_len": content_len,
            "block_start": p, "block_end": l, "p_empty": p_empty, "cconf": cconf}


def extract_block_content(block_ids: torch.Tensor, eob_id: int, empty_id: int) -> dict:
    """Read out generated block tokens per SPEC §4.

    block_ids: [B, S] token ids of one block INCLUDING leading cat token.
    Returns dict of lists: content (token ids after cat, up to & incl first eob),
    terminated (bool), empty (bool: first content token is <empty>).
    """
    contents, terminated, empties = [], [], []
    for row in block_ids.tolist():
        row = row[1:]  # drop cat token
        if row and row[0] == empty_id:
            contents.append([empty_id]); terminated.append(True); empties.append(True)
            continue
        if eob_id in row:
            idx = row.index(eob_id)
            contents.append(row[: idx + 1]); terminated.append(True); empties.append(False)
        else:
            contents.append(row); terminated.append(False); empties.append(False)
    return {"content": contents, "terminated": terminated, "empty": empties}
