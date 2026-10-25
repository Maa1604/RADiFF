"""Training loop for anatomy-block diffusion RRG variants. SPEC.md Sections 3, 6.

Usage:
    conda activate rrg-cxr
    python train.py --variant anatomical --train_samples -1 --epochs 20

Variants share this exact script; they differ only in block boundaries / attention
masks (src/diffusion.py) and, for `ar`, in using plain causal shifted-CE instead of
masked-diffusion denoising loss (SPEC.md Section 2: ar uses constant t=0).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/lustre/scratch/maasala@upvnet.upv.es/hf_cache")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import data as D
from src.diffusion import VARIANTS, sample_t
from src.generate import FIXED_CHUNK, generate_batch
from src.model import RRGModel, causal_allow

TOKENIZER_PATH = "/lustre/scratch/maasala@upvnet.upv.es/anatomy_block_isbi/data/tokenizer_701479/tokenizer.json"
VOCAB_SIZE = 8192
if os.environ.get("RRG_TOK_DIR"):  # phrase-level tokenizer study (src/data.py)
    TOKENIZER_PATH = os.path.join(os.environ["RRG_TOK_DIR"], "tokenizer.json")
    VOCAB_SIZE = len(json.load(open(TOKENIZER_PATH))["model"]["vocab"])
RUNS_ROOT = "/lustre/scratch/maasala@upvnet.upv.es/anatomy_block_isbi/runs"


def collate(batch):
    ids = torch.stack([item["ids"] for item in batch])
    block_spans = [item["block_spans"] for item in batch]
    # Diffusion variants train on the fixed-capacity layout (data.fixed_layout): every block
    # has the same width and absolute position as the slots denoise_block generates into.
    # `ar` keeps the compact variable-length sequence in `ids`.
    fixed = [D.fixed_layout(item["ids"], item["block_spans"]) for item in batch]
    b = len(batch)
    images = torch.zeros(b, D.MAX_VIEWS, 1, D.IMG_LOAD_SIZE, D.IMG_LOAD_SIZE)
    view_mask = torch.ones(b, D.MAX_VIEWS, dtype=torch.bool)  # True = pad (no view)
    for i, item in enumerate(batch):
        for v, p in enumerate(item["image_paths"][:D.MAX_VIEWS]):
            images[i, v, 0] = D.load_image_tensor(p).squeeze(0)
            view_mask[i, v] = False
    # Clinical context padded to the batch max rather than MAX_CONTEXT_LEN: a batch of short
    # indications then carries only the memory slots it needs. Length is floored at 1 so the
    # tensor is always well-formed; an all-pad context row is harmless because the visual part of
    # `memory` is never fully padded, so attention always has something real to attend to.
    ctx_ids = ctx_pad = None
    if any(item.get("ctx_ids") for item in batch):
        cl = max(1, max(len(item.get("ctx_ids") or []) for item in batch))
        ctx_ids = torch.zeros(b, cl, dtype=torch.long)
        ctx_pad = torch.ones(b, cl, dtype=torch.bool)
        for i, item in enumerate(batch):
            c = item.get("ctx_ids") or []
            if c:
                ctx_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)
                ctx_pad[i, :len(c)] = False
    return {
        "ids": ids, "images": images, "view_mask": view_mask,
        "ctx_ids": ctx_ids, "ctx_pad": ctx_pad,
        "block_spans": block_spans, "ex_ids": [item["id"] for item in batch],
        "fixed_ids": torch.tensor([f[0] for f in fixed], dtype=torch.long),
        "filler": torch.tensor([f[1] for f in fixed], dtype=torch.bool),
    }


FIXED_SPANS = D.fixed_block_spans()
FIXED_LEN = FIXED_SPANS[-1][1] + 1  # + final <eor>


def layout_index(variant: str, device):
    """Per-position block id and "clean prompt" flag for the fixed layout. Anatomical
    variants: block k = category k, its leading <cat_k> is a prompt that is never
    corrupted (denoise_block always gives it clean). `fixed`: positional chunks of
    FIXED_CHUNK over the same layout, with only position 0 (<cat_0>) as a prompt, matching
    generate_fixed. The final <eor> has block id -1 in the anatomical layout (never
    corrupted, never attended)."""
    pos = torch.arange(FIXED_LEN, device=device)
    if variant == "fixed":
        return pos // FIXED_CHUNK, pos == 0
    blk = torch.full((FIXED_LEN,), -1, dtype=torch.long, device=device)
    prompt = torch.zeros(FIXED_LEN, dtype=torch.bool, device=device)
    for k, (s, e) in enumerate(FIXED_SPANS):
        blk[s:e] = k
        prompt[s] = True
    return blk, prompt


def n_blocks_for(variant: str) -> int:
    return -(-FIXED_LEN // FIXED_CHUNK) if variant == "fixed" else len(FIXED_SPANS)


CTC = False  # set from --ctc in main(); alignment-aware block loss (src/ctc.py)
# Neighbour unlikelihood (fluency objective, 2026-10-11): weight of -log(1 - p_j(y_{j+-1})) at a masked slot j
# whose masked neighbour holds a different content token. Penalises the position-hedge ("X X" repeats).
UL_NEIGHBOR = 0.0


def compute_loss(model, batch, variant: str, device, t_fixed: float | None = None,
                 block_fixed: int | None = None, filler_weight: float = 1.0,
                 block_sample_weights: torch.Tensor | None = None,
                 self_correct: float = 0.0, glat: float = 0.0, glat_ratio: float = 0.5):
    images = batch["images"].to(device)
    view_mask = batch["view_mask"].to(device)
    ctx_ids = batch.get("ctx_ids")
    ctx_pad = batch.get("ctx_pad")
    if ctx_ids is not None:
        ctx_ids, ctx_pad = ctx_ids.to(device), ctx_pad.to(device)

    if variant == "ar":
        ids = batch["ids"].to(device)
        b, l = ids.shape
        allow = causal_allow(l, device)[None].expand(b, l, l)
        t = torch.zeros(b, device=device)
        pad = ids == D.PAD_ID
        logits = model(images, view_mask, ids, allow, t, pad, ctx_ids=ctx_ids, ctx_pad=ctx_pad)
        pred = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        target = ids[:, 1:].reshape(-1)
        return F.cross_entropy(pred, target, ignore_index=D.PAD_ID)

    # One active block per batch, and the sequence cut right after it: later blocks are
    # hidden from every query that matters (future blocks are never attended), so dropping
    # them leaves the loss unchanged while saving ~1.8x compute on the fixed 362-token
    # layout. Only a suffix is removed, so absolute positions are preserved.
    if block_fixed is not None:
        k = block_fixed
    elif block_sample_weights is not None:
        # oversample under-represented categories: draws block k with probability
        # proportional to block_sample_weights[k] instead of uniformly, so a category
        # with weight 2.0 gets ~2x the training steps (and thus optimizer updates) of a
        # category with weight 1.0 - a more Adam-robust lever than scaling the loss
        # value, since it changes how many actual gradient steps a category receives
        # rather than just the magnitude of a step that gets re-normalized away.
        k = int(torch.multinomial(block_sample_weights, 1))
    else:
        k = int(torch.randint(0, n_blocks_for(variant), (1,)))
    blk, prompt = layout_index(variant, device)
    end = int((blk == k).nonzero().max()) + 1
    ids = batch["fixed_ids"][:, :end].to(device)
    filler = batch["filler"][:, :end].to(device)
    blk, prompt = blk[:end], prompt[:end]
    b, l = ids.shape
    active = torch.full((b,), k, dtype=torch.long, device=device)
    t = torch.full((b,), t_fixed, device=device) if t_fixed is not None else sample_t(b, device)

    region = (blk[None, :] == active[:, None]) & ~prompt[None, :]
    corr_mask = region & (torch.rand(b, l, device=device) < t[:, None])
    corrupted = ids.masked_fill(corr_mask, D.MASK_ID)
    if variant == "anatomical_nocontext":
        visible = blk[None, :] == active[:, None]
    else:  # clean earlier blocks + the active block; future blocks hidden
        visible = (blk[None, :] <= active[:, None]) & (blk[None, :] >= 0)
    allow = visible[:, None, :].expand(b, l, l)
    pad = torch.zeros(b, l, dtype=torch.bool, device=device)
    if CTC and variant != "fixed":
        from src.ctc import ctc_block_loss
        return ctc_block_loss(model, images, view_mask, ids, region, filler, allow, pad,
                              ctx_ids, ctx_pad, t, filler_weight)
    if self_correct > 0 and variant != "fixed" and float(torch.rand(())) < self_correct:
        # Self-correction batch: draft every corrupted slot with the current model (argmax, as the
        # sampler commits), then fill a random fraction q ~ U(0,1) of them with those drafts - the
        # state after a parallel commit step. t becomes the fraction of the block still masked,
        # matching the sampler's t. The loss below still covers every originally corrupted slot,
        # so the model learns both to predict given its own (possibly wrong) neighbours and to
        # correct a wrong draft.
        with torch.no_grad():
            lg0 = model(images, view_mask, corrupted, allow, t, pad, ctx_ids=ctx_ids, ctx_pad=ctx_pad)
            lg0[..., D.MASK_ID] = float("-inf")
            draft = lg0.argmax(dim=-1)
        q = torch.rand(b, 1, device=device)
        filled = corr_mask & (torch.rand(b, l, device=device) < q)
        corrupted = torch.where(filled, draft, corrupted)
        t = (corr_mask & ~filled).sum(dim=1).float() / region.sum(dim=1).clamp(min=1).float()
    if glat > 0 and variant != "fixed" and float(torch.rand(())) < glat:
        # Glancing batch (GLAT, Qian et al. 2021): predict the whole block in one parallel pass
        # (t=1, the sampler first step), count the slots it gets wrong, reveal ceil(ratio * wrong)
        # random reference tokens as hints and train on the rest. Hard reports get more hints,
        # easy ones train in the all-masked state, so the model learns to predict jointly
        # consistent tokens from few anchors instead of averaging phrasings.
        with torch.no_grad():
            full = ids.masked_fill(region, D.MASK_ID)
            lg0 = model(images, view_mask, full, allow, torch.ones(b, device=device), pad,
                        ctx_ids=ctx_ids, ctx_pad=ctx_pad)
            lg0[..., D.MASK_ID] = float("-inf")
            wrong = ((lg0.argmax(dim=-1) != ids) & region).sum(dim=1).float()
        n_rev = torch.ceil(glat_ratio * wrong)
        score = torch.rand(b, l, device=device).masked_fill(~region, 2.0)
        rank = score.argsort(dim=1).argsort(dim=1)
        corr_mask = region & ~(rank < n_rev[:, None])
        corrupted = ids.masked_fill(corr_mask, D.MASK_ID)
        t = corr_mask.sum(dim=1).float() / region.sum(dim=1).clamp(min=1).float()
    logits = model(images, view_mask, corrupted, allow, t, pad, ctx_ids=ctx_ids, ctx_pad=ctx_pad)

    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.reshape(-1),
                         reduction="none").reshape(b, l)
    w = corr_mask.float() * torch.where(filler, filler_weight, 1.0)
    denom = w.sum(dim=1)
    has = denom > 0
    if not has.any():
        return (ce * 0).sum()
    loss = ((ce * w).sum(dim=1)[has] / denom[has]).mean()
    if UL_NEIGHBOR > 0:
        lse = logits.float().logsumexp(dim=-1)
        ul = torch.zeros(b, l, device=device)
        for sh in (1, -1):  # sh=1: neighbour on the left (nb[j] = ids[j-1]); -1: on the right
            nb = torch.roll(ids, sh, dims=1)
            ok = corr_mask & torch.roll(corr_mask, sh, dims=1) & (nb != ids) & (nb >= len(D.SPECIAL_IDS))
            if sh == 1:
                ok[:, 0] = False
            else:
                ok[:, -1] = False
            p_nb = (logits.gather(-1, nb[..., None]).squeeze(-1).float() - lse).exp()
            ul = ul + torch.where(ok, -torch.log1p(-p_nb.clamp(max=1 - 1e-4)), torch.zeros_like(p_nb))
        loss = loss + UL_NEIGHBOR * ((ul * w).sum(dim=1)[has] / denom[has]).mean()
    return loss


def load_sample_batch(hf_dataset, n: int, device, tokenizer=None, with_context: bool = False):
    """A handful of fixed validate examples for qualitative tracking across training
    (never parses findings_section, so it can't fail on a non-canonical category label —
    see test.py's ImageOnlyDataset for the same reasoning). Returns
    (images, view_mask, ids, raw_ref_texts)."""
    roots = D.dataset_roots()["image_roots"]
    examples = [hf_dataset[i] for i in range(min(n, len(hf_dataset)))]
    b = len(examples)
    ctx_lists = ([D.tokenize_context(ex, tokenizer) for ex in examples]
                 if with_context else [[] for _ in examples])
    images = torch.zeros(b, D.MAX_VIEWS, 1, D.IMG_LOAD_SIZE, D.IMG_LOAD_SIZE)
    view_mask = torch.ones(b, D.MAX_VIEWS, dtype=torch.bool)
    ex_ids, refs = [], []
    for i, ex in enumerate(examples):
        paths = D.resolve_image_paths(list(ex["image_paths"])[:D.MAX_VIEWS], roots)
        for v, p in enumerate(paths):
            images[i, v, 0] = D.load_image_tensor(p).squeeze(0)
            view_mask[i, v] = False
        ex_ids.append(ex["id"])
        refs.append(ex["findings_section"])
    ctx_ids = ctx_pad = None
    if any(ctx_lists):
        cl = max(1, max(len(c) for c in ctx_lists))
        ctx_ids = torch.zeros(b, cl, dtype=torch.long)
        ctx_pad = torch.ones(b, cl, dtype=torch.bool)
        for i, c in enumerate(ctx_lists):
            if c:
                ctx_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long)
                ctx_pad[i, :len(c)] = False
        ctx_ids, ctx_pad = ctx_ids.to(device), ctx_pad.to(device)
    return images.to(device), view_mask.to(device), ex_ids, refs, (ctx_ids, ctx_pad)


def log_epoch_samples(model, tok, sample_images, sample_view_mask, sample_ids, sample_refs,
                      variant: str, epoch: int, step: int, samples_csv_path: str, steps: int = 8,
                      sample_ctx=(None, None)) -> None:
    """Print + append a few live generations from the CURRENT checkpoint, on the same
    fixed examples every time, so quality (or lack of it) is visible epoch-over-epoch
    without waiting for training to finish or eyeballing loss numbers alone. Cheap:
    a handful of examples at a reduced step budget, not the full eval sampler."""
    model.eval()
    results = []
    for i in range(0, sample_images.shape[0], 8):
        cids, cpad = sample_ctx
        results += generate_batch(model, tok, sample_images[i:i + 8], sample_view_mask[i:i + 8],
                                  variant, steps=steps,
                                  ctx_ids=None if cids is None else cids[i:i + 8],
                                  ctx_pad=None if cpad is None else cpad[i:i + 8])
    model.train()

    def n_sections(text):
        return sum(1 for line in text.split("\n") if line.strip().rstrip(":") in D.CATEGORIES
                   and line.strip().endswith(":"))

    n = max(len(results), 1)
    hyp_sec = sum(n_sections(r["text"]) for r in results) / n
    ref_sec = sum(n_sections(r) for r in sample_refs) / n
    miss = sum(r["diagnostics"].get("missing_terminator", 0) for r in results) / n
    empty = sum(r["diagnostics"].get("empty", 0) for r in results) / n
    print(f"[samples] epoch={epoch} step={step} n={n} sections hyp={hyp_sec:.2f} ref={ref_sec:.2f} "
          f"missing_terminator={miss:.2f}/8 empty={empty:.2f}/8", flush=True)
    print(f"\n=== epoch={epoch} step={step} generation samples ===", flush=True)
    rows = []
    for j, (ex_id, ref, res) in enumerate(zip(sample_ids, sample_refs, results)):
        if j >= 3:  # print a few, but log all to the CSV
            rows.append({"epoch": epoch, "step": step, "id": ex_id, "ref": ref, "hyp": res["text"]})
            continue
        hyp = res["text"]
        print(f"--- id={ex_id}", flush=True)
        print(f"REF: {ref[:300].replace(chr(10), ' | ')}", flush=True)
        print(f"HYP: {hyp[:300].replace(chr(10), ' | ')}", flush=True)
        rows.append({"epoch": epoch, "step": step, "id": ex_id, "ref": ref, "hyp": hyp})
    write_header = not os.path.exists(samples_csv_path)
    with open(samples_csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "step", "id", "ref", "hyp"])
        if write_header:
            w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def validation_loss(model, loader, variant: str, device, filler_weight: float = 1.0) -> float:
    """Every block at t=0.5, with a fixed corruption seed so successive checkpoints are
    scored on identical masks (comparable across evaluations, not just noisy samples)."""
    model.eval()
    total, count = 0.0, 0
    devices = [torch.device(device)] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1234)
        for batch in loader:
            b = batch["ids"].shape[0]
            if variant == "ar":
                loss = compute_loss(model, batch, variant, device)
                total += loss.item() * b
                count += b
                continue
            for blk in range(n_blocks_for(variant)):
                loss = compute_loss(model, batch, variant, device, t_fixed=0.5, block_fixed=blk,
                                    filler_weight=filler_weight)
                total += loss.item() * b
                count += b
    model.train()
    return total / max(count, 1)


def find_batch_size(make_model, dataset, variant: str, device, candidates: list[int]) -> tuple[int, float]:
    best_bs, best_mem = candidates[0], 0.0
    for bs in candidates:
        try:
            model = make_model().to(device)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            loader = DataLoader(dataset, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=4)
            it = iter(loader)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(20):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                opt.zero_grad()
                # size for the worst case: the last block keeps the full sequence length
                worst = None if variant == "ar" else n_blocks_for(variant) - 1
                loss = compute_loss(model, batch, variant, device, block_fixed=worst)
                loss.backward()
                opt.step()
            mem = torch.cuda.max_memory_allocated() / 1e9
            util_line = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read().strip()
            print(f"[batch_size_finder] bs={bs} peak_mem={mem:.1f}GB util={util_line}%", flush=True)
            best_bs, best_mem = bs, mem
            del model, opt, loader, it
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[batch_size_finder] bs={bs} OOM, stopping search", flush=True)
            torch.cuda.empty_cache()
            break
    return best_bs, best_mem


def build_model(config) -> RRGModel:
    """Architecture comes from the run config so every consumer (test.py, sweeps, mechanism
    checks) rebuilds exactly the trained model rather than a hardcoded default."""
    get = config.get if isinstance(config, dict) else lambda k, d: getattr(config, k, d)
    return RRGModel(vocab_size=VOCAB_SIZE, max_len=384, d_model=get("d_model", 768),
                    nhead=get("nhead", 8), num_layers=get("n_layers", 6),
                    dim_ffn=get("d_ffn", 3072),
                    unfreeze_encoder=get("unfreeze_encoder", False),
                    use_context=get("use_context", False),
                    max_context_len=get("max_context_len", D.MAX_CONTEXT_LEN))


def make_model_fn(args):
    return lambda: build_model(args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--train_samples", type=int, default=-1, help="-1 = full train split")
    p.add_argument("--val_samples", type=int, default=-1, help="-1 = full validate split")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=0, help="0 = auto-tune")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--use_context", action="store_true",
                   help="condition on indication/technique/comparison (the MAIRA-2 +context "
                        "setting). Off by default: the image-only comparison in objective.md "
                        "requires the headline model not to consume these fields.")
    p.add_argument("--unfreeze_encoder", action="store_true",
                   help="train rad-dino instead of keeping it frozen (fp32, own LR group)")
    p.add_argument("--encoder_lr", type=float, default=None,
                   help="LR for the encoder when unfrozen (default: --lr / 10)")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="micro-batches per optimizer step; effective batch = batch_size * this. "
                        "Lets a memory-bound run (e.g. a trainable ViT) match another run's "
                        "effective batch instead of confounding it with a smaller one.")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="path to a checkpoint's model state_dict to warm-start from "
                        "(fine-tuning), instead of random init")
    p.add_argument("--block_sample_weights", type=str, default=None,
                   help="comma-separated per-category weight (len = len(CATEGORIES)) for how "
                        "often each block is drawn as the active training block; default "
                        "(unset) is uniform. E.g. up-weight rare/under-recalled categories "
                        "like Other/Abdominal so they get proportionally more gradient steps.")
    p.add_argument("--filler_weight", type=float, default=1.0,
                   help="loss weight on the trailing <eob> fillers after a block's first <eob> "
                        "(the first <eob> itself always has weight 1). 1.0 = treat them as normal "
                        "targets, as LLaDA does with EOS padding so the model controls length")
    p.add_argument("--lr_min_ratio", type=float, default=0.1,
                   help="cosine decay floor as a fraction of peak lr, reached by the planned "
                        "end of training (epochs * steps_per_epoch)")
    p.add_argument("--self_correct", type=float, default=0.0,
                   help="probability per batch of a self-correction batch (model drafts, a random "
                        "fraction of drafts is kept as context, loss on all corrupted slots); 0 = off")
    p.add_argument("--shuffle_bullets", action="store_true",
                   help="train on a fresh random bullet order inside each category every time a "
                        "report is loaded (validation keeps the original order)")
    p.add_argument("--ctc", action="store_true",
                   help="alignment-aware (CTC/Imputer) loss for present sections; decode with test.py --ctc")
    p.add_argument("--ul_neighbor", type=float, default=0.0,
                   help="weight of the neighbour-unlikelihood term (penalises predicting a neighbour's token)")
    p.add_argument("--glat", type=float, default=0.0,
                   help="probability per batch of a glancing (GLAT) batch; 0 = off")
    p.add_argument("--glat_ratio", type=float, default=0.5,
                   help="GLAT: reference tokens revealed = ceil(ratio * wrong first-pass slots)")
    p.add_argument("--kd_targets", type=str, default=None,
                   help="jsonl of {id, findings_section} written by a teacher (kd_generate.py); "
                        "replaces the train references for matching ids (validation unchanged)")
    p.add_argument("--kd_mix", action="store_true",
                   help="with --kd_targets: train on original AND teacher references (2x rows)")
    p.add_argument("--patience", type=int, default=3, help="epochs without val improvement before stopping")
    p.add_argument("--eval_every_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--d_ffn", type=int, default=3072)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None, help="hard cap for smoke tests")
    p.add_argument("--stop_after_epochs", type=int, default=None,
                   help="stop after this many completed epochs; the LR schedule still follows --epochs")
    p.add_argument("--sample_n", type=int, default=4,
                   help="fixed validate examples to generate+print at every val checkpoint, "
                        "so quality is visible epoch-over-epoch without waiting for the run "
                        "to finish; 0 disables")
    p.add_argument("--sample_steps", type=int, default=8,
                   help="denoising steps for the cheap live-sample generations (lower than "
                        "the real eval's step budget, this is a trend indicator not a score)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="cpu is for correctness-only smoke tests when no GPU is free; real "
                        "training runs must use cuda per this project's GPU-saturation rule")
    return p.parse_args()


def main():
    args = parse_args()
    global CTC, UL_NEIGHBOR
    CTC = args.ctc
    UL_NEIGHBOR = args.ul_neighbor
    torch.manual_seed(args.seed)
    device = args.device
    if device == "cuda":
        assert torch.cuda.is_available(), "train.py requires a GPU node (srun/sbatch on the gpu partition)"

    tok = D.load_tokenizer(TOKENIZER_PATH)
    from datasets import load_dataset

    print("loading dataset splits...", flush=True)
    train_hf = load_dataset("StanfordAIMI/srrg_findings", split="train")
    val_hf = load_dataset("StanfordAIMI/srrg_findings", split="validate")
    if args.train_samples >= 0:
        train_hf = train_hf.select(range(min(args.train_samples, len(train_hf))))
    if args.val_samples >= 0:
        val_hf = val_hf.select(range(min(args.val_samples, len(val_hf))))

    print(f"encoding {len(train_hf)} train / {len(val_hf)} val rows (parsing + image path resolution)...", flush=True)
    t0 = time.time()
    train_ds = D.build_dataset(train_hf, tok, D.DEFAULT_CAPACITIES,
                               with_context=args.use_context,
                               shuffle_bullets=args.shuffle_bullets)
    val_ds = D.build_dataset(val_hf, tok, D.DEFAULT_CAPACITIES,
                             with_context=args.use_context)
    if args.kd_targets:
        kd = {}
        with open(args.kd_targets) as f:
            for line in f:
                r = json.loads(line)
                kd[r["id"]] = r["findings_section"]
        kd_hf = train_hf.map(lambda ex: {"findings_section": kd.get(ex["id"], ex["findings_section"])})
        n_kd = sum(1 for i in train_hf["id"] if i in kd)
        print(f"KD targets: {n_kd}/{len(train_hf)} train rows replaced from {args.kd_targets}"
              f" (mix={args.kd_mix})", flush=True)
        kd_ds = D.build_dataset(kd_hf, tok, D.DEFAULT_CAPACITIES, with_context=args.use_context,
                                shuffle_bullets=args.shuffle_bullets)
        # collate stacks the padded `ids`, so both halves of a mix must pad to the same length
        train_ds._max_len = kd_ds._max_len = max(train_ds._max_len, kd_ds._max_len)
        train_ds = torch.utils.data.ConcatDataset([train_ds, kd_ds]) if args.kd_mix else kd_ds
    print(f"encoded in {time.time() - t0:.1f}s", flush=True)

    model_fn = make_model_fn(args)
    if args.batch_size == 0:
        print("auto-tuning batch size for GPU saturation...", flush=True)
        candidates = [4, 8, 16, 32, 48, 64, 96, 128, 160, 224, 288, 352, 416, 480]
        batch_size, peak_mem = find_batch_size(model_fn, train_ds, args.variant, device, candidates)
        print(f"selected batch_size={batch_size} (peak {peak_mem:.1f}GB)", flush=True)
    else:
        batch_size = args.batch_size
        peak_mem = None

    model = model_fn().to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"warm-started from {args.init_checkpoint} (step={ckpt.get('step')}, "
              f"val_loss={ckpt.get('val_loss')})", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M "
          f"(layers={args.n_layers} d_model={args.d_model} ffn={args.d_ffn})", flush=True)
    if args.unfreeze_encoder:
        # Separate group at a lower LR: rad-dino is pretrained and 2e-4 would wreck it.
        # LambdaLR scales each group by its own initial_lr, so warmup/cosine still applies.
        enc_params = [p for p in model.visual.vit.parameters() if p.requires_grad]
        enc_ids = {id(p) for p in enc_params}
        rest = [p for p in trainable if id(p) not in enc_ids]
        enc_lr = args.encoder_lr if args.encoder_lr is not None else args.lr / 10.0
        opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                                 {"params": enc_params, "lr": enc_lr}], lr=args.lr)
        print(f"encoder UNFROZEN: {sum(p.numel() for p in enc_params) / 1e6:.1f}M encoder params "
              f"@ lr={enc_lr:g}; {sum(p.numel() for p in rest) / 1e6:.1f}M rest @ lr={args.lr:g}",
              flush=True)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr)

    block_sample_weights = None
    if args.block_sample_weights:
        w = [float(x) for x in args.block_sample_weights.split(",")]
        assert len(w) == len(D.CATEGORIES), \
            f"--block_sample_weights needs {len(D.CATEGORIES)} values, got {len(w)}"
        block_sample_weights = torch.tensor(w, device=device)
        print(f"block_sample_weights: {dict(zip(D.CATEGORIES, w))}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, 32), shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)

    # Linear warmup, then cosine decay to `lr_min_ratio` of peak by the planned end of
    # training (args.epochs * steps_per_epoch) - previously this held constant at 1.0
    # forever after warmup, fine for a short run but not well suited to converging a
    # much longer one (see experiments.md: val_loss hadn't plateaued at the 20-epoch
    # ceiling, motivating a longer run that needs an actual decay target to converge to).
    opt_steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = max(args.epochs * opt_steps_per_epoch, args.warmup_steps + 1)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        progress = min(1.0, (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.lr_min_ratio + (1 - args.lr_min_ratio) * cosine

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    run_name = args.run_name or args.variant
    job_id = os.environ.get("SLURM_JOB_ID", str(int(time.time())))
    run_dir = os.path.join(RUNS_ROOT, f"{run_name}_{job_id}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    config = vars(args) | {
        "batch_size_used": batch_size, "peak_mem_gb": peak_mem, "job_id": job_id,
        "run_dir": run_dir, "n_train": len(train_ds), "n_val": len(val_ds),
        "capacities": D.DEFAULT_CAPACITIES, "tokenizer_path": TOKENIZER_PATH,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    if args.sample_n > 0:
        sample_images, sample_view_mask, sample_ids, sample_refs, sample_ctx = load_sample_batch(
            val_hf, args.sample_n, device, tokenizer=tok, with_context=args.use_context)
        samples_csv_path = os.path.join(run_dir, "epoch_samples.csv")

    best_val = float("inf")
    epochs_no_improve = 0
    step = 0
    train_trace, val_trace = [], []
    t_start = time.time()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        opt.zero_grad()
        micro, accum_loss = 0, 0.0
        for batch in train_loader:
            # Gradients are summed over `grad_accum` micro-batches before stepping, so the
            # effective batch is batch_size * grad_accum. Each micro-batch draws its own random
            # block k, which only increases block diversity per optimizer step. Loss is scaled
            # by 1/accum so the gradient magnitude matches a single batch of that size.
            loss = compute_loss(model, batch, args.variant, device,
                                filler_weight=args.filler_weight,
                                block_sample_weights=block_sample_weights,
                                self_correct=args.self_correct, glat=args.glat, glat_ratio=args.glat_ratio)
            accum_loss += loss.item()
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            loss = accum_loss / args.grad_accum   # report the unscaled mean for logging
            accum_loss = 0.0
            step += 1
            if step % 50 == 0:
                train_trace.append({"step": step, "loss": float(loss)})
                print(f"epoch={epoch} step={step} loss={float(loss):.4f}", flush=True)
            if step % args.eval_every_steps == 0:
                vloss = validation_loss(model, val_loader, args.variant, device, args.filler_weight)
                val_trace.append({"step": step, "val_loss": vloss})
                print(f"  [val] step={step} val_loss={vloss:.4f}", flush=True)
                torch.save({"model": model.state_dict(), "step": step, "val_loss": vloss},
                          os.path.join(ckpt_dir, "last.pt"))
                if vloss < best_val:
                    best_val = vloss
                    torch.save({"model": model.state_dict(), "step": step, "val_loss": vloss},
                              os.path.join(ckpt_dir, "best.pt"))
                if args.sample_n > 0:
                    log_epoch_samples(model, tok, sample_images, sample_view_mask, sample_ids,
                                      sample_refs, args.variant, epoch, step, samples_csv_path,
                                      steps=args.sample_steps, sample_ctx=sample_ctx)
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        if stop:
            break
        vloss = validation_loss(model, val_loader, args.variant, device, args.filler_weight)
        val_trace.append({"step": step, "val_loss": vloss, "epoch_end": True})
        print(f"epoch={epoch} done, val_loss={vloss:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "step": step, "val_loss": vloss},
                  os.path.join(ckpt_dir, "last.pt"))
        if args.sample_n > 0:
            log_epoch_samples(model, tok, sample_images, sample_view_mask, sample_ids,
                              sample_refs, args.variant, epoch, step, samples_csv_path,
                              steps=args.sample_steps, sample_ctx=sample_ctx)
        if vloss < best_val - 1e-4:
            best_val = vloss
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "step": step, "val_loss": vloss},
                      os.path.join(ckpt_dir, "best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch={epoch} (patience={args.patience})", flush=True)
                break
        if args.stop_after_epochs is not None and epoch + 1 >= args.stop_after_epochs:
            print(f"stopping after epoch={epoch} (--stop_after_epochs {args.stop_after_epochs})", flush=True)
            break

    elapsed = time.time() - t_start
    with open(os.path.join(run_dir, "train_trace.json"), "w") as f:
        json.dump({"train": train_trace, "val": val_trace, "elapsed_s": elapsed, "best_val": best_val}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        if train_trace:
            ax.plot([r["step"] for r in train_trace], [r["loss"] for r in train_trace], label="train loss", alpha=0.7)
        if val_trace:
            ax.plot([r["step"] for r in val_trace], [r["val_loss"] for r in val_trace], label="val loss", marker="o")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.set_title(f"{run_name} loss curve")
        fig.savefig(os.path.join(ckpt_dir, "loss_curve.png"), dpi=120)
    except ImportError:
        print("matplotlib not available, skipping loss_curve.png", flush=True)

    experiments_md = os.path.expanduser("~/anatomy_block_isbi/experiments.md")
    entry = f"""
## {run_name} (job {job_id}, {time.strftime('%Y-%m-%d %H:%M')})

- **Variant**: {args.variant}
- **Config**: train_samples={args.train_samples}, epochs_ceiling={args.epochs}, batch_size={batch_size} (peak_mem={peak_mem}), lr={args.lr}, warmup={args.warmup_steps}, patience={args.patience}, seed={args.seed}
- **Data**: n_train={len(train_ds)}, n_val={len(val_ds)}
- **Result**: best_val_loss={best_val:.4f}, total_steps={step}, elapsed={elapsed/3600:.2f}h
- **Checkpoint**: {ckpt_dir}/best.pt
- **Next**: run test.py --split validate to sanity-check generation, then eval_srrg.py
"""
    with open(experiments_md, "a") as f:
        f.write(entry)
    print(f"training done. best_val_loss={best_val:.4f} run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
