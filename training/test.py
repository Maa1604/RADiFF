"""Sanity-check generation on a checkpoint + write a predictions CSV. workflow.md step 4.

Usage:
    conda activate rrg-cxr
    python test.py --run_dir /lustre/.../runs/anatomical_12345 --split validate \
        --output_csv predictions_validate.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/lustre/scratch/maasala@upvnet.upv.es/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from src import data as D
from src.generate import generate_batch
from src.model import RRGModel
from train import TOKENIZER_PATH, build_model


class ImageOnlyDataset(torch.utils.data.Dataset):
    """Generation needs only images, never tokenized targets, so this skips
    parse_findings/tokenize_report entirely — unlike RRGDataset, it cannot fail on
    a findings_section that doesn't match the canonical 8-category schema (e.g.
    test_reviewed's human-reviewed rows use some non-canonical category labels;
    SPEC.md's audit already recorded 18/233 such parse errors there). References
    for scoring come straight from the raw findings_section (SPEC.md Section 5),
    never through this dataset."""

    def __init__(self, hf_dataset, max_views=D.MAX_VIEWS, tokenizer=None, with_context=False):
        roots = D.dataset_roots()["image_roots"]
        self.rows = [
            {"image_paths": D.resolve_image_paths(list(ex["image_paths"])[:max_views], roots),
             "id": ex["id"],
             "ctx_ids": (D.tokenize_context(ex, tokenizer) if with_context else [])}
            for ex in hf_dataset
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def gen_collate(batch):
    b = len(batch)
    images = torch.zeros(b, D.MAX_VIEWS, 1, D.IMG_LOAD_SIZE, D.IMG_LOAD_SIZE)
    view_mask = torch.ones(b, D.MAX_VIEWS, dtype=torch.bool)  # True = pad (no view)
    for i, item in enumerate(batch):
        for v, p in enumerate(item["image_paths"][:D.MAX_VIEWS]):
            images[i, v, 0] = D.load_image_tensor(p).squeeze(0)
            view_mask[i, v] = False
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
    return {"images": images, "view_mask": view_mask, "ex_ids": [item["id"] for item in batch],
            "ctx_ids": ctx_ids, "ctx_pad": ctx_pad}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--split", default="validate", choices=["validate", "test", "test_reviewed"])
    p.add_argument("--limit", type=int, default=-1, help="-1 = full split")
    p.add_argument("--ids", type=str, default=None, help="comma-separated study ids to decode (e.g. mimic-59114520)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=12, help="denoising steps per block (diffusion variants)")
    p.add_argument("--output_csv", default=None)
    p.add_argument("--chunk", type=int, default=None,
                   help="semi-autoregressive window (slots) for the block sampler; default: one pool")
    p.add_argument("--empty_threshold", type=float, default=None,
                   help="emit an empty section only when P(<empty>) >= this; default: argmax")
    p.add_argument("--empty_thresholds", type=str, default=None,
                   help="comma-separated per-category thresholds (8 values, canonical order); "
                        "overrides --empty_threshold. Tune with tune_empty_thresholds.py on validate")
    p.add_argument("--collect_p_empty", action="store_true",
                   help="also write predictions_<split>_p_empty.csv (P(<empty>) per category), "
                        "the input tune_empty_thresholds.py needs")
    p.add_argument("--remask_threshold", type=float, default=None,
                   help="confidence floor below which an already-committed slot is returned "
                        "to the masked pool for a later step to revise (anatomical only)")
    p.add_argument("--remask_reserve", type=int, default=2,
                   help="no remasking in the last N steps, so the tail always converges. "
                        "At S=3 the default 2 allows exactly one firing, after step 0.")
    p.add_argument("--kv_cache", action="store_true",
                   help="reuse the per-study cross-attention K/V (output-identical, ~1.3x)")
    p.add_argument("--cat_bias", type=str, default=None,
                   help="comma-separated per-category <cat_k> logit bias for the ar/fixed "
                        "variants, which have no per-category emission slot. Tune with "
                        "tune_cat_bias.py on validate.")
    p.add_argument("--eob_threshold", type=float, default=None,
                   help="accept <eob> only when its probability >= this; higher = longer sections")
    p.add_argument("--no_repeat", action="store_true",
                   help="never commit a content token equal to an adjacent slot's token; take the "
                        "best non-colliding content token instead (diffusion variants)")
    p.add_argument("--spread", action="store_true",
                   help="never commit two adjacent slots in the same denoising step")
    p.add_argument("--revise", action="store_true",
                   help="Option A: checkerboard revision of each step's commits (1 extra call/step)")
    p.add_argument("--bigram_path", default=None, help="Option B: [V,V] log P(next|prev) tensor")
    p.add_argument("--bigram_lambda", type=float, default=0.0, help="Option B weight; 0 = off")
    p.add_argument("--ctc", action="store_true", help="decode a --ctc trained model (src/ctc.py)")
    p.add_argument("--sched", type=str, default=None,
                   help="sequential sampler: fraction of still-masked slots committed per step, e.g. 0.15,0.4")
    p.add_argument("--no_adjacent2", action="store_true",
                   help="(--parallel_blocks) --no_adjacent, case/space-insensitive and across a period")
    p.add_argument("--compact", action="store_true",
                   help="(--parallel_blocks) drop empty sections from denoiser calls after the emptiness decision")
    p.add_argument("--bullet_conf", action="store_true",
                   help="(--parallel_blocks) also write predictions_<split>_bullets.jsonl with per-bullet commit confidence")
    p.add_argument("--parallel_blocks", action="store_true",
                   help="anatomical_nocontext only: denoise all 8 sections in one batch "
                        "(src/parallel.py); S steps cost 1+S calls instead of 8*(1+S)")
    p.add_argument("--no_adjacent", action="store_true",
                   help="packed parallel decoder: ban committing a neighbour's word token (src/parallel.py)")
    p.add_argument("--struct", action="store_true",
                   help="packed parallel decoder: bullet grammar constraints (src/parallel.py STRUCT)")
    p.add_argument("--defer_boundary", type=float, default=None,
                   help="multiply '.'/newline commit confidence by this before the last step")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda":
        assert torch.cuda.is_available(), "test.py requires a GPU node"

    with open(os.path.join(args.run_dir, "config.json")) as f:
        config = json.load(f)
    variant = config["variant"]

    tok = D.load_tokenizer(TOKENIZER_PATH)
    from datasets import load_dataset

    hf = load_dataset("StanfordAIMI/srrg_findings", split=args.split)
    if args.limit >= 0:
        hf = hf.select(range(min(args.limit, len(hf))))
    if args.ids:
        keep = set(args.ids.split(","))
        hf = hf.select([i for i, r in enumerate(hf["id"]) if r in keep])
    ds = ImageOnlyDataset(hf, tokenizer=tok, with_context=config.get("use_context", False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=gen_collate, num_workers=8)

    model = build_model(config).to(device)
    ckpt = torch.load(os.path.join(args.run_dir, "checkpoints", args.checkpoint), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded checkpoint step={ckpt.get('step')} val_loss={ckpt.get('val_loss')}", flush=True)

    # References are the raw findings_section text (SPEC.md Section 5) — never parsed, so a
    # split with non-canonical category labels (e.g. test_reviewed) can't break scoring.
    refs = [row["findings_section"] for row in hf]

    empty_thresholds = args.empty_threshold
    if args.empty_thresholds:
        empty_thresholds = [float(x) for x in args.empty_thresholds.split(",")]
        assert len(empty_thresholds) == len(D.CATEGORIES), "need one threshold per category"

    if args.no_adjacent:
        import src.parallel as _par
        _par.NO_ADJ = True
    if args.no_adjacent2:
        import src.parallel as _par
        _par.NO_ADJ = 2
        if not args.parallel_blocks:  # sequential sampler: same ban in src/diffusion.py
            import src.diffusion as _diff
            from src.generate import word_token_ids
            V = model.decoder.tok_embed.num_embeddings
            wtok = torch.zeros(V, dtype=torch.bool, device=device)
            wtok[list(word_token_ids(tok))] = True
            wtok[:len(D.SPECIAL_IDS)] = False
            gid, var = _par._word_groups(tok, V, device)
            _diff.ADJ_BAN = (wtok, gid, var)
            _diff.DOT_TOK = tok.token_to_id(".")  # 27 with the paper tokenizer; differs for RRG_TOK_DIR tokenizers
    if args.compact:
        import src.parallel as _par
        _par.COMPACT = True
    if args.bullet_conf:
        import src.parallel as _par
        _par.BULLET_CONF = True
        import src.generate as _gen
        _gen.BULLET_CONF = True
        _gen.NL_TOK = tok.token_to_id("\u010a")  # byte-level newline; 212 with the paper tokenizer
    if args.struct:
        import src.parallel as _par
        _par.STRUCT = True
        if not args.parallel_blocks:
            import src.diffusion as _diff
            _diff.STRUCT_M = _par._struct_masks(tok, model.decoder.tok_embed.num_embeddings, device)
    if args.sched:
        import src.diffusion as _diff
        _diff.SCHED = [float(x) for x in args.sched.split(",")]
    if args.defer_boundary is not None:
        import src.diffusion as _diff
        _diff.DEFER_ALPHA = args.defer_boundary
    bigram = torch.load(args.bigram_path, map_location=device) if args.bigram_path else None
    output_csv = args.output_csv or os.path.join(args.run_dir, f"predictions_{args.split}.csv")
    rows, p_empty_rows, bullet_rows = [], [], []
    diag_totals = {"missing_terminator": 0, "empty": 0, "unexpected_special": 0, "n": 0}
    latencies = []
    idx = 0
    t0 = time.time()
    for batch in loader:
        images = batch["images"].to(device)
        view_mask = batch["view_mask"].to(device)
        ctx_ids = batch.get("ctx_ids")
        ctx_pad = batch.get("ctx_pad")
        if ctx_ids is not None:
            ctx_ids, ctx_pad = ctx_ids.to(device), ctx_pad.to(device)
        if args.ctc:
            from src.ctc import generate_anatomical_ctc
            results = generate_anatomical_ctc(model, tok, images, view_mask, steps=args.steps,
                                              empty_threshold=empty_thresholds,
                                              use_kv_cache=args.kv_cache,
                                              collect_p_empty=args.collect_p_empty)
        elif args.parallel_blocks:
            assert variant == "anatomical_nocontext", "parallel sections need a no-context model"
            from src.parallel import generate_parallel
            results = generate_parallel(model, tok, images, view_mask, steps=args.steps,
                                        empty_threshold=empty_thresholds,
                                        eob_threshold=args.eob_threshold,
                                        use_kv_cache=args.kv_cache,
                                        collect_p_empty=args.collect_p_empty)
        else:
          results = generate_batch(model, tok, images, view_mask, variant, steps=args.steps,
                                 ctx_ids=ctx_ids, ctx_pad=ctx_pad,
                                 chunk=args.chunk, empty_threshold=empty_thresholds,
                                 remask_threshold=args.remask_threshold,
                                 remask_reserve=args.remask_reserve,
                                 use_kv_cache=args.kv_cache,
                                 cat_bias=([float(x) for x in args.cat_bias.split(',')]
                                           if args.cat_bias else None),
                                 eob_threshold=args.eob_threshold,
                                 collect_p_empty=args.collect_p_empty,
                                 no_repeat=args.no_repeat, spread=args.spread,
                                 revise=args.revise, bigram=bigram, bigram_lambda=args.bigram_lambda)
        for i, r in enumerate(results):
            ex_id = batch["ex_ids"][i]
            rows.append({"id": ex_id, "ref": refs[idx], "hyp": r["text"]})
            diag_totals["missing_terminator"] += r["diagnostics"].get("missing_terminator", 0)
            diag_totals["empty"] += r["diagnostics"].get("empty", 0)
            diag_totals["unexpected_special"] += r["diagnostics"].get("unexpected_special", 0)
            diag_totals["n"] += 1
            latencies.append(r["diagnostics"].get("latency_s", 0.0))
            if "bullets" in r["diagnostics"]:
                bullet_rows.append({"id": ex_id, "ref": refs[idx], "bullets": r["diagnostics"]["bullets"]})
            if "p_empty" in r["diagnostics"]:
                p_empty_rows.append({"id": ex_id,
                                     **{c: r["diagnostics"]["p_empty"].get(c) for c in D.CATEGORIES}})
            idx += 1
        print(f"generated {idx}/{len(ds)}", flush=True)
    elapsed = time.time() - t0

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "ref", "hyp"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions to {output_csv}", flush=True)

    if bullet_rows:
        with open(output_csv.replace(".csv", "_bullets.jsonl"), "w") as f:
            for br in bullet_rows:
                f.write(json.dumps(br) + "\n")
    if p_empty_rows:
        pe_path = output_csv.replace(".csv", "_p_empty.csv")
        with open(pe_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id"] + list(D.CATEGORIES))
            w.writeheader()
            w.writerows(p_empty_rows)
        print(f"wrote P(<empty>) per category to {pe_path}", flush=True)

    diag_totals["mean_latency_s"] = sum(latencies) / max(len(latencies), 1)
    diag_totals["total_elapsed_s"] = elapsed
    diag_path = output_csv.replace(".csv", "_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diag_totals, f, indent=2)
    print(f"diagnostics: {diag_totals}", flush=True)

    n_empty_hyp = sum(1 for r in rows if not r["hyp"].strip())
    n_degenerate_repeat = sum(1 for r in rows if len(set(r["hyp"].split())) < 3 and r["hyp"].strip())
    print(f"sanity: {n_empty_hyp}/{len(rows)} fully-empty generations, "
          f"{n_degenerate_repeat}/{len(rows)} near-degenerate (<3 unique words)", flush=True)
    print("\n--- 3 example generations ---")
    for r in rows[:3]:
        print(f"\nid={r['id']}\nREF:\n{r['ref'][:400]}\nHYP:\n{r['hyp'][:400]}")


if __name__ == "__main__":
    main()
