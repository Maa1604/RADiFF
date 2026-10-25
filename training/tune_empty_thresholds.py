"""Tune the per-category <empty> decision threshold on VALIDATE only (workflow.md: decode
settings are selected on validation, then frozen for test/test_reviewed).

Which sections a report contains is decided by one number per block, P(<empty>) at its first
slot. Taking the argmax there collapses coverage, because most categories are absent in
48-90% of reports, so <empty> wins by plurality even when its probability is ~0.2. This script
reads the P(<empty>) values recorded by `test.py --collect_p_empty` and picks the threshold
vector maximizing section-presence F1, computed exactly as StructEval does it (per-report
"samples avg" of precision/recall/F1 over the 8 categories, averaged across reports).

Usage:
    python tune_empty_thresholds.py --p_empty_csv <predictions_validate_p_empty.csv> \
        [--split validate] [--grid 0.02] [--rounds 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/lustre/scratch/maasala@upvnet.upv.es/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import data as D


def reference_sections(split: str) -> dict[str, set[str]]:
    """{example id: set of categories with content} straight from findings_section text,
    parsed line-wise (tolerant of the non-canonical labels test_reviewed contains)."""
    from datasets import load_dataset

    out = {}
    for row in load_dataset("StanfordAIMI/srrg_findings", split=split):
        present, current = set(), None
        for line in row["findings_section"].split("\n"):
            stripped = line.strip()
            if stripped.endswith(":") and stripped[:-1].strip() in D.CATEGORIES:
                current = stripped[:-1].strip()
            elif stripped.startswith("-") and current and any(ch.isalnum() for ch in stripped):
                present.add(current)
        out[row["id"]] = present
    return out


def samples_avg_prf(pred_sets, ref_sets):
    """StructEval's section scores: P/R/F1 per report, then averaged (sklearn 'samples avg',
    which scores a report with no predictions and no references as 0)."""
    ps, rs, fs = [], [], []
    for pred, ref in zip(pred_sets, ref_sets):
        tp = len(pred & ref)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(ref) if ref else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        ps.append(p); rs.append(r); fs.append(f)
    n = max(len(fs), 1)
    return sum(ps) / n, sum(rs) / n, sum(fs) / n


def evaluate(thresholds, p_empty, refs):
    preds = [{c for c in D.CATEGORIES if row[c] < thresholds[c]} for row in p_empty]
    return samples_avg_prf(preds, refs)


# objective.md's per-column bar for the two section-presence columns (best across the three
# image-only baselines on Test). Category recall is the column this project is behind on.
BAR_P, BAR_R = 0.7765, 0.8600


def objective(thresholds, p_empty, refs, mode: str):
    """Score a threshold vector. `f1` maximizes section-presence F1 - which is what set
    Abdominal to "never emit", trading away the recall column we actually need. `columns`
    instead maximizes how many of the two section-presence bars are cleared, then the slack
    above them, which is the rule workflow.md's stopping condition implies."""
    p, r, f = evaluate(thresholds, p_empty, refs)
    if mode == "f1":
        return f
    cleared = (p >= BAR_P) + (r >= BAR_R)
    return cleared + 0.1 * min(p - BAR_P, r - BAR_R) + 0.01 * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_empty_csv", required=True)
    ap.add_argument("--split", default="validate")
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--bar_p", type=float, default=None, help="override BAR_P for --mode columns (fraction)")
    ap.add_argument("--bar_r", type=float, default=None, help="override BAR_R for --mode columns (fraction)")
    ap.add_argument("--mode", default="f1", choices=["f1", "columns"],
                    help="f1: maximize section-presence F1. columns: maximize how many of "
                         "objective.md's Category precision/recall bars are cleared")
    args = ap.parse_args()
    global BAR_P, BAR_R
    BAR_P = args.bar_p if args.bar_p is not None else BAR_P
    BAR_R = args.bar_r if args.bar_r is not None else BAR_R

    rows = list(csv.DictReader(open(args.p_empty_csv)))
    ref_by_id = reference_sections(args.split)
    p_empty = [{c: float(r[c]) for c in D.CATEGORIES} for r in rows]
    refs = [ref_by_id[r["id"]] for r in rows]
    print(f"{len(rows)} reports; mean reference sections = "
          f"{sum(len(r) for r in refs) / max(len(refs), 1):.2f}")

    grid = [round(i * args.grid, 4) for i in range(int(1 / args.grid) + 1)]
    thresholds = {c: 0.5 for c in D.CATEGORIES}  # 0.5 ~ "empty unless clearly not"
    print(f"objective={args.mode}; start: P/R/F1 = "
          f"{tuple(round(100 * v, 2) for v in evaluate(thresholds, p_empty, refs))}")
    for rnd in range(args.rounds):
        for cat in D.CATEGORIES:
            best = thresholds[cat]
            best_score = objective(thresholds, p_empty, refs, args.mode)
            for tau in grid:
                trial = dict(thresholds, **{cat: tau})
                score = objective(trial, p_empty, refs, args.mode)
                if score > best_score + 1e-9:
                    best, best_score = tau, score
            thresholds[cat] = best
        p, r, f = evaluate(thresholds, p_empty, refs)
        print(f"round {rnd}: P={100 * p:.2f} R={100 * r:.2f} F1={100 * f:.2f}")

    mean_sections = sum(sum(1 for c in D.CATEGORIES if row[c] < thresholds[c]) for row in p_empty)
    print("\nper-category thresholds (emit section when P(<empty>) < tau):")
    for c in D.CATEGORIES:
        print(f"  {c:<40} tau={thresholds[c]:.2f}")
    print(f"mean sections emitted = {mean_sections / max(len(p_empty), 1):.2f}")
    print("\n--empty_thresholds " + ",".join(f"{thresholds[c]:.4f}" for c in D.CATEGORIES))
    if args.out_json:
        json.dump({"thresholds": thresholds}, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
