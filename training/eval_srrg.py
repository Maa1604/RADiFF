"""Official SRRG-Findings benchmark metrics (BLEU, ROUGE-L, BERTScore, F1-RadGraph,
F1-SRR-BERT, Category presence P/R/F1), matching https://stanford-aimi.github.io/srrg.html

Ported from the reference project's eval/eval_srrg.py (see SPEC.md Section 5), pointed at
the StructEval install on Sirius. Must be run with the `srrg_eval` conda env, NOT `rrg-cxr`.

Model-agnostic: given a predictions_<split>.csv (id, ref, hyp columns, written by
test.py), runs the official scorer and exports srrg_metrics.json/.md plus
organ_level_srr_bert_<split>.md, matching this project's objective.md table format.
"""
import argparse
import re
import json
import os
import sys

STRUCTEVAL_REPO = os.path.expanduser("~/StructEval")
if STRUCTEVAL_REPO not in sys.path:
    sys.path.insert(0, STRUCTEVAL_REPO)

from structeval import StructEval


# --- surface-form detokenization -------------------------------------------------------
# See reference project's eval/eval_srrg.py for the full rationale: a word-level tokenizer
# splits punctuation and hyphenated/slashed compounds off from their neighbors; references
# are ordinary prose, so BLEU needs this reattached first. Keep this if this model's decoder
# tokenizes the same way (BPE trained on the same content strings, standard whitespace-ish
# split behavior); harmless no-op if it doesn't apply.
#
# Every rule here is confined to horizontal whitespace ([ \t]) and applied line by line.
# StructEval's parse_findings splits on '\n' and only recognizes an organ header when a whole
# line is exactly "<Organ>:", so newlines are structure, not spacing. The reference project's
# version collapsed r"\s{2,}" (which matches "\n\n") into a single space, fusing every section
# into one line of the first section's bullets: StructEval then saw only "Lungs and Airways"
# in every hypothesis (Category P ~95% / R ~24%, every other organ scored 0). Its baselines
# were unaffected only because their outputs don't put blank lines between sections.
_PUNCT_LEFT = re.compile(r"[ \t]+([.,;:!?%)\]}])")
_PUNCT_RIGHT = re.compile(r"([(\[{])[ \t]+")
_INFIX = re.compile(r"(?<=\w)[ \t]*([-/])[ \t]*(?=\w)")
_DECIMAL = re.compile(r"(?<=\d)\.[ \t]+(?=\d)")
_HAS_WORD = re.compile(r"[A-Za-z0-9]")


def detokenize_line(line):
    t = _PUNCT_LEFT.sub(r"\1", line)
    t = _PUNCT_RIGHT.sub(r"\1", t)
    t = _INFIX.sub(r"\1", t)
    t = _DECIMAL.sub(".", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def detokenize(text):
    """Line-structure-preserving detokenization. Bullets with no word content (e.g. a bare
    "- " or "- .." left by the generator) carry no finding and are dropped; a section left with
    no bullets is then ignored by StructEval's parser exactly as an absent section would be."""
    out = []
    for line in text.split("\n"):
        t = detokenize_line(line)
        if t.startswith("-") and not _HAS_WORD.search(t):
            continue
        out.append(t)
    return "\n".join(out).strip()


def print_summary(model_name, split, results):
    overall = results["overall_avg_scores"]
    section = results["section_scores"]
    print(f"\n{model_name} | {split}")
    print(f"  BLEU:       {overall.get('bleu', float('nan')):.4f}")
    print(f"  ROUGE-L:    {overall.get('rougeL', float('nan')):.4f}")
    print(f"  BERTScore:  {overall.get('bertscore', float('nan')):.4f}")
    print(
        "  F1-RadGraph (simple/partial/complete): "
        f"{overall.get('radgraph_simple', float('nan')):.4f} / "
        f"{overall.get('radgraph_partial', float('nan')):.4f} / "
        f"{overall.get('radgraph_complete', float('nan')):.4f}"
    )
    print(
        f"  F1-SRR-BERT   P={overall.get('samples_avg_precision', float('nan')):.4f} "
        f"R={overall.get('samples_avg_recall', float('nan')):.4f} "
        f"F1={overall.get('samples_avg_f1-score', float('nan')):.4f}"
    )
    print(
        f"  Category      P={section['section_avg_precision']:.4f} "
        f"R={section['section_avg_recall']:.4f} "
        f"F1={section['section_avg_f1-score']:.4f}"
    )


def save_markdown_table(all_results, model_name, output_path, diseases_mode):
    def pct(x):
        return f"{x * 100:.2f}"

    lines = [
        "| Model | Split | BLEU | ROUGE-L | BERTScore | F1-RadGraph "
        "| F1-SRR-BERT P | F1-SRR-BERT R | F1-SRR-BERT F1 | Category P | Category R | Category F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for split, results in all_results.items():
        overall = results["overall_avg_scores"]
        section = results["section_scores"]
        lines.append(
            f"| {model_name} | {split} "
            f"| {pct(overall.get('bleu', 0.0))} "
            f"| {pct(overall.get('rougeL', 0.0))} "
            f"| {pct(overall.get('bertscore', 0.0))} "
            f"| {pct(overall.get('radgraph_partial', 0.0))} "
            f"| {pct(overall.get('samples_avg_precision', 0.0))} "
            f"| {pct(overall.get('samples_avg_recall', 0.0))} "
            f"| {pct(overall.get('samples_avg_f1-score', 0.0))} "
            f"| {pct(section['section_avg_precision'])} "
            f"| {pct(section['section_avg_recall'])} "
            f"| {pct(section['section_avg_f1-score'])} |"
        )

    note = (
        f"\n*Values scaled to 0-100 to match the SRRG-Findings paper's table. "
        f"F1-RadGraph is the 'partial' reward level. F1-SRR-BERT uses the "
        f"`{diseases_mode}` taxonomy. Evaluated with `aligned=False` "
        f"(\"SRRG-Findings (unaligned)\"), `do_lower_case=True`.*\n"
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n" + note)
    print(f"markdown table saved to {output_path}")


ORGAN_TABLE_ORDER = [
    "Pleura",
    "Abdominal",
    "Hila and Mediastinum",
    "Other",
    "Lungs and Airways",
    "Cardiovascular",
    "Musculoskeletal and Chest Wall",
    "Tubes, Catheters, and Support Devices",
]


def save_organ_level_srr_bert_table(all_results, model_name, output_dir, diseases_mode):
    def pct(x):
        return f"{x * 100:.2f}"

    for split, results in all_results.items():
        organ_scores = results["organ_avg_scores"]
        lines = [
            f"# Organ-level F1-SRR-BERT ({split})",
            "",
            f"`{model_name}`, `{diseases_mode}` taxonomy, `{split}` split. Same row order as "
            "the SRRG-Findings paper's Table 8.",
            "",
            "| Organ | Precision | Recall | F1-Score |",
            "|---|---|---|---|",
        ]
        for organ in ORGAN_TABLE_ORDER:
            key = organ.lower() + ":"
            s = organ_scores.get(key)
            if s is None:
                lines.append(f"| {organ} | N/A | N/A | N/A |")
                continue
            lines.append(
                f"| {organ} "
                f"| {pct(s.get('samples_avg_precision', 0.0))} "
                f"| {pct(s.get('samples_avg_recall', 0.0))} "
                f"| {pct(s.get('samples_avg_f1-score', 0.0))} |"
            )

        output_path = os.path.join(output_dir, f"organ_level_srr_bert_{split}.md")
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"organ-level F1-SRR-BERT table saved to {output_path}")


def score_and_report(refs_by_split, hyps_by_split, model_name, diseases_mode, output_json, output_md):
    print("loading official StructEval scorers (BLEU, ROUGE-L, BERTScore, F1-RadGraph, F1-SRR-BERT)...")
    evaluator = StructEval(
        do_bleu=True, do_rouge=True, do_bertscore=True, do_radgraph=True,
        do_green=False, do_diseases=True, diseases_mode=diseases_mode,
    )

    all_results = {}
    for split, refs in refs_by_split.items():
        hyps = hyps_by_split[split]
        print(f"\n=== split: {split} ===")
        print(f"scoring {len(hyps)} reports with StructEval (aligned=False, "
              "matching the official 'SRRG-Findings (unaligned)' benchmark)...")
        results = evaluator.forward(refs=refs, hyps=hyps, section="findings", aligned=False, do_lower_case=True)
        all_results[split] = results
        print_summary(model_name, split, results)

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nfull results (incl. per-organ breakdown) saved to {output_json}")

    save_markdown_table(all_results, model_name, output_md, diseases_mode)
    save_organ_level_srr_bert_table(all_results, model_name, os.path.dirname(output_md), diseases_mode)
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions_dir", type=str, required=True,
        help="directory containing predictions_<split>.csv files (id,ref,hyp columns), "
        "as written by test.py",
    )
    parser.add_argument("--splits", type=str, nargs="+", default=["validate", "test", "test_reviewed"])
    parser.add_argument("--diseases_mode", type=str, default="leaves",
                        choices=["leaves", "upper", "leaves_with_statuses", "upper_with_statuses"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--no_detokenize", action="store_true")
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = os.path.join(args.predictions_dir, "srrg_metrics.json")
    if args.output_md is None:
        args.output_md = os.path.join(args.predictions_dir, "srrg_metrics.md")

    import csv
    refs_by_split, hyps_by_split = {}, {}
    for split in args.splits:
        csv_path = os.path.join(args.predictions_dir, f"predictions_{split}.csv")
        refs, hyps = [], []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                refs.append(row["ref"])
                hyp = row["hyp"]
                hyps.append(hyp if args.no_detokenize else detokenize(hyp))
        refs_by_split[split] = refs
        hyps_by_split[split] = hyps

    score_and_report(
        refs_by_split, hyps_by_split, args.model_name, args.diseases_mode,
        args.output_json, args.output_md,
    )


if __name__ == "__main__":
    main()
