"""RADiFF inference: chest X-ray study (1-2 views) -> structured Findings report.

Decodes at the operating point reported in the paper (3 denoising steps per anatomical block,
per-category emission thresholds and <eob> threshold from config.json).

Usage:
    python inference.py --images frontal.jpg [lateral.jpg]
    python inference.py --studies studies.txt --output reports.jsonl
        # studies.txt: one study per line, image paths separated by commas, frontal first

From Python:
    from inference import load_radiff, generate_reports
    model, tok, cfg = load_radiff(".", device="cuda")
    print(generate_reports(model, tok, cfg, [["frontal.jpg"]])[0])
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from radiff import data as D  # noqa: E402
from radiff.generate import generate_batch  # noqa: E402
from radiff.model import RRGModel  # noqa: E402

# Same detokenization used for every score in the paper: re-attaches punctuation split off by the
# tokenizer and drops bullets with no word content.
_PUNCT_LEFT = re.compile(r"[ \t]+([.,;:!?%)\]}])")
_PUNCT_RIGHT = re.compile(r"([(\[{])[ \t]+")
_INFIX = re.compile(r"(?<=\w)[ \t]*([-/])[ \t]*(?=\w)")
_DECIMAL = re.compile(r"(?<=\d)\.[ \t]+(?=\d)")
_HAS_WORD = re.compile(r"[A-Za-z0-9]")


def detokenize(text: str) -> str:
    out = []
    for line in text.split("\n"):
        t = _PUNCT_LEFT.sub(r"\1", line)
        t = _PUNCT_RIGHT.sub(r"\1", t)
        t = _INFIX.sub(r"\1", t)
        t = _DECIMAL.sub(".", t)
        t = re.sub(r"[ \t]{2,}", " ", t).strip()
        if t.startswith("-") and not _HAS_WORD.search(t):
            continue
        out.append(t)
    return "\n".join(out).strip()


def load_radiff(model_dir: str = HERE, device: str = "cuda"):
    """Returns (model, tokenizer, config). microsoft/rad-dino is fetched from the Hub on first use."""
    from safetensors.torch import load_file

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    a = cfg["architecture"]
    model = RRGModel(vocab_size=a["vocab_size"], max_len=a["max_len"], d_model=a["d_model"],
                     nhead=a["nhead"], num_layers=a["n_layers"], dim_ffn=a["d_ffn"])
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # model.safetensors holds everything except the frozen RAD-DINO backbone, which comes from
    # its own repository at the pinned revision.
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all(k.startswith("visual.vit.") for k in missing), f"missing keys: {missing[:5]}"
    model.to(device).eval()
    tok = D.load_tokenizer(os.path.join(model_dir, "tokenizer.json"))
    return model, tok, cfg


def _collate(studies: list[list[str]]):
    b = len(studies)
    images = torch.zeros(b, D.MAX_VIEWS, 1, D.IMG_LOAD_SIZE, D.IMG_LOAD_SIZE)
    view_mask = torch.ones(b, D.MAX_VIEWS, dtype=torch.bool)  # True = no view
    for i, paths in enumerate(studies):
        assert 1 <= len(paths), "each study needs at least one image"
        for v, p in enumerate(paths[:D.MAX_VIEWS]):
            images[i, v, 0] = D.load_image_tensor(p).squeeze(0)
            view_mask[i, v] = False
    return images, view_mask


@torch.no_grad()
def generate_reports(model, tok, cfg, studies: list[list[str]], batch_size: int = 32) -> list[str]:
    """studies: list of studies, each a list of 1-2 image paths (frontal first)."""
    dec = cfg["decoding"]
    device = next(model.parameters()).device
    reports = []
    for s in range(0, len(studies), batch_size):
        images, view_mask = _collate(studies[s:s + batch_size])
        out = generate_batch(model, tok, images.to(device), view_mask.to(device), "anatomical",
                             steps=dec["steps"], empty_threshold=dec["empty_thresholds"],
                             eob_threshold=dec["eob_threshold"], use_kv_cache=dec["kv_cache"])
        reports.extend(detokenize(r["text"]) for r in out)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", nargs="+", help="one study: 1-2 image paths, frontal first")
    g.add_argument("--studies", help="text file, one study per line, comma-separated paths")
    p.add_argument("--output", help="JSONL output (default: print)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.images:
        studies = [args.images]
    else:
        with open(args.studies) as f:
            studies = [[x.strip() for x in line.split(",") if x.strip()] for line in f if line.strip()]
    model, tok, cfg = load_radiff(HERE, args.device)
    reports = generate_reports(model, tok, cfg, studies, args.batch_size)
    if args.output:
        with open(args.output, "w") as f:
            for paths, r in zip(studies, reports):
                f.write(json.dumps({"images": paths, "findings": r}) + "\n")
    else:
        for r in reports:
            print(r + "\n")


if __name__ == "__main__":
    main()
