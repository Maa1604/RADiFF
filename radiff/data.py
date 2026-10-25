"""Data module for anatomy-block diffusion RRG.

Implements SPEC.md §1: parsing of `findings_section`, block serialization with
fixed special-token ids, per-category capacities, image-path prefix mapping,
and a torch-optional dataset. Parsing/tokenization logic is pure Python; torch,
tokenizers, datasets, and PIL are imported lazily so this module can be used on
machines without them.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Categories and special tokens (SPEC §1, frozen)
# ---------------------------------------------------------------------------

CATEGORIES: tuple[str, ...] = (
    "Lungs and Airways",
    "Cardiovascular",
    "Pleura",
    "Musculoskeletal and Chest Wall",
    "Tubes, Catheters, and Support Devices",
    "Hila and Mediastinum",
    "Other",
    "Abdominal",
)

SPECIAL_IDS: dict[str, int] = {
    "<pad>": 0,
    "<mask>": 1,
    "<bos>": 2,
    "<eob>": 3,
    "<empty>": 4,
    "<eor>": 5,
    **{f"<cat_{k}>": 6 + k for k in range(8)},
}

PAD_ID: int = SPECIAL_IDS["<pad>"]
MASK_ID: int = SPECIAL_IDS["<mask>"]
BOS_ID: int = SPECIAL_IDS["<bos>"]
EOB_ID: int = SPECIAL_IDS["<eob>"]
EMPTY_ID: int = SPECIAL_IDS["<empty>"]
EOR_ID: int = SPECIAL_IDS["<eor>"]
CAT_IDS: list[int] = [SPECIAL_IDS[f"<cat_{k}>"] for k in range(8)]

# Frozen initial capacities from SPEC §1 (profile 0.995 + margin).
DEFAULT_CAPACITIES: dict[str, int] = {
    "Lungs and Airways": 56,
    "Cardiovascular": 33,
    "Pleura": 42,
    "Musculoskeletal and Chest Wall": 43,
    "Tubes, Catheters, and Support Devices": 74,
    "Hila and Mediastinum": 35,
    "Other": 37,
    "Abdominal": 25,
}

MAX_VIEWS = 2

# Clinical context, matched field-for-field to the MAIRA-2 "+ indication/technique/comparison"
# row in objective.md so the comparison is like-for-like. exam_type_section is deliberately NOT
# used: MAIRA-2 does not receive it, and it is near-constant boilerplate.
CONTEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("indication", "history_section"),
    ("technique", "technique_section"),
    ("comparison", "comparison_section"),
)
# Coverage measured on the real splits: indication present 97.9-99.4% (median 9-11 words, p95 26),
# technique 97.9-99.6% (median 6-7), comparison 42-72% (median 3-4). 96 BPE tokens clears the p95
# of all three concatenated with room to spare.
MAX_CONTEXT_LEN = 96


class ParseError(ValueError):
    """Raised when a `findings_section` cannot be parsed per SPEC §1."""


# ---------------------------------------------------------------------------
# Parsing / serialization (SPEC §1 Parsing)
# ---------------------------------------------------------------------------

def parse_findings(findings_section: str) -> dict[str, str]:
    """Parse a `findings_section` string into {category: content}.

    Sections are separated by blank lines; each section starts with an exact
    `Category Name:` header line and continues with `- ` bullet lines. Content
    is the bullets joined by `\n` with the `- ` prefix stripped, otherwise
    verbatim. Absent categories do not appear in the result. Raises
    ParseError on unknown category names or malformed sections.
    """
    text = (findings_section or "").strip()
    if not text:
        return {}

    sections: dict[str, str] = {}
    for block in _split_blocks(text):
        lines = block.split("\n")
        header = lines[0].rstrip()
        if not header.endswith(":"):
            raise ParseError(f"section header missing ':': {lines[0]!r}")
        name = header[:-1].rstrip()
        if name not in CATEGORIES:
            raise ParseError(f"unknown category: {name!r}")
        if name in sections:
            raise ParseError(f"duplicate category: {name!r}")
        bullets: list[str] = []
        for line in lines[1:]:
            if not line.startswith("- "):
                raise ParseError(f"malformed bullet line: {line!r}")
            bullets.append(line[2:])
        sections[name] = "\n".join(bullets)
    return sections


def _split_blocks(text: str) -> list[str]:
    """Split on blank lines (lines containing only whitespace)."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip() == "":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(b) for b in blocks]


def serialize_findings(sections: Mapping[str, str]) -> str:
    """Inverse of parse_findings for evaluation output.

    Sections are emitted in canonical order; only non-empty ones appear, in
    the dataset style `Category:\n- line\n- line\n\n`.
    """
    out: list[str] = []
    for cat in CATEGORIES:
        content = sections.get(cat, "")
        if content.strip() == "":
            continue
        out.append(f"{cat}:\n")
        for line in content.split("\n"):
            out.append(f"- {line}\n")
        out.append("\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(path: str) -> Any:
    """Load a `tokenizers.Tokenizer` from a tokenizer.json file."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


# ---------------------------------------------------------------------------
# Block serialization (SPEC §1 Block serialization)
# ---------------------------------------------------------------------------

def tokenize_report(
    sections: Mapping[str, str],
    tokenizer: Any,
    capacities: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Serialize parsed sections into the fixed 8-block token sequence.

    Per section k: `[cat_k] t1 ... tn [eob]` (n>=1 content tokens, or the
    single `<empty>` token), concatenated in canonical order, plus a final
    `[eor]` token. `capacities` limits content tokens per category; truncated
    content sets `overflow[cat] = True`. `empty[cat]` is True for sections
    without content. `block_spans` gives (start, end-exclusive) of each block
    in the returned ids.
    """
    ids: list[int] = []
    block_spans: list[tuple[int, int]] = []
    overflow: dict[str, bool] = {}
    empty: dict[str, bool] = {}

    for k, cat in enumerate(CATEGORIES):
        start = len(ids)
        ids.append(CAT_IDS[k])
        content = sections.get(cat, "")
        if content.strip() == "":
            ids.append(EMPTY_ID)
            empty[cat] = True
            overflow[cat] = False
        else:
            empty[cat] = False
            tokens = list(tokenizer.encode(content).ids)
            cap = capacities.get(cat) if capacities is not None else None
            if cap is not None and len(tokens) > cap:
                overflow[cat] = True
                tokens = tokens[:cap]
            else:
                overflow[cat] = False
            ids.extend(tokens)
        ids.append(EOB_ID)
        block_spans.append((start, len(ids)))

    ids.append(EOR_ID)
    return {
        "ids": ids,
        "block_spans": block_spans,
        "overflow": overflow,
        "empty": empty,
    }


def build_context_string(ex: Mapping[str, Any]) -> str:
    """Flatten the clinical-context fields into one labelled string.

    Labels are plain words, not new special tokens, so the existing tokenizer.json is reused
    unchanged and vocab_size stays 8192 - a new special token would invalidate every existing
    checkpoint. Absent fields are dropped rather than emitted empty, so the model never learns
    that "comparison:" with nothing after it is meaningful.
    """
    parts: list[str] = []
    for label, col in CONTEXT_FIELDS:
        raw = ex.get(col)
        if raw is None:
            continue
        val = " ".join(str(raw).split())
        if not val or val.lower() in ("none", "nan", "n/a"):
            continue
        parts.append(f"{label}: {val}")
    return " ".join(parts)


def tokenize_context(ex: Mapping[str, Any], tokenizer: Any,
                     max_len: int = MAX_CONTEXT_LEN) -> list[int]:
    """Context field ids, truncated to `max_len`. Empty list when no context exists."""
    text = build_context_string(ex)
    if not text:
        return []
    return list(tokenizer.encode(text).ids)[:max_len]


def fixed_block_spans(capacities: Mapping[str, int] | None = None) -> list[tuple[int, int]]:
    """Absolute (start, end) of each block in the fixed-capacity layout: block k is
    exactly `capacities[k] + 2` tokens wide (cat + content/eob slots), so it always
    starts at the same position regardless of what earlier blocks contain."""
    caps = capacities if capacities is not None else DEFAULT_CAPACITIES
    spans, start = [], 0
    for cat in CATEGORIES:
        width = caps[cat] + 2
        spans.append((start, start + width))
        start += width
    return spans


def fixed_layout(
    ids: Sequence[int],
    block_spans: Sequence[tuple[int, int]],
    capacities: Mapping[str, int] | None = None,
) -> tuple[list[int], list[bool]]:
    """Re-lay a `tokenize_report` sequence into fixed-capacity blocks (README: "use the
    same capacities at training and inference ... explicit trailing padding targets").

    Block k becomes `[cat_k] content [eob] [eob]*fill`, exactly `cap_k + 2` tokens, i.e.
    the same `[cat] + (cap + 1)` slot geometry `denoise_block` generates into. Without
    this, training blocks had their true length (the last visible slot was always
    `<eob>`, leaking the target length through the block boundary) while inference blocks
    had `cap + 1` slots, so the model filled every slot and rarely terminated; later
    blocks also started at absolute positions never seen in training.

    Returns (ids, filler) where `filler[i]` marks the trailing `<eob>` fillers that
    follow a block's first `<eob>`. Total length is `sum(cap + 2) + 1` (final `<eor>`).
    """
    caps = capacities if capacities is not None else DEFAULT_CAPACITIES
    out: list[int] = []
    filler: list[bool] = []
    for k, (s, e) in enumerate(block_spans):
        width = caps[CATEGORIES[k]] + 2
        block = [int(t) for t in ids[s:e]]
        if len(block) > width:
            raise ValueError(f"block {k} has {len(block)} tokens > fixed width {width}")
        out.extend(block)
        filler.extend([False] * len(block))
        out.extend([EOB_ID] * (width - len(block)))
        filler.extend([True] * (width - len(block)))
    out.append(EOR_ID)
    filler.append(False)
    return out, filler


# ---------------------------------------------------------------------------
# Capacities (SPEC §1 Capacities)
# ---------------------------------------------------------------------------

def compute_capacities(
    lengths: Mapping[str, Sequence[int]],
    quantile: float = 0.995,
    margin: int = 4,
    min_cap: int = 8,
) -> dict[str, int]:
    """capacity_k = quantile_q(n_k) + margin, floored at min_cap.

    The quantile is the empirical quantile over the given per-category content
    token lengths n (ceil-based index into the sorted values).
    """
    capacities: dict[str, int] = {}
    for cat in CATEGORIES:
        values = sorted(lengths.get(cat, []))
        if not values:
            capacities[cat] = min_cap
            continue
        idx = min(len(values) - 1, max(0, math.ceil(quantile * len(values)) - 1))
        capacities[cat] = max(min_cap, values[idx] + margin)
    return capacities


# ---------------------------------------------------------------------------
# Image path resolution (SPEC §1 image roots)
# ---------------------------------------------------------------------------

def dataset_roots() -> dict[str, Any]:
    """Dataset image roots (training/evaluation only), overridable via environment variables.

    Returns {"image_roots": {prefix: root}, "hf_home": path}.
    """
    base = os.environ.get("RRG_DATA_ROOT", ".")
    return {
        "image_roots": {
            "mimic-cxr-images-512/files/": os.environ.get(
                "RRG_MIMIC_ROOT", f"{base}/mimic_cxr/files_jpg_512/files"
            ),
            "CheXpert-v1.0-512/": os.environ.get(
                "RRG_CHEXPERT_ROOT",
                f"{base}/anatomy_block_isbi/data/chexpert_jpg_512",
            ),
        },
        "hf_home": os.environ.get("RRG_HF_HOME", f"{base}/hf_cache"),
    }


def resolve_image_paths(
    image_paths: Sequence[str], roots: Mapping[str, str]
) -> list[str]:
    """Map dataset image paths to local absolute paths by exact prefix.

    The first matching prefix is replaced by its root. Raises FileNotFoundError
    if no prefix matches or the resolved file is missing/unreadable. Never
    substitutes placeholders and never silently drops examples.
    """
    resolved: list[str] = []
    for path in image_paths:
        match = next(
            ((p, r) for p, r in roots.items() if path.startswith(p)), None
        )
        if match is None:
            raise FileNotFoundError(f"no image root matches: {path!r}")
        prefix, root = match
        full = os.path.normpath(os.path.join(root, path[len(prefix):]))
        if not os.path.isfile(full):
            raise FileNotFoundError(f"missing image: {full!r}")
        _verify_image(full)
        resolved.append(full)
    return resolved


def _verify_image(path: str) -> None:
    """Cheap readability check: PIL verify when PIL is available."""
    try:
        from PIL import Image
    except ImportError:
        return
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        raise FileNotFoundError(f"unreadable image: {path!r}") from exc


# ---------------------------------------------------------------------------
# Image loading (model/collation side; never inside the dataset)
# ---------------------------------------------------------------------------

IMG_LOAD_SIZE = 518  # rad-dino input: resize shortest edge + center crop (SPEC.md §2)


def load_image_tensor(path: str) -> Any:
    """PIL -> grayscale -> 518x518 -> float tensor [1, 518, 518], values 0..1."""
    import torch
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("L")
        w, h = img.size
        scale = IMG_LOAD_SIZE / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)))
        # center crop to square
        w, h = img.size
        left, top = (w - IMG_LOAD_SIZE) // 2, (h - IMG_LOAD_SIZE) // 2
        img = img.crop((left, top, left + IMG_LOAD_SIZE, top + IMG_LOAD_SIZE))
        buffer = bytearray(img.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).clone()
    return tensor.reshape(1, IMG_LOAD_SIZE, IMG_LOAD_SIZE).float().div_(255.0)


# ---------------------------------------------------------------------------
# Dataset (torch-optional)
# ---------------------------------------------------------------------------

def _encode_rows(
    examples: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    capacities: Mapping[str, int] | None,
    roots: Mapping[str, str],
    max_views: int,
    with_context: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex in examples:
        sections = parse_findings(ex["findings_section"])
        encoded = tokenize_report(sections, tokenizer, capacities)
        image_paths = resolve_image_paths(
            list(ex["image_paths"])[:max_views], roots
        )
        rows.append(
            {
                "sections": sections,
                "ids": encoded["ids"],
                "block_spans": encoded["block_spans"],
                "empty": encoded["empty"],
                "overflow": encoded["overflow"],
                "image_paths": image_paths,
                "id": ex["id"],
                "ctx_ids": tokenize_context(ex, tokenizer) if with_context else [],
            }
        )
    return rows


try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover - exercised on torch-less machines
    torch = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[assignment]


class RRGDataset(_TorchDataset):  # type: ignore[misc]
    """Torch Dataset over an srrg_findings HF split.

    Pre-encodes reports (parse + block serialization) and resolves image
    paths at construction; images are NOT loaded here — the collate step or
    the model side calls `load_image_tensor`. Each item is a dict with
    `ids` (LongTensor padded with PAD_ID), `block_spans`, `empty`,
    `overflow`, `image_paths` (resolved, capped at `max_views`), and `id`.
    """

    def __init__(
        self,
        hf_dataset: Iterable[Mapping[str, Any]],
        tokenizer: Any,
        capacities: Mapping[str, int] | None = None,
        roots: Mapping[str, str] | None = None,
        max_views: int = MAX_VIEWS,
        with_context: bool = False,
        shuffle_bullets: bool = False,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for RRGDataset")
        if roots is None:
            roots = dataset_roots()["image_roots"]
        self.with_context = with_context
        # On-the-fly augmentation: re-tokenize each report with the bullets of every category in a
        # fresh random order. torch.randperm draws from the per-worker torch RNG, which the
        # DataLoader reseeds every epoch, so each epoch sees new orders.
        self.shuffle_bullets = shuffle_bullets
        self._tokenizer, self._capacities = tokenizer, capacities
        self._rows = _encode_rows(hf_dataset, tokenizer, capacities, roots, max_views,
                                  with_context)
        self._max_len = max((len(r["ids"]) for r in self._rows), default=0)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[index]
        row_ids, row_spans = row["ids"], row["block_spans"]
        if self.shuffle_bullets:
            shuffled = {}
            for cat, content in row["sections"].items():
                lines = content.split("\n")
                if len(lines) > 1:
                    lines = [lines[i] for i in torch.randperm(len(lines)).tolist()]
                shuffled[cat] = "\n".join(lines)
            enc = tokenize_report(shuffled, self._tokenizer, self._capacities)
            row_ids, row_spans = enc["ids"], enc["block_spans"]
        ids = torch.full((self._max_len,), PAD_ID, dtype=torch.long)
        ids[: len(row_ids)] = torch.tensor(row_ids, dtype=torch.long)
        return {
            "ids": ids,
            "block_spans": list(row_spans),
            "empty": dict(row["empty"]),
            "overflow": dict(row["overflow"]),
            "image_paths": list(row["image_paths"]),
            "id": row["id"],
            # left as a list: collate pads to the batch max rather than to MAX_CONTEXT_LEN, so
            # a batch of short indications does not carry 96 masked memory slots
            "ctx_ids": list(row["ctx_ids"]),
        }


def build_dataset(
    hf_dataset: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    capacities: Mapping[str, int] | None = None,
    roots: Mapping[str, str] | None = None,
    max_views: int = MAX_VIEWS,
    with_context: bool = False,
    shuffle_bullets: bool = False,
) -> RRGDataset:
    """Factory for RRGDataset; imports torch lazily."""
    import torch as _torch  # noqa: F401

    return RRGDataset(hf_dataset, tokenizer, capacities, roots, max_views, with_context,
                      shuffle_bullets)
