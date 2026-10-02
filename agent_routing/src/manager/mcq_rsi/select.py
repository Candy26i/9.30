"""Label selection by arm (design §3.4): counterfactual records -> Manager SFT rows.

- ``dynamic``: ``choose_preferred_sequence`` (shortest success, tie-break
  ``Random(tie_break_seed + example_id)``, the collection seed 42),
  ``_balance_records(rho, seed)``, ``_make_sft_rows``. The balance seed is per
  benchmark (registry ``balance_seed``, the default of ``write_selection`` and
  the CLI): the round-1 ratio files reproduce row for row with seed 0 (MedQA,
  MMLU-Pro, AQuA, from ``round1/records.jsonl``) and seed 42 (GPQA, from its
  depth-1 ``round1/label_records.jsonl``).
- ``success``: the same rho-capped records; each kept record trains one option
  drawn uniformly (one ``Random(seed)`` stream over the balanced order) from its
  successful options, commit included when the root is correct
  (``src/verifiable/experiment.py:189-208``).
- ``static``: the pinned round-1 ratio file S_1 was trained on, copied byte for
  byte after its sha256 is checked against the registry.

Rows carry ``split="train"`` and ``question_hash`` and pass ``validate_sft_splits``.
``write_selection`` removes the old ``.report.json``, then writes the label file
and its report, each atomically (unique ``mkstemp`` temp + fsync + ``os.replace``).
The report holds the label sha256 and is the completion marker: a crash in
between leaves labels without a report, never a stale report over new labels.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...subagents.train import validate_sft_splits
from ..marginal_value import _balance_records, _make_sft_rows, choose_preferred_sequence, summarize_counterfactuals
from . import benchmarks as registry
from . import importer
from .collect import _atomic_write

SELECT_VERSION = "mcq_rsi_select/1"
ARMS = registry.ARMS
TIE_BREAK_SEED = 42  # paper collection seed: Random(42 + example_id)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relabel(records: Sequence[Dict[str, Any]], max_depth: Optional[int] = None,
            tie_break_seed: int = TIE_BREAK_SEED) -> List[Dict[str, Any]]:
    """Train records, cut to ``max_depth``: deeper branches dropped, ``preferred_sequence`` re-chosen.

    Uncut records keep the collector's label (``choose_preferred_sequence`` with
    ``Random(tie_break_seed + example_id)``, the collection seed).
    """
    out = []
    for r in records:
        if r.get("split", "train") != "train":
            raise ValueError(f"example {r.get('example_id')}: never select SFT labels from split {r.get('split')!r}")
        branches = [b for b in r.get("branches", []) if max_depth is None or len(b["sequence"]) <= max_depth]
        if len(branches) == len(r.get("branches", [])):
            out.append(dict(r))
            continue
        preferred = choose_preferred_sequence(bool(r.get("direct_correct")), branches,
                                              tie_break_seed=tie_break_seed + int(r["example_id"]))
        out.append({**r, "branches": branches, "preferred_sequence": list(preferred) if preferred is not None else None})
    return out


def success_options(record: Dict[str, Any]) -> List[List[str]]:
    """Every successful sequence of a record; ``[]`` (commit) when the root is correct."""
    options = [[]] if record.get("direct_correct") else []
    return options + [list(b["sequence"]) for b in record.get("branches", []) if b.get("correct")]


def _train_rows(selected: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(itertools.chain.from_iterable(_make_sft_rows(r) for r in selected))
    return [{**row, "split": "train"} for row in rows]


def select(records: Sequence[Dict[str, Any]], arm: str, rho: float, seed: int,
           max_depth: Optional[int] = None, tie_break_seed: int = TIE_BREAK_SEED) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """SFT rows + report for the ``dynamic`` or ``success`` arm (``static``: ``select_static``).

    ``seed`` is the ``_balance_records`` / success-draw seed; the paper's is the registry ``balance_seed``.
    """
    if arm not in ("dynamic", "success"):
        raise ValueError(f"select() handles dynamic/success; arm {arm!r} uses select_static" if arm == "static"
                         else f"Unknown arm {arm!r}")
    labelled = relabel(records, max_depth, tie_break_seed)
    selected = _balance_records(labelled, rho, seed)
    if arm == "success":
        rng = random.Random(seed)
        selected = [{**r, "preferred_sequence": rng.choice(success_options(r))} for r in selected]
    rows = _train_rows(selected)
    if rows:
        validate_sft_splits(rows)
    report = summarize_counterfactuals(labelled)
    report.update({
        "select_version": SELECT_VERSION, "arm": arm, "rho": rho, "seed": seed, "max_depth": max_depth,
        "tie_break_seed": tie_break_seed,
        "n_selected_decisions": len(selected),
        "n_selected_rescue_decisions": sum(bool(r["preferred_sequence"]) for r in selected),
        "n_selected_commit_decisions": sum(r["preferred_sequence"] == [] for r in selected),
        "n_sft_turns": len(rows),
        "decision_types": {t: sum(r["decision_type"] == t for r in rows) for t in ("commit", "call", "commit_after_call")},
        "selected_depth_counts": {str(d): sum(len(r["preferred_sequence"]) == d for r in selected)
                                  for d in sorted({len(r["preferred_sequence"]) for r in selected})},
    })
    return rows, report


def static_labels_path(bench: registry.Benchmark, import_dir=None) -> Path:
    root = import_dir or (registry.PACKAGE_ROOT / importer.DEFAULT_OUT)
    return Path(importer.default_paths(bench, root)["labels"])


def select_static(bench: registry.Benchmark, out, import_dir=None) -> Dict[str, Any]:
    """Copy the pinned round-1 ratio file after checking its sha256; refuse any other bytes."""
    src = static_labels_path(bench, import_dir)
    expected = bench.round1_labels.sha256
    if not src.is_file():
        raise FileNotFoundError(f"{src}: round-1 labels not imported (python -m src.manager.mcq_rsi import)")
    if _sha256(src) != expected:
        raise ValueError(f"{src}: sha256 differs from the registry round-1 labels {expected}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, open(src, "rb") as g:
            for chunk in iter(lambda: g.read(1 << 20), b""):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        if _sha256(tmp) != expected:
            raise ValueError(f"{out}: copy does not match {expected}")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"select_version": SELECT_VERSION, "arm": "static", "source": bench.round1_labels.uri,
            "sha256": expected, "n_sft_turns": sum(1 for line in out.open(encoding="utf-8") if line.strip())}


def write_selection(bench: registry.Benchmark, arm: str, records_path, out, *, rho: Optional[float] = None,
                    seed: Optional[int] = None, max_depth: Optional[int] = None, tie_break_seed: int = TIE_BREAK_SEED,
                    import_dir=None) -> Dict[str, Any]:
    """``select`` to ``out`` (jsonl) and ``out``'s ``.report.json``; static ignores ``records_path``.

    ``rho`` and ``seed`` default to the registry ``rho`` and ``balance_seed``.
    """
    out = Path(out)
    report_path = out.with_name(out.stem + ".report.json")
    report_path.unlink(missing_ok=True)  # the completion marker never vouches for labels being replaced
    if arm == "static":
        report = select_static(bench, out, import_dir)
    else:
        with open(records_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        rows, report = select(records, arm, bench.rho if rho is None else rho,
                              bench.balance_seed if seed is None else seed, max_depth, tie_break_seed)
        if not rows:
            raise ValueError(f"{records_path}: no SFT rows selected")
        _atomic_write(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        report.update(records=str(records_path), records_sha256=_sha256(records_path), sha256=_sha256(out))
    report["benchmark"] = bench.name
    _atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
