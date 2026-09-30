"""Answer-only math datasets, deterministic splits, and test-set exclusion."""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from ..benchmarks.base import StandardRow
from ..utils.io import read_jsonl, write_json, write_jsonl
from .answers import valid_gold

SOURCES = {
    "numina": ("AI-MO/NuminaMath-1.5", "train"),
    # The upstream split is named 'train'; its role here is always locked test.
    "aime2026": ("MathArena/aime_2026", "train"),
    "beyondaime": ("ByteDance-Seed/BeyondAIME", "test"),
}


def parquet_rows(files):
    """Read bounded HF streams without Arrow's asynchronous dataset scanner."""
    import fsspec
    import pyarrow.parquet as pq
    for path in files:
        with fsspec.open(str(path), "rb") as stream:
            with pq.ParquetFile(stream) as parquet:
                for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
                    yield from batch.to_pylist()


def identity(question):
    text = unicodedata.normalize("NFKC", question)
    text = re.sub(r"\s+", "", text).casefold()
    return hashlib.sha256(text.encode()).hexdigest()


def normalize(rec, source, index):
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    answer = rec.get("answer", rec.get("ground_truth"))
    if not question or answer is None:
        return None
    answer = str(answer).strip()
    if source == "numina":
        if rec.get("problem_is_valid") != "Yes" or rec.get("solution_is_valid") != "Yes":
            return None
        qtype = str(rec.get("question_type", "")).lower().replace("_", "-").replace(" ", "-")
        if qtype in {"proof", "multiple-choice", "mcq"} or rec.get("choices"):
            return None
    # Text-only pipeline: do not silently drop required figures.
    if re.search(r"!\[.*?\]\(|<img\b", question, re.I):
        return None
    if not valid_gold(answer):
        return None
    return StandardRow(index, source, "free_response_math", question, {}, answer,
                       metadata={"answer_type": "math", "verification_scope": "terminal_answer",
                                 "content_hash": identity(question),
                                 "source_id": str(rec.get("problem_idx", index)),
                                 "category": str(rec.get("problem_type", ""))},
                       split="test" if source in {"aime2026", "beyondaime"} else "train")


def partition(rows, excluded, train_size, dev_size, seed):
    """Dedup BEFORE splitting; remove normalized exact matches with either test."""
    if train_size < 1 or dev_size < 1:
        raise ValueError("Train and dev splits must be nonempty")
    unique, seen, counts = [], set(excluded), Counter()
    for row in rows:
        key = identity(row.question)
        if key in seen:
            counts["duplicate_or_test_overlap"] += 1
            continue
        seen.add(key)
        unique.append(row)
    random.Random(seed).shuffle(unique)
    needed = train_size + dev_size
    if len(unique) < needed:
        raise ValueError(f"Need {needed} eligible unique questions; found {len(unique)}")
    dev, train = unique[:dev_size], unique[dev_size:needed]
    for i, row in enumerate(train + dev):
        row.example_id = i
        row.split = "train" if i < len(train) else "dev"
    return train, dev, dict(counts)


def load_rows(path, required_split=None):
    names = {f.name for f in fields(StandardRow)}
    rows = [StandardRow(**{k: v for k, v in rec.items() if k in names})
            for rec in read_jsonl(str(path))]
    if not rows:
        raise ValueError(f"Empty data file: {path}")
    if any(r.choices or r.metadata.get("answer_type") != "math" for r in rows):
        raise ValueError("Expected normalized free-response math rows; run prepare first")
    if any(not isinstance(r.question, str) or not r.question.strip() or not valid_gold(r.ground_truth) for r in rows):
        raise ValueError("Normalized math rows require a nonempty question and a verifiable gold answer")
    if any(r.metadata.get("content_hash", identity(r.question)) != identity(r.question) for r in rows):
        raise ValueError("Normalized content hash disagrees with question text")
    if required_split and any(r.split != required_split for r in rows):
        raise ValueError(f"This operation only accepts split={required_split}")
    if len({identity(r.question) for r in rows}) != len(rows):
        raise ValueError("Duplicate questions in normalized data")
    if len({r.example_id for r in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")
    return rows


def prepare(out_dir, train_size=1024, dev_size=256, seed=42, scan_limit=30000,
            local_sources=None):
    if min(train_size, dev_size, scan_limit) < 1:
        raise ValueError("train_size, dev_size and scan_limit must be positive")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Prepared data already exists; use another directory to change splits")
    out.mkdir(parents=True, exist_ok=True)
    local_sources = local_sources or {}
    provenance, normalized, stats = {}, {}, {}
    for name in ("aime2026", "beyondaime", "numina"):
        if name in local_sources:
            path = Path(local_sources[name])
            raw = read_jsonl(str(path))
            provenance[name] = {"local_file": str(path.resolve()),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            from datasets import IterableDataset, load_dataset_builder
            from huggingface_hub import HfApi
            dataset_id, split = SOURCES[name]
            revision = HfApi().dataset_info(dataset_id).sha
            builder = load_dataset_builder(dataset_id, revision=revision)
            if builder.info.builder_name != "parquet":
                raise ValueError(f"Expected upstream Parquet files for {dataset_id}")
            raw = IterableDataset.from_generator(parquet_rows,
                       gen_kwargs={"files": list(builder.config.data_files[split])})
            if name == "numina":
                raw = raw.shuffle(seed=seed, buffer_size=10000)
            provenance[name] = {"dataset": dataset_id, "revision": revision, "source_split": split}
        rows, scanned, rejected = [], 0, 0
        iterator = iter(raw)
        try:
            for i, rec in enumerate(iterator):
                if name == "numina" and i >= scan_limit:
                    break
                scanned += 1
                row = normalize(rec, name, i)
                if row is not None:
                    rows.append(row)
                else:
                    rejected += 1
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
            del iterator, raw
        # Official eval sets must never silently shrink due to parsing/filtering.
        expected = {"aime2026": 30, "beyondaime": 100}.get(name)
        if name not in local_sources and expected and (rejected or len(rows) != expected):
            raise ValueError(f"{name}: expected {expected} intact rows, got {len(rows)}, rejected {rejected}")
        if not rows:
            raise ValueError(f"No valid rows from {name}")
        if name != "numina" and len({identity(r.question) for r in rows}) != len(rows):
            raise ValueError(f"Duplicate official evaluation questions in {name}")
        normalized[name] = rows
        stats[name] = {"scanned": scanned, "eligible": len(rows), "rejected": rejected}
    excluded = {identity(r.question) for key in ("aime2026", "beyondaime") for r in normalized[key]}
    train, dev, dedup = partition(normalized["numina"], excluded, train_size, dev_size, seed)
    splits = {"train": train, "dev": dev, "aime2026": normalized["aime2026"],
              "beyondaime": normalized["beyondaime"]}
    test_overlap = {identity(r.question) for r in normalized["aime2026"]} & {identity(r.question) for r in normalized["beyondaime"]}
    if test_overlap:
        raise ValueError("Question overlap between official evaluation sets; preserve source records and resolve before reporting")
    checksums = {}
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        write_jsonl(str(path), [r.to_dict() for r in rows])
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"seed": seed, "sources": provenance, "stats": stats, "dedup": dedup,
                "counts": {k: len(v) for k, v in splits.items()}, "sha256": checksums,
                "verification_scope": "terminal_answer",
                "dedup_scope": "NFKC, case and whitespace normalized exact text; not semantic decontamination"}
    write_json(str(out / "manifest.json"), manifest)
    return manifest


def verify_manifest(data_dir):
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    for filename, expected in manifest["sha256"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Data changed after preparation: {filename}")
    # A derived pool may lock fewer held-out sets, e.g. when BeyondAIME becomes train/dev.
    tests = manifest.get("test_sets", ["aime2026", "beyondaime"])
    if type(tests) is not list or not tests or len(set(tests)) != len(tests) or set(tests) - {"aime2026", "beyondaime"}:
        raise ValueError("test_sets must be a nonempty subset of aime2026/beyondaime")
    expected = {"train.jsonl", "dev.jsonl"} if manifest.get("smoke_only") else {"train.jsonl", "dev.jsonl", *(t + ".jsonl" for t in tests)}
    if set(manifest["sha256"]) != expected:
        raise ValueError("Manifest must include exactly the required split files")
    seen = set()
    for filename in sorted(expected):
        name = filename.removesuffix(".jsonl")
        rows = load_rows(root / filename, required_split=name if name in {"train", "dev"} else "test")
        keys = {identity(row.question) for row in rows}
        if seen & keys:
            raise ValueError("Question overlap between frozen data splits")
        seen.update(keys)
        if manifest.get("counts") and manifest["counts"].get(name) != len(rows):
            raise ValueError("Manifest count differs from data rows")
    return manifest
