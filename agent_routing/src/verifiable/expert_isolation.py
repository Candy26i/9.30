"""Keep Manager examples disjoint from a frozen expert's train and dev pools.

Ordinary prompt-only configurations are unchanged. Generated expert Manager
configs bind the data manifest by SHA-256. Text rows get the same lexical near
matching as the expert builder; historical SFT rows with only a question hash
can only receive an exact-identity check, and that limitation is returned.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from collections.abc import Mapping

from .data import identity
from .expert_data import _NearIndex, NEAR_THRESHOLD
from .expert_train import digest, read_dataset
from .protocol import KINDS


def _expert_questions(config):
    path = Path(config["expert_data_manifest"]).resolve()
    if path.name != "manifest.json":
        raise ValueError("expert_data_manifest must identify the published manifest.json")
    expected = config.get("expert_data_manifest_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or digest(path) != expected:
        raise ValueError("Frozen expert data manifest fingerprint mismatch")
    # Verifies the manifest content, six files, sidecars, template, row identity,
    # splits and serving prompts before any overlap decision is trusted.
    read_dataset(path.parent, KINDS[0])
    if config.get("advisor_expert_bundle"):
        from .serve import load_expert_bundle
        bundle = load_expert_bundle(config["advisor_expert_bundle"])
        for role in KINDS:
            summary_path = Path(bundle["roles"][role]["checkpoint"]) / "summary.json"
            if not summary_path.is_file() or json.loads(summary_path.read_text()).get("data_manifest_sha256") != expected:
                raise ValueError(f"{role} expert adapter does not bind the configured expert data manifest")
    questions = {}
    for role in KINDS:
        for split in ("train", "dev"):
            with (path.parent / role / f"{split}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        questions[row["question_hash"]] = row["question"]
    index = _NearIndex()
    for key, question in sorted(questions.items()):
        index.add(question, key)
    return expected, questions, index


def verify_manager_rows(config, rows):
    """Reject exact/lexical overlap for StandardRow or SFT-dict input rows.

    Returns counts and the actual comparison scope for experiment metadata.
    Call on the full input before limiting/shuffling and before allocating a GPU.
    """
    if not config.get("expert_data_manifest"):
        if config.get("expert_data_manifest_sha256"):
            raise ValueError("Expert data fingerprint is configured without its manifest path")
        return {"checked": False, "reason": "expert data isolation not configured"}
    manifest_hash, questions, index = _expert_questions(config)
    checked = exact_only = near_checked = 0
    for row in rows:
        if isinstance(row, Mapping):
            question, key = row.get("question"), row.get("question_hash")
        else:
            question, key = getattr(row, "question", None), getattr(row, "question_hash", None)
        if question is not None and (not isinstance(question, str) or not question.strip()):
            raise ValueError("Manager isolation requires a nonempty question when question text is supplied")
        if key is not None and (not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key)):
            raise ValueError("Manager isolation requires a valid question_hash")
        if question:
            computed = identity(question)
            if key is not None and key != computed:
                raise ValueError("Manager question text and question_hash disagree")
            key = computed
        elif key is None:
            raise ValueError("Manager isolation cannot verify a row without question text or question_hash")
        if key in questions:
            raise ValueError(f"Manager input overlaps expert train/dev pool (exact): {key}")
        if question:
            match = index.match(question)
            if match:
                raise ValueError(f"Manager input overlaps expert train/dev pool ({match[1]}): {key} -> {match[0]}")
            near_checked += 1
        else:
            exact_only += 1
        checked += 1
    return {"checked": True, "expert_data_manifest_sha256": manifest_hash,
            "expert_question_groups": len(questions), "checked_rows": checked,
            "text_rows_with_near_check": near_checked, "exact_only_rows": exact_only,
            "near_jaccard_threshold": NEAR_THRESHOLD,
            "scope": "expert train and dev; normalized exact identity and builder lexical shingle matching; not semantic decontamination"}
