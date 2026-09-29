"""Pinned, auditable Numina weak supervision for the three math sub-agents.

This builder does not call a teacher or certify Numina's full derivations. It
keeps source solutions in a separate sidecar, extracts question-grounded facts,
uses reference solutions as explicitly weak reasoner targets, and teaches the
verifier *local arithmetic audits* with executable evidence. All three roles
share question groups. This limited pilot corpus is not a verifier benchmark.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from . import protocol
from .data import identity, load_rows, normalize, parquet_rows, verify_manifest

NUMINA_DATASET = "AI-MO/NuminaMath-1.5"
NUMINA_REVISION = "1b05109f9e5c1ad06c0663519502416c30b300f8"
SCHEMA_VERSION = 1
NEAR_THRESHOLD = 0.85
MANAGER_FILES = {"train.jsonl", "dev.jsonl", "aime2026.jsonl", "beyondaime.jsonl"}
TEMPLATE = Path(__file__).with_name("chat_template.jinja")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(_canonical(row) + "\n")


def _local_rows(path):
    # Streaming input: do not materialize a potentially multi-GB raw export.
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid raw JSONL at line {number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Raw JSONL line {number} must be an object")
            yield row


def _remote_rows(revision):
    from datasets import load_dataset_builder
    builder = load_dataset_builder(NUMINA_DATASET, revision=revision, token=False)
    if builder.info.builder_name != "parquet":
        raise ValueError("Expected pinned Numina Parquet source")
    yield from parquet_rows(list(builder.config.data_files["train"]))


def _features(question):
    # Numbers remain literal. This is lexical near-dedup, not semantic matching.
    text = unicodedata.normalize("NFKC", question).casefold()
    tokens = re.findall(r"[\w]+|[^\s\w]", text)
    return {tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)}


class _NearIndex:
    """Exact Jaccard threshold search with a lossless prefix candidate filter.

    A >=t Jaccard match overlaps at least t*len(stored) stored shingles. Its
    first len(stored)-ceil(t*len(stored))+1 sorted shingles therefore contain a
    match. Querying all query shingles cannot miss a threshold match. Very
    short questions use normalized exact matching only to avoid broad filters.
    """
    def __init__(self):
        self.exact = {}
        self.features = {}
        self.postings = defaultdict(set)

    def match(self, question):
        key = identity(question)
        if key in self.exact:
            return self.exact[key], "exact"
        query = _features(question)
        if len(query) < 6:
            return None
        candidates = set()
        for feature in query:
            candidates.update(self.postings.get(feature, ()))
        for other in sorted(candidates):
            stored = self.features[other]
            if min(len(query), len(stored)) / max(len(query), len(stored)) < NEAR_THRESHOLD:
                continue
            if len(query & stored) / len(query | stored) >= NEAR_THRESHOLD:
                return other, "near"
        return None

    def add(self, question, key):
        self.exact[identity(question)] = key
        features = _features(question)
        if len(features) < 6:
            return
        self.features[key] = features
        prefix_size = len(features) - math.ceil(NEAR_THRESHOLD * len(features)) + 1
        for feature in sorted(features)[:prefix_size]:
            self.postings[feature].add(key)


_MATH_SPAN = re.compile(r"\$\$[\s\S]*?\$\$|\$(?:\\.|[^$])*\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)")


def _source_clauses(question):
    """Split prose clauses while retaining complete, possibly multiline math."""
    text = re.sub(r"^\s*(?:#+\s*)?(?:(?:problem|question|exercise)\s+\d+[.:)]?\s*|\d+[.)]\s+)", "", question, flags=re.I)
    spans = []

    def protect(match):
        spans.append(match.group())
        return f"\ufff0{len(spans) - 1}\ufff1"

    text = _MATH_SPAN.sub(protect, text)
    clauses = []
    for part in re.split(r"(?<=[.!?])\s+|;\s*|\n+", text):
        part = re.sub(r"^\s*\(?[a-z]\)\s*", "", part, flags=re.I).strip(" \n.;")
        if not part:
            continue
        part = re.sub(r"\ufff0(\d+)\ufff1", lambda m: spans[int(m[1])], part)
        clauses.append(part)
    return clauses


def _extract_facts(question):
    """Question-only structured extraction, with no invented deductions.

    All targets and post-request constraints are retained. Numbered headings
    are not mathematical givens, and math blocks are never split into bullets.
    This remains explicit weak supervision, not a claim of expert annotation.
    """
    goal_pattern = r"\b(?:find|determine|calculate|compute|evaluate|what\s+is|how\s+many)\b"
    givens, goals = [], []
    for clause in _source_clauses(question):
        match = re.search(goal_pattern, clause, re.I)
        if not match:
            givens.append(clause)
            continue
        before, goal = clause[:match.start()].strip(" \n,;:"), clause[match.start():].strip()
        if before:
            givens.append(before)
        condition = re.search(r"\b(?:such that|given that|if|where)\b", goal, re.I)
        if condition:
            givens.append(goal[condition.start():].strip())
            goal = goal[:condition.start()].strip()
        goals.append(goal)
    if not goals or sum(map(len, givens)) < 20 or sum(map(len, goals)) < 8:
        return None
    facts = "\n".join(givens)
    if not re.search(r"\d|[=<>]|\\(?:leq?|geq?|in)\b|\b(?:integer|positive|negative|triangle|circle|parallel|perpendicular)\b", facts, re.I):
        return None
    quantities = list(dict.fromkeys(re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w|\.\d)", facts)))
    math_text = "\n".join(span.group() for span in _MATH_SPAN.finditer(facts))
    symbol_pattern = r"(?<![A-Za-z\\])([a-zA-Z])(?![A-Za-z])"
    # a/A/I in ordinary prose are ambiguous; inside a math span they are real
    # lexical symbols and must not be dropped as if they were English articles.
    symbols = sorted(set(re.findall(symbol_pattern, math_text)) |
                     (set(re.findall(symbol_pattern, facts)) - {"a", "A", "I"}))
    parts = ["Stated givens and constraints (source clauses):", *[f"- {c}" for c in givens],
             "Requested target:", *[f"- {goal}" for goal in goals]]
    if quantities:
        parts += ["Explicit numeric quantities in the givens: " + ", ".join(quantities)]
    if symbols:
        parts += ["Single-letter symbols appearing in the givens: " + ", ".join(symbols)]
    return {"response": "\n".join(parts), "quality_checks": {
        "question_only": True, "givens_goal_separated": True,
        "grounding": "verbatim_source_clauses_and_lexical_quantities",
        "semantic_equivalent_formulations_generated": False,
        "quality_level": "deterministic_weak_supervision"}}


def evaluate_arithmetic(expression):
    """Evaluate a bounded arithmetic expression exactly, without eval/sympy.

    Deliberately narrow: rational arithmetic and small integer powers only;
    reject names, functions, comparisons, enormous integers and deep trees.
    """
    expression = expression.strip().replace("^", "**")
    if len(expression) > 160 or not re.fullmatch(r"[0-9.\s()+*/-]+", expression):
        raise ValueError("Unsupported arithmetic expression")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid arithmetic expression") from exc
    if len(list(ast.walk(tree))) > 64:
        raise ValueError("Arithmetic expression too complex")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            token = ast.get_source_segment(expression, node)
            if len(token) > 30:
                raise ValueError("Arithmetic constant too large")
            value = Fraction(token)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = a + b
            elif isinstance(node.op, ast.Sub):
                value = a - b
            elif isinstance(node.op, ast.Mult):
                value = a * b
            elif isinstance(node.op, ast.Div):
                if not b:
                    raise ValueError("Division by zero")
                value = a / b
            else:
                if b.denominator != 1 or abs(b) > 8 or (not a and b < 0):
                    raise ValueError("Unsupported arithmetic exponent")
                value = a ** int(b)
        else:
            raise ValueError("Unsupported arithmetic syntax")
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 256:
            raise ValueError("Arithmetic magnitude too large")
        return value

    return visit(tree.body)


def _number(value):
    return str(value.numerator) if value.denominator == 1 else f"({value.numerator}/{value.denominator})"


def _arithmetic_step(solution):
    # Restrict the candidate to a contiguous pure numeric equality in the source.
    # Do not splice fragments from symbolic equations into an alleged proof.
    numeric = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    left = r"[0-9.\s()+*/^\-]+"
    pattern = re.compile(rf"(?<![\w.\\+*/^\-])(?P<lhs>{left})\s*=\s*(?P<rhs>{numeric})(?![\w.\\+*/^\-!%])")
    for match in pattern.finditer(solution):
        lhs, rhs = match.group("lhs").strip(), match.group("rhs").strip()
        remaining = solution[match.end():].lstrip()
        if re.match(r"[+*/^=!%\-]|\\(?:times|cdot|div|frac|sqrt)\b|[A-Za-z](?:\s*[$}\]]|\s*[+*/^=])", remaining):
            continue  # never label a truncated RHS prefix as the original step
        if not re.search(r"[+*/^\-]", lhs):
            continue
        # A captured suffix beginning after a variable's coefficient is not a
        # self-contained source expression; do not silently drop the variable.
        start = match.start("lhs")
        previous = solution[:start].rstrip()
        if previous and (previous[-1].isalnum() or previous[-1] in "+-*/^\\"):
            continue
        try:
            value, claimed = evaluate_arithmetic(lhs), evaluate_arithmetic(rhs)
        except ValueError:
            continue
        if value == claimed:
            return {"lhs": lhs, "rhs": rhs, "value": _number(value),
                    "source_span": [match.start(), match.end()],
                    "source_text": solution[match.start():match.end()]}
    return None


def _requires_image(text):
    return bool(re.search(
        r"!\[[^\]]*\]\(|<img\b|<svg\b|\\includegraphics|\\begin\{(?:tikzpicture|asy)\}"
        r"|\b(?:diagram|figure|image)\s+(?:below|above|shown|\d+)"
        r"|\b(?:accompanying|following)\s+(?:figure|diagram)"
        r"|\b(?:as\s+shown|illustrated)\s+(?:below|above|in\s+(?:the\s+)?(?:figure|diagram))", text, re.I))


def _targets(row, solution, max_response_chars):
    if len(solution.strip()) > max_response_chars:
        return None, "response_exceeds_character_cap"
    if _requires_image(solution):
        return None, "reference_requires_image"
    if _requires_image(row.question):
        return None, "question_requires_image"
    extractor = _extract_facts(row.question)
    if extractor is None:
        return None, "no_grounded_extractor_target"
    # A scalar answer is not a reasoner demonstration. Keep the complete source
    # solution rather than silently truncating a derivation to a target budget.
    if len(solution.strip()) < 60 or not re.search(r"[=\n]|\b(?:therefore|because|hence|since|thus)\b", solution, re.I):
        return None, "insufficient_reasoner_reference"
    step = _arithmetic_step(solution)
    if step is None:
        return None, "no_executable_reference_arithmetic"
    right = evaluate_arithmetic(step["rhs"])
    bad_rhs = _number(right + 1)
    candidate = f"{step['lhs']} = {step['rhs']}"
    corrupted = f"{step['lhs']} = {bad_rhs}"
    correct = ("Verdict: correct\nEvidence: This supplied arithmetic step evaluates to "
               f"{step['value']} on both sides. This local check does not certify any unstated full solution.\nCorrection: None needed")
    incorrect = (f"Verdict: incorrect\nEvidence: The left side evaluates to {step['value']}, "
                 f"but the right side is {bad_rhs}; these values differ.\nCorrection: Replace the right side with {step['value']}.")
    targets = {
        "extractor": [{**extractor, "candidate": "", "label_source": "question_span_extraction_weak"}],
        "reasoner": [{"response": solution.strip(), "candidate": "", "label_source": "numina_reference_weak",
                      "quality_checks": {"source_validity_flags": True, "derivation_verified": False,
                                         "quality_level": "reference_weak_supervision", "reference_preserved_without_truncation": True}}],
        "verifier": [
            {"response": correct, "candidate": candidate, "verdict": "correct", "label_source": "executable_reference_arithmetic",
             "corruption": None, "arithmetic_evidence": step,
             "quality_checks": {"arithmetic_verified": True, "scope": "local_arithmetic_step", "full_solution_verified": False}},
            {"response": incorrect, "candidate": corrupted, "verdict": "incorrect", "label_source": "executable_arithmetic_corruption",
             "corruption": {"operation": "replace_rhs_with_exact_rhs_plus_one", "original_rhs": step["rhs"], "replacement_rhs": bad_rhs},
             "arithmetic_evidence": step,
             "quality_checks": {"arithmetic_verified": True, "scope": "local_arithmetic_step", "full_solution_verified": False}},
        ],
    }
    if any(len(target["response"]) > max_response_chars for values in targets.values() for target in values):
        return None, "response_exceeds_character_cap"
    return targets, None


def _manager_snapshot(root):
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("smoke_only") or set(manifest.get("sha256", {})) != MANAGER_FILES:
        raise ValueError("Expert preparation requires the complete frozen Manager train/dev and both locked test sets")
    verify_manifest(root)
    index = _NearIndex()
    counts = {}
    for filename in sorted(MANAGER_FILES):
        rows = load_rows(root / filename)
        counts[filename] = len(rows)
        for row in rows:
            index.add(row.question, identity(row.question))
    return index, {"manifest_sha256": _sha(manifest_path), "sha256": manifest["sha256"], "counts": counts}


def _verify_existing(out, request):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    if manifest.get("manifest_content_sha256") != _digest(payload):
        raise ValueError("Expert manifest content changed after preparation")
    if manifest.get("request") != request or manifest.get("request_sha256") != _digest(request):
        raise ValueError("Expert build resume request differs from the frozen source/configuration")
    required = {f"{role}/{split}.jsonl" for role in protocol.KINDS for split in ("train", "dev")} | {"references.jsonl", "exclusions.jsonl"}
    if set(manifest.get("sha256", {})) != required:
        raise ValueError("Expert manifest must hash all six role files and both audit sidecars")
    for relative, expected in manifest["sha256"].items():
        path = out / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Expert data changed after preparation: {relative}")
    # Manifest metadata is authenticated by recomputing the deterministic layout
    # from rows, as the manifest cannot include its own hash.
    references = list(_local_rows(out / "references.jsonl"))
    groups = {r["question_hash"]: r for r in references}
    if len(groups) != len(references):
        raise ValueError("Duplicate expert reference question groups")
    group_counts = dict(Counter(r["split"] for r in references))
    if group_counts != manifest.get("question_counts"):
        raise ValueError("Expert manifest question counts changed")
    for role in protocol.KINDS:
        for split in ("train", "dev"):
            rows = list(_local_rows(out / role / f"{split}.jsonl"))
            if not rows or len(rows) != manifest["counts"][role][split]:
                raise ValueError("Expert manifest role counts changed")
            for row in rows:
                if row["role"] != role or row["split"] != split or groups[row["question_hash"]]["split"] != split:
                    raise ValueError("Expert data role/split identity mismatch")
    return manifest


def build_expert_data(out_dir, manager_data_dir, *, raw_jsonl=None, train_size=128,
                      dev_size=32, seed=42, scan_limit=30000,
                      source_revision=NUMINA_REVISION, max_response_chars=12000,
                      resume=False):
    """Build all six expert SFT files atomically; return the frozen manifest.

    train_size/dev_size are *question groups*, so verifier rows number twice
    those counts. A fixed 20% hash bucket supplies dev; expansion of requested
    sizes never moves an existing question to the opposite split. Resume only
    reuses byte-identical artifacts with identical inputs and builder code.
    Local JSONL must preserve raw Numina validity flags, answer and solution;
    its SHA identifies the export, not independent proof of its claimed pin.
    """
    if any(type(x) is not int or x < 1 for x in (train_size, dev_size, scan_limit, max_response_chars)):
        raise ValueError("Sizes, scan_limit and max_response_chars must be positive integers")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", source_revision):
        raise ValueError("source_revision must be a pinned 40-character commit SHA")
    source_revision = source_revision.lower()
    out, manager = Path(out_dir), Path(manager_data_dir)
    manager_index, exclusion_source = _manager_snapshot(manager)
    source = {"dataset": NUMINA_DATASET, "revision": source_revision, "source_split": "train",
              "mode": "local_raw_jsonl" if raw_jsonl else "huggingface_pinned_parquet",
              "upstream_revision_verified": not bool(raw_jsonl)}
    if raw_jsonl:
        source["local_sha256"] = _sha(raw_jsonl)
        source["upstream_revision_note"] = "Declared provenance; verify export origin separately. Local bytes are frozen by SHA256."
    template_sha, protocol_sha = _sha(TEMPLATE), _sha(Path(protocol.__file__))
    request = {"schema_version": SCHEMA_VERSION, "source": source, "manager_exclusion": exclusion_source,
               "train_size": train_size, "dev_size": dev_size, "seed": seed, "scan_limit": scan_limit,
               "max_response_chars": max_response_chars, "near_threshold": NEAR_THRESHOLD,
               "template_sha256": template_sha, "protocol_sha256": protocol_sha,
               "builder_sha256": _sha(__file__),
               "builder_dependencies_sha256": {name: _sha(Path(__file__).with_name(name)) for name in ("data.py", "answers.py")}}
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError("Expert data already exists; pass --resume to verify and reuse, or choose another output")
        return _verify_existing(out, request)
    if resume:
        raise FileNotFoundError("No completed expert build to resume")
    # Complete all inspection/selection before writing outputs; never leave a
    # seemingly trainable half-build after a source or quality failure.
    raw = _local_rows(raw_jsonl) if raw_jsonl else _remote_rows(source_revision)
    accepted, exclusions, stats = {}, [], Counter()
    observed_hash = hashlib.sha256()
    try:
        for i, record in enumerate(itertools.islice(raw, scan_limit)):
            stats["scanned"] += 1
            observed_hash.update((_canonical(record) + "\n").encode())
            row = normalize(record, "numina", i)
            if row is None:
                stats["source_filter_rejected"] += 1
                continue
            key = identity(row.question)
            overlap = manager_index.match(row.question)
            if overlap:
                stats[f"manager_{overlap[1]}_overlap"] += 1
                exclusions.append({"question_hash": key, "matched_hash": overlap[0], "reason": f"manager_{overlap[1]}_overlap"})
                continue
            solution = record.get("solution")
            if not isinstance(solution, str) or not solution.strip():
                stats["missing_reference_solution"] += 1
                continue
            targets, reason = _targets(row, solution, max_response_chars)
            if reason:
                stats[reason] += 1
                continue
            stable_field = next((field for field in ("problem_idx", "id", "problem_id")
                                 if record.get(field) is not None and str(record[field]).strip()), None)
            row.metadata["source_id"] = str(record[stable_field]) if stable_field else f"source-index:{i}"
            item = {"row": row, "solution": solution, "targets": targets, "source_record_sha256": _digest(record),
                    "source_id_kind": stable_field or "pinned_source_stream_index", "source_ordinal": i}
            if key in accepted:
                stats["source_exact_duplicate"] += 1
                # Do not let repeated raw records make selection order-sensitive.
                if item["source_record_sha256"] >= accepted[key]["source_record_sha256"]:
                    continue
            accepted[key] = item
    finally:
        if hasattr(raw, "close"):
            raw.close()
    # Check source inputs again before publication to catch concurrent edits.
    if raw_jsonl and _sha(raw_jsonl) != source["local_sha256"]:
        raise ValueError("Raw source changed during expert preparation")
    if _sha(manager / "manifest.json") != exclusion_source["manifest_sha256"]:
        raise ValueError("Manager manifest changed during expert preparation")
    verify_manifest(manager)
    selected = {"train": [], "dev": []}
    selected_index = _NearIndex()
    candidates = sorted(accepted, key=lambda key: (_digest([seed, key]), key))
    for key in candidates:
        item = accepted[key]
        overlap = selected_index.match(item["row"].question)
        if overlap:
            stats["source_near_duplicate"] += 1
            exclusions.append({"question_hash": key, "matched_hash": overlap[0], "reason": "source_near_duplicate"})
            continue
        # Dedup all eligible groups before filling either split, including pool
        # groups outside requested sizes, so expansion keeps previous members.
        selected_index.add(item["row"].question, key)
        split = "dev" if int(_digest(["expert-split-v1", seed, key]), 16) % 5 == 0 else "train"
        stats[f"eligible_{split}_groups"] += 1
        limit = dev_size if split == "dev" else train_size
        if len(selected[split]) < limit:
            selected[split].append((key, item))
    if len(selected["train"]) < train_size or len(selected["dev"]) < dev_size:
        raise ValueError(f"Not enough eligible expert question groups for all three roles: requested train={train_size}, dev={dev_size}; found train={stats['eligible_train_groups']}, dev={stats['eligible_dev_groups']}. Increase scan_limit/source pool, not weaken split or labels. Audit: {_canonical(dict(stats))}")
    outputs = {f"{role}/{split}.jsonl": [] for role in protocol.KINDS for split in ("train", "dev")}
    references = []
    for split in ("train", "dev"):
        for key, item in selected[split]:
            row, solution = item["row"], item["solution"]
            common = {"schema_version": SCHEMA_VERSION, "question_hash": key, "question": row.question,
                      "context": row.context, "source_id": row.metadata["source_id"], "split": split,
                      "source_dataset": NUMINA_DATASET, "source_revision": source_revision,
                      "source_record_sha256": item["source_record_sha256"], "source_id_kind": item["source_id_kind"],
                      "source_ordinal": item["source_ordinal"], "reference_sha256": hashlib.sha256(solution.encode()).hexdigest(),
                      "template_sha256": template_sha, "protocol_sha256": protocol_sha,
                      "teacher_revision": None, "reviewed": False}
            references.append({**common, "ground_truth": row.ground_truth, "solution": solution,
                               "label_basis": "Numina validity flags are source claims, not independent proof checking"})
            for role, targets in item["targets"].items():
                for target in targets:
                    entry = {**common, **target, "role": role,
                             "prompt": protocol.advisor_messages(role, row, target["candidate"])}
                    entry["response_sha256"] = hashlib.sha256(entry["response"].encode()).hexdigest()
                    if role == "verifier":
                        entry["candidate_hash"] = hashlib.sha256(entry["candidate"].encode()).hexdigest()
                    outputs[f"{role}/{split}.jsonl"].append(entry)
    outputs["references.jsonl"] = references
    outputs["exclusions.jsonl"] = sorted(exclusions, key=_canonical)
    counts = {role: {split: len(outputs[f"{role}/{split}.jsonl"]) for split in ("train", "dev")} for role in protocol.KINDS}
    verifier_candidates = {split: {row["candidate_hash"] for row in outputs[f"verifier/{split}.jsonl"]}
                           for split in ("train", "dev")}
    candidate_overlap = verifier_candidates["train"] & verifier_candidates["dev"]
    verifier_candidate_audit = {
        "identity": "SHA256 of exact candidate UTF-8 text; not mathematical equivalence",
        "unique_counts": {**{split: len(keys) for split, keys in verifier_candidates.items()},
                          "all": len(verifier_candidates["train"] | verifier_candidates["dev"])},
        "cross_split_overlap_count": len(candidate_overlap),
        "dev_rows_with_train_candidate": sum(row["candidate_hash"] in candidate_overlap
                                             for row in outputs["verifier/dev.jsonl"]),
        "interpretation": "Question groups are disjoint, but repeated arithmetic candidates can make expert dev optimistic; this is an engineering diagnostic, not independent proof-generalization evidence",
    }
    manifest = {"schema_version": SCHEMA_VERSION, "request": request, "request_sha256": _digest(request),
                "source": {**source, "scanned_records_sha256": observed_hash.hexdigest()},
                "template_sha256": template_sha, "protocol_sha256": protocol_sha,
                "manager_exclusion": exclusion_source, "counts": counts,
                "question_counts": {k: len(v) for k, v in selected.items()}, "stats": dict(stats),
                "split_rule": {"unit": "normalized_question_hash", "dev_bucket_fraction": 0.2,
                               "algorithm": "sha256(canonical_json(['expert-split-v1', seed, question_hash])) modulo 5 == 0",
                               "selection_order": "sha256(canonical_json([seed, question_hash]))",
                               "expansion": "prefix stable for identical source scan, seed, filtering and dedup configuration"},
                "dedup": {"exact": "NFKC, case and whitespace normalized question SHA256",
                          "near": "token 3-shingle set Jaccard >= 0.85 (at least 6 shingles), literal numbers retained; lossless prefix search",
                          "scope": "all Manager train/dev/locked tests and all eligible expert groups before splitting",
                          "semantic_decontamination_proven": False},
                "quality_audit": {"reviewed": False, "reviewed_examples": 0,
                                  "teacher_used": False, "extractor": "grounded source spans and lexical inventory; weak supervision, no invented deductions",
                                  "reasoner": "complete source reference; weak supervision; mathematical derivation not independently verified",
                                  "verifier": "only local exact rational-arithmetic checks and known RHS corruption; not full proof verification",
                                  "verifier_verdict_counts": {"correct": train_size + dev_size, "incorrect": train_size + dev_size, "uncertain": 0},
                                  "verifier_candidate_audit": verifier_candidate_audit,
                                  "source_id_policy": "raw problem_idx/id/problem_id, else pinned source stream index; fallback IDs require identical source order",
                                  "selection_bias": "requires extractable givens/goal and an executable reference arithmetic equality; not representative of all Numina",
                                  "response_budget": "over character cap rejected, never truncated; tokenizer-level context validation required before training",
                                  "manual_review_required_before_quality_claim": True}}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.building-", dir=out.parent))
    try:
        checksums = {}
        for relative, rows in outputs.items():
            path = staging / relative
            _jsonl(path, rows)
            checksums[relative] = _sha(path)
        manifest["sha256"] = checksums
        manifest["manifest_content_sha256"] = _digest(manifest)
        (staging / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
        if out.exists():
            out.rmdir()  # only an unchanged empty destination may be replaced
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manager-data-dir", required=True)
    parser.add_argument("--raw-jsonl", help="Local raw Numina JSONL; omit for pinned Hugging Face source")
    parser.add_argument("--source-revision", default=NUMINA_REVISION)
    parser.add_argument("--train-size", type=int, default=128, help="Question groups, not verifier row count")
    parser.add_argument("--dev-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-limit", type=int, default=30000)
    parser.add_argument("--max-response-chars", type=int, default=12000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_expert_data(args.out, args.manager_data_dir, raw_jsonl=args.raw_jsonl,
                                source_revision=args.source_revision, train_size=args.train_size,
                                dev_size=args.dev_size, seed=args.seed, scan_limit=args.scan_limit,
                                max_response_chars=args.max_response_chars, resume=args.resume)
    print(json.dumps({"output": str(Path(args.out).resolve()), "counts": manifest["counts"],
                      "question_counts": manifest["question_counts"], "reviewed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
