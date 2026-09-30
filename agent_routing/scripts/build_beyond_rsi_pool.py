"""Build a Manager RSI pool whose train/dev come from BeyondAIME and whose only held-out test is AIME2026.

The experts stay bound to the original four-file manager/ pool; this script never edits it. BeyondAIME rows
are relabelled split=train/dev by the standard seeded partition (question, gold and metadata unchanged), and
AIME2026 is copied byte for byte as the single locked test set.

Usage (from agent_routing/):
    python scripts/build_beyond_rsi_pool.py
    python scripts/build_beyond_rsi_pool.py --train-n 64 --dev-n 36 --out data/manager_beyond_rsi_20260930
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import write_json, write_jsonl  # noqa: E402
from src.verifiable.data import identity, load_rows, partition, verify_manifest  # noqa: E402
from src.verifiable.expert_data import NEAR_THRESHOLD, _NearIndex  # noqa: E402

SOURCE = Path("data/math_luna_codex_pilot_20260929/manager")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default="data/manager_beyond_rsi_20260930")
    parser.add_argument("--train-n", type=int, default=64)
    parser.add_argument("--dev-n", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.out)
    source_manifest = verify_manifest(source)
    if out.exists():
        raise SystemExit(f"Output exists; use a new directory: {out}")

    beyond = load_rows(source / "beyondaime.jsonl", required_split="test")
    aime = load_rows(source / "aime2026.jsonl", required_split="test")
    before = {identity(r.question): (r.question, r.ground_truth, dict(r.metadata)) for r in beyond}
    train, dev, dedup = partition(beyond, {identity(r.question) for r in aime}, args.train_n, args.dev_n, args.seed)
    for r in train + dev:
        if before[identity(r.question)] != (r.question, r.ground_truth, r.metadata):
            raise RuntimeError("Relabelling changed question content")

    # Exact overlap is also enforced by verify_manifest; record the lexical near-duplicate audit too.
    index = _NearIndex()
    for r in aime:
        index.add(r.question, r.example_id)
    near = [{"example_id": r.example_id, "split": r.split, "aime2026_id": hit[0], "kind": hit[1]}
            for r in train + dev if (hit := index.match(r.question))]
    if near:
        raise SystemExit(f"BeyondAIME rows overlap AIME2026: {near}")

    out.mkdir(parents=True)
    write_jsonl(str(out / "train.jsonl"), [r.to_dict() for r in train])
    write_jsonl(str(out / "dev.jsonl"), [r.to_dict() for r in dev])
    shutil.copyfile(source / "aime2026.jsonl", out / "aime2026.jsonl")
    files = ("train.jsonl", "dev.jsonl", "aime2026.jsonl")
    write_json(str(out / "manifest.json"), {
        "test_sets": ["aime2026"],
        "seed": args.seed,
        "counts": {"train": len(train), "dev": len(dev), "aime2026": len(aime)},
        "sha256": {name: sha256(out / name) for name in files},
        "sources": {name: source_manifest["sources"][name] for name in ("aime2026", "beyondaime")},
        "derivation": {
            "source_manifest_sha256": sha256(source / "manifest.json"),
            "role_change": "BeyondAIME is Manager RSI train/dev only; AIME2026 is the only held-out test",
            "rule": "data.partition(beyondaime rows, excluded=AIME2026 identities, train_n, dev_n, seed)",
            "aime2026_near_duplicates": {"threshold": NEAR_THRESHOLD, "matches": len(near)},
        },
        "dedup": dedup,
        "verification_scope": "terminal_answer",
        "dedup_scope": source_manifest["dedup_scope"],
    })
    manifest = verify_manifest(out)
    print(json.dumps({"out": str(out), "counts": manifest["counts"], "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
