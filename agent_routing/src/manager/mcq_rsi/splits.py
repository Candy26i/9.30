"""Frozen question-hash split manifests for MCQ RSI.

Question identity is ``src.benchmarks.base.question_hash``: sha1 of the
stripped, lower-cased, whitespace-collapsed question text, first 16 hex chars.
It is the key the paper records (``question_hash``), ``--exclude_sft_example_ids``
and ``validate_sft_splits`` already use, so manifests join against them directly.
Choices are deliberately not hashed: the same stem with reordered options is
treated as the same question, which only makes pools more conservative.
The hash is a disjointness key, not a row key: same-stem pairs with different
options or gold exist (AQuA 1144/893, 483/861; MMLU-Pro 8859/9680), so any
runtime lookup (advisor cache, shards, labels) keys on ``example_id`` of the
pool's cache and checks ``question_hash``; ``pool_rows`` does exactly that.

Pools per benchmark: ``collect_r1`` (paper roots), ``collect_r2``/``collect_r3``
(400 new train questions each, never advisor-SFT questions), ``grpo_r1..r3``
(256 each from the paper GRPO pool = paper train pool minus collection),
``dev``, ``test`` and, for MMLU-Pro only, the flagged ``test_paper`` (the paper's
test set, which overlaps dev, collect_r1 and the GRPO pools; ``meta.flagged_overlap``
counts it) and ``test_paper_clean`` (its questions in no other pool or advisor-SFT
file). All pools are pairwise hash-disjoint except declared reuse (GPQA) and
``test_paper``.
Duplicates and skipped candidates are recorded, never raised.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...benchmarks.base import question_hash
from ...utils.io import read_jsonl
from . import benchmarks as registry

BUILDER_VERSION = "mcq_rsi_splits/1"
POOLS = ("collect_r1", "collect_r2", "collect_r3", "grpo_r1", "grpo_r2", "grpo_r3", "dev", "test")
FLAGGED_POOLS = ("test_paper", "test_paper_clean")
PAPER_SEED = 42
N_COLLECT = 400
N_GRPO = 256
ROUNDS = 3
HASH_DEFINITION = "src.benchmarks.base.question_hash: sha1(collapse_ws(strip(question)).lower())[:16]"


def identity(row: Dict[str, Any]) -> str:
    return question_hash(row["question"])


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"example_id": int(r["example_id"]), "question_hash": identity(r)} for r in rows]


def _rng(bench: str, tag: str, seed: int) -> random.Random:
    return random.Random(f"mcq_rsi:{bench}:{tag}:{seed}")


def _paper_sample(rows: Sequence[Dict[str, Any]], n: int, seed: int = PAPER_SEED) -> List[int]:
    """marginal_value.build_marginal_value_sft's root sampler."""
    sample = list(rows)
    random.Random(seed).shuffle(sample)
    return [int(r["example_id"]) for r in sample[:n]]


def _resolve(by_id: Dict[int, Dict[str, Any]], items: Sequence[Dict[str, Any]], what: str) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        row = by_id.get(int(item["example_id"]))
        if row is None or identity(row) != item["question_hash"]:
            raise ValueError(f"{what} example {item['example_id']} does not match the cache (loader drift)")
        out.append(row)
    return out


def _dup_groups(rows: Sequence[Dict[str, Any]]) -> List[List[int]]:
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for r in rows:
        groups[identity(r)].append(int(r["example_id"]))
    return [ids for ids in groups.values() if len(ids) > 1]


class _Claims:
    def __init__(self, excluded_ids=(), excluded_hashes=()):
        self.owner: Dict[str, str] = {}
        self.excluded_ids = set(excluded_ids)
        self.excluded_hashes = set(excluded_hashes)
        self.skipped: Dict[str, Dict[str, int]] = {}

    def claim(self, pool: str, rows: Sequence[Dict[str, Any]]) -> None:
        for r in rows:
            self.owner.setdefault(identity(r), pool)

    def take(self, pool: str, candidates: Sequence[Dict[str, Any]], n: int, exclude_advisor: bool) -> List[Dict[str, Any]]:
        skipped = collections.Counter()
        taken: List[Dict[str, Any]] = []
        for r in candidates:
            if len(taken) == n:
                break
            h = identity(r)
            if h in self.owner:
                skipped["claimed_by_" + self.owner[h]] += 1
            elif exclude_advisor and (int(r["example_id"]) in self.excluded_ids or h in self.excluded_hashes):
                skipped["advisor_sft"] += 1
            else:
                taken.append(r)
                self.owner[h] = pool
        if len(taken) < n:
            raise ValueError(f"{pool}: only {len(taken)} eligible questions, need {n}")
        self.skipped[pool] = dict(sorted(skipped.items()))
        return taken


def _chunks(rows: List[Dict[str, Any]], prefix: str, size: int) -> Dict[str, List[Dict[str, Any]]]:
    return {f"{prefix}{k + 1}": rows[k * size:(k + 1) * size] for k in range(len(rows) // size)}


def build_splits(
    bench: str,
    rows: Sequence[Dict[str, Any]],
    collection: Sequence[Dict[str, Any]],
    advisor_sft: Sequence[Dict[str, Any]],
    *,
    test_rows: Optional[Sequence[Dict[str, Any]]] = None,
    seed: int = PAPER_SEED,
) -> Dict[str, Any]:
    """Build one benchmark's manifest from its normalized cache and paper inputs.

    ``collection`` / ``advisor_sft`` are ``{example_id, question_hash}`` items from
    the round-1 counterfactual records and the advisor SFT files.
    """
    rows = list(rows)
    by_id = {int(r["example_id"]): r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{bench}: duplicate example_id in cache")
    coll_rows = _resolve(by_id, collection, "collection")
    advisor_ids = {int(a["example_id"]) for a in advisor_sft}
    advisor_hashes = {a["question_hash"] for a in advisor_sft}
    _resolve(by_id, advisor_sft, "advisor-SFT")
    coll_ids = {int(r["example_id"]) for r in coll_rows}
    coll_hashes = {identity(r) for r in coll_rows}
    in_collection = lambda r: int(r["example_id"]) in coll_ids or identity(r) in coll_hashes
    split = lambda name: [r for r in rows if (r.get("split") or "").lower() == name]
    claims = _Claims(advisor_ids, advisor_hashes)
    pools: Dict[str, List[Dict[str, Any]]] = {"collect_r1": coll_rows}
    reuse: Dict[str, str] = {}
    flags: Dict[str, str] = {}
    meta: Dict[str, Any] = {}

    if bench in {"medqa", "aqua"}:
        train = split("train")
        paper_train = train[:1200]
        meta["paper_train_pool"] = "first 1200 train rows (train_size 1200, explicit splits)"
        pools["dev"] = split("dev")[:200] if bench == "medqa" else split("dev")
        pools["test"] = split("test")[:500] if bench == "medqa" else split("test")
        claims.claim("collect_r1", coll_rows)
        for pool in ("dev", "test"):
            claims.claim(pool, pools[pool])
        grpo_candidates = [r for r in paper_train if not in_collection(r)]
        fresh_candidates = train[1200:]
        meta["fresh_source"] = "train rows after the first 1200"
    elif bench == "mmlu_pro":
        order = list(rows)
        random.Random(PAPER_SEED).shuffle(order)
        position = {int(r["example_id"]): i for i, r in enumerate(order)}
        coll_pos = sorted(position[int(r["example_id"])] for r in coll_rows)
        meta.update(
            order="random.Random(42).shuffle(cache rows) (paper random split path)",
            collect_r1_positions=[coll_pos[0], coll_pos[-1]],
            paper_train_pool="shuffled positions 300-1499 (test_size 200, dev_size 100, train_size 1200)",
        )
        paper_train = order[300:1500]
        pools["dev"] = order[:200]
        pools["test_paper"] = order[:500]
        claims.claim("collect_r1", coll_rows)
        claims.claim("dev", pools["dev"])
        grpo_candidates = [r for r in paper_train if not in_collection(r)]
        pools["test"] = claims.take("test", order[1500:], 500, exclude_advisor=True)
        meta["test_source"] = "first 500 eligible shuffled positions >= 1500 (not advisor-SFT, hash-new)"
        fresh_candidates = order[2000:]
        meta["fresh_source"] = "shuffled positions >= 2000"
    elif bench == "gpqa":
        if test_rows is None:
            raise ValueError("gpqa needs the Diamond-100 test rows")
        paper_train = rows[:406]
        meta["paper_train_pool"] = "first 406 rows of gpqa_train446 (dev_size 40 carved from the tail)"
        leftover_diamond = 98
        meta["leftover_diamond_rows"] = ("first 98 rows of gpqa_train446 are the Diamond questions not in "
                                         "Diamond-100 (scripts/build_gpqa_splits.py order)")
        pools["test"] = list(test_rows)
        claims.claim("collect_r1", coll_rows)
        claims.claim("test", pools["test"])
        diamond_ids = {int(r["example_id"]) for r in rows[:leftover_diamond]}
        spare = [r for r in rows if not in_collection(r)]
        diamond_spare = [r for r in spare if int(r["example_id"]) in diamond_ids]
        others = [r for r in spare if int(r["example_id"]) not in diamond_ids]
        _rng(bench, "dev", seed).shuffle(others)
        dev = claims.take("dev", diamond_spare, len(diamond_spare), exclude_advisor=True)
        dev += claims.take("dev_others", others, 100 - len(dev), exclude_advisor=True)
        pools["dev"] = dev
        meta["dev_source"] = f"all {len(diamond_spare)} non-collection leftover-Diamond rows + seeded others"
        remaining = [r for r in others if identity(r) not in claims.owner]
        grpo = claims.take("grpo_r1", remaining, len(remaining), exclude_advisor=False)
        pools.update(grpo_r1=grpo, grpo_r2=list(grpo), grpo_r3=list(grpo),
                     collect_r2=list(coll_rows), collect_r3=list(coll_rows))
        reuse = {"collect_r2": "collect_r1", "collect_r3": "collect_r1", "grpo_r2": "grpo_r1", "grpo_r3": "grpo_r1"}
        flags["collect_r2"] = flags["collect_r3"] = "GPQA has no spare questions: recollection reuses the collect_r1 roots"
        flags["grpo_r1"] = "GPQA GRPO pool is the 146 rows left after collection and dev, reused every round"
        grpo_candidates = fresh_candidates = None
    else:
        raise KeyError(f"Unknown benchmark {bench}")

    meta["paper_sampler_reproduced"] = _paper_sample(paper_train, len(coll_rows)) == [int(r["example_id"]) for r in coll_rows]
    if grpo_candidates is not None:
        meta["paper_grpo_pool"] = len(grpo_candidates)
        _rng(bench, "grpo", seed).shuffle(grpo_candidates)
        pools.update(_chunks(claims.take("grpo", grpo_candidates, ROUNDS * N_GRPO, exclude_advisor=False), "grpo_r", N_GRPO))
        _rng(bench, "collect", seed).shuffle(fresh_candidates)
        fresh = claims.take("collect", fresh_candidates, (ROUNDS - 1) * N_COLLECT, exclude_advisor=True)
        pools.update({f"collect_r{k + 2}": fresh[k * N_COLLECT:(k + 1) * N_COLLECT] for k in range(ROUNDS - 1)})
    if "test_paper" in pools:
        used = {p: {identity(r) for r in pools[p]} for p in POOLS}
        is_advisor = lambda r: int(r["example_id"]) in advisor_ids or identity(r) in advisor_hashes
        overlap = {p: n for p in POOLS if (n := sum(identity(r) in used[p] for r in pools["test_paper"]))}
        overlap["advisor_sft"] = sum(map(is_advisor, pools["test_paper"]))
        pools["test_paper_clean"] = [r for r in pools["test_paper"]
                                     if not is_advisor(r) and not any(identity(r) in used[p] for p in POOLS)]
        overlap["clean"] = len(pools["test_paper_clean"])
        meta["flagged_overlap"] = {"test_paper": overlap}
        grpo = "/".join(str(overlap.get(f"grpo_r{k + 1}", 0)) for k in range(ROUNDS))
        flags["test_paper"] = (
            "paper-reconstructed --test_size 500 set (positions 0-499); shares question hashes with dev "
            f"({overlap.get('dev', 0)}), collect_r1 ({overlap.get('collect_r1', 0)}), grpo_r1/r2/r3 ({grpo}; every "
            f"arm's G_k trains on them) and advisor SFT ({overlap['advisor_sft']}); report separately and on test_paper_clean")
        flags["test_paper_clean"] = (f"the {overlap['clean']} test_paper questions in no other pool and no advisor-SFT "
                                     "file; the only uncontaminated view of the paper test set")

    names = POOLS + tuple(p for p in FLAGGED_POOLS if p in pools)
    manifest = {
        "benchmark": bench,
        "builder_version": BUILDER_VERSION,
        "seed": seed,
        "question_hash": HASH_DEFINITION,
        "counts": {p: len(pools[p]) for p in names},
        "reuse": reuse,
        "flags": flags,
        "meta": meta,
        "duplicates": {
            "cache_extra_rows": len(rows) - len({identity(r) for r in rows}),
            "within_pool": {p: g for p in names if (g := _dup_groups(pools[p]))},
            "skipped_candidates": claims.skipped,
        },
        "advisor_sft": {
            "n": len(advisor_ids),
            "overlap": {p: sum(int(r["example_id"]) in advisor_ids or identity(r) in advisor_hashes for r in pools[p])
                        for p in names},
            "entries": sorted(({"example_id": int(a["example_id"]), "question_hash": a["question_hash"]} for a in advisor_sft),
                              key=lambda e: (e["example_id"], e["question_hash"])),
        },
        "pools": {p: _entries(pools[p]) for p in names},
    }
    check_manifest(manifest)
    return manifest


def check_manifest(manifest: Dict[str, Any]) -> None:
    """Raise if pools overlap by hash, except declared reuse and flagged pools."""
    pools = manifest["pools"]
    missing = [p for p in POOLS if p not in pools]
    if missing:
        raise ValueError(f"missing pools {missing}")
    reuse = manifest.get("reuse", {})
    for pool, src in reuse.items():
        if pools[pool] != pools[src]:
            raise ValueError(f"{pool} must equal {src}")
    canonical = [p for p in POOLS if p not in reuse]
    hashes = {p: {e["question_hash"] for e in pools[p]} for p in canonical}
    for i, a in enumerate(canonical):
        for b in canonical[i + 1:]:
            shared = hashes[a] & hashes[b]
            if shared:
                raise ValueError(f"pools {a} and {b} share {len(shared)} question hashes")
    fresh = [p for p in ("collect_r2", "collect_r3") if p not in reuse]
    advisor = {e["question_hash"] for e in manifest["advisor_sft"]["entries"]}
    for p in fresh:
        if hashes[p] & advisor:
            raise ValueError(f"{p} contains advisor-SFT questions")
    if "test_paper_clean" in pools:
        clean = {e["question_hash"] for e in pools["test_paper_clean"]}
        if not clean <= {e["question_hash"] for e in pools["test_paper"]}:
            raise ValueError("test_paper_clean must be a subset of test_paper")
        if clean & advisor or any(clean & hashes[p] for p in canonical):
            raise ValueError("test_paper_clean overlaps another pool or advisor-SFT data")
    for p, entries in pools.items():
        if manifest["counts"][p] != len(entries):
            raise ValueError(f"count mismatch for {p}")


# ----------------------------------------------------------------- I/O helpers

def load_cache(path, expected_sha256: str = "") -> List[Dict[str, Any]]:
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError(f"{path}: sha256 differs from the registry")
    return read_jsonl(str(path))


def build_aqua_rows(hf_cache_dir: Optional[str] = None, downloader=None, sources=None) -> List[Dict[str, Any]]:
    """The 9.30 loader over the registry-pinned deepmind/aqua_rat raw parquet files (network on a cold cache)."""
    import pyarrow.parquet as pq

    from ...benchmarks.aqua_rat import load_aqua_rat

    if downloader is None:
        from huggingface_hub import hf_hub_download as downloader
    sources = registry.get("aqua").cache_sources if sources is None else sources
    with tempfile.TemporaryDirectory() as raw:
        for split, f in sources:
            local = downloader(repo_id=f.repo_id, filename=f.path, repo_type=f.repo_type,
                               revision=f.revision, cache_dir=hf_cache_dir)
            if file_sha256(local) != f.sha256:
                raise ValueError(f"{f.uri}: sha256 differs from the registry")
            records = pq.read_table(local).to_pylist()
            Path(raw, f"{split}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return [r.to_dict() for r in load_aqua_rat(source="local", local_path=raw)]


def ensure_aqua_cache(path, expected_sha256: str, hf_cache_dir: Optional[str] = None, downloader=None) -> Path:
    """Materialize the AQuA normalized cache; nothing lands at ``path`` unless its sha256 matches."""
    path = Path(path)
    if path.exists():
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"{path}: AQuA cache sha256 differs from the registry; delete it to rebuild")
        return path
    rows = build_aqua_rows(hf_cache_dir, downloader)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if file_sha256(tmp) != expected_sha256:
        tmp.unlink()
        raise ValueError(f"{path}: rebuilt AQuA cache sha256 differs from the registry (loader drift)")
    os.replace(tmp, path)
    return path


def pool_cache(bench: str, pool: str) -> Tuple[Path, str]:
    """(path, sha256) of the cache whose ``example_id``s a pool indexes (GPQA ``test``: Diamond-100)."""
    spec = registry.get(bench)
    aux = {role: (rel, digest) for role, rel, digest in spec.aux_caches}
    if pool == "test" and "test" in aux:
        return spec.path(aux["test"][0]), aux["test"][1]
    return spec.path(spec.cache), spec.cache_sha256


def pool_rows(manifest: Dict[str, Any], pool: str, rows: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """A pool's cache rows, resolved by ``example_id`` with every ``question_hash`` checked."""
    if rows is None:
        rows = load_cache(*pool_cache(manifest["benchmark"], pool))
    return _resolve({int(r["example_id"]): r for r in rows}, manifest["pools"][pool], pool)


def identities(path) -> List[Dict[str, Any]]:
    """``{example_id, question_hash}`` items of a records / SFT jsonl file."""
    return [{"example_id": int(r["example_id"]), "question_hash": str(r["question_hash"])} for r in read_jsonl(str(path))]


def union_identities(paths: Iterable) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for path in paths:
        for item in identities(path):
            key = (item["example_id"], item["question_hash"])
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def write_manifest(manifest: Dict[str, Any], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_manifest(path) -> Dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise ValueError(f"{path}: builder_version {manifest.get('builder_version')} != {BUILDER_VERSION}")
    check_manifest(manifest)
    return manifest


def write_frozen(manifest: Dict[str, Any], path, force: bool = False) -> Path:
    """``write_manifest`` that refuses to replace a different existing manifest unless ``force``."""
    path = Path(path)
    if path.exists() and not force:
        old = json.loads(path.read_text(encoding="utf-8"))
        new = json.loads(json.dumps(manifest))
        if old != new:
            keys = sorted(k for k in set(old) | set(new) if k != "pools" and old.get(k) != new.get(k))
            pools = sorted(p for p in set(old.get("pools", {})) | set(new["pools"])
                           if old.get("pools", {}).get(p) != new["pools"].get(p))
            raise ValueError(f"{path} exists and differs (pools {pools}, keys {keys}); use force to overwrite")
    return write_manifest(manifest, path)


def prepare_benchmark(
    bench: str,
    records: str,
    advisor_sft: Sequence[str],
    *,
    cache: Optional[str] = None,
    out: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    seed: int = PAPER_SEED,
    force: bool = False,
) -> Dict[str, Any]:
    if seed != PAPER_SEED and out is None:
        raise ValueError(f"seed {seed} needs an explicit out path; the tracked manifests are frozen at seed {PAPER_SEED}")
    spec = registry.get(bench)
    cache_path = Path(cache) if cache else spec.path(spec.cache)
    if bench == "aqua":
        ensure_aqua_cache(cache_path, spec.cache_sha256, hf_cache_dir)
    rows = load_cache(cache_path, spec.cache_sha256)
    sources: Dict[str, Any] = {
        "cache": {"path": spec.cache, "sha256": spec.cache_sha256, "rows": len(rows)},
        "collect_r1": {"uri": spec.round1_records.uri, "sha256": file_sha256(records)},
        "advisor_sft": [{"uri": m.uri, "sha256": file_sha256(p)} for (_, m), p in zip(spec.advisor_sft, advisor_sft)],
    }
    if sources["collect_r1"]["sha256"] != spec.round1_records.sha256:
        raise ValueError(f"{records}: sha256 differs from the registry")
    for (kind, member), item in zip(spec.advisor_sft, sources["advisor_sft"]):
        if item["sha256"] != member.sha256:
            raise ValueError(f"advisor-SFT {kind}: sha256 differs from the registry")
    aux = {}
    for role, rel, digest in spec.aux_caches:
        aux[role] = load_cache(spec.path(rel), digest)
        sources[f"{role}_cache"] = {"path": rel, "sha256": digest, "rows": len(aux[role])}
    manifest = build_splits(bench, rows, identities(records), union_identities(advisor_sft),
                            test_rows=aux.get("test"), seed=seed)
    if "dev" in aux and _entries(aux["dev"]) != manifest["pools"]["dev"]:
        raise ValueError(f"{bench}: dev cache does not match the dev pool")
    manifest["sources"] = sources
    write_frozen(manifest, out or spec.path(spec.split_manifest), force)
    return manifest
