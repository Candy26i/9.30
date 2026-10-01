import copy
import json
import random
from functools import lru_cache
from pathlib import Path

import pytest

from src.benchmarks.base import question_hash
from src.manager.mcq_rsi import splits
from src.manager.mcq_rsi.benchmarks import BENCHMARKS

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def manifest(name):
    return splits.read_manifest(ROOT / BENCHMARKS[name].split_manifest)


@lru_cache(maxsize=None)
def cache(rel):
    return tuple(json.loads(line) for line in (ROOT / rel).read_text(encoding="utf-8").splitlines() if line.strip())


def ids(m, pool):
    return [e["example_id"] for e in m["pools"][pool]]


def hashes(m, pool):
    return {e["question_hash"] for e in m["pools"][pool]}


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_manifests_are_frozen_and_disjoint(name):
    m = manifest(name)
    assert m["benchmark"] == name and m["builder_version"] == splits.BUILDER_VERSION and m["seed"] == 42
    assert m["meta"]["paper_sampler_reproduced"] is True
    sizes = {"gpqa": (200, 146, 100, 100), "aqua": (400, 256, 254, 254)}.get(name, (400, 256, 200, 500))
    assert [m["counts"][p] for p in ("collect_r1", "grpo_r1", "dev", "test")] == list(sizes)
    canonical = [p for p in splits.POOLS if p not in m["reuse"]]
    for i, a in enumerate(canonical):
        for b in canonical[i + 1:]:
            assert not hashes(m, a) & hashes(m, b), (a, b)
    if name != "gpqa":
        assert m["reuse"] == {}
        for p in ("collect_r2", "collect_r3"):
            assert m["counts"][p] == 400 and m["advisor_sft"]["overlap"][p] == 0
    assert m["sources"]["cache"]["sha256"] == BENCHMARKS[name].cache_sha256
    assert m["sources"]["collect_r1"]["sha256"] == BENCHMARKS[name].round1_records.sha256


@pytest.mark.parametrize("name", ["medqa", "mmlu_pro", "gpqa"])
def test_manifest_hashes_match_tracked_caches(name):
    m = manifest(name)
    b = BENCHMARKS[name]
    rows = {r["example_id"]: r for r in cache(b.cache)}
    for _, rel, _ in b.aux_caches:
        rows.update({r["example_id"]: r for r in cache(rel)})
    for pool, entries in m["pools"].items():
        for e in entries:
            assert question_hash(rows[e["example_id"]]["question"]) == e["question_hash"], (pool, e)


def test_medqa_paper_parity():
    m = manifest("medqa")
    assert ids(m, "dev") == list(range(10178, 10378))
    assert ids(m, "test") == list(range(11450, 11950))
    assert all(i < 1200 for p in ("collect_r1", "grpo_r1", "grpo_r2", "grpo_r3") for i in ids(m, p))
    assert m["meta"]["paper_grpo_pool"] == 800
    assert all(1200 <= i < 10178 for p in ("collect_r2", "collect_r3") for i in ids(m, p))
    rows = cache("outputs/data/medqa_dev200.jsonl")
    assert [r["example_id"] for r in rows] == ids(m, "dev")


def test_mmlu_pro_paper_parity():
    m = manifest("mmlu_pro")
    order = list(cache(BENCHMARKS["mmlu_pro"].cache))
    random.Random(42).shuffle(order)
    pos = {r["example_id"]: i for i, r in enumerate(order)}
    coll = [pos[i] for i in ids(m, "collect_r1")]
    assert (min(coll), max(coll)) == (304, 1495) and m["meta"]["collect_r1_positions"] == [304, 1495]
    assert [pos[i] for i in ids(m, "dev")] == list(range(200))
    assert [pos[i] for i in ids(m, "test_paper")] == list(range(500))
    assert len(set(ids(m, "test_paper")) & set(ids(m, "collect_r1"))) == 68
    assert "test_paper" in m["flags"] and "test_paper_clean" in m["flags"]
    overlap = m["meta"]["flagged_overlap"]["test_paper"]
    assert overlap == {"dev": 200, "collect_r1": 68, "grpo_r1": 47, "grpo_r2": 50, "grpo_r3": 31,
                       "advisor_sft": 4, "clean": 104}
    assert "47/50/31" in m["flags"]["test_paper"]
    dirty = set().union(*(hashes(m, p) for p in splits.POOLS))
    dirty |= {e["question_hash"] for e in m["advisor_sft"]["entries"]}
    assert hashes(m, "test_paper_clean") == hashes(m, "test_paper") - dirty
    assert m["counts"]["test_paper_clean"] == 104
    assert all(300 <= pos[i] < 1500 for p in ("grpo_r1", "grpo_r2", "grpo_r3") for i in ids(m, p))
    assert all(pos[i] >= 1500 for i in ids(m, "test"))
    assert all(pos[i] >= 2000 for p in ("collect_r2", "collect_r3") for i in ids(m, p))
    assert m["duplicates"]["within_pool"] == {"collect_r1": [[9680, 8859]]}
    assert not any(k.startswith("test_paper") for k in splits.read_manifest(ROOT / BENCHMARKS["medqa"].split_manifest)["pools"])
    assert m["advisor_sft"]["overlap"]["test"] == 0


def test_gpqa_dev_is_disjoint_from_collection_and_diamond():
    m = manifest("gpqa")
    train = cache("outputs/data/gpqa_train446.jsonl")
    diamond = cache("outputs/data/gpqa_diamond_eval100.jsonl")
    dev, coll = set(ids(m, "dev")), set(ids(m, "collect_r1"))
    assert not dev & coll and not hashes(m, "dev") & {question_hash(r["question"]) for r in diamond}
    assert ids(m, "test") == [r["example_id"] for r in diamond]
    leftover = {r["example_id"] for r in train[:98]} - coll
    assert len(leftover) == 51 and leftover <= dev
    assert len(coll & {r["example_id"] for r in train[:98]}) == 47
    assert not coll & {r["example_id"] for r in train[406:]}
    assert dev | coll | set(ids(m, "grpo_r1")) == {r["example_id"] for r in train}
    assert m["reuse"] == {"collect_r2": "collect_r1", "collect_r3": "collect_r1", "grpo_r2": "grpo_r1", "grpo_r3": "grpo_r1"}
    assert m["advisor_sft"]["n"] == 160 and m["advisor_sft"]["overlap"]["collect_r1"] == 160


def test_aqua_manifest_records_paper_duplicates():
    m = manifest("aqua")
    assert m["duplicates"]["within_pool"] == {"collect_r1": [[483, 861], [1144, 893]]}
    assert m["advisor_sft"]["overlap"]["collect_r1"] == 400
    assert all(i < 1200 for p in ("collect_r1", "grpo_r1", "grpo_r2", "grpo_r3") for i in ids(m, p))
    assert all(1200 <= i < 97467 for p in ("collect_r2", "collect_r3") for i in ids(m, p))
    assert ids(m, "dev") == list(range(97467, 97721)) and ids(m, "test") == list(range(97721, 97975))


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_rebuild_from_cache_is_identical(name):
    m = manifest(name)
    b = BENCHMARKS[name]
    if not (ROOT / b.cache).exists():
        pytest.skip(f"{b.cache} not built")
    test_rows = cache(b.aux_caches[0][1]) if name == "gpqa" else None
    rebuilt = splits.build_splits(name, cache(b.cache), m["pools"]["collect_r1"], m["advisor_sft"]["entries"],
                                  test_rows=test_rows)
    assert rebuilt == {k: v for k, v in m.items() if k != "sources"}


def _synthetic(n_train=2400, dup_every=97):
    rows = []
    for i in range(n_train + 400):
        q = f"question {i // dup_every}" if i % dup_every == 5 and i > 5 else f"question {i}"
        rows.append({"example_id": i, "question": q, "choices": {"A": "x", "B": "y"},
                     "ground_truth": "A", "context": "", "split": "train" if i < n_train else ("dev" if i < n_train + 200 else "test")})
    train = rows[:1200]
    coll = random.Random(42).sample(train, 400)
    return rows, [{"example_id": r["example_id"], "question_hash": question_hash(r["question"])} for r in coll]


def test_builder_is_deterministic_and_records_duplicates():
    rows, coll = _synthetic()
    rows[coll[1]["example_id"]]["question"] = rows[coll[0]["example_id"]]["question"]  # paper-style duplicate root
    coll[1]["question_hash"] = coll[0]["question_hash"]
    advisor = [{"example_id": r["example_id"], "question_hash": question_hash(r["question"])} for r in rows[1200:1300]]
    a = splits.build_splits("aqua", rows, coll, advisor)
    b = splits.build_splits("aqua", list(rows), list(coll), list(advisor))
    assert a == b
    assert [coll[0]["example_id"], coll[1]["example_id"]] in a["duplicates"]["within_pool"]["collect_r1"]
    assert a["duplicates"]["cache_extra_rows"] > 0
    assert a["duplicates"]["skipped_candidates"]["collect"].get("advisor_sft", 0) > 0
    fresh = {e["example_id"] for p in ("collect_r2", "collect_r3") for e in a["pools"][p]}
    assert not fresh & set(range(1200, 1300))
    assert splits.build_splits("aqua", rows, coll, advisor, seed=7)["pools"]["grpo_r1"] != a["pools"]["grpo_r1"]


def test_builder_rejects_loader_drift_and_overlap():
    rows, coll = _synthetic()
    drift = [dict(coll[0], question_hash="0" * 16)] + coll[1:]
    with pytest.raises(ValueError, match="loader drift"):
        splits.build_splits("aqua", rows, drift, [])
    m = splits.build_splits("aqua", rows, coll, [])
    bad = copy.deepcopy(m)
    bad["pools"]["dev"].append(bad["pools"]["test"][0])
    bad["counts"]["dev"] += 1
    with pytest.raises(ValueError, match="share"):
        splits.check_manifest(bad)
    bad = copy.deepcopy(m)
    bad["advisor_sft"]["entries"] = [bad["pools"]["collect_r2"][0]]
    with pytest.raises(ValueError, match="advisor-SFT"):
        splits.check_manifest(bad)


def test_write_manifest_is_compact_and_round_trips(tmp_path):
    rows, coll = _synthetic()
    m = splits.build_splits("medqa", rows, coll, [])
    path = splits.write_manifest(m, tmp_path / "x_splits.json")
    text = path.read_text()
    assert text.count("\n") == 1 and text.endswith("\n")
    assert splits.read_manifest(path) == m


def test_aqua_cache_builder_checks_digest(tmp_path, monkeypatch):
    row = {"example_id": 0, "question": "1+1?", "choices": {"A": "2", "B": "3"}, "ground_truth": "A", "split": "train"}
    monkeypatch.setattr(splits, "build_aqua_rows", lambda *a, **kw: [row])
    path = tmp_path / "aqua.jsonl"
    with pytest.raises(ValueError, match="drift"):
        splits.ensure_aqua_cache(path, "0" * 64)
    assert not path.exists() and not list(tmp_path.iterdir())
    digest = splits.file_sha256(_write(tmp_path / "ref.jsonl", [row]))
    assert splits.ensure_aqua_cache(path, digest) == path and splits.file_sha256(path) == digest
    path.write_text("tampered\n")
    with pytest.raises(ValueError, match="delete it"):
        splits.ensure_aqua_cache(path, digest)


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_aqua_rows_come_from_pinned_raw_files(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import dataclasses
    import pyarrow as pa

    raw = {"question": ["Two plus two?"], "options": [["A)3", "B)4", "C)5", "D)6", "E)7"]],
           "rationale": ["2+2=4"], "correct": ["B"]}
    pq.write_table(pa.table(raw), tmp_path / "raw.parquet")
    sha = splits.file_sha256(tmp_path / "raw.parquet")
    sources = [(split, dataclasses.replace(f, sha256=sha)) for split, f in BENCHMARKS["aqua"].cache_sources]
    calls = []

    def downloader(repo_id, filename, repo_type, revision, cache_dir=None):
        calls.append((repo_id, filename, revision))
        return str(tmp_path / "raw.parquet")

    rows = splits.build_aqua_rows(downloader=downloader, sources=sources)
    assert [r["split"] for r in rows] == ["train", "dev", "test"] and [r["example_id"] for r in rows] == [0, 1, 2]
    assert rows[0]["choices"]["B"] == "4" and "2+2=4" not in json.dumps(rows)
    assert {c[2] for c in calls} == {"33301c6a050c96af81f63cad5562cb5363e88971"}
    bad = [(s, dataclasses.replace(f, sha256="0" * 64)) for s, f in sources]
    with pytest.raises(ValueError, match="sha256"):
        splits.build_aqua_rows(downloader=downloader, sources=bad)


@pytest.mark.parametrize("name", ["medqa", "mmlu_pro", "gpqa"])
def test_pool_rows_resolve_by_example_id(name):
    m = manifest(name)
    for pool in m["pools"]:
        rows = splits.pool_rows(m, pool)
        assert [r["example_id"] for r in rows] == ids(m, pool)
        assert [question_hash(r["question"]) for r in rows] == [e["question_hash"] for e in m["pools"][pool]]
    if name == "gpqa":
        assert splits.pool_cache("gpqa", "test")[0].name == "gpqa_diamond_eval100.jsonl"
        with pytest.raises(ValueError, match="does not match"):
            splits.pool_rows(m, "test", rows=cache(BENCHMARKS["gpqa"].cache))


def test_frozen_manifest_is_not_silently_overwritten(tmp_path):
    rows, coll = _synthetic()
    m = splits.build_splits("medqa", rows, coll, [])
    path = splits.write_frozen(m, tmp_path / "x_splits.json")
    assert splits.write_frozen(m, path) == path
    other = splits.build_splits("medqa", rows, coll, [], seed=7)
    with pytest.raises(ValueError, match="grpo_r1"):
        splits.write_frozen(other, path)
    assert splits.read_manifest(path) == m
    splits.write_frozen(other, path, force=True)
    assert splits.read_manifest(path) == other
    with pytest.raises(ValueError, match="explicit out"):
        splits.prepare_benchmark("medqa", "unused", [], seed=7)
