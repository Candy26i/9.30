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


# ------------------------------------------------------------------ extended rounds (r4, r5)

EXT = ("collect_r4", "grpo_r4", "collect_r5", "grpo_r5")


@pytest.mark.parametrize("name", ["mmlu_pro", "aqua"])
def test_tracked_r5_manifests_keep_every_frozen_pool_and_add_disjoint_rounds(name):
    base_path = ROOT / BENCHMARKS[name].split_manifest
    path = splits.extended_manifest_path(name, 5)
    assert path == base_path.with_name(f"{name}_splits_r5.json") and path.is_file()
    base, ext = manifest(name), splits.read_manifest(path)
    assert ext["meta"]["extension"]["source_manifest_sha256"] == splits.file_sha256(base_path)
    assert ext["meta"]["extension"]["source_manifest"] == BENCHMARKS[name].split_manifest
    assert ext["meta"]["extension"]["pools"] == list(EXT) and ext["meta"]["extension"]["rule"]
    # Every original pool byte-identical (entries and order); the rest of the manifest only gains new-pool records.
    assert all(ext["pools"][p] == entries for p, entries in base["pools"].items())
    assert set(ext["pools"]) == set(base["pools"]) | set(EXT)
    for key in ("benchmark", "builder_version", "seed", "question_hash", "reuse", "sources"):
        assert ext[key] == base[key], key
    # Only flags for the extended GRPO pools are added: their population (and advisor-SFT share) differs.
    assert {k: v for k, v in ext["flags"].items() if k not in ("grpo_r4", "grpo_r5")} == base["flags"]
    overlap = ext["advisor_sft"]["overlap"]
    assert ext["meta"]["extension"]["grpo_advisor_sft_overlap"] == {
        "base": {p: overlap[p] for p in ("grpo_r1", "grpo_r2", "grpo_r3")},
        "extended": {p: overlap[p] for p in ("grpo_r4", "grpo_r5")}, "pool_size": 256}
    for p in ("grpo_r4", "grpo_r5"):
        assert "round-3/4 boundary" in ext["flags"][p] and f"{p} {overlap[p]}" in ext["flags"][p]
        assert f"grpo_r1 {overlap['grpo_r1']}" in ext["flags"][p]
    assert sum(overlap[f"grpo_r{k}"] for k in (1, 2, 3)) > 100 and sum(overlap[p] for p in ("grpo_r4", "grpo_r5")) < 10
    assert {k: v for k, v in ext["meta"].items() if k != "extension"} == base["meta"]
    assert {p: ext["counts"][p] for p in base["counts"]} == base["counts"]
    assert [ext["counts"][p] for p in EXT] == [400, 256, 400, 256]
    assert ext["advisor_sft"]["entries"] == base["advisor_sft"]["entries"]
    assert ext["advisor_sft"]["overlap"]["collect_r4"] == ext["advisor_sft"]["overlap"]["collect_r5"] == 0
    old = set().union(*(hashes(base, p) for p in base["pools"]))
    advisor = {e["question_hash"] for e in base["advisor_sft"]["entries"]}
    advisor_ids = {e["example_id"] for e in base["advisor_sft"]["entries"]}
    seen = set()
    for p in EXT:
        assert not hashes(ext, p) & old and not hashes(ext, p) & seen, p
        seen |= hashes(ext, p)
    for p in ("collect_r4", "collect_r5"):
        assert not hashes(ext, p) & advisor and not set(ids(ext, p)) & advisor_ids
    # Drawn from the population of collect_r2/r3.
    if name == "aqua":
        assert all(1200 <= i < 97467 for p in EXT for i in ids(ext, p))
    else:
        order = list(cache(BENCHMARKS[name].cache))
        random.Random(42).shuffle(order)
        pos = {r["example_id"]: i for i, r in enumerate(order)}
        assert all(pos[i] >= 2000 for p in EXT for i in ids(ext, p))
    # Regenerating from the cache reproduces the tracked file exactly.
    if not (ROOT / BENCHMARKS[name].cache).exists():
        pytest.skip(f"{BENCHMARKS[name].cache} not built")
    again = splits.extend_splits(base, cache(BENCHMARKS[name].cache), 5,
                                 source_sha256=splits.file_sha256(base_path), source_name=BENCHMARKS[name].split_manifest)
    assert again == json.loads(path.read_text())


def test_extend_benchmark_writes_a_new_file_and_never_touches_the_frozen_manifest(tmp_path):
    name = "mmlu_pro"
    base_path = ROOT / BENCHMARKS[name].split_manifest
    if not (ROOT / BENCHMARKS[name].cache).exists():
        pytest.skip("cache not built")
    before = splits.file_sha256(base_path)
    m, path = splits.extend_benchmark(name, 5, out=str(tmp_path / "x_r5.json"))
    assert path == tmp_path / "x_r5.json" and splits.file_sha256(base_path) == before
    assert splits.read_manifest(path) == json.loads(splits.extended_manifest_path(name, 5).read_text())
    with pytest.raises(ValueError, match="never replaces its source"):
        splits.extend_benchmark(name, 5, out=str(base_path))
    with pytest.raises(ValueError, match="no spare questions"):
        splits.extend_benchmark("gpqa", 5)
    with pytest.raises(ValueError, match="already extended"):
        splits.extend_benchmark(name, 5, manifest=str(path), out=str(tmp_path / "y.json"))
    from src.manager.mcq_rsi import __main__ as cli
    with pytest.raises(SystemExit):
        cli.main(["extend-splits", "--bench", "gpqa"])
    assert cli.main(["extend-splits", "--bench", name, "--rounds", "4", "--out", str(tmp_path / "x_r4.json")]) == 0
    r4 = splits.read_manifest(tmp_path / "x_r4.json")
    assert [p for p in r4["pools"] if p not in manifest(name)["pools"]] == ["collect_r4", "grpo_r4"]
    assert all(r4["pools"][p] == m["pools"][p] for p in ("collect_r4", "grpo_r4"))


def test_extend_splits_is_deterministic_round_stable_and_refuses_gpqa():
    rows, coll = _synthetic(n_train=6000)
    for r in rows[6000:]:  # duplicates only among train rows (dev/test stay disjoint from collect_r1)
        r["question"] = f"question {r['example_id']}"
    advisor = [{"example_id": r["example_id"], "question_hash": question_hash(r["question"])} for r in rows[1200:1500]]
    base = splits.build_splits("aqua", rows, coll, advisor)
    ext = splits.extend_splits(base, rows, 5, source_sha256="s" * 64)
    assert ext == splits.extend_splits(copy.deepcopy(base), list(rows), 5, source_sha256="s" * 64)
    assert all(ext["pools"][p] == base["pools"][p] for p in base["pools"]) and base == splits.build_splits("aqua", rows, coll, advisor)
    r4 = splits.extend_splits(base, rows, 4, source_sha256="s" * 64)
    assert {p: r4["pools"][p] for p in ("collect_r4", "grpo_r4")} == {p: ext["pools"][p] for p in ("collect_r4", "grpo_r4")}
    assert "collect_r5" not in r4["pools"] and splits.manifest_rounds(ext["pools"]) == 5
    fresh = {e["example_id"] for p in EXT for e in ext["pools"][p]}
    assert fresh <= set(range(1200, 6000)) and not {e["example_id"] for p in ("collect_r4", "collect_r5")
                                                     for e in ext["pools"][p]} & set(range(1200, 1500))
    assert ext["duplicates"]["skipped_candidates"]["collect_r4"]["advisor_sft"] > 0
    # Duplicate-question rows of an existing pool are never drawn (same hash, other example_id).
    dup_of_old = [r for r in rows if r["example_id"] >= 1200 and question_hash(r["question"]) in
                  set().union(*(hashes(base, p) for p in base["pools"])) and
                  r["example_id"] not in {e["example_id"] for p in base["pools"] for e in base["pools"][p]}]
    assert dup_of_old and not {r["example_id"] for r in dup_of_old} & fresh
    for p in EXT:
        assert ext["meta"]["extension"]["pools"] == list(EXT) and ext["counts"][p] == len(ext["pools"][p])
    with pytest.raises(ValueError, match="already extended"):
        splits.extend_splits(ext, rows, 5, source_sha256="")
    for bad in (3, 6):
        with pytest.raises(ValueError, match="rounds must be in"):
            splits.extend_splits(base, rows, bad, source_sha256="")
    with pytest.raises(ValueError, match="only .* eligible"):  # not enough fresh questions
        small, small_coll = _synthetic(n_train=2600)
        splits.extend_splits(splits.build_splits("aqua", small, small_coll, []), small, 5, source_sha256="")
    with pytest.raises(ValueError, match="no spare questions"):
        splits.extend_splits(manifest("gpqa"), cache(BENCHMARKS["gpqa"].cache), 5, source_sha256="")
    # check_manifest: extended pools are validated like the base ones.
    bad = copy.deepcopy(ext)
    del bad["pools"]["grpo_r5"], bad["counts"]["grpo_r5"]
    with pytest.raises(ValueError, match="extended pools"):
        splits.check_manifest(bad)
    bad = copy.deepcopy(ext)
    bad["pools"]["grpo_r4"][0] = bad["pools"]["dev"][0]
    with pytest.raises(ValueError, match="pools dev and grpo_r4 share"):
        splits.check_manifest(bad)
    bad = copy.deepcopy(ext)
    bad["advisor_sft"]["entries"] = bad["advisor_sft"]["entries"] + [bad["pools"]["collect_r5"][0]]
    with pytest.raises(ValueError, match="collect_r5 contains advisor-SFT"):
        splits.check_manifest(bad)
    assert splits.extension_pools(["dev", "grpo_r5", "collect_r4", "grpo_r4", "collect_r5", "collect_r2"]) == list(EXT)
