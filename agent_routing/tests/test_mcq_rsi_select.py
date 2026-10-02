import dataclasses
import hashlib
import itertools
import json
import os
import random
from pathlib import Path

import pytest

from mcq_rsi_helpers import FakeManager, advisor_server, make_rows
from src.manager.marginal_value import _balance_records, _make_sft_rows, choose_preferred_sequence
from src.manager.mcq_rsi import __main__ as cli
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import importer
from src.manager.mcq_rsi import collect as C
from src.manager.mcq_rsi import select as S
from src.manager.mcq_rsi.advisors import CachedAdvisorPool
from src.subagents.train import validate_sft_splits

ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = Path(os.environ.get("MCQ_RSI_IMPORT_DIR", ROOT / "outputs/mcq_rsi/import"))
MEDQA = registry.get("medqa")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    """24 collected records: 12 correct roots, 6 rescues (some with ties / depth 2), 6 unsolved."""
    tmp = tmp_path_factory.mktemp("sel")
    rows = make_rows(24)
    script = {}
    for i, row in enumerate(rows):
        gold, eid = row["ground_truth"], row["example_id"]
        wrong = "A" if gold != "A" else "B"
        if i % 2 == 0:  # correct root; some calls keep it correct, the reasoner corrupts it
            script[eid] = {"root": (gold, "commit"), "rev": {("reasoner",): wrong}}
        elif i % 4 == 1:  # rescue, sometimes a tie, every 8th only at depth 2
            rev = {("reasoner", "verifier"): gold} if i % 8 == 1 else {("verifier",): gold, ("extractor",): gold}
            script[eid] = {"root": (wrong, "verifier"), "rev": rev}
        else:
            script[eid] = {"root": (wrong, "commit")}
    http, adapters = advisor_server(tmp)
    pool = CachedAdvisorPool("medqa", tmp / "cache", "http://x", adapters=adapters, http=http)
    out = C.collect(rows, MEDQA, FakeManager(script), pool, tmp / "run", pool_name="collect_r2", round_index=2)
    return [json.loads(line) for line in open(out["records_jsonl"])]


def test_dynamic_rho_cap_equals_balance_records(records):
    for rho in (0.0, 1.0, 1.5, 3.0, -1.0):
        rows, report = S.select(records, "dynamic", rho, seed=42)
        expected = list(itertools.chain.from_iterable(_make_sft_rows(r) for r in _balance_records(records, rho, 42)))
        assert rows == [{**r, "split": "train"} for r in expected]
        n_commit = min(len([r for r in records if r["preferred_sequence"] == []]), int(rho * 6)) if rho >= 0 else 12
        assert report["n_selected_commit_decisions"] == n_commit and report["n_selected_rescue_decisions"] == 6
        validate_sft_splits(rows)
        assert all(r["split"] == "train" and r["question_hash"] for r in rows)


def test_success_arm_semantics(records):
    rows, report = S.select(records, "success", 2.0, seed=7)
    kept = _balance_records(S.relabel(records), 2.0, 7)
    rng = random.Random(7)  # experiment.py: one stream over the balanced order
    chosen = [rng.choice(S.success_options(r)) for r in kept]
    assert report["n_selected_decisions"] == len(kept) == 6 + 12
    expected = list(itertools.chain.from_iterable(
        _make_sft_rows({**r, "preferred_sequence": c}) for r, c in zip(kept, chosen)))
    assert rows == [{**r, "split": "train"} for r in expected]
    for r in records:
        options = S.success_options(r)
        if r["direct_correct"]:  # commit plus the calls that keep the correct root correct
            assert options == [[], ["extractor"], ["verifier"]]
        elif r["preferred_sequence"] is not None:
            assert r["preferred_sequence"] in options and all(len(o) == len(r["preferred_sequence"]) for o in options)
        else:
            assert options == []
    # Correct roots sometimes train a call that stays correct, never the corrupting one.
    calls_on_correct = [c for r, c in zip(kept, chosen) if r["direct_correct"] and c]
    assert calls_on_correct and all(c != ["reasoner"] for c in calls_on_correct)
    validate_sft_splits(rows)


def test_max_depth_cut_and_split_guard(records):
    cut = S.relabel(records, max_depth=1)
    deep = [r for r in records if r["preferred_sequence"] and len(r["preferred_sequence"]) == 2]
    assert deep and all(c["preferred_sequence"] is None for c in cut if c["example_id"] in {d["example_id"] for d in deep})
    assert all(len(b["sequence"]) == 1 for c in cut for b in c["branches"])
    assert S.relabel(records) == records
    with pytest.raises(ValueError, match="never select"):
        S.select([{**records[0], "split": "dev"}], "dynamic", 1.0, 0)
    with pytest.raises(ValueError, match="select_static"):
        S.select(records, "static", 1.0, 0)


def test_static_arm_copies_pinned_file_after_sha_check(tmp_path):
    labels = tmp_path / "import" / "medqa" / "round1" / "labels.jsonl"
    labels.parent.mkdir(parents=True)
    labels.write_bytes(b'{"example_id": 1, "prompt": [], "response": []}\n')
    good = dataclasses.replace(MEDQA.round1_labels, sha256=hashlib.sha256(labels.read_bytes()).hexdigest())
    bench = dataclasses.replace(MEDQA, round1_labels=good)
    report = S.write_selection(bench, "static", None, tmp_path / "out.jsonl", import_dir=tmp_path / "import")
    assert (tmp_path / "out.jsonl").read_bytes() == labels.read_bytes() and report["n_sft_turns"] == 1
    assert json.loads((tmp_path / "out.report.json").read_text())["sha256"] == good.sha256
    with pytest.raises(ValueError, match="sha256"):
        S.select_static(MEDQA, tmp_path / "bad.jsonl", import_dir=tmp_path / "import")
    assert not (tmp_path / "bad.jsonl").exists()
    with pytest.raises(FileNotFoundError):
        S.select_static(MEDQA, tmp_path / "x.jsonl", import_dir=tmp_path / "nowhere")


def test_write_selection_is_atomic_and_unique(records, tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    out = tmp_path / "sel" / "dyn.jsonl"
    report = S.write_selection(MEDQA, "dynamic", path, out, rho=1.0)
    assert report["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert json.loads(out.with_name("dyn.report.json").read_text())["sha256"] == report["sha256"]
    before = out.read_bytes()
    # A crash before the report is written leaves no report vouching for the replaced labels,
    # and no partial file or fixed-name temp another writer could clobber.
    real = S._atomic_write

    def crash_on_report(p, text):
        if p.name.endswith(".report.json"):
            raise KeyboardInterrupt
        return real(p, text)
    monkeypatch.setattr(S, "_atomic_write", crash_on_report)
    with pytest.raises(KeyboardInterrupt):
        S.write_selection(MEDQA, "dynamic", path, out, rho=3.0)
    assert out.read_bytes() != before and not out.with_name("dyn.report.json").exists()  # no completion marker
    assert not list(out.parent.glob("*.part")) and not list(out.parent.glob(".*"))


def test_cli_select(records, tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert cli.main(["select", "--bench", "medqa", "--arm", "dynamic", "--records", str(path),
                     "--out", str(tmp_path / "dyn.jsonl")]) == 0
    rows = [json.loads(line) for line in open(tmp_path / "dyn.jsonl")]
    report = json.loads((tmp_path / "dyn.report.json").read_text())
    assert report["rho"] == MEDQA.rho == 3.0 and report["n_sft_turns"] == len(rows)
    assert report["seed"] == MEDQA.balance_seed == 0
    assert rows == S.select(records, "dynamic", 3.0, 0)[0]
    # --seed overrides the registry balance seed.
    assert cli.main(["select", "--bench", "medqa", "--arm", "dynamic", "--records", str(path), "--seed", "5",
                     "--out", str(tmp_path / "dyn5.jsonl")]) == 0
    assert json.loads((tmp_path / "dyn5.report.json").read_text())["seed"] == 5
    with pytest.raises(SystemExit):
        cli.main(["select", "--bench", "medqa", "--arm", "success", "--out", str(tmp_path / "s.jsonl")])


def test_write_selection_defaults_to_the_registry_balance_seed(records, tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    bench = dataclasses.replace(MEDQA, balance_seed=7, rho=1.0)
    report = S.write_selection(bench, "dynamic", path, tmp_path / "d.jsonl")
    assert report["seed"] == 7 and report["rho"] == 1.0
    assert [json.loads(line) for line in open(tmp_path / "d.jsonl")] == S.select(records, "dynamic", 1.0, 7)[0]
    assert S.select(records, "dynamic", 1.0, 7)[0] != S.select(records, "dynamic", 1.0, 0)[0]
    assert S.write_selection(bench, "dynamic", path, tmp_path / "o.jsonl", seed=0)["seed"] == 0


@pytest.mark.parametrize("bench", ["medqa", "mmlu_pro", "gpqa", "aqua"])
def test_paper_round1_labels_reproduced(bench, tmp_path):
    """The locked round-1 ratio files, row for row (modulo the added ``split``), from the records they came from.

    Both seeds are pinned: the records' labels are ``Random(42 + example_id)`` tie-breaks and the ratio file
    is ``_balance_records(rho, registry balance_seed)`` (0, but 42 for GPQA, whose file was balanced from the
    depth-1 ``label_records``); the wrong seed in either place is detected.
    """
    b = registry.get(bench)
    paths = importer.default_paths(b, IMPORT_DIR)
    if not (paths["labels"].exists() and paths["label_records"].exists()):
        pytest.skip("round-1 data not imported (set MCQ_RSI_IMPORT_DIR)")
    assert hashlib.sha256(paths["labels"].read_bytes()).hexdigest() == b.round1_labels.sha256
    assert (paths["label_records"] == paths["records"]) == (bench != "gpqa")
    records = [json.loads(line) for line in open(paths["label_records"])]
    paper = [json.loads(line) for line in open(paths["labels"])]
    strip = lambda rows: [{k: v for k, v in r.items() if k != "split"} for r in rows]
    # Tie-break seed: every stored label is the collector's choice with Random(TIE_BREAK_SEED + example_id) ...
    relabelled = lambda seed: [choose_preferred_sequence(bool(r["direct_correct"]), r["branches"], tie_break_seed=seed
                                                         + int(r["example_id"])) for r in records]
    stored = [tuple(r["preferred_sequence"]) if r["preferred_sequence"] is not None else None for r in records]
    assert relabelled(S.TIE_BREAK_SEED) == stored
    # ... and select.relabel re-derives it when a cut forces a re-choice: every record gets a failing branch one
    # step deeper than the tree, which max_depth drops, so no record takes the keep-the-stored-label path.
    deepest = max(len(br["sequence"]) for r in records for br in r["branches"])
    padded = [{**r, "branches": r["branches"] + [{"sequence": ["extractor"] * (deepest + 1), "correct": False}]}
              for r in records]
    recut = lambda **kw: [r["preferred_sequence"] for r in S.relabel(padded, max_depth=deepest, **kw)]
    assert recut() == [r["preferred_sequence"] for r in records]
    assert recut(tie_break_seed=0) != [r["preferred_sequence"] for r in records]  # ties exist; the seed matters
    # Balance seed: the registry seed reproduces the file exactly; the other paper seed does not.
    assert b.balance_seed == (42 if bench == "gpqa" else 0)
    rows, report = S.select(records, "dynamic", b.rho, b.balance_seed)
    assert strip(rows) == paper and report["n_sft_turns"] == len(paper)
    assert strip(S.select(records, "dynamic", b.rho, 42 - b.balance_seed)[0]) != paper
    # The CLI default (registry rho and balance_seed) writes the same rows.
    src = tmp_path / "records.jsonl"
    src.write_bytes(paths["label_records"].read_bytes())
    assert cli.main(["select", "--bench", bench, "--arm", "dynamic", "--records", str(src),
                     "--out", str(tmp_path / "labels.jsonl")]) == 0
    assert strip(json.loads(line) for line in open(tmp_path / "labels.jsonl")) == paper
    assert json.loads((tmp_path / "labels.report.json").read_text())["seed"] == b.balance_seed
