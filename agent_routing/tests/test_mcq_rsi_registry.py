import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from src.manager.marginal_value import ADVISOR_KINDS
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi.benchmarks import BENCHMARKS, MEDQA, GPQA

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mcq_rsi"
IMPORT_DIR = Path(os.environ.get("MCQ_RSI_IMPORT_DIR", ROOT / "outputs/mcq_rsi/import"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_registry_is_internally_consistent():
    registry.validate()
    assert list(BENCHMARKS) == ["medqa", "mmlu_pro", "gpqa", "aqua"]
    assert {b: BENCHMARKS[b].rho for b in BENCHMARKS} == {"medqa": 3.0, "mmlu_pro": 2.0, "gpqa": 2.0, "aqua": 1.0}
    assert {b.depth for b in BENCHMARKS.values()} == {2}
    # The _balance_records seed and records each round-1 ratio file reproduces from (GPQA: depth-1, seed 42).
    assert {b: (BENCHMARKS[b].balance_seed, BENCHMARKS[b].label_records) for b in BENCHMARKS} == {
        "medqa": (0, "round1_records"), "mmlu_pro": (0, "round1_records"), "gpqa": (42, "label_records"),
        "aqua": (0, "round1_records")}
    with pytest.raises(ValueError, match="label_records"):
        registry.validate(dataclasses.replace(MEDQA, label_records="label_records"))
    assert [len(b.choice_keys) for b in BENCHMARKS.values()] == [4, 10, 4, 5]


def test_lora_names_and_advisor_layout():
    names = [b.lora_name(k) for b in BENCHMARKS.values() for k in ADVISOR_KINDS]
    assert len(set(names)) == 12 and "medqa_verifier" in names and "mmlu_pro_extractor" in names
    for b in BENCHMARKS.values():
        for kind in ADVISOR_KINDS:
            adapter = b.advisor(kind)
            assert adapter.repo_id == f"MaliDDD/agent-routing-advisors-{b.name}-9b"
            assert adapter.subfolder == f"{kind}_adapter"
            assert all("checkpoint-" not in f.path for f in adapter.hf_files())
    # The MedQA Verifier top level (checkpoint-600) ships no tokenizer/template files.
    assert [n for n, _, _ in MEDQA.advisor("verifier").files] == list(registry.ADAPTER_FILES)
    with pytest.raises(ValueError):
        MEDQA.lora_name("planner")


def test_task_descriptions_reproduce_paper_system_prompts():
    for b in BENCHMARKS.values():
        prompt = b.manager_system_prompt(b.choice_keys)
        assert prompt.startswith(b.task_description + "\n\nYou have THREE")
        assert hashlib.sha256(prompt.encode()).hexdigest() == b.manager_system_sha256


def test_manager_prompt_uses_row_choice_keys():
    # Every distinct round-1 system prompt (records and labels), keyed by the row's option count.
    observed = json.loads((FIXTURES / "manager_system_prompts.json").read_text())
    assert {b: sorted(map(int, v)) for b, v in observed.items()} == {
        "medqa": [4], "mmlu_pro": [4, 6, 7, 8, 9, 10], "gpqa": [4], "aqua": [5]}
    for name, by_n in observed.items():
        b = BENCHMARKS[name]
        for n, sha in by_n.items():
            prompt = b.manager_system_prompt(b.choice_keys[:int(n)])
            assert hashlib.sha256(prompt.encode()).hexdigest() == sha, (name, n)
    mmlu = BENCHMARKS["mmlu_pro"]
    with pytest.raises(TypeError):
        mmlu.manager_system_prompt()
    for keys in (["A", "C"], ["A"], list("ABCDEFGHIJK"), ["B", "C"]):
        with pytest.raises(ValueError):
            mmlu.manager_system_prompt(keys)


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_round1_system_prompts_rebuild_from_row_keys(name):
    b = BENCHMARKS[name]
    base = IMPORT_DIR / name / "round1"
    if not (base / "records.jsonl").exists() or not (ROOT / b.cache).exists():
        pytest.skip(f"round-1 data or {b.cache} not present (set MCQ_RSI_IMPORT_DIR)")
    rows = {r["example_id"]: r for r in map(json.loads, (ROOT / b.cache).read_text(encoding="utf-8").splitlines())}
    n = 0
    for fname, key in (("records.jsonl", "base_messages"), ("labels.jsonl", "prompt")):
        for line in (base / fname).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            assert rec[key][0]["content"] == b.manager_system_prompt(rows[rec["example_id"]]["choices"]), rec["example_id"]
            n += 1
    assert n > 0


@pytest.mark.parametrize("change", [
    {"rho": -1.0},
    {"depth": 3},
    {"task_description": "You are a manager agent solving a multiple-choice question."},
    {"split_manifest": "data/elsewhere.json"},
    {"cache_sha256": "abc"},
])
def test_validate_rejects_inconsistent_entries(change):
    with pytest.raises(ValueError):
        registry.validate(dataclasses.replace(MEDQA, **change))


def test_validate_rejects_unpinned_or_checkpoint_adapters():
    bad = dataclasses.replace(MEDQA.round1_adapter, subfolder="managers/x/checkpoint-400")
    with pytest.raises(ValueError):
        registry.validate(dataclasses.replace(MEDQA, round1_adapter=bad))
    bad = dataclasses.replace(GPQA.round1_adapter, revision="main")
    with pytest.raises(ValueError):
        registry.validate(dataclasses.replace(GPQA, round1_adapter=bad))
    bad = dataclasses.replace(MEDQA.round1_labels, sha256="")
    with pytest.raises(ValueError):
        registry.validate(dataclasses.replace(MEDQA, round1_labels=bad))


def test_round1_artifacts_follow_design():
    assert MEDQA.round1_adapter.subfolder == "managers/medqa_marginal_9b_d2_400/sft_3.0"
    assert BENCHMARKS["mmlu_pro"].round1_adapter.subfolder == "managers/mmlu_marginal_9b_d2/sft_2.0"
    assert GPQA.round1_adapter.repo_id == "MaliDDD/gpqa_marginal_9b-sft_2.0"
    assert BENCHMARKS["aqua"].round1_adapter.subfolder == "managers/aqua_marginal_9b_v1/sft_1.0"
    assert [b.round1_labels.rows for b in (MEDQA, BENCHMARKS["mmlu_pro"], BENCHMARKS["aqua"])] == [315, 331, 402]
    assert all(b.round1_records.uri.startswith("hf://datasets/MaliDDD/") for b in BENCHMARKS.values())
    aqua = BENCHMARKS["aqua"]
    alt = dict(aqua.extra_sources)["round1_labels_alt"]
    assert alt.archive == aqua.round1_labels.archive and alt.sha256 != aqua.round1_labels.sha256
    assert not any("avg_tool_calls" in k for k, _ in aqua.paper_targets)
    assert [s for s, _ in aqua.cache_sources] == ["train", "validation", "test"]
    assert {f.revision for _, f in aqua.cache_sources} == {"33301c6a050c96af81f63cad5562cb5363e88971"}


@pytest.mark.parametrize("name", ["medqa", "mmlu_pro", "gpqa"])
def test_tracked_caches_match_registry(name):
    b = BENCHMARKS[name]
    assert _sha256(ROOT / b.cache) == b.cache_sha256
    for _, rel, digest in b.aux_caches:
        assert _sha256(ROOT / rel) == digest


def test_aqua_cache_matches_registry_when_built():
    b = BENCHMARKS["aqua"]
    path = ROOT / b.cache
    if not path.exists():
        pytest.skip("AQuA cache not built; `python -m src.manager.mcq_rsi prepare-splits --bench aqua` builds it")
    assert _sha256(path) == b.cache_sha256
