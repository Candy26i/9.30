"""Round-k Manager SFT wrapper and the SFT tokenisation parity check (design §3.4, §7.1 item 7)."""
import copy
import json
import os
from pathlib import Path

import pytest

from mcq_rsi_helpers import FakeManager, advisor_server, make_rows, tiny_model, tiny_tokenizer
from src.manager.marginal_value import _make_sft_rows
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import collect as C
from src.manager.mcq_rsi import sft as S
from src.manager.mcq_rsi.advisors import CachedAdvisorPool

MEDQA = registry.get("medqa")
BENCHES = ("medqa", "mmlu_pro", "gpqa", "aqua")
ROUND1_ROWS = {"medqa": 315, "mmlu_pro": 331, "gpqa": 148, "aqua": 402}


@pytest.fixture(autouse=True)
def _threads():
    import torch
    torch.set_num_threads(1)


def label_rows(tmp_path):
    rows = make_rows(4)
    e = [r["example_id"] for r in rows]
    script = {e[0]: {"root": ("A", "commit")}, e[1]: {"root": ("A", "verifier"), "rev": {("extractor",): "B"}},
              e[2]: {"root": ("A", "commit"), "rev": {("reasoner", "verifier"): "C"}},
              e[3]: {"root": ("A", "commit"), "rev": {("verifier",): "D"}}}
    http, adapters = advisor_server(tmp_path)
    pool = CachedAdvisorPool("medqa", Path(tmp_path) / "cache", "http://x", adapters=adapters, http=http,
                             sleep=lambda s: None)
    out = []
    for r in rows:
        out += [{**x, "split": "train"} for x in _make_sft_rows(C.collect_question(r, MEDQA, FakeManager(script), pool))]
    assert {r["decision_type"] for r in out} == set(S.DECISION_TYPES)
    return out


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return Path(path)


def test_label_validation(tmp_path):
    rows = label_rows(tmp_path)
    report = S.validate_labels(write(tmp_path / "labels.jsonl", rows), MEDQA)
    assert report["rows"] == len(rows) and len(report["sha256"]) == 64 and report["questions"] == 4

    def broken(fn, match):
        bad = copy.deepcopy(rows)
        fn(bad)
        with pytest.raises(ValueError, match=match):
            S.validate_labels(bad, MEDQA)

    call = next(i for i, r in enumerate(rows) if r["decision_type"] == "call")
    verifier = next(i for i, r in enumerate(rows) if r["decision_type"] == "call"
                    and r["response"][0]["tool_calls"][0]["function"]["name"] == "verifier_tool")
    commit = next(i for i, r in enumerate(rows) if r["decision_type"] == "commit")
    after = next(i for i, r in enumerate(rows) if r["decision_type"] == "commit_after_call")
    broken(lambda b: b[0].update(split="dev"), "not train")
    broken(lambda b: b[0].update(question_hash=""), "question_hash")
    broken(lambda b: b[0].update(decision_type="route"), "decision_type")
    broken(lambda b: b[commit]["response"][0].update(content="DRAFT_ANSWER_A\nANSWER_B"), "ANSWER_K")
    broken(lambda b: b[commit]["response"].append({"role": "assistant", "content": "x"}), "exactly one")
    broken(lambda b: b[call]["response"][0]["tool_calls"][0]["function"].update(arguments='{"example_id": 1}'),
           "environment binding")
    broken(lambda b: b[verifier]["response"][0]["tool_calls"][0]["function"].update(arguments='{"current_draft": "Z"}'),
           "environment binding")
    broken(lambda b: b[call]["response"][0]["tool_calls"].append(b[call]["response"][0]["tool_calls"][0]), "2 tool calls")
    broken(lambda b: b[after].update(prompt=b[after]["prompt"][:2]), "commit")
    broken(lambda b: b[commit]["prompt"][0].update(content="You are a manager."), "manager prompt")
    with pytest.raises(ValueError, match="empty"):
        S.validate_labels([])
    S.validate_labels(rows)  # without a benchmark the system prompt is not checked


def test_tokenization_parity_tiny_tokenizer(tmp_path):
    tok, rows = tiny_tokenizer(), label_rows(tmp_path)
    paper = S.tokenization_parity(rows, tok, 4096, "paper")
    assert paper["supervised_mismatch"] == 0 and paper["input_mismatch"] == 0 and paper["first_difference"] is None
    evolve = S.tokenization_parity(rows, tok, 4096, "evolve")
    assert evolve["supervised_mismatch"] == 0 and evolve["input_mismatch"] == len(rows)
    assert "tool_calls" in evolve["first_difference"]["ours"] and "tool_calls" not in evolve["first_difference"]["paper"]
    assert S.check_tokenization_parity(rows, tok, context="evolve")["input_mismatch"] == len(rows)
    with pytest.raises(ValueError, match="differs"):
        S.check_tokenization_parity(rows, tok, context="evolve", require_inputs=True)


def _real_inputs():
    from mcq_rsi_helpers import real_tokenizer_dir
    imp, tok = os.environ.get("MCQ_RSI_IMPORT_DIR"), real_tokenizer_dir()
    if not imp or not os.environ.get("MCQ_RSI_TOKENIZER_DIR") or tok is None:
        return None
    files = {b: Path(imp) / b / "round1" / "labels.jsonl" for b in BENCHES}
    return (tok, files) if all(f.is_file() for f in files.values()) else None


@pytest.mark.skipif(_real_inputs() is None, reason="set MCQ_RSI_IMPORT_DIR and MCQ_RSI_TOKENIZER_DIR (round-1 labels)")
def test_tokenization_parity_on_round1_label_files():
    """Supervised ids of 9.30 SFT == paper-era masking on all four round-1 ratio files (Qwen3.5 tokenizer)."""
    from transformers import AutoTokenizer
    tok_dir, files = _real_inputs()
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    for bench, path in files.items():
        assert registry.get(bench).round1_labels.sha256 == S._sha256(path)
        labels = S.validate_labels(path, registry.get(bench))
        paper = S.check_tokenization_parity(path, tok, 4096, "paper")
        assert paper["rows"] == labels["rows"] == ROUND1_ROWS[bench]
        assert paper["supervised_mismatch"] == 0 and paper["input_mismatch"] == 0 and paper["max_tokens"] <= 4096
        evolve = S.tokenization_parity(path, tok, 4096, "evolve")
        # 9.30 as-is: same labels, but every rendered tool schema gains "tool_calls": null.
        assert evolve["supervised_mismatch"] == 0 and evolve["input_mismatch"] == labels["rows"]
        assert evolve["supervised_tokens"] == paper["supervised_tokens"]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    root = tmp_path_factory.mktemp("round_sft")
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    model.save_pretrained(root / "base")
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, init_lora_weights=False,
                                            target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"]))
    peft.save_pretrained(root / "g_prev")
    tok.save_pretrained(root / "g_prev")
    return root


def test_commit_rows_draft_supervision_masks_only_the_call_rows_drafts(tmp_path):
    """``sft.draft_supervision = "commit_rows"``: the SFT tokeniser gives call rows no loss on their (wrong)
    DRAFT_ANSWER_X and leaves commit / commit_after_call rows exactly as the paper's full supervision."""
    from src.manager import evolve
    from src.manager.routing_anchor import build_anchor_features
    tok, rows = tiny_tokenizer(), label_rows(tmp_path)
    tools = S.sft_tools("paper")
    full, _ = build_anchor_features(rows, tok, 4096, "full", tools)
    with S.sft_context("paper", "commit_rows"):
        ds = evolve._tokenize_manager_sft(rows, tok, 4096, tools=list(tools))
    with S.sft_context("paper"):
        same = evolve._tokenize_manager_sft(rows, tok, 4096, tools=list(tools))
    assert len(ds) == len(rows) == len(same)
    masked = 0
    for row, f_full, f_masked, f_same in zip(rows, full, ds, same):
        assert f_same["labels"] == f_full["labels"]  # "all" is the paper's tokenisation
        sup = tok.decode([t for t in f_masked["labels"] if t != -100])
        if row["decision_type"] == "call":
            assert "DRAFT_ANSWER_" not in sup and "_tool" in sup
            assert sum(t != -100 for t in f_masked["labels"]) < sum(t != -100 for t in f_full["labels"])
            masked += 1
        else:
            assert f_masked["labels"] == f_full["labels"] and "DRAFT_ANSWER_" in sup
    assert masked > 0
    assert S.RoundSFTConfig(draft_supervision="commit_rows").draft_supervision == "commit_rows"
    with pytest.raises(ValueError, match="draft_supervision"):
        S.RoundSFTConfig(draft_supervision="calls").validate()
    with pytest.raises(ValueError, match="draft_supervision"):
        with S.sft_context("paper", "nothing"):
            pass
    # "none": no row trains its draft; commit rows keep ANSWER_X, call rows keep the tool call.
    with S.sft_context("paper", "none"):
        ds3 = evolve._tokenize_manager_sft(rows, tok, 4096, tools=list(tools))
    assert len(ds3) == len(rows)
    for row, f, f_full in zip(rows, ds3, full):
        sup = tok.decode([t for t in f["labels"] if t != -100])
        assert "DRAFT_ANSWER_" not in sup and sum(t != -100 for t in f["labels"]) < sum(t != -100 for t in f_full["labels"])
        assert ("_tool" in sup) == (row["decision_type"] == "call") and ("ANSWER_" in sup) == (row["decision_type"] != "call")
    # Also under the evolve tool container (another prompt render, which "all" leaves unpatched): the same rule.
    with S.sft_context("evolve", "commit_rows"):
        ds2 = evolve._tokenize_manager_sft(rows, tok, 4096, tools=list(tools))
    assert len(ds2) == len(rows)
    for row, f in zip(rows, ds2):
        sup = tok.decode([t for t in f["labels"] if t != -100])
        assert ("DRAFT_ANSWER_" not in sup) == (row["decision_type"] == "call") and ("_tool" in sup) == (row["decision_type"] == "call")


def test_continuation_keeps_lora_config_and_loads(ckpt, tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file
    from src.manager import evolve, routing_anchor
    from src.manager.mcq_rsi import protocol
    labels = write(tmp_path / "labels.jsonl", label_rows(tmp_path))
    seen = []
    real = routing_anchor.build_anchor_features

    def spy(rows, tok, max_seq_len, mode, tools=None):
        seen.append((mode, type(tools).__name__, max_seq_len))
        return real(rows, tok, max_seq_len, mode, tools)

    monkeypatch.setattr(routing_anchor, "build_anchor_features", spy)
    original = evolve._tokenize_manager_sft
    cfg = {"max_steps": 2, "gradient_accumulation_steps": 2}
    report = S.train_round_sft(labels, ckpt / "g_prev", tmp_path / "sft", base_model=str(ckpt / "base"),
                               bench="medqa", config=cfg)
    assert evolve._tokenize_manager_sft is original  # the paper-context wrapper is removed afterwards
    assert ("full", "tuple", 4096) in seen  # train_manager_sft tokenised with the paper tool block
    model_dir = Path(report["model_dir"])
    assert model_dir == tmp_path / "sft" / "model"
    assert (report["lora"]["r"], report["lora"]["lora_alpha"]) == (2, 4)
    assert report["lora"]["target_modules"] == sorted(["q_proj", "v_proj", "gate_proj", "down_proj"])
    assert report["tokenization_parity"]["input_mismatch"] == 0 and report["labels"]["rows"] > 0
    assert report["training_metrics"]["optimizer_steps"] == 2
    assert report["config"]["num_train_epochs"] == 3 and report["config"]["learning_rate"] == 1e-5
    assert report["config"]["draft_supervision"] == "all"  # the default: the paper's tokenisation
    before, after = load_file(str(ckpt / "g_prev" / "adapter_model.safetensors")), load_file(str(model_dir / "adapter_model.safetensors"))
    assert set(before) == set(after) and any(not torch.equal(before[k], after[k]) for k in before)
    # The continued adapter loads through the MCQ manager loader (the eval/collection path).
    backend = protocol.load_hf_manager(str(model_dir), str(ckpt / "base"), device="cpu")
    assert backend.identity["adapter_sha256"] == report["adapter_sha256"]
    # The signature pins the code that shapes SFT (this wrapper, train_manager_sft, the renderer) and the base.
    signature = json.loads((tmp_path / "sft" / "run_signature.json").read_text())
    assert signature["source_sha256"] == S.source_sha256()
    assert signature["base_identity"]["path"] == str((ckpt / "base").resolve())
    assert signature["base_revision"] == registry.BASE_REVISION  # recorded (a local base is used as given)
    assert "config.json" in signature["base_identity"]["files"]
    # Complete: a rerun returns the report; changed inputs are refused.
    assert S.train_round_sft(labels, ckpt / "g_prev", tmp_path / "sft", base_model=str(ckpt / "base"),
                             bench="medqa", config=cfg) == report
    with pytest.raises(ValueError, match="inputs changed"):
        S.train_round_sft(labels, ckpt / "g_prev", tmp_path / "sft", base_model=str(ckpt / "base"),
                          bench="medqa", config={**cfg, "context": "evolve"})
    with pytest.raises(ValueError, match="LoRA adapter"):
        S.train_round_sft(labels, ckpt / "base", tmp_path / "other", base_model=str(ckpt / "base"))


def test_cli_sft_parity_only(tmp_path, capsys):
    from src.manager.mcq_rsi import __main__ as cli
    tiny_tokenizer().save_pretrained(tmp_path / "tok")
    labels = write(tmp_path / "labels.jsonl", label_rows(tmp_path))
    argv = ["sft", "--bench", "medqa", "--labels", str(labels), "--init", str(tmp_path / "tok"), "--parity-only"]
    assert cli.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["parity"]["input_mismatch"] == 0 and out["labels"]["rows"] > 0
    assert cli.main(argv + ["--context", "evolve"]) == 0  # supervised ids match; inputs differ (reported)
    with pytest.raises(SystemExit):
        cli.main(argv[:-1])  # training needs --out


class _Resolved(Exception):
    pass


def test_round_sft_pins_the_base_revision(tmp_path, monkeypatch):
    """A hub id is resolved to its BASE_REVISION snapshot before anything else (and '' / None stays unpinned)."""
    from src.manager.mcq_rsi import evaluate as E
    seen = []

    def resolve(model, revision):
        seen.append((model, revision))
        raise _Resolved

    monkeypatch.setattr(E, "resolve_base", resolve)
    with pytest.raises(_Resolved):
        S.train_round_sft(tmp_path / "labels.jsonl", tmp_path / "init", tmp_path / "out")
    with pytest.raises(_Resolved):
        S.train_round_sft(tmp_path / "labels.jsonl", tmp_path / "init", tmp_path / "out", base_revision="")
    assert seen == [(registry.BASE_MODEL, registry.BASE_REVISION), (registry.BASE_MODEL, None)]


def test_source_hashes_cover_the_code_that_shapes_inputs(monkeypatch):
    """Resume signatures of round SFT and FA-GRPO change with every file that shapes their inputs."""
    from src.manager.mcq_rsi import grpo as G
    files = G.source_files()
    names = {str(f.relative_to(G.PACKAGE.parents[1])) for f in files}
    for f in ("manager/mcq_rsi/protocol.py", "manager/mcq_rsi/prompts.py", "manager/mcq_rsi/select.py",
              "manager/mcq_rsi/collect.py", "manager/mcq_rsi/sft.py", "manager/mcq_rsi/grpo.py",
              "manager/mcq_rsi/benchmarks.py", "manager/routing_anchor.py", "manager/evolve.py",
              "manager/prompt.py", "pipeline/stages.py", "subagents/train.py"):
        assert f in names, f
    assert any(n.startswith("manager/mcq_rsi/prompts/") for n in names)  # advisor prompt templates
    base_sft, base_grpo = S.source_sha256(), G.source_sha256()
    real = Path.read_bytes
    for target in ("protocol.py", "routing_anchor.py", "evolve.py", "stages.py", "select.py"):
        monkeypatch.setattr(Path, "read_bytes", lambda self, t=target: real(self) + (b"#" if self.name == t else b""))
        assert S.source_sha256() != base_sft and G.source_sha256() != base_grpo, target
        monkeypatch.setattr(Path, "read_bytes", real)
    assert S.source_sha256() == base_sft
