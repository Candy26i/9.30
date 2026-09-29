"""No downloaded models: real tiny CPU expert SFT and interruption recovery."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.verifiable import expert_train as training
from src.verifiable.data import identity
from src.verifiable.protocol import KINDS, advisor_messages


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))


def write_manifest(path, manifest):
    payload = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    content_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    write_json(path, {**payload, "manifest_content_sha256": content_hash})


def make_data(root):
    template = training.digest(Path(training.__file__).with_name("chat_template.jinja"))
    protocol = training.digest(Path(training.__file__).with_name("protocol.py"))
    hashes = {}
    for role in KINDS:
        for split in ("train", "dev"):
            rows = []
            for index in range(4 if split == "train" else 2):
                question = f"{split} question {index}: one plus one"
                candidate = "one plus one is two" if role == "verifier" else ""
                rows.append({"role": role, "split": split, "source_id": f"{split}-{index}",
                    "question": question, "context": "", "candidate": candidate,
                    "question_hash": identity(question), "template_sha256": template,
                    "protocol_sha256": protocol, "reviewed": False,
                    "prompt": advisor_messages(role, SimpleNamespace(question=question, context=""), candidate),
                    "response": "two" if role != "verifier" else "Verdict: correct. Evidence: one plus one is two. Correction: None needed."})
            relative = f"{role}/{split}.jsonl"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            hashes[relative] = training.digest(path)
    write_manifest(root / "manifest.json", {"schema_version": 1, "template_sha256": template,
        "protocol_sha256": protocol, "sha256": hashes})
    return root


def rewrite_rows(data, relative, change):
    path = data / relative
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["sha256"][relative] = training.digest(path)
    write_manifest(data / "manifest.json", manifest)


def test_expert_bundle_verifies_files_and_serving_prompt(tmp_path):
    data = make_data(tmp_path / "data")
    _, train, dev = training.read_dataset(data, "reasoner")
    assert len(train) == 4 and len(dev) == 2
    path = data / "reasoner/train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="fingerprint"):
        training.read_dataset(data, "reasoner")
    make_data(data)
    rewrite_rows(data, "reasoner/train.jsonl", lambda rows: rows[0]["prompt"][1].update(content="secret answer key"))
    with pytest.raises(ValueError, match="serving protocol"):
        training.read_dataset(data, "reasoner")


def test_expert_bundle_rejects_cross_role_question_overlap(tmp_path):
    data = make_data(tmp_path / "data")
    def overlap(rows):
        row = rows[0]
        row["question"] = "train question 0: one plus one"
        row["question_hash"] = identity(row["question"])
        row["prompt"] = advisor_messages("verifier", SimpleNamespace(question=row["question"], context=""), row["candidate"])
    rewrite_rows(data, "verifier/dev.jsonl", overlap)
    with pytest.raises(ValueError, match="overlap"):
        training.read_dataset(data, "extractor")


def test_expert_config_rejects_unpinned_or_inherited_base_and_bad_budget(tmp_path):
    with pytest.raises(ValueError, match="pinned"):
        training.load_config({"base_model": "Qwen/Qwen3.5-9B", "base_model_revision": "main"})
    cfg = {"base_model": str(tmp_path), "max_steps": 0}
    with pytest.raises(ValueError, match="max_steps"):
        training.load_config(cfg)
    write_json(tmp_path / "adapter_config.json", {})
    with pytest.raises(ValueError, match="independently"):
        training.load_config({"base_model": str(tmp_path)})


def test_expert_masks_exact_inference_prefix_and_reports_length_filter(monkeypatch):
    class Tok:
        def __call__(self, text, **kwargs):
            return {"input_ids": {"P ": [1, 2], " T": [3, 4], "P  T": [1, 99, 4]}[text]}
    monkeypatch.setattr(training, "render", lambda tok, messages, generation=True: "P " if generation else "P  T")
    row = {"prompt": [], "response": "T", "source_id": "one", "question_hash": "a" * 64}
    features, report, excluded = training.tokenize_rows([row], Tok(), 4)
    assert features[0]["input_ids"] == [1, 2, 3, 4]
    assert features[0]["labels"] == [-100, -100, 3, 4]
    assert report["supervised_tokens_per_epoch"] == 2 and not excluded
    features, report, excluded = training.tokenize_rows([row], Tok(), 3)
    assert features == [] and report["overlength_rows"] == 1
    assert excluded[0]["input_tokens"] == 4


def tiny_base(root):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from src.verifiable.backend import configure_tokenizer
    torch.manual_seed(7)
    words = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "one", "two", "plus", "is",
             "train", "dev", "question", "0", "1", "2", "3", ":", "Verdict", "correct", "Evidence"]
    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>",
        eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    configure_tokenizer(tok).save_pretrained(root)
    Qwen3ForCausalLM(Qwen3Config(vocab_size=len(words), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=1024, pad_token_id=0, eos_token_id=3)).save_pretrained(root)
    return root


@pytest.fixture
def tiny_cpu(monkeypatch):
    import torch
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_real_expert_sft_resume_preserves_weights_steps_rng_and_server_reload(tmp_path, monkeypatch, tiny_cpu):
    import torch
    from safetensors.torch import load_file
    from src.verifiable.backend import load_model
    base = tiny_base(tmp_path / "base")
    data = make_data(tmp_path / "data")
    cfg = {"base_model": str(base), "max_steps": 3, "save_steps": 1, "max_seq_len": 1024,
        "gradient_accumulation_steps": 2, "lora_rank": 2, "lora_alpha": 4,
        "lora_dropout": .05, "learning_rate": 1e-3, "bf16": False}
    uninterrupted = tmp_path / "complete"
    result = training.train_expert(cfg, data, "extractor", uninterrupted)
    assert result["training_complete"] and result["optimizer_steps"] == 3 and result["eval_n"] == 2
    expected = load_file(str(uninterrupted / "adapter_model.safetensors"))
    assert any("lora_B" in name and value.abs().sum() > 0 for name, value in expected.items())
    assert result["observed_supervised_tokens"] > 0
    interrupted = tmp_path / "resume"
    original = training._commit_checkpoint
    def stop_after_saved_step(path, step, signature_hash):
        original(path, step, signature_hash)
        if step == 1:
            raise KeyboardInterrupt("simulated shutdown after a committed checkpoint")
    monkeypatch.setattr(training, "_commit_checkpoint", stop_after_saved_step)
    with pytest.raises(KeyboardInterrupt):
        training.train_expert(cfg, data, "extractor", interrupted)
    assert not json.loads((interrupted / "summary.json").read_text())["training_complete"]
    assert "simulated shutdown" in (interrupted / "errors.log").read_text()
    monkeypatch.setattr(training, "_commit_checkpoint", original)
    resumed = training.train_expert(cfg, data, "extractor", interrupted, resume=True)
    actual = load_file(str(interrupted / "adapter_model.safetensors"))
    assert resumed["optimizer_steps"] == 3 and resumed["training_complete"]
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    checkpoint = Path(resumed["final_checkpoint"])
    assert all((checkpoint / name).exists() for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "expert_checkpoint.json"))
    _, model = load_model(str(base), str(interrupted))
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.peft_config["default"].lora_dropout == .05
    assert training.train_expert(cfg, data, "extractor", interrupted, resume=True)["training_complete"]
    with pytest.raises(FileExistsError, match="resume"):
        training.train_expert(cfg, data, "extractor", interrupted)
    with pytest.raises(ValueError, match="changed"):
        training.train_expert({**cfg, "max_steps": 4}, data, "extractor", interrupted, resume=True)


def test_final_export_resume_never_exceeds_budget_and_roles_start_at_base(tmp_path, monkeypatch, tiny_cpu):
    import torch
    import peft
    from transformers import Trainer
    base = tiny_base(tmp_path / "base")
    data = make_data(tmp_path / "data")
    cfg = {"base_model": str(base), "max_steps": 1, "save_steps": 2, "max_seq_len": 1024,
        "gradient_accumulation_steps": 1, "lora_rank": 2, "lora_alpha": 4, "bf16": False}
    initial_weights = []
    attach_adapter = peft.get_peft_model
    def track_initial_base(model, *args, **kwargs):
        assert not hasattr(model, "peft_config")
        initial_weights.append(model.get_input_embeddings().weight.detach().clone())
        return attach_adapter(model, *args, **kwargs)
    monkeypatch.setattr(peft, "get_peft_model", track_initial_base)
    reasoner = tmp_path / "reasoner"
    training.train_expert(cfg, data, "reasoner", reasoner)
    verifier = tmp_path / "verifier"
    original_hash = training._adapter_hash
    def fail_export(output):
        if Path(output) == verifier:
            raise OSError("simulated final export interruption")
        return original_hash(output)
    monkeypatch.setattr(training, "_adapter_hash", fail_export)
    with pytest.raises(OSError, match="export interruption"):
        training.train_expert(cfg, data, "verifier", verifier)
    assert len(initial_weights) == 2 and torch.equal(initial_weights[0], initial_weights[1])
    assert (verifier / "checkpoint-1/expert_checkpoint.json").is_file()
    monkeypatch.setattr(training, "_adapter_hash", original_hash)
    monkeypatch.setattr(Trainer, "train", lambda *a, **kw: pytest.fail("Must not update after restored max_steps"))
    final = training.train_expert(cfg, data, "verifier", verifier, resume=True)
    assert final["training_complete"] and final["optimizer_steps"] == final["resumed_optimizer_steps"] == 1
    assert final["role"] == "verifier" and final["observed_supervised_tokens"] > 0
    # A changed tokenizer/adapter config invalidates a formerly complete run.
    path = verifier / "adapter_config.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="artifacts changed"):
        training.train_expert(cfg, data, "verifier", verifier, resume=True)


def test_checkpoint_recovery_ignores_torn_save_and_rejects_modified_state(tmp_path):
    good = tmp_path / "checkpoint-1"
    good.mkdir()
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
        (good / name).write_bytes(b"state")
    write_json(good / "trainer_state.json", {"global_step": 1})
    training._commit_checkpoint(good, 1, "manifest-hash")
    torn = tmp_path / "checkpoint-2"
    torn.mkdir()
    (torn / "optimizer.pt").write_bytes(b"incomplete")
    assert training._resume_checkpoint(tmp_path, "manifest-hash", 3) == str(good)
    (good / "optimizer.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or incompatible"):
        training._resume_checkpoint(tmp_path, "manifest-hash", 3)


def test_expert_manifest_metadata_change_is_rejected(tmp_path):
    data = make_data(tmp_path / "data")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["counts"] = {"extractor": {"train": 99999, "dev": 99999}}
    write_json(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest content fingerprint"):
        training.read_dataset(data, "extractor")
