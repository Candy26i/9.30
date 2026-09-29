import copy
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.verifiable.backend import ContextBudgetExceeded, HTTPAdvisors
from src.verifiable.protocol import KINDS
from src.verifiable.runner import checkpoint_identity, verify_advisor
from src.verifiable.serve import (EXPERT_ADAPTERS, FrozenExpertBackend, expert_bundle_sha256,
                                  generate_advisor_request, load_expert_bundle, template_sha256)


def write_bundle(root, base="base-model", revision="pinned"):
    roles = {}
    for role in KINDS:
        directory = root / role
        directory.mkdir()
        (directory / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": base, "revision": revision}))
        (directory / "adapter_model.safetensors").write_bytes(role.encode())
        roles[role] = {"checkpoint": role, "identity": checkpoint_identity(str(directory))}
    payload = {"schema_version": 1, "frozen": True, "base_model": base,
               "base_model_revision": revision, "template_sha256": template_sha256(), "roles": roles}
    path = root / "expert_bundle.json"
    path.write_text(json.dumps(payload))
    return path, payload


def test_manifest_resolves_all_roles_and_rejects_missing_duplicate_and_tampered(tmp_path):
    path, payload = write_bundle(tmp_path)
    result = load_expert_bundle(path, "base-model", "pinned")
    assert set(result["roles"]) == set(KINDS)
    assert all(r["checkpoint"].startswith(str(tmp_path)) for r in result["roles"].values())
    for mutate, error in [
        (lambda x: x["roles"].pop("verifier"), "exactly"),
        (lambda x: x["roles"]["reasoner"].update(checkpoint="extractor"), "distinct"),
        (lambda x: x["roles"]["reasoner"].update(identity={}), "fingerprint"),
        (lambda x: x.update(template_sha256="wrong"), "template"),
        (lambda x: x.update(frozen=False), "frozen"),
        (lambda x: x.update(base_model="different"), "base_model"),
        (lambda x: x.update(base_model_revision="different"), "revision"),
    ]:
        changed = copy.deepcopy(payload)
        mutate(changed)
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=error):
            load_expert_bundle(path, "base-model", "pinned")
    path.write_text(json.dumps(payload))
    (tmp_path / "reasoner/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load_expert_bundle(path)


def test_manifest_rejects_adapter_base_mismatch_even_with_updated_hash(tmp_path):
    path, payload = write_bundle(tmp_path)
    (tmp_path / "verifier/adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": "wrong-base"}))
    payload["roles"]["verifier"]["identity"] = checkpoint_identity(str(tmp_path / "verifier"))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="base/model"):
        load_expert_bundle(path)


def test_manifest_rejects_wrong_trained_role_even_with_matching_file_identity(tmp_path):
    path, payload = write_bundle(tmp_path)
    summary = {"training_complete": True, "role": "extractor", "base_model": "base-model",
               "base_model_revision": "pinned", "template_sha256": template_sha256()}
    (tmp_path / "verifier/summary.json").write_text(json.dumps(summary))
    payload["roles"]["verifier"]["identity"] = checkpoint_identity(str(tmp_path / "verifier"))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="summary role"):
        load_expert_bundle(path)


def fake_experts(bundle, blocked=None, release=None):
    class Model:
        active_adapters = ["default"]
        def load_adapter(self, path, adapter_name, is_trainable):
            assert not is_trainable
        def requires_grad_(self, value):
            self.grad = value
        def eval(self):
            self.training = False
        def set_adapter(self, name):
            self.active_adapters = [name]
            self.grad = True  # PEFT can do this when switching.
            self.training = True
    tokenizer = SimpleNamespace(eos_token="<|im_end|>", eos_token_id=1,
                                get_vocab=lambda: {"<|im_start|>": 0, "<|im_end|>": 1})
    model = Model()
    def generate(messages, **kwargs):
        selected = model.active_adapters[:]
        assert not model.grad and not model.training
        if blocked is not None and selected == ["default"]:
            blocked.set()
            assert release.wait(3)
        assert model.active_adapters == selected
        if messages == ["over-budget"]:
            raise ContextBudgetExceeded(100, 10, 50)
        return {"text": selected[0], "prompt_tokens": 2, "completion_tokens": 1, "truncated": False}
    base = SimpleNamespace(model=model, tokenizer=tokenizer, generate=generate)
    with patch("src.verifiable.serve.HFBackend", return_value=base) as constructor, \
         patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
        experts = FrozenExpertBackend(bundle)
    assert constructor.call_count == 1
    return experts


def request(role, messages=None):
    return {"model": role, "messages": messages or [], "max_tokens": 10}


def test_alternating_roles_and_budget_failures_return_actual_frozen_adapter(tmp_path):
    path, _ = write_bundle(tmp_path)
    experts = fake_experts(load_expert_bundle(path))
    for role in ("reasoner", "extractor", "verifier", "reasoner"):
        result, _ = generate_advisor_request(experts, request(role))
        assert result["text"] == EXPERT_ADAPTERS[role]
        assert result["margent_role"] == {"role": role, "adapter_name": EXPERT_ADAPTERS[role],
                                         "identity": experts.bundle["roles"][role]["identity"]}
    result, _ = generate_advisor_request(experts, request("verifier", ["over-budget"]))
    assert result["error"] == "context_budget_exceeded"
    assert result["margent_role"]["role"] == "verifier"


def test_role_switch_and_generation_are_serialized(tmp_path):
    path, _ = write_bundle(tmp_path)
    entered, release = threading.Event(), threading.Event()
    experts = fake_experts(load_expert_bundle(path), entered, release)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(generate_advisor_request, experts, request("extractor"))
        assert entered.wait(3)
        second = pool.submit(generate_advisor_request, experts, request("verifier"))
        assert experts.model.active_adapters == ["default"]
        release.set()
        assert first.result()[0]["text"] == "default"
        assert second.result()[0]["text"] == "verifier"


def response_for(bundle, role="reasoner"):
    fingerprint = {"model": bundle["base_model"], "expert_bundle": bundle,
                   "expert_bundle_sha256": expert_bundle_sha256(bundle)}
    return {"margent_advisor": fingerprint, "actual_role": role,
            "margent_role": {"role": role, "adapter_name": EXPERT_ADAPTERS[role],
                             "identity": bundle["roles"][role]["identity"]},
            "choices": [{"message": {"content": "hint"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1}}


def test_client_rejects_wrong_missing_or_tampered_role_and_restarted_bundle(tmp_path):
    path, _ = write_bundle(tmp_path)
    data = response_for(load_expert_bundle(path))
    row = SimpleNamespace(question="q", context="")
    for mutate in [lambda d: d.pop("margent_role"), lambda d: d.update(actual_role="verifier"),
                   lambda d: d["margent_role"].update(adapter_name="default"),
                   lambda d: d["margent_role"].update(identity={}),
                   lambda d: d["margent_advisor"].update(expert_bundle_sha256="wrong")]:
        bad = copy.deepcopy(data)
        mutate(bad)
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: bad)
        with patch("requests.post", return_value=response), pytest.raises(RuntimeError, match="expert"):
            HTTPAdvisors("http://fake").call("reasoner", row)
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    client = HTTPAdvisors("http://fake")
    with patch("requests.post", return_value=response):
        assert client.call("reasoner", row)["actual_role"] == "reasoner"
        data["margent_advisor"] = {**data["margent_advisor"], "restart": "changed"}
        with pytest.raises(RuntimeError, match="identity changed"):
            client.call("reasoner", SimpleNamespace(question="different q", context=""))


def test_verify_health_requires_the_entire_expected_expert_bundle(tmp_path):
    path, _ = write_bundle(tmp_path)
    bundle = load_expert_bundle(path)
    value = response_for(bundle)
    value["margent_advisor"]["requested_revision"] = "pinned"
    cfg = {"advisor_url": "http://fake", "advisor_base_model": "base-model",
           "advisor_revision": "pinned", "advisor_expert_bundle": str(path)}
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: value)
    with patch("requests.get", return_value=response):
        assert verify_advisor(cfg, tmp_path / "run")["expert_bundle"] == bundle
        value["margent_advisor"].pop("expert_bundle")
        with pytest.raises(ValueError, match="expert bundle"):
            verify_advisor(cfg, tmp_path / "run")


def test_real_cpu_one_base_three_loras_remain_frozen(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file
    from test_rsi import tiny_checkpoint
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    base, checkpoint = tiny_checkpoint(tmp_path)
    roles = {}
    for i, role in enumerate(KINDS):
        directory = tmp_path / role
        shutil.copytree(checkpoint, directory)
        weights = load_file(str(directory / "adapter_model.safetensors"))
        for name in weights:
            if "lora_B" in name:
                weights[name].fill_(0.01 * (i + 1))
        save_file(weights, str(directory / "adapter_model.safetensors"))
        roles[role] = {"checkpoint": role, "identity": checkpoint_identity(str(directory))}
    path = tmp_path / "expert_bundle.json"
    path.write_text(json.dumps({"schema_version": 1, "frozen": True, "base_model": str(base),
        "base_model_revision": None, "template_sha256": template_sha256(), "roles": roles}))
    experts = FrozenExpertBackend(load_expert_bundle(path), max_context=4096)
    assert set(experts.model.peft_config) == set(EXPERT_ADAPTERS.values())
    shared_model = id(experts.model.base_model.model)
    for role in ("reasoner", "verifier", "extractor", "reasoner"):
        result, _ = generate_advisor_request(experts, {"model": role, "max_tokens": 1,
            "messages": [{"role": "user", "content": "one two"}]})
        assert result["margent_role"]["role"] == role
        assert experts.model.active_adapters == [EXPERT_ADAPTERS[role]]
        assert not experts.model.training and not any(p.requires_grad for p in experts.model.parameters())
        assert id(experts.model.base_model.model) == shared_model
        assert all(p.device == torch.device("cpu") for p in experts.model.parameters())


def test_checkpoint_and_expert_bundle_cli_flags_are_mutually_exclusive(monkeypatch):
    from src.verifiable.serve import main
    monkeypatch.setattr("sys.argv", ["serve", "--model", "base", "--checkpoint", "adapter",
                                    "--expert-bundle", "bundle.json"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
