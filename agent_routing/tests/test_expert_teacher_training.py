"""Teacher provenance checks use recorded fixtures only: no API or downloaded model."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.verifiable import expert_train as training
from src.verifiable.expert_isolation import verify_manager_rows
from src.verifiable.protocol import KINDS, advisor_messages
from test_expert_train import make_data, rewrite_rows, tiny_base, tiny_cpu, write_manifest


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def write_records(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def rebind_file(data, name):
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["sha256"][name] = training.digest(data / name)
    write_manifest(data / "manifest.json", manifest)


def make_teacher_data(data):
    make_data(data)
    manifest = json.loads((data / "manifest.json").read_text())
    config = {"teacher_provider": "openai", "teacher_model": "gpt-4o", "teacher_revision": None,
              "temperature": 0.2, "max_tokens": 128}
    config_hash, run_hash = canonical_hash(config), canonical_hash({"fixture": "teacher-data"})
    requests, responses = [], []

    def call(row, kind, text, candidate_id=None, candidate_hash=None):
        messages = [{"role": "system", "content": "Teacher synthesis instructions"},
                    {"role": "user", "content": row["question"]}]
        request = {"schema_version": 1, "task_id": f'{row["question_hash"]}:{kind}:0',
            "kind": kind, "variant": 0, "question_hash": row["question_hash"], "split": row["split"],
            "provider": "openai", "model": "gpt-4o", "teacher_revision": None,
            "config_sha256": config_hash, "run_sha256": run_hash, "messages": messages,
            "prompt_sha256": canonical_hash(messages), "temperature": .2, "max_tokens": 128,
            "candidate_request_id": candidate_id, "candidate_response_sha256": candidate_hash}
        request["request_id"] = canonical_hash(request)
        requests.append(request)
        response = {"text": text, "provider": "openai", "model": "gpt-4o", "actual_model": None,
            "usage": {"input_tokens": 12, "output_tokens": 4}, "finish_reason": "stop",
            "request_id": "provider-response-id", "system_fingerprint": None,
            "latency_seconds": .1, "request_attempts": 1, "provider_usage": {"total_tokens": 16}}
        response_hash = canonical_hash(response)
        responses.append({"schema_version": 1, "request_id": request["request_id"], "attempt": 1,
            "origin": "external_import", "response": response, "response_sha256": response_hash,
            "validation": {"accepted": True, "reasons": [], "verdict": "correct" if kind == "verifier" else None}})
        return request, response_hash

    for role in KINDS:
        for split in ("train", "dev"):
            relative = f"{role}/{split}.jsonl"
            rows = [json.loads(line) for line in (data / relative).read_text().splitlines()]
            for row in rows:
                candidate_id = candidate_hash = None
                if role == "verifier":
                    candidate_request, candidate_hash = call(row, "candidate", row["candidate"])
                    candidate_id = candidate_request["request_id"]
                    row.update(candidate_request_id=candidate_id, candidate_response_sha256=candidate_hash, verdict="correct")
                request, response_hash = call(row, role, row["response"], candidate_id, candidate_hash)
                row.update(label_source="teacher_synthetic", teacher_provider="openai", teacher_model="gpt-4o",
                    teacher_actual_model=None, teacher_revision=None, teacher_request_id=request["request_id"],
                    teacher_prompt_sha256=request["prompt_sha256"], teacher_response_sha256=response_hash,
                    quality_checks={"label_verification": "unverified"})
            write_records(data / relative, rows)
            manifest["sha256"][relative] = training.digest(data / relative)
    responses.insert(0, {"schema_version": 1, "request_id": requests[0]["request_id"], "attempt": 0,
        "origin": "api", "response": None, "response_sha256": None,
        "validation": {"accepted": False, "reasons": ["transient failure"], "verdict": None}})
    for name, records in (("teacher_requests.jsonl", requests), ("teacher_responses.jsonl", responses),
                          ("references.jsonl", []), ("exclusions.jsonl", [])):
        write_records(data / name, records)
        manifest["sha256"][name] = training.digest(data / name)
    manifest.update(supervision="teacher_synthetic", teacher={"provider": "openai", "model": "gpt-4o", "revision": None},
        synthesis={"config": config, "config_sha256": config_hash, "run_sha256": run_hash,
                   "accepted": len(requests), "rejected": 1, "budget": {"actual_requests": len(responses)}},
        quality_audit={"teacher_used": True, "reviewed": False})
    write_manifest(data / "manifest.json", manifest)
    return data


def test_teacher_rows_load_runtime_prompts_and_remain_isolated(tmp_path):
    data = make_teacher_data(tmp_path / "data")
    for role in KINDS:
        manifest, train, dev = training.read_dataset(data, role)
        assert len(train) == 4 and len(dev) == 2
        assert manifest["teacher"]["revision"] is None
        assert train[0]["prompt"][0]["content"] != "Teacher synthesis instructions"
    cfg = {"expert_data_manifest": str(data / "manifest.json"),
           "expert_data_manifest_sha256": training.digest(data / "manifest.json")}
    with pytest.raises(ValueError, match="overlaps"):
        verify_manager_rows(cfg, [{"question_hash": dev[0]["question_hash"]}])
    assert verify_manager_rows(cfg, [{"question": "An entirely unrelated Manager question"}])["checked"]


def test_official_teacher_requirement_cannot_be_bypassed_by_direct_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "load_model", lambda *args, **kwargs: pytest.fail("Must reject data before loading model"))
    cfg = {"base_model": str(tmp_path), "expert_data_mode": "teacher_synthetic",
           "expected_teacher": {"provider": "openai", "model": "gpt-4o"}}
    weak = make_data(tmp_path / "weak")
    with pytest.raises(ValueError, match="requires teacher-synthesized"):
        training.train_expert(cfg, weak, "extractor", tmp_path / "output")
    assert not (tmp_path / "output").exists()
    teacher = make_teacher_data(tmp_path / "teacher")
    with pytest.raises(ValueError, match="expected_teacher"):
        training.train_expert({**cfg, "expected_teacher": {"provider": "openai", "model": "wrong"}},
                             teacher, "extractor", tmp_path / "output")
    with pytest.raises(ValueError, match="weak_debug"):
        training.train_expert({"base_model": str(tmp_path), "expert_data_mode": "weak_debug"},
                             teacher, "extractor", tmp_path / "output")


@pytest.mark.parametrize("field,value", [("response", "forged target"), ("teacher_model", "different-model"),
    ("teacher_actual_model", "unrecorded-snapshot"), ("teacher_prompt_sha256", "a" * 64),
    ("label_source", "numina_reference_weak")])
def test_teacher_rows_cannot_diverge_from_accepted_evidence(tmp_path, field, value):
    data = make_teacher_data(tmp_path / "data")
    rewrite_rows(data, "reasoner/train.jsonl", lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError, match="accepted request/response evidence"):
        training.read_dataset(data, "reasoner")


def test_teacher_verifier_candidate_requires_accepted_source(tmp_path):
    data = make_teacher_data(tmp_path / "data")
    def change(rows):
        rows[0]["candidate"] = "one plus one is three"
        rows[0]["prompt"] = advisor_messages("verifier", SimpleNamespace(question=rows[0]["question"], context=""),
                                             rows[0]["candidate"])
    rewrite_rows(data, "verifier/train.jsonl", change)
    with pytest.raises(ValueError, match="verifier candidate"):
        training.read_dataset(data, "verifier")


def test_teacher_verdict_cannot_be_relabelled_after_generation(tmp_path):
    data = make_teacher_data(tmp_path / "data")
    rewrite_rows(data, "verifier/dev.jsonl", lambda rows: rows[0].update(verdict="incorrect"))
    with pytest.raises(ValueError, match="verdict differs"):
        training.read_dataset(data, "verifier")


@pytest.mark.parametrize("location", ["row", "manifest"])
def test_teacher_cannot_claim_human_review_without_review_evidence(tmp_path, location):
    data = make_teacher_data(tmp_path / "data")
    if location == "row":
        rewrite_rows(data, "extractor/train.jsonl", lambda rows: rows[0].update(reviewed=True))
    else:
        manifest = json.loads((data / "manifest.json").read_text())
        manifest["quality_audit"].update(reviewed=True, reviewed_examples=1)
        write_manifest(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="review evidence"):
        training.read_dataset(data, "extractor")


@pytest.mark.parametrize("filename,field", [("teacher_requests.jsonl", "temperature"),
                                           ("teacher_responses.jsonl", "response_sha256")])
def test_teacher_evidence_rehashing_file_does_not_hide_changed_call(tmp_path, filename, field):
    data = make_teacher_data(tmp_path / "data")
    records = [json.loads(line) for line in (data / filename).read_text().splitlines()]
    records[-1][field] = .9 if field == "temperature" else "a" * 64
    write_records(data / filename, records)
    rebind_file(data, filename)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        training.read_dataset(data, "extractor")


def test_rejected_response_cannot_supply_a_target(tmp_path):
    data = make_teacher_data(tmp_path / "data")
    filename = "teacher_responses.jsonl"
    records = [json.loads(line) for line in (data / filename).read_text().splitlines()]
    records[1]["validation"]["accepted"] = False
    write_records(data / filename, records)
    rebind_file(data, filename)
    with pytest.raises(ValueError, match="accepted request/response evidence"):
        training.read_dataset(data, "extractor")


def test_all_manifest_sidecars_are_verified(tmp_path):
    data = make_teacher_data(tmp_path / "data")
    (data / "additional_audit.json").write_text('{"requests": 1}')
    rebind_file(data, "additional_audit.json")
    training.read_dataset(data, "extractor")
    (data / "additional_audit.json").write_text('{"requests": 2}')
    with pytest.raises(ValueError, match="additional_audit.json"):
        training.read_dataset(data, "extractor")


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json", "audit/../outside.json", "audit\\outside.json", "./audit.json"])
def test_manifest_paths_cannot_escape_or_alias(tmp_path, path):
    data = make_data(tmp_path / "data")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["sha256"][path] = "a" * 64
    write_manifest(data / "manifest.json", manifest)
    with pytest.raises(ValueError, match="safe relative"):
        training.read_dataset(data, "extractor")


def test_manifest_symlink_cannot_read_outside_data(tmp_path):
    data = make_data(tmp_path / "data")
    outside = tmp_path / "external.json"
    outside.write_text("private")
    (data / "linked.json").symlink_to(outside)
    rebind_file(data, "linked.json")
    with pytest.raises(ValueError, match="escapes"):
        training.read_dataset(data, "extractor")


def test_review_status_is_independent_of_label_origin(monkeypatch):
    monkeypatch.setattr(training, "render", lambda tokenizer, messages, generation=True: "P" if generation else "PT")
    class Tok:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1] * len(text)}
    base = {"prompt": [], "response": "T", "source_id": "one", "question_hash": "a" * 64}
    rows = [{**base, "label_source": "teacher_synthetic", "reviewed": False},
            {**base, "label_source": "teacher_synthetic", "reviewed": True},
            {**base, "label_source": "numina_reference_weak", "reviewed": False}, base]
    _, report, _ = training.tokenize_rows(rows, Tok(), 4)
    assert report["teacher_generated_rows"] == 2
    assert report["weak_supervision_rows"] == 1 and report["other_supervision_rows"] == 1
    assert report["reviewed_rows"] == 1 and report["unreviewed_rows"] == 3
    assert report["label_source_counts"] == {"teacher_synthetic": 2, "numina_reference_weak": 1, "unspecified": 1}


def test_teacher_sft_real_cpu_save_resume_retains_provenance(tmp_path, monkeypatch, tiny_cpu):
    base = tiny_base(tmp_path / "base")
    data = make_teacher_data(tmp_path / "data")
    cfg = {"base_model": str(base), "max_steps": 2, "save_steps": 1, "max_seq_len": 1024,
           "gradient_accumulation_steps": 1, "lora_rank": 2, "lora_alpha": 4, "bf16": False,
           "expert_data_mode": "teacher_synthetic", "expected_teacher": {"provider": "openai", "model": "gpt-4o"}}
    output = tmp_path / "trained"
    original = training._commit_checkpoint
    def interrupt(path, step, signature_hash):
        original(path, step, signature_hash)
        if step == 1:
            raise KeyboardInterrupt("interrupted teacher SFT")
    monkeypatch.setattr(training, "_commit_checkpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.train_expert(cfg, data, "verifier", output)
    monkeypatch.setattr(training, "_commit_checkpoint", original)
    summary = training.train_expert(cfg, data, "verifier", output, resume=True)
    assert summary["training_complete"] and summary["resumed_optimizer_steps"] == 1
    assert summary["optimizer_steps"] == 2 and summary["supervision"] == "teacher_synthetic"
    assert summary["teacher"] == {"provider": "openai", "model": "gpt-4o", "revision": None}
    assert summary["teacher_actual_models"] == [] and summary["teacher_actual_model_unknown_rows"] == 6
    assert summary["synthesis"]["rejected"] == 1
    assert summary["data_report"]["train"]["teacher_generated_rows"] == 4
    assert summary["data_report"]["train"]["weak_supervision_rows"] == 0
    assert training.train_expert(cfg, data, "verifier", output, resume=True) == summary
