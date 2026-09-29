"""Independent, resumable math-role LoRA SFT using the serving chat protocol.

Each role starts from the pinned base. Dataset construction and role-quality
assessment are separate stages: development cross-entropy is not a benchmark
accuracy or evidence that an expert helps the Manager.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .backend import load_model, render
from .protocol import KINDS, advisor_messages
from .telemetry import Monitor, atomic_json, metrics, progress, training_callback, usage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_files(root, hashes):
    """Authenticate every published file, including optional provenance evidence."""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Expert manifest requires a nonempty file fingerprint map")
    root = Path(root).resolve()
    for relative, expected in hashes.items():
        if (not isinstance(relative, str) or not relative or "\\" in relative
            or "\x00" in relative or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))):
            raise ValueError(f"Expert manifest requires a safe relative file path: {relative!r}")
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Expert manifest file escapes the data directory: {relative}")
        if not path.is_file():
            raise ValueError(f"Expert manifest file is missing or not a file: {relative}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or digest(path) != expected:
            raise ValueError(f"Expert data fingerprint mismatch: {relative}")


_WEAK_LABEL_SOURCES = frozenset({"question_span_extraction_weak", "numina_reference_weak",
    "executable_reference_arithmetic", "executable_arithmetic_corruption"})


def supervision_metadata(manifest):
    """Keep label origin separate from correctness and human-review claims."""
    mode = manifest.get("supervision")
    if mode is None:
        mode = ("legacy_rule_weak" if manifest.get("quality_audit", {}).get("teacher_used") is False
                else "legacy_unspecified")
    return {"supervision": mode, "teacher": manifest.get("teacher"),
        "synthesis": manifest.get("synthesis"), "data_quality_audit": manifest.get("quality_audit", {}),
        "supervision_scope": "label provenance only; teacher output and unreviewed labels are not correctness evidence"}


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _teacher_evidence(root, manifest):
    """Validate archived teacher calls without contacting the teacher service."""
    if manifest.get("supervision") != "teacher_synthetic":
        return None
    from ..utils.io import read_jsonl
    teacher = manifest.get("teacher")
    if (not isinstance(teacher, dict) or any(not isinstance(teacher.get(k), str) or not teacher[k].strip()
                                          for k in ("provider", "model"))
        or "revision" not in teacher or teacher["revision"] is not None and not isinstance(teacher["revision"], str)
        or not isinstance(manifest.get("synthesis"), dict)):
        raise ValueError("Teacher supervision requires explicit teacher identity and synthesis metadata")
    synthesis = manifest["synthesis"]
    if (not isinstance(synthesis.get("config"), dict)
        or synthesis.get("config_sha256") != _json_digest(synthesis["config"])
        or not re.fullmatch(r"[0-9a-f]{64}", str(synthesis.get("run_sha256", "")))):
        raise ValueError("Teacher synthesis configuration or run fingerprint is invalid")
    quality = manifest.get("quality_audit")
    if (not isinstance(quality, dict) or quality.get("teacher_used") is not True
        or quality.get("reviewed") is not False or quality.get("reviewed_examples", 0) != 0):
        raise ValueError("Teacher-synthetic exports are unreviewed; human review requires separate review evidence")
    for name in ("teacher_requests.jsonl", "teacher_responses.jsonl"):
        if name not in manifest["sha256"]:
            raise ValueError(f"Teacher supervision is missing fingerprinted evidence: {name}")
    requests, accepted = {}, {}
    for request in read_jsonl(str(root / "teacher_requests.jsonl")):
        request_id = request.get("request_id")
        if (request.get("schema_version") != 1 or request_id in requests
            or request_id != _json_digest({k: v for k, v in request.items() if k != "request_id"})
            or request.get("prompt_sha256") != _json_digest(request.get("messages"))
            or not isinstance(request.get("messages"), list) or not request["messages"]
            or request.get("provider") != teacher["provider"] or request.get("model") != teacher["model"]
            or request.get("teacher_revision") != teacher["revision"]
            or request.get("config_sha256") != synthesis["config_sha256"]
            or request.get("run_sha256") != synthesis["run_sha256"]):
            raise ValueError("Teacher request identity, prompt or fingerprint mismatch")
        requests[request_id] = request
    for evidence in read_jsonl(str(root / "teacher_responses.jsonl")):
        request = requests.get(evidence.get("request_id"))
        response = evidence.get("response")
        validation = evidence.get("validation")
        if (evidence.get("schema_version") != 1 or request is None or not isinstance(validation, dict)
            or type(validation.get("accepted")) is not bool):
            raise ValueError("Teacher response has missing request or validation evidence")
        if response is None:
            if evidence.get("response_sha256") is not None or validation["accepted"]:
                raise ValueError("Empty teacher response cannot be accepted")
            continue
        if (not isinstance(response, dict) or evidence.get("response_sha256") != _json_digest(response)
            or response.get("provider") != request["provider"] or response.get("model") != request["model"]):
            raise ValueError("Teacher response identity or fingerprint mismatch")
        if validation["accepted"]:
            if not isinstance(response.get("text"), str) or not response["text"].strip():
                raise ValueError("Accepted teacher response requires nonempty target text")
            accepted[(evidence["request_id"], evidence["response_sha256"])] = evidence
    return teacher, requests, accepted


def _verify_teacher_row(row, evidence):
    if evidence is None:
        if row.get("label_source") == "teacher_synthetic":
            raise ValueError("Teacher targets require a teacher_synthetic manifest and archived evidence")
        return
    teacher, requests, accepted = evidence
    request = requests.get(row.get("teacher_request_id"))
    response_evidence = accepted.get((row.get("teacher_request_id"), row.get("teacher_response_sha256")))
    response = response_evidence["response"] if response_evidence else None
    if (row.get("label_source") != "teacher_synthetic" or request is None or response is None
        or row.get("teacher_provider") != teacher["provider"] or row.get("teacher_model") != teacher["model"]
        or "teacher_revision" not in row or row["teacher_revision"] != teacher["revision"]
        or "teacher_actual_model" not in row or row["teacher_actual_model"] != response.get("actual_model")
        or row.get("teacher_prompt_sha256") != request["prompt_sha256"]
        or row.get("response") != response["text"]
        or row.get("role") != request.get("kind") or row.get("split") != request.get("split")
        or row.get("question_hash") != request.get("question_hash")):
        raise ValueError("Expert teacher target does not match its accepted request/response evidence")
    if row.get("reviewed") is not False:
        raise ValueError("Teacher-synthetic targets cannot claim human review without review evidence")
    if row["role"] == "verifier":
        verdict = response_evidence["validation"].get("verdict")
        if verdict not in ("correct", "incorrect", "uncertain") or row.get("verdict") != verdict:
            raise ValueError("Expert verifier verdict differs from its accepted teacher validation evidence")
        candidate_id = row.get("candidate_request_id")
        candidate_hash = row.get("candidate_response_sha256")
        candidate_request = requests.get(candidate_id)
        candidate = accepted.get((candidate_id, candidate_hash))
        if (candidate_request is None or candidate is None or candidate_request.get("kind") != "candidate"
            or request.get("candidate_request_id") != candidate_id
            or request.get("candidate_response_sha256") != candidate_hash
            or candidate_request.get("question_hash") != row["question_hash"]
            or candidate_request.get("split") != row["split"] or candidate["response"]["text"] != row.get("candidate")):
            raise ValueError("Expert verifier candidate does not match accepted teacher evidence")


def _verify_supervision_config(config, manifest):
    mode = config.get("expert_data_mode")
    if mode == "teacher_synthetic" and manifest.get("supervision") != "teacher_synthetic":
        raise ValueError("Configured teacher_synthetic training requires teacher-synthesized data")
    if mode == "weak_debug" and manifest.get("supervision") == "teacher_synthetic":
        raise ValueError("weak_debug is reserved for the explicit legacy weak-supervision ablation")
    expected = config.get("expected_teacher")
    if expected is not None:
        actual = manifest.get("teacher")
        if not isinstance(actual, dict) or any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Expert data teacher does not match the configured expected_teacher")


def load_config(source):
    config = dict(source) if isinstance(source, dict) else json.loads(Path(source).read_text())
    for alias, name in (("lr", "learning_rate"), ("batch", "per_device_batch_size"),
                        ("accum", "gradient_accumulation_steps")):
        if alias in config:
            if name in config and config[name] != config[alias]:
                raise ValueError(f"Conflicting {alias} and {name}")
            config[name] = config.pop(alias)
    defaults = dict(seed=42, max_seq_len=8192, learning_rate=2e-5,
        per_device_batch_size=1, gradient_accumulation_steps=8, max_steps=16,
        lora_rank=16, lora_alpha=32, lora_dropout=.05, save_steps=8,
        bf16=True, gradient_checkpointing=True, save_total_limit=2,
        max_grad_norm=1.0, weight_decay=0.0, warmup_ratio=0.0)
    config = {**defaults, **config}
    if not isinstance(config.get("base_model"), str) or not config["base_model"]:
        raise ValueError("base_model is required")
    if config.get("expert_data_mode") not in (None, "teacher_synthetic", "weak_debug"):
        raise ValueError("expert_data_mode must be teacher_synthetic or weak_debug")
    if "expected_teacher" in config:
        teacher = config["expected_teacher"]
        if (not isinstance(teacher, dict) or not teacher or set(teacher) - {"provider", "model", "revision"}
            or any(not isinstance(value, str) or not value.strip() for key, value in teacher.items()
                   if key != "revision" or value is not None)):
            raise ValueError("expected_teacher requires named provider/model/revision constraints")
        if config.get("expert_data_mode") == "weak_debug":
            raise ValueError("weak_debug cannot require a teacher")
    base = Path(config["base_model"])
    if (base / "adapter_config.json").exists():
        raise ValueError("Each expert must start independently from the base, not an adapter")
    if not base.is_dir() and not re.fullmatch(r"[0-9a-fA-F]{40}", str(config.get("base_model_revision", ""))):
        raise ValueError("Remote expert base requires a pinned 40-character model revision")
    for key in ("max_seq_len", "per_device_batch_size", "gradient_accumulation_steps", "max_steps",
                "lora_rank", "lora_alpha", "save_steps", "save_total_limit"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for key in ("learning_rate", "max_grad_norm", "weight_decay", "warmup_ratio", "lora_dropout"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["learning_rate"] <= 0 or config["max_grad_norm"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Learning rate / max grad norm must be positive; weight decay nonnegative")
    if not 0 <= config["lora_dropout"] < 1 or not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("Dropout and warmup ratio must be in [0, 1)")
    for key in ("bf16", "gradient_checkpointing"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    return config


def read_dataset(data_dir, role):
    """Verify the exported bundle before loading any model weights."""
    from types import SimpleNamespace
    from ..utils.io import read_jsonl
    from .data import identity
    if role not in KINDS:
        raise ValueError(f"Unknown expert role: {role}")
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported expert data manifest schema")
    payload = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    content_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    if manifest.get("manifest_content_sha256") != content_hash:
        raise ValueError("Expert manifest content fingerprint changed after preparation")
    template_hash = digest(Path(__file__).with_name("chat_template.jinja"))
    if manifest.get("template_sha256") != template_hash:
        raise ValueError("Expert data chat template differs from the current serving template")
    protocol_hash = digest(Path(__file__).with_name("protocol.py"))
    if manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("Expert data advisor protocol differs from the current serving protocol")
    hashes = manifest.get("sha256", {})
    _verified_files(root, hashes)
    teacher_evidence = _teacher_evidence(root, manifest)
    all_rows = {}
    questions = {"train": set(), "dev": set()}
    source_ids = {"train": set(), "dev": set()}
    for kind in KINDS:
        for split in ("train", "dev"):
            relative = f"{kind}/{split}.jsonl"
            path = root / relative
            if not isinstance(hashes.get(relative), str) or digest(path) != hashes[relative]:
                raise ValueError(f"Expert data fingerprint mismatch: {relative}")
            rows = read_jsonl(str(path))
            if not rows:
                raise ValueError(f"Empty expert dataset: {relative}")
            system = advisor_messages(kind, SimpleNamespace(question="", context=""), "")[0]
            for row in rows:
                if row.get("role") != kind or row.get("split") != split:
                    raise ValueError(f"Expert role/split mismatch in {relative}")
                if row.get("template_sha256") != template_hash or row.get("protocol_sha256") != protocol_hash:
                    raise ValueError(f"Expert row template fingerprint mismatch in {relative}")
                if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("question_hash", ""))):
                    raise ValueError(f"Missing question hash in {relative}")
                if not isinstance(row.get("source_id"), (int, str)) or not str(row["source_id"]).strip():
                    raise ValueError(f"Missing source identity in {relative}")
                if not isinstance(row.get("question"), str) or identity(row["question"]) != row["question_hash"]:
                    raise ValueError(f"Expert question content/hash mismatch in {relative}")
                expected_prompt = advisor_messages(kind, SimpleNamespace(question=row["question"],
                    context=row.get("context", "")), row.get("candidate", ""))
                prompt = row.get("prompt")
                if prompt != expected_prompt:
                    raise ValueError(f"Expert input differs from serving protocol: {relative}")
                if (not isinstance(prompt, list) or len(prompt) != 2 or prompt[0] != system
                    or prompt[1].get("role") != "user" or not isinstance(prompt[1].get("content"), str)
                    or not prompt[1]["content"].strip()):
                    raise ValueError(f"Expert prompt does not follow the serving role protocol: {relative}")
                if not isinstance(row.get("response"), str) or not row["response"].strip():
                    raise ValueError(f"Expert target must be nonempty assistant text: {relative}")
                if "label_source" in row and (not isinstance(row["label_source"], str) or not row["label_source"].strip()):
                    raise ValueError(f"Expert label_source must be nonempty text: {relative}")
                _verify_teacher_row(row, teacher_evidence)
                questions[split].add(row["question_hash"])
                source_ids[split].add(str(row["source_id"]))
            all_rows[(kind, split)] = rows
    if questions["train"] & questions["dev"] or source_ids["train"] & source_ids["dev"]:
        raise ValueError("Expert train/dev question overlap, including across roles and draft variants")
    return manifest, all_rows[(role, "train")], all_rows[(role, "dev")]


def tokenize_rows(rows, tokenizer, max_seq_len):
    features, excluded = [], []
    report = {"input_rows": len(rows), "kept_rows": 0, "overlength_rows": 0,
              "empty_token_rows": 0, "input_tokens_per_epoch": 0,
              "supervised_tokens_per_epoch": 0, "reviewed_rows": 0,
              "teacher_generated_rows": 0, "weak_supervision_rows": 0,
              "other_supervision_rows": 0, "label_source_counts": {}}
    for index, row in enumerate(rows):
        prompt = render(tokenizer, row["prompt"])
        full = render(tokenizer, [*row["prompt"], {"role": "assistant", "content": row["response"]}], generation=False)
        if not full.startswith(prompt):
            raise ValueError("Serving template is not prefix preserving; response masking would be invalid")
        # Preserve the exact inference prefix even when a BPE token could cross
        # the text boundary. The serving model never receives that merged token.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(full[len(prompt):], add_special_tokens=False)["input_ids"]
        size = len(prompt_ids) + len(target_ids)
        reason = "overlength" if size > max_seq_len else "empty_token" if not prompt_ids or not target_ids else None
        if reason:
            report[reason + "_rows"] += 1
            excluded.append({"row_index": index, "source_id": row["source_id"],
                             "question_hash": row["question_hash"], "reason": reason,
                             "input_tokens": size, "target_tokens": len(target_ids)})
            continue
        features.append({"input_ids": prompt_ids + target_ids, "attention_mask": [1] * size,
                         "labels": [-100] * len(prompt_ids) + target_ids})
        report["kept_rows"] += 1
        report["reviewed_rows"] += int(row.get("reviewed") is True)
        source = row.get("label_source") or "unspecified"
        report["label_source_counts"][source] = report["label_source_counts"].get(source, 0) + 1
        supervision = ("teacher_generated" if source == "teacher_synthetic" else
                       "weak_supervision" if source in _WEAK_LABEL_SOURCES else "other_supervision")
        report[supervision + "_rows"] += 1
        report["input_tokens_per_epoch"] += size
        report["supervised_tokens_per_epoch"] += len(target_ids)
    report["dropped_rows"] = len(excluded)
    report["unreviewed_rows"] = report["kept_rows"] - report["reviewed_rows"]
    return features, report, excluded


def _checkpoint_files(path):
    return {p.name: digest(p) for p in sorted(Path(path).iterdir())
            if p.is_file() and p.name != "expert_checkpoint.json"}


def _commit_checkpoint(path, step, signature_hash):
    root = Path(path)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "adapter_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Incomplete resumable expert checkpoint: {name}")
    if not any((root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Expert checkpoint is missing adapter weights")
    state = json.loads((root / "trainer_state.json").read_text())
    if state.get("global_step") != step:
        raise ValueError("Expert checkpoint optimizer step disagrees with Trainer state")
    atomic_json(root / "expert_checkpoint.json", {"schema_version": 1, "optimizer_steps": step,
        "training_run_sha256": signature_hash, "sha256": _checkpoint_files(root)})


def _resume_checkpoint(output, signature_hash, max_steps):
    candidates = []
    for path in Path(output).glob("checkpoint-*"):
        marker = path / "expert_checkpoint.json"
        if not path.is_dir() or not marker.exists():
            continue  # A killed save is not a committed checkpoint.
        data = json.loads(marker.read_text())
        step = data.get("optimizer_steps")
        if (type(step) is not int or not 0 < step <= max_steps or path.name != f"checkpoint-{step}"
                or data.get("training_run_sha256") != signature_hash
                or data.get("sha256") != _checkpoint_files(path)):
            raise ValueError(f"Changed or incompatible expert checkpoint: {path.name}")
        if json.loads((path / "trainer_state.json").read_text()).get("global_step") != step:
            raise ValueError(f"Inconsistent expert checkpoint step: {path.name}")
        candidates.append((step, path))
    return str(max(candidates)[1]) if candidates else None


def _adapter_hash(output):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = Path(output) / name
        if path.is_file():
            return digest(path)
    raise ValueError("Completed expert training is missing adapter weights")


def _export_hashes(output):
    names = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt")
    return {name: digest(Path(output) / name) for name in names if (Path(output) / name).is_file()}


def train_expert(config, data_dir, role, output, resume=False):
    """Train one independent role; explicit --resume preserves the step budget."""
    from .provenance import harness_identity
    from .runner import checkpoint_identity
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Math expert SFT supports one training process per role")
    config = load_config(config)
    manifest, train_rows, dev_rows = read_dataset(data_dir, role)
    _verify_supervision_config(config, manifest)
    root = Path(output)
    signature = {"schema_version": 1, "stage": "expert_sft", "role": role,
        "config": config, "base_identity": checkpoint_identity(config["base_model"]),
        "harness": harness_identity(), "data_manifest_sha256": digest(Path(data_dir) / "manifest.json"),
        "data_sha256": manifest["sha256"], "template_sha256": manifest["template_sha256"],
        "protocol_sha256": manifest["protocol_sha256"]}
    root.mkdir(parents=True, exist_ok=True)
    run_path = root / "training_run.json"
    if run_path.exists():
        if json.loads(run_path.read_text()) != signature:
            raise ValueError("Expert training configuration, data, or source changed; use a new output directory")
        if not resume:
            raise FileExistsError("Expert training output already exists; pass --resume to continue it")
    elif any(root.iterdir()):
        raise FileExistsError("Expert output contains files without a matching training manifest")
    else:
        atomic_json(run_path, signature)
    signature_hash = digest(run_path)
    final_path = root / "summary.json"
    if final_path.exists():
        complete = json.loads(final_path.read_text())
        if complete.get("training_complete"):
            if (complete.get("optimizer_steps") != config["max_steps"]
                or complete.get("adapter_sha256") != _adapter_hash(root)
                or complete.get("training_run_sha256") != signature_hash
                or complete.get("export_sha256") != _export_hashes(root)
                or not (root / "adapter_config.json").is_file()):
                raise ValueError("Completed expert training artifacts changed or are incomplete")
            saved = _resume_checkpoint(root, signature_hash, config["max_steps"])
            if not saved or Path(saved).name != f"checkpoint-{config['max_steps']}":
                raise ValueError("Completed expert training is missing final resumable checkpoint evidence")
            return complete
    checkpoint = _resume_checkpoint(root, signature_hash, config["max_steps"]) if resume else None
    with Monitor(root, "expert_sft") as monitor:
        metadata = {"role": role, "controller_status": "running", "current_stage": "expert_sft",
            "training_complete": False, "training_algorithm": "math_expert_response_only_lora_sft",
            "base_model": config["base_model"], "base_model_revision": config.get("base_model_revision"),
            "template_sha256": manifest["template_sha256"], "protocol_sha256": manifest["protocol_sha256"],
            "data_manifest_sha256": signature["data_manifest_sha256"],
            "training_run_sha256": signature_hash, "resumed_checkpoint": checkpoint,
            "optimizer_step_budget": config["max_steps"], "training_config": config,
            **supervision_metadata(manifest)}
        if manifest.get("supervision") == "teacher_synthetic":
            metadata.update(teacher_actual_models=sorted({row["teacher_actual_model"] for row in train_rows + dev_rows
                                                        if row["teacher_actual_model"] is not None}),
                teacher_actual_model_unknown_rows=sum(row["teacher_actual_model"] is None for row in train_rows + dev_rows))
        monitor.summary(metadata)
        try:
            result = _train(config, train_rows, dev_rows, role, root, checkpoint, signature_hash, monitor)
            complete = {**metadata, **result, "training_complete": True,
                        "controller_status": "completed", "current_stage": "complete", "error": None}
            atomic_json(final_path, complete)
            monitor.summary(complete)
            return complete
        except BaseException as exc:
            failure = {**metadata, "controller_status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       "failed_stage": "expert_sft", "error": str(exc), "error_type": type(exc).__name__}
            atomic_json(final_path, failure)
            monitor.summary(failure)
            raise


def _train(config, train_rows, dev_rows, role, root, checkpoint, signature_hash, monitor):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments, set_seed
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA device per expert training process")
    set_seed(config["seed"])
    tokenizer, model = load_model(config["base_model"], checkpoint=checkpoint, trainable=bool(checkpoint),
                                   revision=config.get("base_model_revision"))
    if not config["bf16"]:
        model.float()
    targets = sorted({name.rsplit(".", 1)[-1] for name, _ in model.named_modules()} &
        {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
         "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"})
    if not targets:
        raise ValueError("Expert base has no supported LoRA projection modules")
    if not checkpoint:
        model = get_peft_model(model, LoraConfig(r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=targets, bias="none", task_type="CAUSAL_LM",
            revision=config.get("base_model_revision")))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.train()
    train, train_report, train_excluded = tokenize_rows(train_rows, tokenizer, config["max_seq_len"])
    dev, dev_report, dev_excluded = tokenize_rows(dev_rows, tokenizer, config["max_seq_len"])
    data_report = {"train": train_report, "dev": dev_report, "excluded": {"train": train_excluded, "dev": dev_excluded},
                   "filter_rule": "drop entire overlength/empty-token rows; never truncate a target",
                   "dev_loss_scope": "kept development rows only; cross-entropy is not role quality or benchmark accuracy"}
    atomic_json(root / "sft_data_report.json", data_report)
    monitor.summary({"data_report": data_report, "lora_target_modules": targets,
        "trainable_module_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "actual_dtype": str(next(model.parameters()).dtype)})
    if not train or not dev:
        raise ValueError("Expert training requires nonempty train and dev after reported length filtering")
    for split, report in (("train", train_report), ("dev", dev_report)):
        metrics(report, f"expert_data/{split}")
    bf16 = config["bf16"] and torch.cuda.is_available()
    args = TrainingArguments(output_dir=str(root), max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_batch_size"],
        per_device_eval_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"], learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"], warmup_ratio=config["warmup_ratio"], max_grad_norm=config["max_grad_norm"],
        bf16=bf16, use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=config["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config["seed"], data_seed=config["seed"], logging_steps=1,
        save_strategy="steps", save_steps=config["save_steps"], save_total_limit=config["save_total_limit"],
        eval_strategy="steps", eval_steps=config["save_steps"], report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, logging_nan_inf_filter=False)

    class CheckpointCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= args.max_steps:
                control.should_save = True
                control.should_evaluate = True
            monitor.summary({"optimizer_steps": state.global_step})

        def on_save(self, args, state, control, **kwargs):
            _commit_checkpoint(root / f"checkpoint-{state.global_step}", state.global_step, signature_hash)

    class ExpertTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                          num_items_in_batch=num_items_in_batch)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite expert SFT loss")
            if not model.training:
                usage("expert_sft_dev", {}, expert_role=role,
                    input_tokens=int(inputs["attention_mask"].sum().item()),
                    supervised_tokens=int((inputs["labels"] != -100).sum().item()), step=self.state.global_step)
            return result

        def training_step(self, model, inputs, num_items_in_batch=None):
            result = super().training_step(model, inputs, num_items_in_batch)
            usage("expert_sft_train", {}, expert_role=role,
                input_tokens=int(inputs["attention_mask"].sum().item()),
                supervised_tokens=int((inputs["labels"] != -100).sum().item()), step=self.state.global_step)
            return result

    trainer = ExpertTrainer(model=model, args=args, train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(dev), processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[training_callback(root), CheckpointCallback()])
    started = time.monotonic()
    progress(phase="training", role=role, max_steps=config["max_steps"])
    restored_step = (json.loads((Path(checkpoint) / "trainer_state.json").read_text())["global_step"]
                     if checkpoint else 0)
    monitor.summary({"resumed_optimizer_steps": restored_step})
    if restored_step == config["max_steps"]:
        # A crash during final export/evaluation must not buy an extra update.
        from types import SimpleNamespace
        from transformers import TrainerState
        trainer.state = TrainerState.load_from_json(str(Path(checkpoint) / "trainer_state.json"))
        outcome = SimpleNamespace(metrics={"resumed_at_step_budget": True})
    else:
        outcome = trainer.train(resume_from_checkpoint=checkpoint)
    if trainer.state.global_step != config["max_steps"]:
        raise RuntimeError("Expert stopped before the declared optimizer-step budget")
    dev_metrics = trainer.evaluate()
    if not math.isfinite(dev_metrics.get("eval_loss", float("nan"))):
        raise FloatingPointError("Missing or nonfinite expert development loss")
    trainer.save_model(str(root))
    tokenizer.save_pretrained(root)
    final_checkpoint = _resume_checkpoint(root, signature_hash, config["max_steps"])
    if not final_checkpoint or Path(final_checkpoint).name != f"checkpoint-{config['max_steps']}":
        raise ValueError("Expert final resumable checkpoint is missing")
    metrics(outcome.metrics, "expert_train", trainer_step=trainer.state.global_step)
    metrics(dev_metrics, "expert_dev", trainer_step=trainer.state.global_step)
    atomic_json(root / "dev_metrics.json", {**dev_metrics, "n": len(dev), "optimizer_steps": trainer.state.global_step,
        "scope": "response-only cross-entropy; not correctness, role quality, or downstream benchmark"})
    result = {"optimizer_steps": trainer.state.global_step, "adapter_sha256": _adapter_hash(root),
        "export_sha256": _export_hashes(root), "resumed_optimizer_steps": restored_step,
        "lora_target_modules": targets,
        "trainable_module_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "actual_dtype": str(next(model.parameters()).dtype),
        "final_checkpoint": final_checkpoint, "eval_loss": dev_metrics["eval_loss"], "eval_n": len(dev),
        "train_metrics": outcome.metrics, "train_loss_scope": "HF Trainer reported aggregate; use step logs when resuming",
        "data_report": data_report, "wall_seconds_this_attempt": time.monotonic() - started,
        "token_count_scope": "observed executed microbatches across attempts, including replay after interruption",
        "observed_training_input_tokens": monitor.totals.get("usage/expert_sft_train/input_tokens", 0),
        "observed_supervised_tokens": monitor.totals.get("usage/expert_sft_train/supervised_tokens", 0),
        "observed_dev_input_tokens": monitor.totals.get("usage/expert_sft_dev/input_tokens", 0),
        "observed_dev_supervised_tokens": monitor.totals.get("usage/expert_sft_dev/supervised_tokens", 0)}
    atomic_json(root / "training_metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--role", choices=KINDS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_expert(args.config, args.data_dir, args.role, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
