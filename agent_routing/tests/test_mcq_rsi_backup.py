"""scripts/backup_mcq_rsi_hf.py: what is staged, incremental copies, the advisor-cache archive, upload wiring."""
import importlib.util
import os
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("backup_mcq_rsi_hf", ROOT / "scripts" / "backup_mcq_rsi_hf.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def _w(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_work(tmp_path: Path) -> Path:
    w = tmp_path / "mcq_rsi"
    g = w / "runs/medqa_pilot/r1/grpo"
    for rel in ("final/adapter_model.safetensors", "final/adapter_config.json", "metrics.jsonl", "resume.json",
                "step-00001-ab/step.json", "step-00001-ab/adapter_config.json",
                "step-00001-ab/adapter_model.safetensors", "step-00001-ab/optimizer.pt",
                "incomplete-zz/adapter_model.safetensors", ".final-qq/adapter_model.safetensors"):
        _w(g / rel)
    for rel in ("rsi_run.json", "controller.lock", "r1/.grpo.stage.lock", "r2/collect/shard_000.jsonl",
                "r2/collect/.shard_001.jsonl.abc.part", "r2/dynamic/sft/model/adapter_model.safetensors",
                "r2/dynamic/sft/checkpoint-10/optimizer.pt", "r1/S1_dev/decision.json"):
        _w(w / "runs/medqa_pilot" / rel)
    _w(w / "logs/pipeline.log")
    _w(w / "advisor_cache/preflight/medqa.json")
    _w(w / "advisor_cache/locked_test/medqa.json")
    _w(w / "advisor_cache/medqa/ab/ab12.json", '{"output": "x"}')
    _w(w / "advisor_cache/medqa/cd/cd34.json", '{"output": "y"}')
    _w(w / "import/medqa/import_manifest.json")
    _w(w / "import/medqa/round1/labels.jsonl")  # re-importable: not backed up
    _w(w / "hf-cache-should-not-exist/x")
    return w


def test_staged_files_keep_results_and_drop_step_weights_temporaries_and_locks(tmp_path):
    w = make_work(tmp_path)
    got = set(B.source_files(w))
    p = "runs/medqa_pilot/"
    assert {p + "r1/grpo/final/adapter_model.safetensors", p + "r1/grpo/final/adapter_config.json",
            p + "r1/grpo/metrics.jsonl", p + "r1/grpo/resume.json", p + "r1/grpo/step-00001-ab/step.json",
            p + "r1/grpo/step-00001-ab/adapter_config.json", p + "rsi_run.json", p + "r2/collect/shard_000.jsonl",
            p + "r2/dynamic/sft/model/adapter_model.safetensors", p + "r1/S1_dev/decision.json", "logs/pipeline.log",
            "advisor_cache/preflight/medqa.json", "advisor_cache/locked_test/medqa.json",
            "import/medqa/import_manifest.json"} == got


def test_mirror_is_incremental_and_removes_files_gone_from_the_source(tmp_path):
    w = make_work(tmp_path)
    stage = w / B.STAGE
    first = B.mirror(w, stage)
    assert first["copied"] == len(B.source_files(w)) and first["removed"] == 0
    assert (stage / "runs/medqa_pilot/r1/grpo/final/adapter_model.safetensors").is_file()
    assert not (stage / "runs/medqa_pilot/r1/grpo/step-00001-ab/optimizer.pt").exists()
    again = B.mirror(w, stage)
    assert again["copied"] == 0 and again["unchanged"] == first["copied"]
    status = w / "runs/medqa_pilot/r1/grpo/metrics.jsonl"
    status.write_text("y")  # same size, new content and mtime
    os.utime(status, ns=(time.time_ns(), time.time_ns() + 5_000))
    (w / "runs/medqa_pilot/r1/S1_dev/decision.json").unlink()
    third = B.mirror(w, stage)
    assert third["copied"] == 1 and third["removed"] == 1
    assert (stage / "runs/medqa_pilot/r1/grpo/metrics.jsonl").read_text() == "y"
    assert not (stage / "runs/medqa_pilot/r1/S1_dev/decision.json").exists()
    dry = tmp_path / "dry"
    assert B.mirror(w, dry, dry_run=True)["copied"] > 0 and not dry.exists()


def test_advisor_cache_archive_is_rewritten_only_when_the_cache_changed(tmp_path):
    w = make_work(tmp_path)
    stage = w / B.STAGE
    assert B.pack_advisor_cache(w, stage)
    with tarfile.open(stage / B.ADVISOR_TAR) as tar:
        assert sorted(tar.getnames()) == ["advisor_cache/medqa/ab/ab12.json", "advisor_cache/medqa/cd/cd34.json"]
    assert B.pack_advisor_cache(w, stage) is None
    new = _w(w / "advisor_cache/medqa/ef/ef56.json")
    later = (stage / B.ADVISOR_TAR).stat().st_mtime + 10
    os.utime(new, (later, later))
    assert B.pack_advisor_cache(w, stage)
    with tarfile.open(stage / B.ADVISOR_TAR) as tar:
        assert "advisor_cache/medqa/ef/ef56.json" in tar.getnames()


def test_one_pass_uploads_the_staging_folder_privately_and_records_the_result(tmp_path, monkeypatch):
    w = make_work(tmp_path)
    calls = []
    monkeypatch.setattr(B, "upload", lambda stage, repo, private: calls.append((stage, repo, private)))
    res = B.one_pass(w, "MaliDDD/test-backup", private=True, dry_run=False)
    assert calls == [(w / B.STAGE, "MaliDDD/test-backup", True)]
    assert res["copied"] > 0 and res["advisor_cache_packed"] and (w / "logs/hf_backup_last.json").is_file()
    calls.clear()
    assert B.one_pass(w, "r", private=True, dry_run=True)["dry_run"] and not calls


def test_per_run_backup_stages_one_run_its_logs_and_the_registries(tmp_path, monkeypatch):
    w = make_work(tmp_path)
    _w(w / "runs/medqa_v2/rsi_run.json")
    _w(w / "runs/medqa_v2/r2/dynamic_sft/sft/model/adapter_model.safetensors")
    _w(w / "logs/medqa_v2.log")
    _w(w / "logs/medqa_v2_final_test.log")
    files = B.source_files(w, run="medqa_v2")
    assert set(files) == {"runs/medqa_v2/rsi_run.json", "runs/medqa_v2/r2/dynamic_sft/sft/model/adapter_model.safetensors",
                          "logs/medqa_v2.log", "logs/medqa_v2_final_test.log", "advisor_cache/preflight/medqa.json",
                          "advisor_cache/locked_test/medqa.json", "import/medqa/import_manifest.json"}
    assert not any(k.startswith("runs/medqa_pilot") for k in files) and "logs/pipeline.log" not in files
    calls = []
    monkeypatch.setattr(B, "upload", lambda stage, repo, private: calls.append((stage, repo, private)))
    result = B.one_pass(w, "u/mcq-medqa_v2", True, False, run="medqa_v2")
    stage = B.stage_dir(w, "medqa_v2")
    assert stage == w / "hf_backup_stage__medqa_v2" and calls == [(stage, "u/mcq-medqa_v2", True)]
    assert (stage / "runs/medqa_v2/rsi_run.json").is_file() and not (stage / "runs/medqa_pilot").exists()
    assert (stage / B.ADVISOR_TAR).is_file() and result["run"] == "medqa_v2" and result["copied"] == 7
    assert (w / "logs/hf_backup_last__medqa_v2.json").is_file()
    assert B.stage_dir(w) == w / B.STAGE and B.source_files(w) == B.source_files(w, None)
    import pytest
    with pytest.raises(SystemExit, match="no run directory"):
        B.one_pass(w, "u/x", True, True, run="missing")


def test_wrapper_exposes_the_backup_step():
    wrapper = (ROOT / "scripts" / "runpod_mcq_rsi.sh").read_text()
    assert "step_backup" in wrapper and "|backup)$" in wrapper and 'BACKUP_EVERY_MIN="${BACKUP_EVERY_MIN:-}"' in wrapper
