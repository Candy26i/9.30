"""Paper evidence survives failures, resume and bounded UI tables (no network)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_math_wandb import FakeWandb
from src.verifiable.telemetry import Monitor, status_snapshot
from src.verifiable.wandb_tracking import WandbTracker


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.sdk = FakeWandb()
        for p in (patch.dict("os.environ", {"MARGENT_WANDB_MODE": "online", "WANDB_ENTITY": "test",
                                            "WANDB_PROJECT": "test", "MARGENT_WANDB_TEXT": "0"}),
                  patch("src.verifiable.wandb_tracking.import_module", return_value=self.sdk),
                  patch("src.verifiable.telemetry.subprocess.run", side_effect=FileNotFoundError)):
            p.start()
            self.addCleanup(p.stop)

    def test_failure_evidence_and_final_status_without_text_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training_run.json").write_text(json.dumps({"config": {"seed": 43}, "data_sha256": "abc"}))
            (root / "generations.jsonl").write_text('{"text":"NOT_OPTED_IN"}\n')
            (root / "adapter_model.safetensors").write_text("MODEL_NEVER_UPLOAD")
            with self.assertRaisesRegex(ValueError, "sample failed"):
                with Monitor(root, "sft") as monitor:
                    monitor.summary({"checkpoint_sha256": "checkpoint-digest"})
                    raise ValueError("sample failed api_key=DO_NOT_UPLOAD")
            run = self.sdk.runs[-1]
            files = run.artifacts[-1].files
            self.assertIn("errors.log", files)
            self.assertIn("sample failed", files["errors.log"])
            self.assertNotIn("DO_NOT_UPLOAD", str(files))
            self.assertNotIn("generations.jsonl", files)
            self.assertNotIn("adapter_model.safetensors", files)
            inventory = json.loads(files["checkpoint_inventory.json"])
            self.assertFalse(inventory["weights_uploaded"])
            self.assertEqual(inventory["files"][0]["path"], "adapter_model.safetensors")
            self.assertEqual(len(inventory["files"][0]["sha256"]), 64)
            self.assertEqual(json.loads(files["status.json"])["status"], "failed")
            self.assertEqual(run.summary["checkpoint_sha256"], "checkpoint-digest")
            self.assertEqual(run.exit_code, 1)

    def test_stage_config_and_provenance_override_parent_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "loop.json").write_text(json.dumps({"config": {"rl_max_steps": 8}, "arm": "dynamic_rl"}))
            stage = root / "round_1" / "grpo"
            stage.mkdir(parents=True)
            (stage / "training_run.json").write_text(json.dumps({"config": {"rl_max_steps": 1}, "checkpoint": "parent-sha", "data_sha256": "train-sha"}))
            tracker = WandbTracker(stage, "grpo", "attempt")
            tracker.start()
            config = self.sdk.calls[-1]["config"]
            self.assertEqual(config["rl_max_steps"], 1)
            self.assertEqual(config["stage_manifest"]["checkpoint"], "parent-sha")
            self.assertEqual(config["arm"], "dynamic_rl")

    def test_rsi_arms_share_group_and_keep_round_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rsi_run.json").write_text(json.dumps({"config": {"seed": 42}, "arms": ["dynamic", "static", "success"]}))
            for relative in ("initial_dev", "dynamic/round_1/sft", "static/round_2/grpo"):
                stage = root / relative
                stage.mkdir(parents=True)
                WandbTracker(stage, "stage", "attempt").start()
            self.assertEqual(len({c["group"] for c in self.sdk.calls}), 1)
            self.assertEqual([c["config"]["arm"] for c in self.sdk.calls], ["shared", "dynamic", "static"])
            self.assertEqual([c["config"]["round"] for c in self.sdk.calls], [0, 1, 2])

    def test_text_evidence_is_complete_and_oversized_files_explicitly_omitted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"MARGENT_WANDB_TEXT": "1"}):
            root = Path(tmp)
            content = json.dumps({"text": "begin" + "x" * 25000 + "end"})
            (root / "generations.jsonl").write_text(content)
            tracker = WandbTracker(root, "evaluate", "attempt")
            tracker.start()
            tracker.snapshot_artifacts()
            self.assertEqual(self.sdk.runs[-1].artifacts[-1].files["generations.jsonl"], content)
            with patch.dict("os.environ", {"MARGENT_WANDB_ARTIFACT_MAX_BYTES": "64"}):
                tracker.snapshot_artifacts()
            index = json.loads(self.sdk.runs[-1].artifacts[-1].files["evidence_index.json"])
            self.assertTrue(any(p["path"] == "generations.jsonl" for p in index["omitted"]))

    def test_transient_metric_failure_can_recover_and_finish_keeps_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Monitor(tmp, "evaluate") as monitor:
                with patch.object(monitor.tracker.run, "log", side_effect=OSError("network")):
                    monitor.metrics({"n": 1}, "eval")
                monitor.retry_after = 0
                monitor.metrics({"n": 2}, "eval")
            self.assertTrue(any(r.get("eval/n") == 2 for r in self.sdk.runs[-1].history))
            self.assertIn("metrics.jsonl", self.sdk.runs[-1].artifacts[-1].files)
            self.assertEqual(len((Path(tmp) / "metrics.jsonl").read_text().splitlines()), 2)

    def test_summary_works_with_tracking_disabled(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"MARGENT_WANDB_MODE": "disabled"}):
            with Monitor(tmp, "baseline") as monitor:
                monitor.summary({"current_stage": "dev_preflight", "nested": [[{"password": "HIDDEN"}]]})
            saved = json.loads((Path(tmp) / "run_summary.json").read_text())
            self.assertEqual(saved["current_stage"], "dev_preflight")
            self.assertNotIn("HIDDEN", json.dumps(saved))
            self.assertFalse(self.sdk.calls)

    def test_resume_recovers_torn_usage_tail_without_inventing_cost(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"MARGENT_WANDB_MODE": "disabled"}):
            path = Path(tmp) / "usage.jsonl"
            path.write_text('{"role":"manager","completion_tokens":7}\n{"role":"manager","completion_to')
            with Monitor(tmp, "evaluate") as monitor:
                monitor.usage("manager", {"completion_tokens": 3})
            self.assertEqual(monitor.totals["usage/manager/generated_tokens"], 10)
            self.assertTrue(monitor.state["usage_incomplete"])
            self.assertEqual(len([json.loads(line) for line in path.read_text().splitlines()]), 2)
            self.assertTrue(list(Path(tmp).glob("usage_torn_tail_*.txt")))
            with Monitor(tmp, "evaluate") as resumed:
                self.assertTrue(resumed.state["usage_incomplete"])

    def test_interior_usage_corruption_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "usage.jsonl").write_text('broken\n{"role":"manager"}\n')
            with self.assertRaises(json.JSONDecodeError):
                with Monitor(tmp, "evaluate"):
                    self.fail("interior corruption was ignored")

    def test_status_ignores_archived_evidence_and_generation_axis_resumes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"MARGENT_WANDB_MODE": "disabled"}):
            for i in range(2):
                with Monitor(tmp, "evaluate") as monitor:
                    monitor.generation("manager", {"text": "answer", "completion_tokens": 2})
                    self.assertEqual(monitor.generation_count, i + 1)
            root = Path(tmp)
            copy = root / "tracking_exports/old/status.json"
            copy.parent.mkdir(parents=True)
            copy.write_bytes((root / "status.json").read_bytes())
            self.assertEqual(len(status_snapshot(root)["states"]), 1)


if __name__ == "__main__":
    unittest.main()
