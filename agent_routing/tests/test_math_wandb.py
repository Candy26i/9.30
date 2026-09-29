"""W&B boundaries, resume identities and metric attribution; no network or GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.utils.io import write_json, write_jsonl
from src.verifiable import telemetry
from src.verifiable.runner import log_diagnostic
from src.verifiable.wandb_tracking import WandbTracker, scalar_metrics


class FakeRun:
    def __init__(self):
        self.summary, self.history, self.axes = {}, [], []
        self.url = "https://wandb.ai/test/project/runs/fake"
        self.exit_code = None
        self.artifacts = []

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def log(self, value):
        self.history.append(value)

    def define_metric(self, *args, **kwargs):
        self.axes.append((args, kwargs))

    def finish(self, exit_code):
        self.exit_code = exit_code


class FakeWandb:
    def __init__(self):
        self.calls, self.runs = [], []

    def Settings(self, **kwargs):
        return kwargs

    class Artifact:
        def __init__(self, name, type, metadata):
            self.name, self.type, self.metadata, self.files = name, type, metadata, {}

        def add_file(self, path, name):
            self.files[name] = Path(path).read_text()

    def init(self, **kwargs):
        self.calls.append(kwargs)
        result = FakeRun()
        self.runs.append(result)
        return result


class WandbTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeWandb()
        self.env = patch.dict("os.environ", {"MARGENT_WANDB_MODE": "online", "WANDB_ENTITY": "test",
                                             "WANDB_PROJECT": "project"})
        self.env.start()
        self.module = patch("src.verifiable.wandb_tracking.import_module", return_value=self.fake)
        self.module.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.module.stop)

    def loop(self, path, seed=42):
        path.mkdir(parents=True, exist_ok=True)
        write_json(str(path / "loop.json"), {"arm": "dynamic_rl", "config": {"seed": seed,
                   "api_key": "NEVER_UPLOAD", "max_new_tokens": 2048}})
        return path

    def test_disabled_does_not_import_sdk_or_write_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"MARGENT_WANDB_MODE": "disabled"}):
            tracker = WandbTracker(tmp, "sft", "attempt")
            tracker.start()
            tracker.log({"loss": 1.})
            tracker.finish("completed")
            self.assertFalse(self.fake.calls)
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_online_resume_group_and_separate_stage_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.loop(Path(tmp) / "experiment")
            for stage in ("sft", "rl", "sft"):
                path = root / "round_1" / stage
                path.mkdir(parents=True, exist_ok=True)
                t = WandbTracker(path, stage, "attempt")
                t.start()
                t.finish("completed")
            a, b, c = self.fake.calls
            self.assertEqual(a["id"], c["id"])
            self.assertNotEqual(a["id"], b["id"])
            self.assertEqual(a["group"], b["group"])
            self.assertEqual(a["resume"], "allow")
            self.assertEqual(a["config"]["round"], 1)
            self.assertEqual(a["config"]["arm"], "dynamic_rl")
            self.assertEqual(a["config"]["api_key"], "[redacted]")
            self.assertEqual(a["config"]["max_new_tokens"], 2048)

    def test_fresh_copy_and_offline_restarts_do_not_merge_runs(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"MARGENT_WANDB_MODE": "offline"}):
            root = self.loop(Path(tmp) / "first")
            for attempt in ("aaaaaaaa", "bbbbbbbb"):
                t = WandbTracker(root, "loop", attempt)
                t.start()
                t.finish("completed")
            other = self.loop(Path(tmp) / "second")
            t = WandbTracker(other, "loop", "cccccccc")
            t.start()
            a, b, c = self.fake.calls
            self.assertNotEqual(a["id"], b["id"])
            self.assertEqual(a["config"]["logical_stage_id"], b["config"]["logical_stage_id"])
            self.assertNotEqual(a["group"], c["group"])
            self.assertNotIn("resume", a)

    def test_changed_destination_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = WandbTracker(tmp, "loop", "attempt")
            t.start()
            with patch.dict("os.environ", {"WANDB_PROJECT": "different"}):
                with self.assertRaisesRegex(ValueError, "project/entity"):
                    WandbTracker(tmp, "loop", "other").start()

    def test_scalar_only_and_trainer_step_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = WandbTracker(tmp, "sft", "attempt")
            t.start()
            for step in (20, 10):
                t.log({"trainer_step": step, "train/loss": .5, "prompt": "PRIVATE",
                       "raw": ["PRIVATE"], "nan": float("nan"), "nested": {"accuracy": .8}})
            self.assertEqual([r["trainer_step"] for r in t.run.history], [20, 10])
            self.assertNotIn("PRIVATE", json.dumps(t.run.history))
            self.assertNotIn("nan", t.run.history[0])
            self.assertEqual(t.run.history[0]["nested/accuracy"], .8)

    def test_init_failure_does_not_replace_active_monitor(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.fake, "init", side_effect=RuntimeError("login")):
            previous = telemetry.ACTIVE
            with self.assertRaisesRegex(RuntimeError, "login"):
                with telemetry.Monitor(tmp, "sft"):
                    self.fail("must not start training")
            self.assertIs(telemetry.ACTIVE, previous)

    def test_upload_failure_keeps_local_metrics_and_original_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.verifiable.telemetry.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, "training failed"):
                with telemetry.Monitor(tmp, "rl") as m:
                    with patch.object(m.tracker.run, "log", side_effect=RuntimeError("network")):
                        m.metrics({"reward": .5}, "train", trainer_step=1)
                    raise ValueError("training failed")
            self.assertEqual(self.fake.runs[-1].exit_code, 1)
            self.assertIn('"train/reward": 0.5', (Path(tmp) / "metrics.jsonl").read_text())
            state = json.loads((Path(tmp) / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["wandb_status"], "recovered_with_local_evidence")
            self.assertTrue(m.had_wandb_error)

    def test_cached_tokens_and_retries_preserve_observed_costs(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.verifiable.telemetry.subprocess.run", side_effect=FileNotFoundError):
            for i in range(2):
                with telemetry.Monitor(tmp, "collect") as m:
                    m.usage("advisor", {"prompt_tokens": 10, "completion_tokens": 5,
                                        "actual_prompt_tokens": 0 if i else 10,
                                        "actual_completion_tokens": 0 if i else 5, "cache_hit": bool(i)})
            self.assertEqual(m.totals["usage/advisor/generated_tokens"], 5)
            self.assertEqual(m.totals["usage/advisor/cache_hits"], 1)

    def test_callback_forwards_loss_and_reward_without_transformers(self):
        fake_transformers = SimpleNamespace(TrainerCallback=object)
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {"transformers": fake_transformers}), \
             patch("src.verifiable.telemetry.subprocess.run", side_effect=FileNotFoundError):
            with telemetry.Monitor(tmp, "rl") as m:
                cb = telemetry.training_callback(tmp)
                cb.on_log(None, SimpleNamespace(global_step=3, epoch=.5), None,
                          logs={"loss": .2, "reward": .75, "kl": .01})
            self.assertTrue(any(r.get("train/reward") == .75 and r["trainer_step"] == 3
                                for r in m.tracker.run.history))

    def test_diagnostic_internalization_uses_fixed_initial_rescue_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.loop(Path(tmp))
            before = [{"question_hash": "a", "direct_correct": False, "branches": [{"correct": True}]},
                      {"question_hash": "b", "direct_correct": True, "branches": [{"correct": True}]}]
            after = [{"question_hash": "a", "direct_correct": True, "branches": [{"correct": True}]},
                     {"question_hash": "b", "direct_correct": False, "branches": [{"correct": True}]}]
            initial, current = root / "initial_dev", root / "round_1/sft_dev"
            initial.mkdir()
            current.mkdir(parents=True)
            write_jsonl(str(initial / "records.jsonl"), before)
            write_jsonl(str(current / "records.jsonl"), after)
            write_json(str(current / "summary.json"), {"independent_accuracy": .5})
            with patch("src.verifiable.runner.metrics") as emit:
                log_diagnostic(root, {"output": str(current)}, 1)
            result = next(call.args[0] for call in emit.call_args_list if call.args[1] == "internalization")
            self.assertEqual(result["rescued_now_independent_rate"], 1.)
            self.assertEqual(result["newly_solved_n"], 1)
            self.assertEqual(result["regressed_n"], 1)


if __name__ == "__main__":
    unittest.main()
