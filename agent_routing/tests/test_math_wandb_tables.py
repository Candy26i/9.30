"""Text diagnostics retain failed generations without changing experiment labels."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_math_wandb import FakeWandb
from test_verifiable import Advisors, Backend, CFG, row
from src.utils.io import write_json, write_jsonl
from src.verifiable import telemetry
from src.verifiable.backend import HTTPAdvisors
from src.verifiable.debug_records import generation_values, upload_records
from src.verifiable.runner import run_data
from src.verifiable.wandb_tracking import WandbTracker


class FakeTable:
    def __init__(self, columns, log_mode):
        self.columns, self.log_mode, self.data = columns, log_mode, []

    def add_data(self, *data):
        self.data.append(list(data))


class TextTablesTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeWandb()
        self.fake.Table = FakeTable
        env = patch.dict("os.environ", {"MARGENT_WANDB_MODE": "online", "MARGENT_WANDB_TEXT": "1",
                        "WANDB_ENTITY": "test", "WANDB_PROJECT": "project",
                        "MARGENT_WANDB_TABLE_MAX_ROWS": "10000", "MARGENT_WANDB_TABLE_MAX_CHARS": "20000"})
        sdk = patch("src.verifiable.wandb_tracking.import_module", return_value=self.fake)
        gpu = patch("src.verifiable.telemetry.subprocess.run", side_effect=FileNotFoundError)
        for mock in (env, sdk, gpu):
            mock.start()
            self.addCleanup(mock.stop)

    def tables(self, tracker, kind):
        return [t for k, t in tracker.tables.items() if k.startswith("debug/" + kind + "s_")]

    def test_text_off_remains_scalar_only_and_keeps_full_local_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"MARGENT_WANDB_TEXT": "0"}):
            with telemetry.Monitor(tmp, "diagnose") as m:
                m.set_question(row())
                m.generation("manager", {"text": "PRIVATE_TEXT", "completion_tokens": 20, "seconds": 2})
            self.assertFalse(m.tracker.tables)
            self.assertNotIn("PRIVATE_TEXT", json.dumps(m.tracker.run.history))
            self.assertIn("PRIVATE_TEXT", (Path(tmp) / "generations.jsonl").read_text())

    def test_truncated_advisor_text_is_saved_without_failing_the_stage(self):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "choices": [{"message": {"content": "unfinished derivation"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 2048}})
        with tempfile.TemporaryDirectory() as tmp, patch("requests.post", return_value=response) as post:
            with telemetry.Monitor(tmp, "diagnose") as m:
                telemetry.question_context(row(answer="GOLD_ONLY_IN_LOGS"))
                telemetry.progress(phase="counterfactual", sequence=["reasoner", "verifier"])
                result = HTTPAdvisors("http://fake", max_tokens=2048).call("verifier", row(), "candidate")
                self.assertEqual(result["error"], "advisor_output_truncated")
            record = json.loads((Path(tmp) / "generations.jsonl").read_text())
            self.assertEqual(record["text"], "unfinished derivation")
            self.assertEqual(record["error"], "advisor_output_truncated")
            self.assertTrue(record["truncated"])
            self.assertEqual(record["sequence"], ["reasoner", "verifier"])
            self.assertNotIn("GOLD_ONLY_IN_LOGS", json.dumps(post.call_args.kwargs["json"]))
            table = self.tables(m.tracker, "generation")[0]
            saved = dict(zip(table.columns, table.data[0]))
            self.assertEqual(saved["ground_truth"], "GOLD_ONLY_IN_LOGS")
            self.assertEqual(saved["advisor"], "verifier")
            self.assertEqual(saved["text"], "unfinished derivation")
            self.assertEqual(table.log_mode, "INCREMENTAL")
            self.assertTrue(any(next(iter(m.tracker.tables)) in entry for entry in m.tracker.run.history))
            self.assertEqual(m.tracker.run.exit_code, 0)
            self.assertFalse((Path(tmp) / "records.jsonl").exists())

    def test_text_upload_failure_preserves_original_error_and_local_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "original generation failure"):
                with telemetry.Monitor(tmp, "diagnose") as m:
                    with patch.object(m.tracker.run, "log", side_effect=RuntimeError("upload failed")):
                        m.generation("advisor", {"text": "truncated text", "truncated": True})
                    raise RuntimeError("original generation failure")
            self.assertTrue(m.had_wandb_error)
            self.assertFalse(m.wandb_failed)
            self.assertIn("truncated text", (Path(tmp) / "generations.jsonl").read_text())
            self.assertEqual(m.tracker.run.exit_code, 1)

    def test_completed_question_has_saved_grades_and_never_changes_labels_or_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "dev.jsonl"
            write_jsonl(str(data), [row(split="dev").to_dict()])
            backend = Backend()
            run_data(CFG, str(data), "fake", str(Path(tmp) / "result"), "diagnose",
                     backend=backend, advisors=Advisors())
            record = json.loads((Path(tmp) / "result/records.jsonl").read_text())
            question_tables = [value for entry in self.fake.runs[-1].history for key, value in entry.items()
                               if key.startswith("debug/questions_")]
            table = question_tables[-1]
            saved = dict(zip(table.columns, table.data[0]))
            self.assertEqual(saved["question"], row().question)
            self.assertEqual(saved["direct_correct"], record["direct_correct"])
            self.assertEqual(saved["policy_correct"], record["policy"]["correct"])
            self.assertEqual(saved["successful_branches"], sum(b["correct"] for b in record["branches"]))
            journal = [json.loads(line) for line in (Path(tmp) / "result/generations.jsonl").read_text().splitlines()]
            self.assertEqual(len(journal), len(backend.inputs))
            self.assertEqual(journal[0]["phase"], "independent")
            self.assertEqual(journal[0]["sequence"], [])
            self.assertEqual(journal[-1]["phase"], "self_continue")

    def test_table_limits_mark_clipping_and_preserve_last_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {
                "MARGENT_WANDB_TABLE_MAX_ROWS": "1", "MARGENT_WANDB_TABLE_MAX_CHARS": "256"}):
            with telemetry.Monitor(tmp, "diagnose") as m:
                for text in ("START" + "x" * 1000 + "END", "second output"):
                    m.generation("manager", {"text": text})
            table = self.tables(m.tracker, "generation")[0]
            record = dict(zip(table.columns, table.data[0]))
            self.assertEqual(len(table.data), 1)
            self.assertEqual(len(record["text"]), 256)
            self.assertTrue(record["text"].startswith("START"))
            self.assertTrue(record["text"].endswith("END"))
            self.assertIn("text", record["clipped_fields"])
            self.assertEqual(len((Path(tmp) / "generations.jsonl").read_text().splitlines()), 2)
            self.assertEqual(next(v for k, v in m.tracker.run.summary.items() if k.endswith("_omitted_rows")), 1)

    def test_resume_uses_same_stage_run_but_separate_table_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            keys = []
            for attempt in ("first", "second"):
                t = WandbTracker(tmp, "diagnose", attempt)
                t.start()
                t.log_text("generation", {"text": attempt})
                t.finish("completed")
                keys.append(set(t.tables))
            self.assertEqual(self.fake.calls[0]["id"], self.fake.calls[1]["id"])
            self.assertFalse(keys[0] & keys[1])

    def test_cache_hit_has_zero_actual_tokens_and_no_invented_rate(self):
        values = generation_values({"completion_tokens": 100, "actual_completion_tokens": 0,
                                    "cache_hit": True, "seconds": 0})
        self.assertEqual(values["actual_completion_tokens"], 0)
        self.assertIsNone(values["tokens_per_second"])

    def test_backfill_old_records_and_failed_generation_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "initial_dev"
            source.mkdir()
            record = {"question_hash": "old", "independent_prompt": [{"role": "user", "content": "Compute 9+10."}],
                      "ground_truth": "19", "direct_text": "bad", "direct_correct": False,
                      "costs": [{"role": "manager", "text": "bad", "completion_tokens": 3, "seconds": 1}]}
            write_jsonl(str(source / "records.jsonl"), [record])
            write_jsonl(str(source / "generations.jsonl"), [{"question_hash": "failed", "text": "unfinished",
                        "role": "advisor", "truncated": True, "error": "advisor_output_truncated"}])
            write_json(str(source / "run.json"), {"config": CFG, "harness": {"source_sha256": "original"}})
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            with patch("src.verifiable.backend.load_model", side_effect=AssertionError("must not load a model")):
                result = upload_records(source, Path(tmp) / "review")
            self.assertEqual(result["text_upload_status"], "submitted")
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            review = json.loads((Path(tmp) / "review/run.json").read_text())
            self.assertEqual(review["config"]["original_harness"]["source_sha256"], "original")
            tables = [value for entry in self.fake.runs[-1].history for key, value in entry.items()
                      if key.startswith("debug/generations_")]
            rows = [dict(zip(tables[-1].columns, r)) for r in tables[-1].data]
            self.assertEqual([r["source"] for r in rows], ["legacy_costs", "saved_generations"])
            self.assertEqual(rows[0]["question"], "Compute 9+10.")
            self.assertTrue(rows[1]["truncated"])

    def test_backfill_requires_opt_in_and_never_overwrites_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "stage"
            source.mkdir()
            write_jsonl(str(source / "records.jsonl"), [{"question_hash": "one"}])
            with self.assertRaisesRegex(ValueError, "separate review"):
                upload_records(source, source)
            with patch.dict("os.environ", {"MARGENT_WANDB_TEXT": "0"}):
                with self.assertRaisesRegex(ValueError, "MARGENT_WANDB_TEXT"):
                    upload_records(source, Path(tmp) / "review")


@unittest.skipUnless(importlib.util.find_spec("wandb"), "W&B SDK is not installed")
class RealOfflineTableTest(unittest.TestCase):
    def test_cli_reviews_saved_records_in_a_fresh_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "old_stage"
            source.mkdir()
            write_jsonl(str(source / "records.jsonl"), [{
                "question_hash": "old-question", "independent_prompt": [{"role": "user", "content": "Compute 9+10."}],
                "ground_truth": "19", "direct_text": "19", "direct_correct": False,
                "costs": [{"role": "manager", "text": "19", "completion_tokens": 2, "seconds": .1}]}])
            result = subprocess.run([sys.executable, "-m", "src.verifiable", "wandb-upload-records",
                                     "--run-dir", str(source), "--out", str(Path(tmp) / "review")],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
                env={**os.environ, "MARGENT_WANDB_MODE": "offline", "MARGENT_WANDB_TEXT": "1",
                     "WANDB_ENTITY": "offline-test", "WANDB_PROJECT": "margent-tables-test", "WANDB_SILENT": "true"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"text_upload_status": "submitted"', result.stdout)
            self.assertTrue(list((Path(tmp) / "review").rglob("*.table.json")))
            self.assertEqual(sorted(p.name for p in source.iterdir()), ["records.jsonl"])

    def test_incremental_sdk_serializes_later_rows_and_flushes_failure(self):
        # Exercise the installed SDK's table type inference and incremental
        # serialization, without credentials, a network connection or a GPU.
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {
                "MARGENT_WANDB_MODE": "offline", "MARGENT_WANDB_TEXT": "1",
                "WANDB_ENTITY": "offline-test", "WANDB_PROJECT": "margent-tables-test",
                "WANDB_SILENT": "true", "MARGENT_WANDB_TABLE_MAX_ROWS": "3",
                "MARGENT_WANDB_TABLE_MAX_CHARS": "20000"}):
            tracker = WandbTracker(tmp, "diagnose", "offline-attempt")
            tracker.start()
            try:
                tracker.log_text("generation", {"role": "manager", "text": "first",
                                 "seconds": 0, "completion_tokens": 0})
                tracker.log_text("generation", {"role": "advisor", "advisor": "verifier", "text": "second",
                                 "seconds": 2.5, "completion_tokens": 10,
                                 "messages": [{"role": "user", "content": "question"}]})
                tracker.flush_tables(force=True)
                tracker.log_text("generation", {"role": "advisor", "text": "last truncated output",
                                 "truncated": True, "error": "advisor_output_truncated"})
                tracker.log_text("generation", {"role": "manager", "text": "over display limit"})
                tracker.log_text("question", {"question_hash": "one", "ground_truth": "19",
                                 "direct_text": "answer", "direct_correct": True})
            finally:
                tracker.finish("failed")
            generations, questions = [], []
            for path in Path(tmp).rglob("*.table.json"):
                saved = json.loads(path.read_text())
                rows = [dict(zip(saved["columns"], r)) for r in saved["data"]]
                (generations if "role" in saved["columns"] else questions).extend(rows)
            self.assertEqual([r["text"] for r in sorted(generations, key=lambda r: r["text"])],
                             ["first", "last truncated output", "second"])
            self.assertEqual(questions[0]["ground_truth"], "19")
            self.assertTrue(next(r for r in generations if r["text"] == "last truncated output")["truncated"])


if __name__ == "__main__":
    unittest.main()
