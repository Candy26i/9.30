from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext

from src.benchmarks.base import StandardRow
from src.utils.io import write_jsonl
from src.verifiable.answers import correct, equivalent, extract_final
from src.verifiable.backend import HFBackend, strip_generation_endings
from src.verifiable.data import identity, normalize, partition, prepare, verify_manifest
from src.verifiable.experiment import collect_one, compare, policy_rollout, sft_rows, summary
from src.verifiable.protocol import advisor_messages, messages, parse_calls
from src.verifiable.runner import build_plan, run_data
from src.verifiable.training import make_environment, reward_function, tokenize_turn

CFG = {"base_model": "fake", "advisor_url": "http://fake", "seed": 42, "max_depth": 2,
       "max_new_tokens": 100, "advisor_max_tokens": 50, "max_context": 10000, "max_seq_len": 10000}


def row(index=0, answer="42", split="train"):
    return StandardRow(index, "math", "free_response_math", f"Compute a value, problem {index}.", {}, answer,
                       metadata={"answer_type": "math"}, split=split)


class Backend:
    def __init__(self, initial="0"):
        self.initial = initial
        self.inputs = []

    def generate(self, history, tools=None, **kwargs):
        self.inputs.append(deepcopy(history))
        helped = any(m.get("role") == "tool" and m.get("name") == "reasoner_tool" for m in history)
        if tools and helped:
            text = "COMMIT"
        elif tools:
            text = '<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call>'
        else:
            text = "A full derivation with intermediate steps.\nFINAL_ANSWER: \\boxed{" + ("42" if helped else self.initial) + "}"
        return {"text": text, "prompt_tokens": 10, "completion_tokens": 20,
                "truncated": False, "seconds": 0.001}


class Advisors:
    def __init__(self):
        self.calls = []

    def call(self, kind, problem, draft=""):
        self.calls.append((kind, problem.question, draft))
        return {"text": "Useful advice", "prompt_tokens": 5, "completion_tokens": 10,
                "seconds": 0.001, "truncated": False}


class GenerationEndingsTest(unittest.TestCase):
    def test_repeated_eos_and_mixed_padding(self):
        answer = "Reasoning.\nFINAL_ANSWER: \\boxed{25}"
        for pad in ("<|im_end|>", "<|endoftext|>", None):
            tok = SimpleNamespace(eos_token="<|im_end|>", pad_token=pad)
            for suffix in ("<|im_end|>", "<|im_end|>" * 3,
                           "<|im_end|> \n" + (pad or "") * 2):
                with self.subTest(pad=pad, suffix=suffix):
                    text = strip_generation_endings(answer + suffix, tok)
                    self.assertEqual(text, answer)
                    self.assertTrue(correct(text, "25"))

    def test_tools_and_commit_preserved(self):
        tok = SimpleNamespace(eos_token="<|im_end|>", pad_token="<|im_end|>")
        call = '<tool_call>{"name":"verifier_tool","arguments":{}}</tool_call>'
        clean = strip_generation_endings(call + "<|im_end|>" * 2, tok)
        content, calls = parse_calls(clean)
        self.assertEqual(content, "")
        self.assertEqual(calls[0]["name"], "verifier_tool")
        self.assertEqual(strip_generation_endings("COMMIT<|im_end|>", tok), "COMMIT")

    def test_does_not_relax_answer_validation(self):
        tok = SimpleNamespace(eos_token="<|im_end|>", pad_token=None)
        for bad in (r"FINAL_ANSWER: \boxed{25} or 26", r"FINAL_ANSWER: \boxed{24}",
                    "FINAL_ANSWER: \\boxed{25}\nI disagree",
                    "FINAL_ANSWER: \\boxed{24}\nFINAL_ANSWER: \\boxed{25}"):
            self.assertFalse(correct(strip_generation_endings(bad + "<|im_end|>", tok), "25"))
        interior = "Quoted <|im_end|> marker\nFINAL_ANSWER: \\boxed{25}"
        self.assertEqual(strip_generation_endings(interior, tok), interior)

    def test_generate_passes_chatml_eos_and_retains_truncation(self):
        class Inputs(dict):
            def to(self, device):
                return self
        class Output:
            def __init__(self, ids):
                self.ids = ids
            def __getitem__(self, key):
                return self.ids
        class Tokenizer:
            eos_token, pad_token = "<|im_end|>", "<|endoftext|>"
            eos_token_id, pad_token_id = 7, 8
            def apply_chat_template(self, *args, **kwargs):
                return "prompt"
            def __call__(self, *args, **kwargs):
                return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))
            def decode(self, ids, **kwargs):
                return "FINAL_ANSWER: \\boxed{25}" + ("<|im_end|>" if ids[-1] == 7 else "")
        class Model:
            device = SimpleNamespace(type="cpu")
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return Output(self.ids)
        backend = HFBackend.__new__(HFBackend)
        backend.tokenizer, backend.model, backend.max_context = Tokenizer(), Model(), 100
        seeds = []
        fake_torch = SimpleNamespace(random=SimpleNamespace(fork_rng=lambda devices: nullcontext()),
                                     inference_mode=nullcontext, manual_seed=seeds.append)
        sampled = {"temperature": 0.7, "seed": 43, "top_p": 0.8, "top_k": 20,
                   "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
        with patch.dict("sys.modules", {"torch": fake_torch,
                                        "transformers": SimpleNamespace(LogitsProcessorList=list)}):
            for ids, truncated, settings in (([1, 7], False, None), ([1, 2], True, None),
                                             ([1, 7], False, sampled)):
                backend.model.ids = ids
                result = backend.generate([], max_tokens=2, generation_options=settings)
                self.assertEqual(backend.model.kwargs["eos_token_id"], 7)
                self.assertEqual(result["truncated"], truncated)
                self.assertEqual(result["completion_tokens"], 2)
                self.assertEqual(result["text"], r"FINAL_ANSWER: \boxed{25}")
                self.assertEqual(backend.model.kwargs["do_sample"], settings is not None)
                if settings:
                    self.assertEqual(seeds[-1], 43)
                    self.assertEqual(backend.model.kwargs["temperature"], 0.7)
                    self.assertEqual(backend.model.kwargs["top_p"], 0.8)
                    self.assertEqual(backend.model.kwargs["top_k"], 20)
                    self.assertEqual(len(backend.model.kwargs["logits_processor"]), 1)


class AnswersTest(unittest.TestCase):
    def test_no_incidental_or_multiple_answers(self):
        for text in ["The intermediate value is 42", r"\boxed{42}",
                     "FINAL_ANSWER: \\boxed{42}\nI disagree", "FINAL_ANSWER: 42",
                     "FINAL_ANSWER: \\boxed{0}\nFINAL_ANSWER: \\boxed{42}"]:
            self.assertFalse(correct(text, "42"))

    def test_nested_latex_and_equivalence(self):
        self.assertEqual(extract_final(r"FINAL_ANSWER: \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertTrue(correct(r"FINAL_ANSWER: \boxed{\frac{1}{2}}", "0.5"))
        self.assertTrue(equivalent("042", "42"))
        self.assertFalse(equivalent("41.99999", "42"))
        self.assertFalse(correct(r"FINAL_ANSWER: \boxed{42} or 43", "42"))


class DataTest(unittest.TestCase):
    def test_zero_not_dropped_and_gold_solution_not_copied(self):
        normalized = normalize({"problem": "Find zero", "answer": 0, "solution": "SECRET"}, "aime2026", 0)
        self.assertEqual(normalized.ground_truth, "0")
        self.assertEqual(normalized.split, "test")
        self.assertNotIn("SECRET", json.dumps(normalized.to_dict()))

    def test_numina_filters(self):
        valid = {"problem": "Find a value", "answer": "1", "problem_is_valid": "Yes", "solution_is_valid": "Yes"}
        self.assertIsNotNone(normalize(valid, "numina", 0))
        for change in [{"answer": "proof"}, {"solution_is_valid": "Incomplete"},
                       {"question_type": "proof"}, {"problem": "Use ![](image.png)"}]:
            self.assertIsNone(normalize({**valid, **change}, "numina", 0))

    def test_dedup_and_split(self):
        data = [row(i) for i in range(7)] + [row(0)]
        train, dev, _ = partition(data, {identity(row(1).question)}, 3, 2, 42)
        self.assertFalse({identity(r.question) for r in train} & {identity(r.question) for r in dev})
        self.assertNotIn(identity(row(1).question), {identity(r.question) for r in train + dev})
        self.assertEqual([r.split for r in train + dev], ["train"] * 3 + ["dev"] * 2)
        self.assertEqual(identity("A  B"), identity("a\nb"))

    def test_prepare_locks_data_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {}
            for name in ("numina", "aime2026", "beyondaime"):
                path = root / (name + ".jsonl")
                raw = [{"problem": name + str(i), "answer": "42", "solution": "SECRET",
                        "problem_is_valid": "Yes", "solution_is_valid": "Yes"} for i in range(10)]
                write_jsonl(str(path), raw)
                sources[name] = str(path)
            out = root / "normalized"
            result = prepare(out, 4, 2, local_sources=sources)
            self.assertEqual(result["counts"]["train"], 4)
            self.assertNotIn("SECRET", (out / "train.jsonl").read_text())
            verify_manifest(out)
            (out / "train.jsonl").write_text("corrupted")
            with self.assertRaises(ValueError):
                verify_manifest(out)

    def test_manifest_can_lock_only_aime_as_held_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train", "dev", "aime2026"):
                split = name if name in {"train", "dev"} else "test"
                write_jsonl(str(root / f"{name}.jsonl"), [row(i, split=split).to_dict() | {
                    "question": f"{name} question {i}"} for i in range(3)])
            sha = {f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                   for f in ("train.jsonl", "dev.jsonl", "aime2026.jsonl")}

            def write(**manifest):
                (root / "manifest.json").write_text(json.dumps({"sha256": sha, **manifest}))

            write(test_sets=["aime2026"], counts={"train": 3, "dev": 3, "aime2026": 3})
            self.assertEqual(verify_manifest(root)["test_sets"], ["aime2026"])
            write()  # Default still requires both locked test sets.
            with self.assertRaisesRegex(ValueError, "required split files"):
                verify_manifest(root)
            for bad in ([], ["numina"], ["aime2026", "aime2026"], "aime2026"):
                write(test_sets=bad)
                with self.assertRaisesRegex(ValueError, "test_sets"):
                    verify_manifest(root)
            write_jsonl(str(root / "aime2026.jsonl"), [row(0, split="test").to_dict() | {"question": "train question 0"}])
            sha["aime2026.jsonl"] = hashlib.sha256((root / "aime2026.jsonl").read_bytes()).hexdigest()
            write(test_sets=["aime2026"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                verify_manifest(root)


class CounterfactualTest(unittest.TestCase):
    def test_full_branching_same_root_and_full_solution_export(self):
        backend, advisors = Backend(), Advisors()
        record = collect_one(row(), backend, advisors, CFG, 12, evaluate_policy=True)
        self.assertEqual(len(record["branches"]), 9)
        self.assertEqual(record["preferred_sequence"], ["reasoner"])
        self.assertTrue(record["policy"]["correct"])
        for kind, _, draft in advisors.calls:
            if kind == "verifier":
                self.assertIn("full derivation", draft)
        for branch in record["branches"]:
            self.assertEqual(branch["steps"][0]["prompt"], record["base_messages"])
        turns = sft_rows([record])
        self.assertEqual([r["decision_type"] for r in turns], ["call", "revision", "commit", "independent_solution"])
        self.assertIn("intermediate steps", turns[-1]["response"][0]["content"])
        self.assertNotIn("Useful advice", json.dumps(turns[-1]["prompt"]))
        self.assertEqual(summary([record])["measured_union_coverage"], 1)

    def test_direct_correct_still_probes_and_no_heldout_training(self):
        record = collect_one(row(split="dev"), Backend("42"), Advisors(), CFG, 12)
        self.assertEqual(record["preferred_sequence"], [])
        self.assertEqual(len(record["branches"]), 9)
        with self.assertRaises(ValueError):
            sft_rows([record])

    def test_labels_do_not_change_runtime_inputs(self):
        a, b = Backend(), Backend()
        collect_one(row(answer="42"), a, Advisors(), CFG, 12)
        collect_one(row(answer="999"), b, Advisors(), CFG, 12)
        self.assertEqual(a.inputs, b.inputs)
        self.assertEqual(messages(row(answer="42")), messages(row(answer="999")))
        self.assertEqual(advisor_messages("reasoner", row(answer="42"), ""),
                         advisor_messages("reasoner", row(answer="999"), ""))

    def test_resume_and_setting_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "data.jsonl"
            write_jsonl(str(source), [row().to_dict()])
            out = str(Path(tmp) / "run")
            run_data(CFG, str(source), "fake", out, "collect", backend=Backend(), advisors=Advisors())
            backend = Backend()
            run_data(CFG, str(source), "fake", out, "collect", resume=True, backend=backend, advisors=Advisors())
            self.assertFalse(backend.inputs)
            with self.assertRaises(ValueError):
                run_data({**CFG, "max_depth": 1}, str(source), "fake", out, "collect", resume=True)

    def test_policy_has_no_oracle_feedback(self):
        result = policy_rollout(row(answer="999"), Backend(), Advisors(), CFG, 12)
        self.assertFalse(result["correct"])
        self.assertEqual(result["sequence"], ["reasoner"])
        self.assertNotIn("999", json.dumps(result["history"]))

    def test_comparison_and_no_fake_coverage(self):
        before = collect_one(row(), Backend(), Advisors(), CFG, 12)
        after = collect_one(row(), Backend("42"), Advisors(), CFG, 12)
        diff = compare([before], [after])
        self.assertEqual(len(diff["previously_rescued_now_independent"]), 1)
        with self.assertRaises(ValueError):
            compare([before], [])
        final_only = {"direct_correct": False, "policy": {"correct": True, "calls": 1, "valid": True}}
        self.assertNotIn("delegation_search_coverage", summary([final_only]))


class ProtocolTest(unittest.TestCase):
    def test_json_subagent_calls(self):
        for text in ['<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call>',
                     '<tool_call>{"name":"verifier_tool","arguments":{}}</tool_call>']:
            _, calls = parse_calls(text)
            self.assertEqual(len(calls), 1)
        for text in ['<tool_call>{bad}</tool_call>', '<tool_call>{"name":"wrong","arguments":{}}</tool_call>', '<tool_call>']:
            with self.assertRaises(ValueError):
                parse_calls(text)

    def test_rl_tool_budget_and_binary_reward(self):
        env = make_environment([row()], Advisors(), 1)()
        env.reset(0)
        env.reasoner_tool()
        with tempfile.TemporaryDirectory() as tmp:
            reward = reward_function(str(Path(tmp) / "trace.jsonl"))
            completions = [[{"role": "assistant", "content": r"FINAL_ANSWER: \boxed{42}"}]]
            self.assertEqual(reward(completions, ["42"], [env]), [1.0])
            env.reasoner_tool()
            self.assertEqual(reward(completions, ["42"], [env]), [0.0])

    def test_two_round_plan_continues_weights_and_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "cfg.json"
            config.write_text(json.dumps(CFG))
            plan = build_plan(config, tmp, str(Path(tmp) / "run"), "dynamic_sft", 2)
            collects = [p for p in plan if p["stage"] == "collect"]
            self.assertEqual(len(collects), 2)
            self.assertIn(str((Path(tmp) / "run/round_1/sft").resolve()), collects[1]["command"])
            fixed = build_plan(config, tmp, str(Path(tmp) / "fixed"), "static_sft", 2)
            self.assertEqual(sum(p["stage"] == "collect" for p in fixed), 1)
            self.assertFalse(any("aime2026.jsonl" in str(p) for p in plan))


if __name__ == "__main__":
    unittest.main()
