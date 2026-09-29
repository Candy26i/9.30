"""Protocol-v2 LoRA SFT, with an inactive legacy GRPO implementation."""
from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import os
import json
import time

from ..utils.io import append_jsonl, read_jsonl, write_json
from .answers import correct, extract_final
from .backend import HTTPAdvisors, load_model, render
from .data import identity, load_rows
from .experiment import root_state
from .protocol import KINDS, tool_schemas
from .telemetry import Monitor, progress, usage, training_callback, metrics


def tokenize_turn(row, tok, max_seq_len):
    schemas = tool_schemas() if row.get("decision_type") in {"call", "commit"} else None
    prompt = render(tok, row["prompt"], schemas)
    full = render(tok, row["prompt"] + row["response"], schemas, generation=False)
    if not full.startswith(prompt):
        raise ValueError("Chat template is not prefix preserving; refusing incorrect loss masking")
    # Keep exactly the tokens supplied at inference. Joint tokenization can merge
    # the prompt's trailing whitespace with the response's leading whitespace
    # (notably Qwen3.5 tool calls), changing a prompt token into a target token.
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
    ids = pids + target_ids
    if len(ids) > max_seq_len:
        return None  # drop and report; do not train on truncated solutions
    if not pids or not target_ids:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(pids) + target_ids}


def _training_output(path, config, checkpoint, data_path, stage):
    from transformers.trainer_utils import get_last_checkpoint
    from .runner import checkpoint_identity
    from .provenance import harness_identity
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    signature = {"harness": harness_identity(), "config": config, "checkpoint": checkpoint_identity(checkpoint), "stage": stage,
                 "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()}
    manifest = output / "training_run.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError("Training settings or data changed; choose a new output directory")
    elif any(output.iterdir()):
        raise FileExistsError("Training directory has no matching run manifest")
    else:
        write_json(str(manifest), signature)
    complete = ((output / "training_metrics.json").exists() and (output / "adapter_config.json").exists()
                and any((output / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin")))
    return complete, get_last_checkpoint(str(output))


def train_sft(config, checkpoint, data_path, output):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Paper SFT runs use one Manager training process; do not use torchrun")
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    if config.get("sft_max_steps", -1) <= 0:
        raise ValueError("Set a positive sft_max_steps shared across comparison arms")
    finished, resume_checkpoint = _training_output(output, config, checkpoint, data_path, "sft")
    if finished:
        from .runner import validate_stage_artifacts
        validate_stage_artifacts(output, "sft")
        return
    with Monitor(output, "sft") as monitor:
        set_seed(config["seed"])
        tok, model = load_model(config["base_model"], checkpoint, trainable=True,
                                lora_rank=config.get("lora_rank", 16), revision=config.get("base_model_revision"))
        monitor.summary({"training_algorithm": "manager_response_only_sft", "source_checkpoint": checkpoint,
            "resumed_checkpoint": resume_checkpoint, "base_model": config["base_model"],
            "base_model_revision": config.get("base_model_revision"),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters())})
        progress(phase="preparing_sft_data")
        rows = read_jsonl(data_path)
        if any(r.get("split") != "train" or r.get("protocol_version") != 2 for r in rows):
            raise ValueError("SFT requires protocol-v2 train-only rows exported by collect")
        if any(not any(str(m.get("content") or "").strip() or m.get("tool_calls")
                       for m in row.get("response", [])) for row in rows):
            raise ValueError("SFT rows must contain a nonempty assistant target")
        features = [tokenize_turn(row, tok, config["max_seq_len"]) for row in rows]
        kept = [f for f in features if f is not None]
        if len(kept) != len(features):
            raise ValueError("SFT targets exceed max_seq_len or have no supervised tokens; fix the pilot configuration instead of dropping examples")
        if not kept:
            raise ValueError("No successful SFT targets")
        report = {"input_turns": len(rows), "kept_turns": len(kept), "dropped_turns": len(rows) - len(kept),
                  "input_tokens_per_epoch": sum(len(f["input_ids"]) for f in kept),
                  "supervised_tokens_per_epoch": sum(sum(y != -100 for y in f["labels"]) for f in kept),
                  "checkpoint": checkpoint, "data": data_path,
                  "optimizer_step_budget": config["sft_max_steps"],
                  "budget_scope": "matched optimizer updates and accumulation; actual tokens reported, not equal FLOPs"}
        write_json(str(Path(output) / "sft_data_report.json"), report)
        metrics(report, "sft_data")
        print(report, flush=True)
        args = TrainingArguments(output_dir=output, per_device_train_batch_size=1,
            gradient_accumulation_steps=config.get("sft_accumulation", 8),
            learning_rate=config.get("sft_learning_rate", 2e-5),
            num_train_epochs=config.get("sft_epochs", 1),
            max_steps=config.get("sft_max_steps", -1), bf16=torch.cuda.is_available(),
            # load_model explicitly supports CUDA or CPU. Keep Trainer on the
            # same device instead of implicitly selecting MPS on macOS tests.
            use_cpu=not torch.cuda.is_available(),
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1, save_strategy="steps", save_steps=config.get("save_steps", 10),
            # W&B is owned by Monitor; avoid a second Trainer integration/run.
            save_total_limit=2, report_to=[], remove_unused_columns=False,
            seed=config["seed"])
        class LoggedSFTTrainer(Trainer):
            def training_step(self, model, inputs, num_items_in_batch=None):
                loss = super().training_step(model, inputs, num_items_in_batch)
                usage("sft_train", {}, input_tokens=int(inputs["attention_mask"].sum().item()),
                      supervised_tokens=int((inputs["labels"] != -100).sum().item()), step=self.state.global_step)
                return loss

        trainer = LoggedSFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(kept),
                          callbacks=[training_callback(output)],
                          data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
        start = time.monotonic()
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        metrics(result.metrics, "train", trainer_step=trainer.state.global_step)
        trainer.save_model(output)
        tok.save_pretrained(output)
        write_json(str(Path(output) / "training_metrics.json"),
                   {**result.metrics, "optimizer_steps": trainer.state.global_step, "wall_seconds": time.monotonic() - start, "config": config})


def make_environment(rows, advisors, max_calls):
    index = {r.example_id: r for r in rows}

    class MathEnvironment:
        def reset(self, example_id: int, **kwargs):
            self.row = index[int(example_id)]
            self.used = []
            self.valid = True
            self.error = None
            self.usage = []
            return None

        def _call(self, kind, draft=""):
            if kind in self.used or len(self.used) >= max_calls:
                self.valid = False
                return '{"error": "advisor repeat or call budget exceeded"}'
            self.used.append(kind)
            try:
                response = advisors.call(kind, self.row, draft)
            except Exception as exc:
                self.error = str(exc)
                self.valid = False
                raise
            self.usage.append(response)
            return response["text"]

        def extractor_tool(self) -> str:
            """Extract the stated facts and constraints.

            Returns:
                Advice on the current problem.
            """
            return self._call("extractor")

        def reasoner_tool(self) -> str:
            """Suggest a solution approach using relevant principles.

            Returns:
                Advice on the current problem.
            """
            return self._call("reasoner")

        def verifier_tool(self, current_draft: str) -> str:
            """Audit the supplied reasoning for errors.

            Args:
                current_draft: Your full current reasoning, including candidate answer.

            Returns:
                An audit of the supplied derivation.
            """
            if not current_draft.strip():
                self.valid = False
                return '{"error": "missing derivation"}'
            return self._call("verifier", current_draft)

    return MathEnvironment


def reward_function(trace_path):
    def reward(completions, ground_truth, environments, question_hash=None, **kwargs):
        values, records = [], []
        batch = time.time_ns()
        state = kwargs.get("trainer_state")
        for i, (completion, gold, env) in enumerate(zip(completions, ground_truth, environments)):
            if env.error:
                raise RuntimeError(f"Advisor infrastructure failed; refusing to turn this into a math reward: {env.error}")
            turns = [m for m in completion if m.get("role") == "assistant"] if isinstance(completion, list) else []
            text = str(turns[-1].get("content") or "") if turns else str(completion or "")
            valid = env.valid and (not turns or not turns[-1].get("tool_calls"))
            # Multiple calls in one assistant turn violate the collection protocol.
            valid = valid and all(len(m.get("tool_calls") or []) <= 1 for m in turns)
            valid = valid and all(not (m.get("tool_calls") and "FINAL_ANSWER:" in str(m.get("content")))
                                  for m in turns)
            valid = valid and "<tool_call" not in text
            value = float(valid and correct(text, str(gold)))
            values.append(value)
            records.append({"question_hash": question_hash[i] if question_hash else None,
                "correct": bool(value), "valid_answer": extract_final(text) is not None,
                "protocol_valid": bool(valid), "reward_batch": batch,
                "step": getattr(state, "global_step", None), "completion": completion,
                "ground_truth": str(gold), "final_answer": extract_final(text),
                "completion_tokens_including_tools": len(kwargs["completion_ids"][i]) if "completion_ids" in kwargs else None,
                "calls": len(env.used), "sequence": env.used, "advisor_usage": env.usage})
        append_jsonl(trace_path, records)
        return values
    reward.__name__ = "verified_math_terminal_correctness"
    return reward


def train_rl(config, checkpoint, data_path, output):
    raise NotImplementedError("The legacy GRPO environment does not implement protocol-v2 immutable COMMIT. Use dynamic_sft, success_sft or static_sft for the paper experiment.")


def _legacy_train_rl(config, checkpoint, data_path, output):
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    from .backend import HFBackend

    if "environment_factory" not in inspect.signature(GRPOTrainer.__init__).parameters:
        raise RuntimeError("TRL with environment_factory is required; use requirements-math.txt")
    finished, resume_checkpoint = _training_output(output, config, checkpoint, data_path, "rl")
    if finished:
        return
    with Monitor(output, "rl"):
        set_seed(config["seed"])
        rows = load_rows(data_path, required_split="train")
        tok, model = load_model(config["base_model"], checkpoint, trainable=True,
                                lora_rank=config.get("lora_rank", 16), revision=config.get("base_model_revision"))
        backend = HFBackend.__new__(HFBackend)
        backend.tokenizer, backend.model, backend.max_context = tok, model, config["max_context"]
        model.eval()
        prompts, root_costs = [], []
        root_file = Path(output) / "rl_prompts.jsonl"
        if root_file.exists():
            prompts = read_jsonl(str(root_file))
            if [x["question_hash"] for x in prompts] != [identity(r.question) for r in rows[:len(prompts)]]:
                raise ValueError("RL draft cache does not match training rows")
        for i, row in enumerate(rows[len(prompts):], start=len(prompts)):
            progress(phase="rl_root_drafts", completed_examples=i, total_examples=len(rows))
            root, history = root_state(row, backend, config, config["seed"] + i * 1000)
            rec = {"prompt": history, "ground_truth": row.ground_truth,
                   "example_id": row.example_id, "question_hash": identity(row.question)}
            append_jsonl(str(root_file), [rec])
            prompts.append(rec)
            root_costs.append({k: root[k] for k in ("prompt_tokens", "completion_tokens", "seconds", "truncated")})
        append_jsonl(str(Path(output) / "root_generation_usage.jsonl"), root_costs)
        model.train()
        advisors = HTTPAdvisors(config["advisor_url"], config["advisor_max_tokens"], config.get("advisor_models"),
                               generation_options=config.get("advisor_generation"))
        count = config.get("num_generations", 4)
        accumulation = config.get("rl_accumulation", count)
        if accumulation % count:
            raise ValueError("On one training GPU, rl_accumulation must be divisible by num_generations")
        args = GRPOConfig(output_dir=output, per_device_train_batch_size=1,
            gradient_accumulation_steps=accumulation, num_generations=count,
            max_completion_length=config.get("rl_max_completion_length", 4096),
            max_tool_calling_iterations=config["max_depth"],
            temperature=config.get("rl_temperature", 0.8), beta=config.get("rl_beta", 0.01),
            learning_rate=config.get("rl_learning_rate", 1e-6),
            max_steps=config.get("rl_max_steps", 50), bf16=torch.cuda.is_available(),
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            chat_template_kwargs={"enable_thinking": False}, use_vllm=False,
            remove_unused_columns=False, report_to=[], logging_steps=1, save_strategy="steps",
            save_steps=config.get("save_steps", 10), save_total_limit=2,
            mask_truncated_completions=True, seed=config["seed"])
        class LoggedGRPOTrainer(GRPOTrainer):
            def _generate_single_turn(self, prompts):
                progress(phase="rl_generation", step=self.state.global_step, max_steps=self.state.max_steps)
                started = time.monotonic()
                result = super()._generate_single_turn(prompts)
                # TRL 0.29 returns actual model-generated tokens for this turn;
                # tool response tokens are not counted as generated completions.
                prompt_ids, completion_ids, _, _ = result
                usage("manager_rl", {"prompt_tokens": sum(map(len, prompt_ids)),
                      "completion_tokens": sum(map(len, completion_ids)), "seconds": time.monotonic() - started},
                      step=self.state.global_step, samples=len(completion_ids))
                return result

        trainer = LoggedGRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(prompts),
            callbacks=[training_callback(output)],
            processing_class=tok, reward_funcs=reward_function(str(Path(output) / "rollouts.jsonl")),
            environment_factory=make_environment(rows, advisors, config["max_depth"]))
        start = time.monotonic()
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        metrics(result.metrics, "train", trainer_step=trainer.state.global_step)
        trainer.save_model(output)
        tok.save_pretrained(output)
        write_json(str(Path(output) / "training_metrics.json"), {**result.metrics,
            "wall_seconds": time.monotonic() - start, "config": config,
            "root_policy": "drafts regenerated from the SFT checkpoint once per RL phase; held fixed within phase"})
