"""Single-GPU, multi-turn GRPO for the immutable-COMMIT protocol.

The independent candidate is a shared, greedy environment state. Only sampled
Manager decisions and revisions receive policy gradients; tool tokens never do.
This intentionally small implementation favors auditable pilot runs over speed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from .backend import ContextBudgetExceeded, HFBackend, HTTPAdvisors, load_model, render, strip_generation_endings
from .data import identity, load_rows
from .experiment import policy_rollout, root_state
from .provenance import harness_identity
from .runner import checkpoint_identity, verify_advisor
from .telemetry import Monitor, atomic_json, metrics, progress, question_context, usage, rollout
from .rollout_reporting import reward_metrics, rollout_record


def group_advantages(rewards, epsilon=1e-4):
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("GRPO needs at least two finite rewards per question")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    return [(x - mean) / (std + epsilon) for x in rewards]


def token_objective(logp, old, reference, advantage, clip=0.2, beta=0.01):
    """Unreduced GRPO loss; caller normalizes per trajectory, then per group."""
    import torch
    ratio = (logp - old).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    if beta == 0:
        return -surrogate
    delta = reference - logp
    # k3 estimator, with the standard sampled-action GRPO approximation.
    kl = delta.expm1() - delta
    return -surrogate + beta * kl


def turn_logprobs(model, turn, temperature):
    """Score exact sampled IDs, including EOS; do not re-tokenize decoded text."""
    import torch
    prompt, response = turn["prompt_ids"], turn["completion_ids"]
    if not prompt or not response or temperature <= 0:
        raise ValueError("Nonempty prompt/response and positive RL temperature required")
    ids = torch.tensor([prompt + response], device=model.device)
    # Qwen supports logits_to_keep. Include the position predicting the first
    # response token, but exclude the unused final position in the loss below.
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=len(response) + 1).logits[0, :-1]
    if logits.shape[0] != len(response):
        raise ValueError("Model did not honor logits_to_keep; refusing misaligned loss")
    targets = ids[0, -len(response):]
    if turn.get("action_paths") is not None:
        from .actions import ActionTrie
        trie = ActionTrie(turn["action_paths"])
        if response not in trie.paths:
            raise ValueError("Scoring incomplete/illegal constrained action")
        values = []
        for index, target in enumerate(response):
            allowed = trie.allowed(response[:index])
            selected = logits[index, allowed].float() / temperature
            values.append(selected[allowed.index(target)] - selected.logsumexp(0))
        return torch.stack(values)
    # Chunk vocabulary normalization to reduce peak float32 activation memory.
    chunks = []
    for start in range(0, len(response), 64):
        chunk = logits[start:start + 64].float() / temperature
        target = targets[start:start + 64]
        chunks.append(chunk.gather(-1, target[:, None]).squeeze(-1) - chunk.logsumexp(-1))
    return torch.cat(chunks)


class RolloutBackend(HFBackend):
    def __init__(self, tokenizer, model, config):
        self.tokenizer, self.model = tokenizer, model
        self.max_context = min(config["max_context"], config["max_seq_len"])
        self.rl_temperature = config["rl_temperature"]
        self.decision_constraint = config.get("decision_constraint", "none")
        self.turns = None

    def generate(self, messages, tools=None, max_tokens=2048, temperature=0., seed=42,
                 generation_options=None):
        if self.turns is None:
            return super().generate(messages, tools, max_tokens, 0., seed)
        import torch
        from transformers import GenerationConfig
        prompt = render(self.tokenizer, messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        n = inputs.input_ids.shape[1]
        if n + max_tokens > self.max_context:
            raise ContextBudgetExceeded(n, max_tokens, self.max_context)
        # Fresh config removes model-specific top-k, repetition and forced-token
        # processors. The exact same temperature-softmax is used in the loss.
        gen = GenerationConfig(do_sample=True, temperature=self.rl_temperature,
            top_k=0, top_p=1., typical_p=1., repetition_penalty=1.,
            max_new_tokens=max_tokens, eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id, use_cache=True)
        devices = [self.model.device.index or 0] if self.model.device.type == "cuda" else []
        start = time.monotonic()
        paths, grammar = None, {}
        if tools and self.decision_constraint == "finite_actions_v1":
            from .actions import ActionTrie, decision_paths
            paths = decision_paths(self.tokenizer, messages, tools, max_tokens)
            grammar = ActionTrie(paths).generation_kwargs(n)
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            ids = self.model.generate(**inputs, generation_config=gen, **grammar)[0, n:].tolist()
        if paths is not None and ids not in paths:
            raise ValueError("Constrained decision did not finish a legal action")
        self.turns.append({"prompt_ids": inputs.input_ids[0].tolist(), "completion_ids": ids,
                           "kind": "decision" if tools else "revision",
                           **({"action_paths": paths} if paths is not None else {})})
        result = {"text": strip_generation_endings(self.tokenizer.decode(ids, skip_special_tokens=False), self.tokenizer),
                  "prompt_tokens": n, "completion_tokens": len(ids),
                  "seconds": time.monotonic() - start,
                  "truncated": len(ids) >= max_tokens and ids[-1] != self.tokenizer.eos_token_id}
        usage("manager", result, sampling_temperature=self.rl_temperature)
        return result


def validate_rl_config(config):
    for key in ("rl_max_steps", "num_generations"):
        value = config.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Set positive integer {key}")
    if config["num_generations"] < 2:
        raise ValueError("num_generations must be >= 2")
    for key, default in (("rl_temperature", 0), ("rl_learning_rate", 1e-6),
                         ("rl_max_grad_norm", 1.)):
        value = config.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Set finite positive {key}")
    beta, clip = config.get("rl_beta", .01), config.get("rl_clip", .2)
    if not math.isfinite(beta) or beta < 0 or not math.isfinite(clip) or not 0 < clip < 1:
        raise ValueError("Invalid KL or clipping coefficient")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one Manager GPU and one separate frozen advisor server")


def committed_step_directories(output):
    """Read only optimizer steps committed by the atomic resume pointer.

    A crash can leave a fully written step directory before resume.json is
    advanced. Such an orphan must not count twice in paper metrics on restart.
    """
    root = Path(output)
    pointer = root / "resume.json"
    if not pointer.exists():
        return []
    resume = json.loads(pointer.read_text())
    last = resume["directory"]
    if Path(last).name != last:
        raise ValueError("Invalid GRPO resume directory")
    last_step = json.loads((root / last / "step.json").read_text())["step"]
    if resume.get("step", last_step) != last_step:
        raise ValueError("GRPO resume pointer and step evidence disagree")
    names = resume.get("committed_directories")
    if names is None:
        # Compatibility with checkpoints written before the explicit chain.
        by_step = {}
        for path in root.glob("step-*/step.json"):
            step = json.loads(path.read_text())["step"]
            if step < last_step:
                if step in by_step:
                    raise ValueError("Ambiguous legacy GRPO step history; cannot count orphaned updates")
                by_step[step] = path.parent.name
        by_step[last_step] = last
        names = [by_step[i] for i in sorted(by_step)]
    if not names or names[-1] != last or len(names) != last_step or len(set(names)) != len(names):
        raise ValueError("Incomplete GRPO committed step chain")
    paths = []
    for expected, name in enumerate(names, 1):
        if Path(name).name != name:
            raise ValueError("Invalid GRPO committed directory")
        path = root / name
        if json.loads((path / "step.json").read_text())["step"] != expected:
            raise ValueError("GRPO committed step sequence differs from optimizer updates")
        paths.append(path)
    return paths


def train_grpo(config, checkpoint, data_path, output):
    validate_rl_config(config)
    if not (Path(checkpoint) / "adapter_config.json").is_file():
        raise ValueError("This pilot requires an SFT LoRA checkpoint before GRPO")
    import torch
    from transformers import set_seed
    rows = sorted(load_rows(data_path, required_split="train"), key=lambda r: identity(r.question))
    from .expert_isolation import verify_manager_rows
    expert_isolation = verify_manager_rows(config, rows)
    if not rows:
        raise ValueError("Empty RL training set")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    signature = {"implementation": "immutable-commit-grpo-v1", "config": config,
        "harness": harness_identity(), "checkpoint": checkpoint_identity(checkpoint),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()}
    manifest = root / "training_run.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError("GRPO inputs changed; use a new output directory")
    elif any(root.iterdir()):
        raise ValueError("Nonempty GRPO output without matching manifest")
    else:
        atomic_json(manifest, signature)
    if (root / "training_metrics.json").exists():
        from .runner import validate_stage_artifacts
        validate_stage_artifacts(root, "rl")
        return
    # Model loading and resume errors belong to this run too, not only the
    # optimization loop. Monitor retains their traceback and terminal status.
    with Monitor(output, "rsi_grpo") as monitor:
        if expert_isolation.get('checked'):
            monitor.summary({'expert_data_isolation': expert_isolation})
        _train_grpo_steps(config, checkpoint, rows, root, monitor)


def _train_grpo_steps(config, checkpoint, rows, root, monitor):
    import torch
    from transformers import set_seed
    frozen = verify_advisor(config, root)
    resume = json.loads((root / "resume.json").read_text()) if (root / "resume.json").exists() else None
    committed = committed_step_directories(root)
    source = str(root / resume["directory"]) if resume else checkpoint
    set_seed(config["seed"])
    tok, model = load_model(config["base_model"], source, trainable=True,
                           lora_rank=config.get("lora_rank", 16), revision=config.get("base_model_revision"))
    # Small frozen adapter on the SAME base; never allocate a second 9B model.
    # Reference is the input SFT checkpoint, including after an interrupted run.
    model.load_adapter(checkpoint, adapter_name="rsi_reference", is_trainable=False)
    model.set_adapter("default")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.get("rl_learning_rate", 1e-6), weight_decay=0.)
    completed = 0
    optimizer_updates = 0
    if resume:
        state = torch.load(root / resume["directory"] / "optimizer.pt", map_location=model.device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        completed = state["step"]
        optimizer_updates = state.get("optimizer_steps", completed)
        saved_report = json.loads((committed[-1] / "step.json").read_text()) if committed else {}
        if optimizer_updates != saved_report.get("optimizer_steps", completed):
            raise ValueError("GRPO optimizer update count and saved report disagree")
        if completed != len(committed) or completed > config["rl_max_steps"]:
            raise ValueError("GRPO optimizer state and committed evidence disagree")
    backend = RolloutBackend(tok, model, config)
    advisors = HTTPAdvisors(config["advisor_url"], config["advisor_max_tokens"], config.get("advisor_models"),
                           generation_options=config.get("advisor_generation"))
    advisors.identity = frozen
    order = list(range(len(rows)))
    random.Random(config["seed"]).shuffle(order)
    monitor.summary({"training_algorithm": "immutable_commit_grpo", "reference_checkpoint": checkpoint,
        "reference_frozen": True, "root_gradient": False, "tool_gradient": False,
        "loss_normalization": "mean_tokens_per_trajectory_then_mean_trajectories",
        "reward_definition": "binary_valid_terminal_correctness", "resumed_optimizer_steps": optimizer_updates,
        "resumed_groups": completed,
        "trainable_parameters": sum(p.numel() for p in params),
        "total_parameters": sum(p.numel() for p in model.parameters())})
    start = time.monotonic()
    for step in range(completed, config["rl_max_steps"]):
        step_started = time.monotonic()
        row = rows[order[step % len(order)]]
        question_context(row)
        progress(phase="grpo_rollout", group_step=step, optimizer_step=optimizer_updates, total_groups=config["rl_max_steps"])
        seed = (config["seed"] + step * 100003) % (2 ** 31)
        model.set_adapter("default")
        model.eval()
        backend.turns = None
        direct, history = root_state(row, backend, config, seed)
        trajectories = []
        for sample in range(config["num_generations"]):
            backend.turns = []
            outcome = policy_rollout(row, backend, advisors, config, seed + sample + 1, direct, history)
            trajectories.append({"turns": backend.turns, "outcome": outcome, "reward": float(outcome["correct"])})
        backend.turns = None
        if not any(t["outcome"]["valid"] for t in trajectories):
            atomic_json(root / "invalid_group.json", {"step": step + 1,
                "reason": "All rollouts invalid; unchanged zero rewards, no outcome learning signal",
                "root": direct, "trajectories": trajectories})
        advantages = group_advantages([t["reward"] for t in trajectories])
        diagnostics = reward_metrics(trajectories, advantages)
        metrics(diagnostics, "grpo", trainer_step=step + 1)
        for index, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
            record = rollout_record(step + 1, index, direct, trajectory, advantage,
                {"question_hash": identity(row.question), "question": row.question})
            rollout({**record, "source": "live"})
        # Score old and frozen reference before any optimizer update.
        for adapter, field in (("default", "old"), ("rsi_reference", "reference")):
            model.set_adapter(adapter)
            model.train()  # checkpointing on; dropout disabled in both paths
            with torch.no_grad():
                for trajectory in trajectories:
                    for turn in trajectory["turns"]:
                        turn[field] = turn_logprobs(model, turn, config["rl_temperature"]).detach().cpu()
        model.set_adapter("default")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        policy_loss_value, kl_loss_value = 0., 0.
        manager_tokens = 0
        token_kl_sum = token_surprisal_sum = token_ratio_sum = 0.
        clipped_tokens = 0
        for trajectory, advantage in zip(trajectories, advantages):
            token_count = sum(len(t["completion_ids"]) for t in trajectory["turns"])
            if not token_count:
                if trajectory["outcome"].get("error") != "context_budget_exceeded":
                    raise ValueError("Empty Manager rollout has no documented context budget failure")
                # Keep its zero reward in the question group, but there is no
                # sampled action to score and no token/gradient to fabricate.
                continue
            manager_tokens += token_count
            for turn in trajectory["turns"]:
                lp = turn_logprobs(model, turn, config["rl_temperature"])
                old, reference = turn["old"].to(lp.device), turn["reference"].to(lp.device)
                with torch.no_grad():
                    ratio = (lp - old).exp()
                    delta = reference - lp
                    token_kl_sum += float((delta.expm1() - delta).sum())
                    token_surprisal_sum += float(-lp.sum())
                    token_ratio_sum += float(ratio.sum())
                    clipped_tokens += int(((ratio < 1 - config.get("rl_clip", .2)) |
                                           (ratio > 1 + config.get("rl_clip", .2))).sum())
                scale = token_count * len(trajectories)
                policy_loss = token_objective(lp, old, reference, advantage, config.get("rl_clip", .2), 0.).sum() / scale
                kl_loss = token_objective(lp, old, reference, 0., config.get("rl_clip", .2), config.get("rl_beta", .01)).sum() / scale
                loss = policy_loss + kl_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GRPO loss")
                loss.backward()
                loss_value += float(loss.detach())
                policy_loss_value += float(policy_loss.detach())
                kl_loss_value += float(kl_loss.detach())
        norm = 0.
        if manager_tokens:
            norm = torch.nn.utils.clip_grad_norm_(params, config.get("rl_max_grad_norm", 1.), error_if_nonfinite=True)
            optimizer.step()
            optimizer_updates += 1
        report = {"step": step + 1, "optimizer_steps": optimizer_updates,
            "optimizer_update_applied": bool(manager_tokens),
            "loss": loss_value, "gradient_norm": float(norm),
            **diagnostics,
            "policy_loss": policy_loss_value, "weighted_kl_loss": kl_loss_value,
            "sampled_kl_per_token": token_kl_sum / manager_tokens if manager_tokens else None,
            # Sampled surprisal is not a full-vocabulary entropy measurement.
            "sampled_surprisal_per_token": token_surprisal_sum / manager_tokens if manager_tokens else None,
            "importance_ratio_mean": token_ratio_sum / manager_tokens if manager_tokens else None,
            "clip_fraction": clipped_tokens / manager_tokens if manager_tokens else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "step_wall_seconds": time.monotonic() - step_started,
            "trajectory_tokens_mean": manager_tokens / len(trajectories),
            "trajectory_tokens_min": min(sum(len(t["completion_ids"]) for t in x["turns"]) for x in trajectories),
            "trajectory_tokens_max": max(sum(len(t["completion_ids"]) for t in x["turns"]) for x in trajectories),
            "old_reference_max_abs_logp_difference": max((
                float((turn["old"] - turn["reference"]).abs().max())
                for t in trajectories for turn in t["turns"]), default=0.),
            "rewards": [t["reward"] for t in trajectories], "advantages": advantages,
            "mixed_reward_group": len({t["reward"] for t in trajectories}) > 1,
            "calls": [t["outcome"]["calls"] for t in trajectories],
            "protocol_valid": [t["outcome"]["valid"] for t in trajectories],
            "manager_supervised_tokens": manager_tokens,
            "scoring_and_training_input_tokens": 3 * sum(len(t["prompt_ids"]) + len(t["completion_ids"])
                for trajectory in trajectories for t in trajectory["turns"]),
            "question_hash": identity(row.question)}
        metrics(report, "grpo", trainer_step=step + 1)
        usage("grpo_train", {}, supervised_tokens=manager_tokens, step=step + 1)
        # Commit optimizer, adapter and evidence together. Ignore partial
        # directories after interruption; resume only from the atomic pointer.
        import tempfile
        stage = Path(tempfile.mkdtemp(prefix="incomplete-", dir=root))
        model.save_pretrained(stage, selected_adapters=["default"])
        tok.save_pretrained(stage)
        torch.save({"optimizer": optimizer.state_dict(), "step": step + 1, "optimizer_steps": optimizer_updates}, stage / "optimizer.pt")
        atomic_json(stage / "step.json", report)
        evidence = [{"reward": t["reward"], "outcome": t["outcome"],
                     "turns": [{k: v for k, v in turn.items() if k not in {"old", "reference"}}
                               for turn in t["turns"]]} for t in trajectories]
        atomic_json(stage / "rollouts.json", {"root": direct, "trajectories": evidence})
        directory = f"step-{step + 1:05d}-{stage.name.removeprefix('incomplete-')}"
        stage.rename(root / directory)
        committed.append(root / directory)
        atomic_json(root / "resume.json", {"directory": directory, "step": step + 1,
                                           "committed_directories": [p.name for p in committed]})
    model.set_adapter("default")
    model.save_pretrained(root, selected_adapters=["default"])
    tok.save_pretrained(root)
    reports = [json.loads((p / "step.json").read_text()) for p in committed]
    monitor.summary({"optimizer_steps": optimizer_updates, "completed_groups": len(reports), "mixed_reward_groups": sum(r["mixed_reward_group"] for r in reports),
                     "committed_supervised_tokens": sum(r["manager_supervised_tokens"] for r in reports)})
    atomic_json(root / "training_metrics.json", {"optimizer_steps": optimizer_updates,
        "completed_groups": len(reports), "groups_without_sampled_tokens": sum(not r.get("optimizer_update_applied", True) for r in reports),
        "wall_seconds_this_attempt": time.monotonic() - start,
        "reference_checkpoint": checkpoint, "reward": "binary_valid_terminal_correctness",
        "call_penalty": 0., "root_gradient": False, "tool_gradient": False,
        "loss_normalization": "mean tokens within trajectory, mean trajectories within question group",
        "rl_temperature": config["rl_temperature"], "config": config})
