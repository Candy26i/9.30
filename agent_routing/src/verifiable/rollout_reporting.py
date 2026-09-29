"""Reward summaries and readable rollout failures; never changes rewards."""
import math


def reward_metrics(trajectories, advantages):
    n = len(trajectories)
    if not n or len(advantages) != n:
        raise ValueError("A nonempty group with aligned advantages is required")
    rewards = [float(t["reward"]) for t in trajectories]
    mean = sum(rewards) / n
    valid = sum(bool(t["outcome"]["valid"]) for t in trajectories)
    return {"reward_mean": mean, "reward_std": math.sqrt(sum((x - mean) ** 2 for x in rewards) / n),
        "valid_rate": valid / n, "invalid_rate": 1 - valid / n,
        "invalid_count": n - valid, "sample_count": n,
        "advantage_abs_mean": sum(abs(a) for a in advantages) / n,
        "zero_advantage_fraction": sum(a == 0 for a in advantages) / n,
        "mean_calls": sum(t["outcome"].get("calls", 0) for t in trajectories) / n,
        "mixed_reward_group": len(set(rewards)) > 1}


def rollout_record(step, sample, root, trajectory, advantage, question=None):
    outcome = trajectory["outcome"]
    turns = trajectory.get("turns", [])
    costs = [c for c in outcome.get("costs", []) if c.get("role") == "manager"]
    outputs = [{"kind": turn.get("kind"), "text": cost.get("text"),
                "truncated": cost.get("truncated"), "tokens": cost.get("completion_tokens")}
               for turn, cost in zip(turns, costs)]
    error = outcome.get("error")
    failure = "none"
    if not outcome["valid"]:
        if error == "context_budget_exceeded":
            failure = "context_budget_exceeded"
        elif error in {"answer_truncated", "answer_format"}:
            revisions = [o for o in outputs if o["kind"] == "revision"]
            failure = ("revision" if revisions else "root") + ("_truncated" if error == "answer_truncated" else "_answer_format")
        elif error:
            failure = "decision_truncated" if error == "truncated_decision" else "decision_protocol"
        else:
            revisions = [o for o in outputs if o["kind"] == "revision"]
            terminal = revisions[-1] if revisions else root
            phase = "revision" if revisions else "root"
            failure = phase + ("_truncated" if terminal.get("truncated") else "_answer_format")
    return {"step": step, "sample": sample, **(question or {}),
        "reward": trajectory["reward"], "advantage": advantage, "valid": outcome["valid"],
        "error": error, "failure_type": failure, "calls": outcome.get("calls"),
        "root_text": root.get("text"), "root_valid": root.get("valid"),
        "root_truncated": root.get("truncated"), "final_text": outcome.get("text"),
        "decisions": outcome.get("decisions"), "manager_outputs": outputs,
        "source": "saved_rollout"}
