"""Finite-action GRPO (FA-GRPO) for the MCQ manager (design §3.5).

Every manager turn is a finite action (``protocol``), so each GRPO quantity is an
exact sum over at most K keys or 4 decision paths instead of a sampled estimate.
Per optimizer step, for ``questions_per_step`` questions of ``grpo_rk``:

1. **Root, no policy gradient.** One forward gives the K-way draft-key
   distribution of the deployment root (``TOOLS_DEPLOY``); X = its greedy key
   (deployment is greedy). Loss: ``beta_root * KL(p_root,theta || p_root,ref)``, exact.
2. **Decision, no policy gradient.** The <= 4 legal turns after ``DRAFT_ANSWER_X``
   (commit, call E/R/V) are scored under theta and ref and renormalised over the
   legal set (trie). Loss: ``beta_dec * KL_dec``, exact.
3. **Revision, policy gradient.** For every advisor a (forced, tool output from
   the advisor cache, Verifier gets ``current_draft=X``, depth cap 1), the
   post-tool state plus the forced ``DRAFT_ANSWER_`` prefix gives the K-way key
   softmax pi(y | s_a); r(y) = 1[y = gold]. ``exact`` (default, the G -> inf limit
   of GRPO): J = pi(gold), A(y) = (r(y) - J) / (sqrt(J(1-J)) + 1e-4),
   ``L_pg = -sum_y sg[pi(y) A(y)] log pi(y)``; ``sampled``: G draws from the same
   forward with ``rsi_grpo.group_advantages``. Plus ``beta_rev * KL(pi || pi_ref)``.
4. **Anchor.** ``anchor_lambda`` times the route-only cross-entropy (token mean)
   of ``anchor_rows_per_step`` rows of this round's SFT file
   (``routing_anchor.build_anchor_features(..., "route_only")``).

Loss of one step: mean over questions of the root and decision terms, mean over
revision states of the revision terms, plus the anchor; AdamW (constant lr,
weight decay 0), gradient clipping. Under a binary reward the reward-optimal
routing is "always call" (design §3.5 table), so decisions and the root draft
never receive a policy gradient; the ``decision_pg`` ablation ('decRL', MedQA
only) adds one with exact action values Q(commit) = 1[X = gold], Q(a) = J_a.

The reference is a frozen copy of the round's SFT checkpoint S_k loaded as a
second, non-trainable adapter on the same base (never the raw base: the paper-era
TRL run used the base as its KL reference); on a bf16 base its weights are
reloaded into the fp32 LoRA layers and checked bitwise against S_k's file. Every
step commits adapter, optimizer and ``step.json`` atomically (``incomplete-*``
directory fsynced and renamed into place, then the ``resume.json`` pointer;
``rsi_grpo.committed_step_directories`` reads only the pointer's chain, so an
orphaned step is redone, never counted twice, and then deleted). Only the newest
step (resume) and the rollback target keep their weights (``_prune``); every
committed step keeps ``step.json``. Every ``guard_every`` steps (and at the last step) the greedy call rate
and mean ``KL_dec`` on a fixed probe of ``grpo_rk`` roots are compared with S_k;
a failure rolls back to the last passing guarded step (or S_k) and ends the stage.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..marginal_value import ADVISOR_KINDS
from . import benchmarks as registry
from . import protocol
from .advisors import AdvisorRequest, _fsync_dir
from .collect import _atomic_write, rows_digest

GRPO_VERSION = "mcq_rsi_fa_grpo/1"
ESTIMATORS = ("exact", "sampled")
ANCHOR_CONTEXTS = ("paper", "evolve")
REFERENCE = "reference"
POLICY = "default"
PACKAGE = Path(__file__).resolve().parent


@dataclass
class FAGRPOConfig:
    bench: str = "medqa"
    base_model: str = registry.BASE_MODEL
    base_revision: Optional[str] = registry.BASE_REVISION
    seed: int = 0
    steps: int = 64
    questions_per_step: int = 4
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    beta_root: float = 0.1
    beta_dec: float = 0.1
    beta_rev: float = 0.02
    anchor_lambda: float = 0.03
    anchor_rows_per_step: int = 4
    anchor_max_seq_len: int = 4096
    # "paper": render the anchor's tool block as paper-era SFT and deployment do;
    # "evolve": exactly as 9.30 train_manager_sft (adds "tool_calls": null to each schema).
    anchor_context: str = "paper"
    estimator: str = "exact"
    num_generations: int = 16
    advantage_eps: float = 1e-4
    guard_every: int = 8
    guard_probe_size: int = 64
    guard_max_call_rate_delta: float = 0.10
    guard_max_kl_dec: float = 0.05
    informative_low: float = 0.02
    informative_high: float = 0.98
    informative_min_fraction: float = 0.15
    decision_pg: bool = False  # 'decRL' ablation (MedQA only); the guard is then logged, not enforced
    decision_pg_epsilon: float = 0.0  # lexicographic variant: 0.05
    # The lexicographic tie set: actions with Q >= max Q - this (exact ties of the continuous J_a never occur).
    decision_pg_tie_tolerance: float = 0.02
    gradient_checkpointing: Optional[bool] = None  # None: on for CUDA
    device: Optional[str] = None

    @classmethod
    def from_dict(cls, config) -> "FAGRPOConfig":
        if isinstance(config, cls):
            return config
        config = dict(config or {})
        unknown = sorted(set(config) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown FA-GRPO config keys: {unknown}")
        return cls(**config)

    def validate(self) -> None:
        registry.get(self.bench)
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}")
        if self.anchor_context not in ANCHOR_CONTEXTS:
            raise ValueError(f"anchor_context must be one of {ANCHOR_CONTEXTS}")
        for name in ("steps", "questions_per_step", "guard_every", "guard_probe_size", "anchor_max_seq_len"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.estimator == "sampled" and self.num_generations < 2:
            raise ValueError("sampled estimator needs num_generations >= 2")
        if self.anchor_lambda > 0 and self.anchor_rows_per_step <= 0:
            raise ValueError("anchor_rows_per_step must be positive when anchor_lambda > 0")
        for name in ("learning_rate", "max_grad_norm", "advantage_eps"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("beta_root", "beta_dec", "beta_rev", "anchor_lambda", "weight_decay", "decision_pg_epsilon",
                     "decision_pg_tie_tolerance"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0):
                raise ValueError(f"{name} must be finite and >= 0")
        if not 0 <= self.informative_low < self.informative_high <= 1:
            raise ValueError("informative_low < informative_high in [0, 1]")
        if self.decision_pg and self.bench != "medqa":
            raise ValueError("the decision-PG ablation (decRL) is defined for MedQA only (design §3.5)")
        if self.decision_pg_epsilon and not self.decision_pg:
            raise ValueError("decision_pg_epsilon needs decision_pg")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("FA-GRPO runs on one manager GPU")


# ------------------------------------------------------------- exact objectives (pure torch)

def exact_kl(logp, logq):
    """KL(p || q) of two distributions given as log-probabilities over the same finite support."""
    return (logp.exp() * (logp - logq)).sum()


def exact_pg_loss(logp, gold: int, eps: float = 1e-4):
    """G -> inf GRPO surrogate on a finite action set with binary reward 1[y = gold].

    Returns ``(loss, J)`` with J = pi(gold). ``-grad loss = grad J / (sqrt(J(1-J)) + eps)``.
    """
    import torch
    p = logp.exp()
    J = p[gold].detach()
    reward = torch.zeros_like(p)
    reward[gold] = 1.0
    advantage = (reward - J) / (torch.sqrt(J * (1 - J)) + eps)
    return -((p * advantage).detach() * logp).sum(), float(J)


def sampled_pg_loss(logp, gold: int, generations: int, generator, eps: float = 1e-4):
    """GRPO with G draws from pi (one shared forward); ``rsi_grpo.group_advantages``."""
    import torch
    from ...verifiable.rsi_grpo import group_advantages
    probs = logp.detach().float().exp().cpu()
    draws = torch.multinomial(probs, generations, replacement=True, generator=generator)
    rewards = [float(int(i) == gold) for i in draws]
    advantages = torch.tensor(group_advantages(rewards, eps), dtype=logp.dtype, device=logp.device)
    return -(advantages * logp[draws.to(logp.device)]).sum() / generations, float(probs[gold])


def tie_break_bonus(values: Sequence[float], calls: Sequence[int], epsilon: float,
                    tolerance: float = 0.0) -> List[float]:
    """Lexicographic decRL term: ``epsilon * (-calls)`` centred among the (near-)argmax-Q actions.

    Q(commit) is 0/1 and Q(a) = J_a is a continuous probability, so exact ties
    (``tolerance`` 0) need a saturated float J and almost never occur; the set is
    therefore every action with Q >= max Q - ``tolerance``. All zeros when the set
    has a single action or a single call count.
    """
    bonus = [0.0] * len(values)
    if not epsilon:
        return bonus
    best = [i for i, v in enumerate(values) if v >= max(values) - tolerance - 1e-12]
    cost = [-float(calls[i]) for i in best]
    mean = sum(cost) / len(cost)
    for i, c in zip(best, cost):
        bonus[i] = epsilon * (c - mean)
    return bonus


def decision_pg_loss(logq, values: Sequence[float], calls: Sequence[int], epsilon: float = 0.0, eps: float = 1e-4,
                     tie_tolerance: float = 0.0):
    """decRL ablation: exact policy gradient over the legal decisions with action values ``values``.

    A(d) = (Q(d) - V) / (sigma + eps) under q; the lexicographic variant adds
    ``tie_break_bonus`` (``epsilon * (-calls)`` centred among the argmax-Q actions
    within ``tie_tolerance``) after standardisation.
    """
    import torch
    q = logq.exp()
    Q = torch.tensor(list(values), dtype=logq.dtype, device=logq.device)
    V = (q.detach() * Q).sum()
    sigma = torch.sqrt((q.detach() * (Q - V) ** 2).sum())
    advantage = (Q - V) / (sigma + eps)
    if epsilon:
        advantage = advantage + torch.tensor(tie_break_bonus(values, calls, epsilon, tie_tolerance),
                                             dtype=logq.dtype, device=logq.device)
    return -((q * advantage).detach() * logq).sum()


def question_objective(root_lp, root_ref, dec_lp, dec_ref, rev_lps, rev_refs, gold: int, cfg: FAGRPOConfig,
                       generators=None, decision_values=None, decision_calls=None) -> Dict[str, Any]:
    """Loss of one question: root/decision exact KL (no PG), revision PG + KL, optional decRL."""
    kl_root = exact_kl(root_lp, root_ref)
    kl_dec = exact_kl(dec_lp, dec_ref)
    pg_terms, kl_terms, Js = [], [], []
    for i, (lp, ref) in enumerate(zip(rev_lps, rev_refs)):
        if cfg.estimator == "exact":
            pg, J = exact_pg_loss(lp, gold, cfg.advantage_eps)
        else:
            pg, J = sampled_pg_loss(lp, gold, cfg.num_generations, generators[i], cfg.advantage_eps)
        pg_terms.append(pg)
        kl_terms.append(exact_kl(lp, ref))
        Js.append(J)
    n = max(1, len(pg_terms))
    pg = sum(pg_terms) / n if pg_terms else kl_root * 0
    kl_rev = sum(kl_terms) / n if kl_terms else kl_root * 0
    loss = cfg.beta_root * kl_root + cfg.beta_dec * kl_dec + pg + cfg.beta_rev * kl_rev
    out = {"kl_root": kl_root, "kl_dec": kl_dec, "pg": pg, "kl_rev": kl_rev, "J": Js}
    if cfg.decision_pg:
        dec_pg = decision_pg_loss(dec_lp, decision_values, decision_calls, cfg.decision_pg_epsilon, cfg.advantage_eps,
                                  cfg.decision_pg_tie_tolerance)
        loss = loss + dec_pg
        out["dec_pg"] = dec_pg
        out["dec_tiebreak"] = any(tie_break_bonus(decision_values, decision_calls, cfg.decision_pg_epsilon,
                                                  cfg.decision_pg_tie_tolerance))
    out["loss"] = loss
    return out


def informativeness(Js: Sequence[float], low: float = 0.02, high: float = 0.98,
                    min_fraction: float = 0.15) -> Dict[str, Any]:
    """Design §3.5(7): share of revision states whose J lies strictly inside (low, high)."""
    n = len(Js)
    k = sum(low < J < high for J in Js)
    fraction = k / n if n else 0.0
    return {"n_states": n, "n_informative": k, "fraction": fraction, "min_fraction": min_fraction,
            "passed": bool(n and fraction >= min_fraction)}


def select_rollback(guards: Sequence[Dict[str, Any]]) -> Tuple[int, Optional[int]]:
    """``(selected_step, failed_step)`` from the guard history in step order.

    Training ends at the first enforced failure; the selected checkpoint is the last
    guarded step that passed before it (0 = S_k). Without a failure it is the last
    passing guarded step.
    """
    selected = 0
    for guard in guards:
        if guard["enforced"] and not guard["passed"]:
            return selected, guard["step"]
        if guard["passed"] or not guard["enforced"]:
            selected = guard["step"]
    return selected, None


# ----------------------------------------------------------------- scoring with a model

def score_tree(model, prompt_ids: Sequence[int], paths, trie=None, pad_id: int = 0):
    """Exact renormalised log-probs of ``paths`` after ``prompt_ids`` and the trie-greedy path index.

    When every path is a shared prefix plus one distinct token (draft keys under
    the Qwen3.5 tokenizer) a single one-row forward suffices; otherwise all paths
    go through one right-padded batch (exact for a causal model). Forced
    (single-choice) positions contribute log 1 = 0. Gradients flow when enabled.
    """
    import torch
    paths = [tuple(p) for p in paths]
    trie = trie or protocol.make_trie(paths)
    device = next(model.parameters()).device
    n = len(paths)
    if n == 1:
        return torch.zeros(1, device=device), 0
    length = len(paths[0])
    if all(len(p) == length and p[:-1] == paths[0][:-1] for p in paths):
        ids = torch.tensor([list(prompt_ids) + list(paths[0][:-1])], device=device)
        row = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    logits_to_keep=1).logits[0, -1].float()
        tokens = [p[-1] for p in paths]
        selected = row[tokens]
        logp = selected - torch.logsumexp(selected, 0)
        values = selected.detach().tolist()
        greedy = max(range(n), key=lambda i: (values[i], -tokens[i]))
        return logp, greedy
    width = max(map(len, paths))
    ids = torch.full((n, len(prompt_ids) + width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(paths):
        seq = list(prompt_ids) + list(p)
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False,
                   logits_to_keep=width + 1).logits.float()
    nodes: Dict[Tuple[int, ...], Tuple[List[int], List[float]]] = {}
    out = []
    for i, p in enumerate(paths):
        total = logits.new_zeros(())
        for t, token in enumerate(p):
            allowed = trie.allowed(p[:t])
            if len(allowed) == 1:
                continue
            selected = logits[i, t, allowed]
            total = total + selected[allowed.index(token)] - torch.logsumexp(selected, 0)
            nodes.setdefault(p[:t], (allowed, selected.detach().tolist()))
        out.append(total)
    prefix: Tuple[int, ...] = ()
    while protocol.complete_path(trie, prefix) is None:
        allowed = trie.allowed(prefix)
        if len(allowed) == 1:
            prefix += (allowed[0],)
            continue
        allowed, values = nodes[prefix]
        prefix += (max(zip(allowed, values), key=lambda x: (x[1], -x[0]))[0],)
    return torch.stack(out), paths.index(protocol.complete_path(trie, prefix))


class _Scorer:
    """The protocol objects of one benchmark under one tokenizer (cached per key set / root)."""

    def __init__(self, tokenizer, bench, kinds):
        self.tok, self.bench, self.kinds = tokenizer, bench, list(kinds)
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self._drafts: Dict[Tuple[str, ...], Tuple[List[Tuple[int, ...]], Any]] = {}
        self._roots: Dict[int, Tuple[List[Dict[str, Any]], List[int]]] = {}
        self._decisions: Dict[Tuple[int, str], Tuple[List[protocol.Action], Any]] = {}

    def prompt_ids(self, messages) -> List[int]:
        return list(self.tok(protocol.render(self.tok, messages, protocol.TOOLS_DEPLOY))["input_ids"])

    def drafts(self, keys) -> Tuple[List[Tuple[int, ...]], Any]:
        keys = tuple(keys)
        if keys not in self._drafts:
            paths = protocol.draft_paths(self.tok, keys)
            ordered = [paths[k] for k in keys]
            self._drafts[keys] = (ordered, protocol.make_trie(ordered))
        return self._drafts[keys]

    def root(self, row) -> Tuple[List[Dict[str, Any]], List[int]]:
        eid = int(row["example_id"])
        if eid not in self._roots:
            messages = protocol.manager_messages(self.bench, row)
            self._roots[eid] = (messages, self.prompt_ids(messages))
        return self._roots[eid]

    def decision(self, row, key: str) -> Tuple[List[protocol.Action], Any]:
        eid = int(row["example_id"])
        if (eid, key) not in self._decisions:
            messages, _ = self.root(row)
            actions = protocol.decision_actions(self.tok, messages, protocol.TOOLS_DEPLOY, [key], self.kinds, eid)
            self._decisions[(eid, key)] = (actions, protocol.make_trie(a.ids for a in actions))
        return self._decisions[(eid, key)]

    def revision_prompt(self, row, kind: str, key: str, output: str) -> List[int]:
        """Post-tool state as the deployed eval keeps it (example_id injected), depth 1."""
        messages, _ = self.root(row)
        eid = int(row["example_id"])
        call_id = f"eval_{eid}_0"
        state = list(messages) + [protocol.eval_call_message(kind, key, eid, call_id),
                                  protocol.tool_message(kind, call_id, output)]
        return self.prompt_ids(state)

    def key_logprobs(self, model, prompt_ids, keys):
        paths, trie = self.drafts(keys)
        return score_tree(model, prompt_ids, paths, trie, self.pad_id)


def _use(model, adapter: str) -> None:
    """Activate an adapter; the reference adapter never requires grad (peft re-enables it on activation)."""
    model.set_adapter(adapter)
    for name, param in model.named_parameters():
        if f".{REFERENCE}." in name:
            param.requires_grad_(False)


def _advise(pool, row, kind: str, draft: str) -> str:
    return pool.call(agent_kind=kind, example_id=int(row["example_id"]), question=row["question"],
                     context=row.get("context") or "", choices=row["choices"], cache_namespace="mcq_rsi_grpo",
                     candidate_answer=draft if kind == "verifier" else "")


def _seed(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:15], 16)


# --------------------------------------------------------------------------- guard

def guard_probe(model, scorer: _Scorer, rows, adapter: str = POLICY, reference: Optional[str] = REFERENCE):
    """Greedy root call rate (and mean KL_dec / KL_root vs ``reference``) of ``adapter`` on ``rows``."""
    import torch
    calls, kl_dec, kl_root, actions = 0, [], [], {}
    model.eval()
    with torch.no_grad():
        for row in rows:
            keys = list(row["choices"])
            _, prompt = scorer.root(row)
            _use(model, adapter)
            root_lp, gi = scorer.key_logprobs(model, prompt, keys)
            key = keys[gi]
            acts, trie = scorer.decision(row, key)
            dec_lp, di = score_tree(model, prompt, [a.ids for a in acts], trie, scorer.pad_id)
            calls += acts[di].kind is not None
            actions[acts[di].name] = actions.get(acts[di].name, 0) + 1
            if reference is not None:
                _use(model, reference)
                ref_root, _ = scorer.key_logprobs(model, prompt, keys)
                ref_dec, _ = score_tree(model, prompt, [a.ids for a in acts], trie, scorer.pad_id)
                kl_dec.append(float(exact_kl(dec_lp, ref_dec)))
                kl_root.append(float(exact_kl(root_lp, ref_root)))
    _use(model, POLICY)
    n = max(1, len(rows))
    out = {"n": len(rows), "call_rate": calls / n, "actions": actions}
    if reference is not None:
        out.update(kl_dec=sum(kl_dec) / n, kl_root=sum(kl_root) / n)
    return out


def _guard(model, scorer, probe_rows, baseline, cfg: FAGRPOConfig, step: int) -> Dict[str, Any]:
    result = guard_probe(model, scorer, probe_rows)
    delta = result["call_rate"] - baseline["call_rate"]
    reasons = []
    if abs(delta) > cfg.guard_max_call_rate_delta:
        reasons.append(f"|delta call rate| {abs(delta):.4f} > {cfg.guard_max_call_rate_delta}")
    if result["kl_dec"] > cfg.guard_max_kl_dec:
        reasons.append(f"KL_dec {result['kl_dec']:.4f} > {cfg.guard_max_kl_dec}")
    return {**result, "step": step, "call_rate_delta": delta, "baseline_call_rate": baseline["call_rate"],
            "passed": not reasons, "reasons": reasons, "enforced": not cfg.decision_pg}


# ---------------------------------------------------------------------- checkpointing

def _write_json(path: Path, value) -> None:
    _atomic_write(Path(path), json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _fsync_tree(path: Path) -> None:
    """fsync every file under ``path`` and every directory (durable before the rename that publishes it)."""
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            with open(f, "rb") as handle:
                os.fsync(handle.fileno())
    for d in sorted({Path(path), *(f for f in Path(path).rglob("*") if f.is_dir())}, reverse=True):
        _fsync_dir(d)


def _commit_step(root: Path, step: int, model, optimizer, report: Dict[str, Any]) -> str:
    """Write one step into ``incomplete-*``, fsync it and rename it into place (the pointer comes after).

    The tokenizer is not saved per step (it is S_k's, copied once into ``final``).
    """
    import torch
    stage = Path(tempfile.mkdtemp(prefix="incomplete-", dir=root))
    model.save_pretrained(stage, selected_adapters=[POLICY])
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, stage / "optimizer.pt")
    _write_json(stage / "step.json", report)
    _fsync_tree(stage)
    name = f"step-{step:05d}-{stage.name.removeprefix('incomplete-')}"
    os.replace(stage, root / name)
    _fsync_dir(root)
    return name


def _advance_pointer(root: Path, names: List[str], step: int) -> None:
    _write_json(root / "resume.json", {"directory": names[-1], "step": step, "committed_directories": list(names)})
    _fsync_dir(root)  # the pointer's rename is durable before anything relies on it


WEIGHTS = ("adapter_model.safetensors", "optimizer.pt")


def _prune(root: Path, names: Sequence[str], keep_optimizer: bool = True) -> Dict[str, int]:
    """Bound the stage's disk use (~370 MB per 9B r16 step otherwise, kept for all 64 steps).

    Committed steps keep ``step.json`` and ``adapter_config.json`` (the chain stays
    verifiable); only the newest step (resume) and the current rollback target (the
    last passing guarded step) keep their adapter weights, and the newest its
    optimizer state (dropped too with ``keep_optimizer=False``, once the stage
    ended). ``step-*`` directories outside the committed chain (orphans of a crash
    between rename and pointer) and ``incomplete-*`` directories are removed.
    """
    names = list(names)
    guards = [r["guard"] for r in _reports([root / n for n in names]) if r.get("guard")]
    selected, _ = select_rollback(guards)
    keep = {names[-1]} if names else set()
    if selected:
        keep.add(names[selected - 1])
    removed = {"files": 0, "directories": 0}
    for name in names:
        for f in WEIGHTS:
            path = root / name / f
            if path.exists() and (name not in keep or (f == "optimizer.pt" and (name != names[-1] or not keep_optimizer))):
                path.unlink()
                removed["files"] += 1
    chain = set(names)
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d.name.startswith("incomplete-") or (d.name.startswith("step-") and d.name not in chain)):
            shutil.rmtree(d)
            removed["directories"] += 1
    _fsync_dir(root)
    return removed


def _reports(committed) -> List[Dict[str, Any]]:
    return [json.loads((p / "step.json").read_text(encoding="utf-8")) for p in committed]


def _write_metrics(root: Path, reports) -> None:
    """Per-step metrics of the committed chain only (rewritten whole: no double counting after a crash)."""
    _atomic_write(root / "metrics.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in reports))


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "chat_template.jinja", "chat_template.json", "vocab.json", "merges.txt", "tokenizer.model")


def _copy_checkpoint(src: Path, dest: Path, tokenizer_src: Path) -> None:
    """The adapter of ``src`` plus S_k's tokenizer and chat template (no optimizer, step evidence or ``checkpoint-*``)."""
    tmp = Path(tempfile.mkdtemp(prefix=".final-", dir=dest.parent))
    try:
        for name in ADAPTER_FILES:
            if not (Path(src) / name).is_file():
                raise FileNotFoundError(f"{src}: no {name} (pruned or never written)")
            shutil.copy2(Path(src) / name, tmp / name)
        for name in TOKENIZER_FILES:
            if (Path(tokenizer_src) / name).is_file():
                shutil.copy2(Path(tokenizer_src) / name, tmp / name)
        _fsync_tree(tmp)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
        _fsync_dir(dest.parent)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def _file_sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> List[Path]:
    """Code that shapes FA-GRPO inputs beyond ``harness_identity`` (which covers ``src/verifiable``,
    ``marginal_value.py`` and ``utils/io.py``): this package (protocol, prompts, select, collect, advisors,
    this file), ``manager/prompt.py``, ``stages.py`` (deployment tool schemas), the route-only anchor
    renderer ``routing_anchor.py``, ``evolve.py`` (its renderer the anchor shares) and the base loader."""
    from .collect import SOURCES
    files = {f for src in SOURCES for f in ([src] if src.is_file() else src.rglob("*"))
             if f.is_file() and f.suffix in (".py", ".txt", ".jinja")}
    files |= {PACKAGE.parent / "routing_anchor.py", PACKAGE.parent / "evolve.py",
              PACKAGE.parents[1] / "subagents" / "train.py"}
    return sorted(files)


def files_sha256(files: Sequence[Path]) -> str:
    root = PACKAGE.parents[1]
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(Path(f).relative_to(root)).encode() + b"\0" + Path(f).read_bytes())
    return digest.hexdigest()


def source_sha256() -> str:
    return files_sha256(source_files())


def _base_identity(cfg: FAGRPOConfig) -> Dict[str, Any]:
    """A local base directory by its files; a hub id is loaded at ``cfg.base_revision``."""
    if os.path.isdir(cfg.base_model):
        from .evaluate import base_identity
        return base_identity(cfg.base_model, cfg.base_revision)
    return {"model_id": cfg.base_model, "revision": cfg.base_revision}


def run_signature(cfg: FAGRPOConfig, sft_checkpoint, pool_rows, anchor_rows, advisor_pool) -> Dict[str, Any]:
    from ...verifiable.provenance import harness_identity
    from ...verifiable.runner import checkpoint_identity
    return json.loads(json.dumps({
        "version": GRPO_VERSION, "config": asdict(cfg), "harness": harness_identity(), "source_sha256": source_sha256(),
        "base_identity": _base_identity(cfg),
        "sft_checkpoint": str(sft_checkpoint), "sft_identity": checkpoint_identity(sft_checkpoint),
        "pool_rows_sha256": rows_digest(pool_rows),
        "anchor_rows_sha256": hashlib.sha256(json.dumps(list(anchor_rows), sort_keys=True).encode()).hexdigest(),
        "advisors": advisor_pool.identity() if hasattr(advisor_pool, "identity") else None,
    }))


# --------------------------------------------------------------------------- loading

def _device(cfg: FAGRPOConfig) -> str:
    import torch
    return cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")


def _dtype(device: str):
    import torch
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def check_adapter_exact(model, adapter: str, checkpoint) -> None:
    """Raise unless ``adapter``'s weights equal ``checkpoint``'s adapter file bit for bit (in the layer dtype)."""
    import torch
    from peft.utils import get_peft_model_state_dict, load_peft_weights
    saved = load_peft_weights(str(checkpoint), device="cpu")
    loaded = get_peft_model_state_dict(model, adapter_name=adapter, save_embedding_layers=False)
    if set(saved) != set(loaded):
        raise RuntimeError(f"adapter {adapter!r} and {checkpoint} hold different tensors")
    for key, value in saved.items():
        have = loaded[key].detach().cpu()
        if have.dtype != torch.float32 or not torch.equal(have, value.to(have.dtype)):
            raise RuntimeError(f"adapter {adapter!r} is not an exact fp32 copy of {checkpoint} ({key})")


def load_policy(cfg: FAGRPOConfig, sft_checkpoint, source=None):
    """Base + trainable policy adapter (``source``, default S_k) + frozen reference adapter (S_k).

    On a bf16 base ``load_adapter`` loads the second adapter into layers still in
    the base dtype and upcasts them only afterwards, so the reference would be a
    bf16-rounded S_k while the policy (``PeftModel.from_pretrained`` upcasts
    first) is the exact fp32 S_k. The reference weights are therefore reloaded
    into the fp32 layers, and both adapters are checked bitwise against their files.
    """
    import torch
    from peft import PeftModel
    from peft.utils import load_peft_weights, set_peft_model_state_dict
    from transformers import AutoTokenizer

    from ...subagents.train import load_text_causal_model
    device = _device(cfg)
    dtype = _dtype(device)
    tok = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    kwargs = {"revision": cfg.base_revision} if cfg.base_revision and not os.path.isdir(cfg.base_model) else {}
    base = load_text_causal_model(cfg.base_model, dtype=dtype, trust_remote_code=True, **kwargs).to(device)
    model = PeftModel.from_pretrained(base, str(source or sft_checkpoint), adapter_name=POLICY, is_trainable=True)
    model.load_adapter(str(sft_checkpoint), adapter_name=REFERENCE, is_trainable=False)
    set_peft_model_state_dict(model, load_peft_weights(str(sft_checkpoint)), adapter_name=REFERENCE)
    model.to(device)
    check_adapter_exact(model, REFERENCE, sft_checkpoint)
    check_adapter_exact(model, POLICY, source or sft_checkpoint)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    _use(model, POLICY)
    if cfg.gradient_checkpointing if cfg.gradient_checkpointing is not None else device.startswith("cuda"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return tok, model


def recorded_sft_context(sft_checkpoint) -> Optional[str]:
    """The tool-block context round-k SFT trained S_k with (``<stage>/sft_report.json`` next to ``<stage>/model``).

    None for a checkpoint not produced by ``sft.train_round_sft`` (e.g. a round-1 S_1 downloaded as is).
    """
    path = Path(sft_checkpoint)
    report = path.parent / "sft_report.json"
    if path.name != "model" or not report.is_file():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    return (data.get("config") or {}).get("context")


def check_anchor_context(cfg: FAGRPOConfig, sft_checkpoint) -> None:
    """The route-only anchor must render S_k's tool block the way S_k's SFT did."""
    recorded = recorded_sft_context(sft_checkpoint)
    if cfg.anchor_lambda > 0 and recorded is not None and recorded != cfg.anchor_context:
        raise ValueError(f"{sft_checkpoint} was trained with SFT context {recorded!r} but anchor_context="
                         f"{cfg.anchor_context!r}; pass anchor_context={recorded!r} (--anchor-context)")


def anchor_features(anchor_rows, tok, cfg: FAGRPOConfig):
    from ..routing_anchor import build_anchor_features
    if cfg.anchor_lambda == 0:
        return [], {}
    if not anchor_rows:
        raise ValueError(f"anchor_lambda={cfg.anchor_lambda} but no anchor rows: the route-only anchor would be off")
    tools = protocol.TOOLS_SFT
    tools = tuple(tools) if cfg.anchor_context == "paper" else list(tools)
    features, stats = build_anchor_features(list(anchor_rows), tok, cfg.anchor_max_seq_len, "route_only", tools)
    if len(features) != len(anchor_rows):
        raise ValueError(f"route-only anchor rows dropped (empty or overlength); no silent truncation: {stats}")
    return features, stats


def anchor_backward(model, features, scale: float) -> Tuple[float, int]:
    """Route-only CE (token mean over the rows) times ``scale``, backpropagated row by row."""
    import torch
    device = next(model.parameters()).device
    total = sum(sum(y != -100 for y in f["labels"]) for f in features)
    loss_sum = 0.0
    for f in features:
        labels = f["labels"]
        start = next(i for i, y in enumerate(labels) if y != -100)
        ids = torch.tensor([f["input_ids"]], device=device)
        keep = len(labels) - start + 1
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=keep).logits[0, :-1].float()
        target = torch.tensor(labels[start:], device=device)
        ce = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100, reduction="sum")
        (scale * ce / total).backward()
        loss_sum += float(ce.detach())
    return loss_sum / max(1, total), total


# ---------------------------------------------------------------------------- training

def _plan(n_rows: int, cfg: FAGRPOConfig) -> List[int]:
    order = list(range(n_rows))
    random.Random(cfg.seed).shuffle(order)
    return order


def _probe_rows(rows, cfg: FAGRPOConfig):
    idx = sorted(random.Random(_seed(cfg.seed, "guard_probe")).sample(range(len(rows)), min(cfg.guard_probe_size, len(rows))))
    return [rows[i] for i in idx]


def _train_step(step: int, model, tok, scorer, optimizer, params, batch, anchor_batch, pool, cfg) -> Dict[str, Any]:
    import torch
    started = time.monotonic()
    kinds = scorer.kinds
    # Phase 1: greedy root X under theta, then the advisor outputs for every (question, advisor) concurrently.
    model.eval()
    _use(model, POLICY)
    roots = []
    with torch.no_grad():
        for row in batch:
            keys = list(row["choices"])
            _, prompt = scorer.root(row)
            _, gi = scorer.key_logprobs(model, prompt, keys)
            roots.append(keys[gi])
    if hasattr(pool, "prefetch"):
        pool.prefetch([AdvisorRequest.for_row(k, row, x) for row, x in zip(batch, roots) for k in kinds])
    outputs = [{k: _advise(pool, row, k, x) for k in kinds} for row, x in zip(batch, roots)]

    optimizer.zero_grad(set_to_none=True)
    per_q, Js, greedy_dec, rev_correct = [], [], [], {k: 0 for k in kinds}
    for qi, (row, x, outs) in enumerate(zip(batch, roots, outputs)):
        keys = list(row["choices"])
        gold = keys.index(row["ground_truth"])
        _, prompt = scorer.root(row)
        acts, trie = scorer.decision(row, x)
        dec_paths = [a.ids for a in acts]
        rev_prompts = [scorer.revision_prompt(row, k, x, outs[k]) for k in kinds]
        model.eval()
        _use(model, REFERENCE)
        with torch.no_grad():
            ref_root, _ = scorer.key_logprobs(model, prompt, keys)
            ref_dec, _ = score_tree(model, prompt, dec_paths, trie, scorer.pad_id)
            ref_revs = [scorer.key_logprobs(model, p, keys)[0] for p in rev_prompts]
        _use(model, POLICY)
        model.train()
        root_lp, _ = scorer.key_logprobs(model, prompt, keys)
        dec_lp, di = score_tree(model, prompt, dec_paths, trie, scorer.pad_id)
        rev = [scorer.key_logprobs(model, p, keys) for p in rev_prompts]
        rev_lps = [lp for lp, _ in rev]
        generators = [torch.Generator().manual_seed(_seed(cfg.seed, step, qi, k)) for k in kinds] \
            if cfg.estimator == "sampled" else None
        values = calls = None
        if cfg.decision_pg:
            J_by_kind = {k: float(lp.detach().exp()[gold]) for k, lp in zip(kinds, rev_lps)}
            values = [float(x == row["ground_truth"]) if a.kind is None else J_by_kind[a.kind] for a in acts]
            calls = [0 if a.kind is None else 1 for a in acts]
        parts = question_objective(root_lp, ref_root, dec_lp, ref_dec, rev_lps, ref_revs, gold, cfg,
                                   generators, values, calls)
        if not torch.isfinite(parts["loss"]):
            raise FloatingPointError(f"non-finite FA-GRPO loss at step {step}")
        (parts["loss"] / len(batch)).backward()
        Js.extend(parts["J"])
        greedy_dec.append(acts[di].name)
        for k, (_, ri) in zip(kinds, rev):
            rev_correct[k] += ri == gold
        per_q.append({"example_id": int(row["example_id"]), "draft": x, "draft_correct": x == row["ground_truth"],
                      "decision": acts[di].name, "J": dict(zip(kinds, parts["J"])),
                      **{k: float(parts[k].detach()) for k in ("loss", "kl_root", "kl_dec", "kl_rev", "pg")},
                      **({"dec_pg": float(parts["dec_pg"].detach()), "dec_tiebreak": parts["dec_tiebreak"]}
                         if "dec_pg" in parts else {})})
    anchor_ce, anchor_tokens = (anchor_backward(model, anchor_batch, cfg.anchor_lambda) if anchor_batch else (None, 0))
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    mean = lambda key: sum(q[key] for q in per_q) / len(per_q)
    info = informativeness(Js, cfg.informative_low, cfg.informative_high, cfg.informative_min_fraction)
    return {
        "step": step, "example_ids": [q["example_id"] for q in per_q], "questions": per_q,
        "loss": mean("loss") + (cfg.anchor_lambda * anchor_ce if anchor_ce is not None else 0.0),
        "kl_root": mean("kl_root"), "kl_dec": mean("kl_dec"), "kl_rev": mean("kl_rev"), "pg_loss": mean("pg"),
        **({"dec_pg_loss": mean("dec_pg"), "dec_tiebreak_states": sum(q["dec_tiebreak"] for q in per_q)}
           if cfg.decision_pg else {}),
        "anchor_ce": anchor_ce, "anchor_supervised_tokens": anchor_tokens,
        "grad_norm": float(norm), "learning_rate": optimizer.param_groups[0]["lr"],
        "root_accuracy": sum(q["draft_correct"] for q in per_q) / len(per_q),
        "train_call_rate": sum(d != "commit" for d in greedy_dec) / len(greedy_dec),
        "J_mean": sum(Js) / max(1, len(Js)),
        "revision_greedy_accuracy": {k: v / len(batch) for k, v in rev_correct.items()},
        "n_states": info["n_states"], "n_informative": info["n_informative"],
        "advisor_stats": dict(getattr(pool, "stats", {}) or {}),
        "step_seconds": time.monotonic() - started,
    }


def train_fa_grpo(config, sft_checkpoint, pool_rows, anchor_rows, advisor_pool, out_dir, resume: bool = True) -> Dict[str, Any]:
    """FA-GRPO from S_k (``sft_checkpoint``) on ``pool_rows`` (grpo_rk); returns the stage summary.

    ``out_dir/final`` holds the selected adapter (S_k itself if the guard rolled
    back to step 0); ``summary.json`` marks completion and is returned on rerun.
    """
    cfg = FAGRPOConfig.from_dict(config)
    cfg.validate()
    sft_checkpoint = Path(sft_checkpoint)
    if not (sft_checkpoint / "adapter_config.json").is_file():
        raise ValueError(f"{sft_checkpoint}: FA-GRPO starts from an SFT LoRA adapter (S_k)")
    rows = list(pool_rows)
    if not rows:
        raise ValueError("empty GRPO pool")
    anchor_rows = list(anchor_rows or [])
    if cfg.anchor_lambda > 0 and not anchor_rows:
        raise ValueError(f"anchor_lambda={cfg.anchor_lambda} but no anchor rows: the route-only anchor would be off")
    check_anchor_context(cfg, sft_checkpoint)
    eids = [int(r["example_id"]) for r in rows]
    if len(set(eids)) != len(eids):
        raise ValueError("duplicate example_id in the GRPO pool")
    for r in rows:
        if r.get("ground_truth") not in r["choices"]:
            raise ValueError(f"example {r['example_id']}: ground truth is not a choice key")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    signature = run_signature(cfg, sft_checkpoint, rows, anchor_rows, advisor_pool)
    manifest = root / "training_run.json"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{root} already holds an FA-GRPO run; pass resume=True to continue it")
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != signature:
            diff = sorted(k for k in set(old) | set(signature) if old.get(k) != signature.get(k))
            raise ValueError(f"{root}: FA-GRPO inputs changed ({diff}); use a new output directory")
    elif any(root.iterdir()):
        raise ValueError(f"{root}: non-empty FA-GRPO output without a matching training_run.json")
    else:
        _write_json(manifest, signature)
    if (root / "summary.json").exists():
        return json.loads((root / "summary.json").read_text(encoding="utf-8"))

    from ...verifiable.rsi_grpo import committed_step_directories
    import torch
    from transformers import set_seed
    committed = committed_step_directories(root)
    reports = _reports(committed)
    if len(reports) != len(committed) or any(r["step"] != i + 1 for i, r in enumerate(reports)):
        raise ValueError("committed FA-GRPO steps are not 1..n")
    _prune(root, [p.name for p in committed])  # orphans / incomplete directories of an interrupted run
    guards = [r["guard"] for r in reports if r.get("guard")]
    _, failed = select_rollback(guards)
    if failed is not None or len(committed) >= cfg.steps:
        return _finalize(root, cfg, committed, sft_checkpoint, signature)

    set_seed(cfg.seed)
    tok, model = load_policy(cfg, sft_checkpoint, committed[-1] if committed else None)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    if not params or any(f".{REFERENCE}." in n for n, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("trainable parameters must be exactly the policy adapter")
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if committed:
        state = torch.load(committed[-1] / "optimizer.pt", map_location=next(model.parameters()).device,
                           weights_only=True)
        if state["step"] != len(committed):
            raise ValueError("optimizer state and committed steps disagree")
        optimizer.load_state_dict(state["optimizer"])
    bench = registry.get(cfg.bench)
    kinds = [k for k in ADVISOR_KINDS if advisor_pool.has(k)]
    if not kinds:
        raise ValueError("the advisor pool serves no advisor")
    scorer = _Scorer(tok, bench, kinds)
    features, anchor_stats = anchor_features(anchor_rows, tok, cfg)
    anchor_order = list(range(len(features)))
    random.Random(_seed(cfg.seed, "anchor")).shuffle(anchor_order)
    order = _plan(len(rows), cfg)
    probe = _probe_rows(rows, cfg)
    if hasattr(advisor_pool, "clear_abort"):
        advisor_pool.clear_abort()
    baseline_path = root / "baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:  # S_k's greedy behaviour on the probe, under the frozen reference adapter
        baseline = guard_probe(model, scorer, probe, adapter=REFERENCE, reference=None)
        baseline.update(probe_example_ids=[int(r["example_id"]) for r in probe], anchor=anchor_stats)
        _write_json(baseline_path, baseline)
    names = [p.name for p in committed]
    try:
        for step in range(len(committed) + 1, cfg.steps + 1):
            batch = [rows[order[((step - 1) * cfg.questions_per_step + i) % len(rows)]]
                     for i in range(cfg.questions_per_step)]
            anchor_batch = [features[anchor_order[((step - 1) * cfg.anchor_rows_per_step + i) % len(features)]]
                            for i in range(cfg.anchor_rows_per_step)] if features else []
            report = _train_step(step, model, tok, scorer, optimizer, params, batch, anchor_batch, advisor_pool, cfg)
            if step % cfg.guard_every == 0 or step == cfg.steps:
                report["guard"] = _guard(model, scorer, probe, baseline, cfg, step)
            names.append(_commit_step(root, step, model, optimizer, report))
            _advance_pointer(root, names, step)
            _prune(root, names)
            committed = [root / n for n in names]
            _write_metrics(root, _reports(committed))
            guard = report.get("guard")
            print(f"[MCQ_RSI/GRPO] step {step}/{cfg.steps} loss={report['loss']:.4f} kl_dec={report['kl_dec']:.2e} "
                  f"J={report['J_mean']:.3f} informative={report['n_informative']}/{report['n_states']}"
                  + (f" guard={'pass' if guard['passed'] else 'FAIL ' + '; '.join(guard['reasons'])}" if guard else ""))
            if guard and guard["enforced"] and not guard["passed"]:
                break
    except BaseException:
        if hasattr(advisor_pool, "abort"):
            advisor_pool.abort()
        raise
    del model, optimizer
    return _finalize(root, cfg, [root / n for n in names], sft_checkpoint, signature)


def _finalize(root: Path, cfg: FAGRPOConfig, committed, sft_checkpoint: Path, signature) -> Dict[str, Any]:
    reports = _reports(committed)
    _write_metrics(root, reports)
    guards = [r["guard"] for r in reports if r.get("guard")]
    selected, failed = select_rollback(guards)
    _copy_checkpoint(sft_checkpoint if selected == 0 else committed[selected - 1], root / "final", sft_checkpoint)
    info = informativeness([j for r in reports for q in r["questions"] for j in q["J"].values()],
                           cfg.informative_low, cfg.informative_high, cfg.informative_min_fraction)
    summary = {
        "version": GRPO_VERSION, "steps": len(reports), "planned_steps": cfg.steps,
        "accepted_by_guard": failed is None, "guard_failed_step": failed,
        "rollback_step": selected if failed is not None else None, "selected_step": selected,
        "final_dir": str(root / "final"),
        "final_adapter_sha256": _file_sha(root / "final" / "adapter_model.safetensors"),
        "reference_checkpoint": str(sft_checkpoint), "guard_enforced": not cfg.decision_pg,
        "guards": guards, "informativeness": info,
        "baseline": json.loads((root / "baseline.json").read_text(encoding="utf-8"))
        if (root / "baseline.json").exists() else None,
        "per_step": [{k: v for k, v in r.items() if k != "questions"} for r in reports],
        "config": asdict(cfg), "signature_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
    }
    _write_json(root / "summary.json", summary)
    _prune(root, [Path(p).name for p in committed], keep_optimizer=False)
    return summary
