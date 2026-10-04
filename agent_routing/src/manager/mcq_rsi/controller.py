"""Round-major MCQ RSI controller (design §3.2, §3.6, §5).

``build_plan`` turns a config, a phase, arms and a round count into a static list of
stages. Every stage has a name (its output directory under the run root), a kind, a
GPU lane and parameters whose checkpoint / label inputs are symbolic references,
resolved only when the stage runs, from files earlier stages wrote:

- ``import:S_1`` / ``import:labels``: the imported round-1 manager and its label file;
- ``out:<stage>::<relative path>``: a file or directory a stage produced;
- ``decision:<stage>[#field]``: a field (default ``checkpoint``) of ``<stage>/decision.json``.

This is what lets gates change the path without changing the plan: a GRPO stage whose
dev eval is rejected writes ``decision.json`` pointing at S_k, so ``G_k := S_k``
propagates to round k+1 (``rejected GRPO propagates S_k``).

Plan shape (main phase)::

    prefetch/advisors                         E/R for every pool, V(q, S_1 greedy root) for dev/test
    r1/S1_dev                                 eval(S_1, dev) + parity gate vs the registry paper targets
    r1/grpo, r1/grpo_dev                      FA-GRPO(S_1, grpo_r1) -> G_1, accept gate (shared by all arms)
    r2/collect                                collect(G_1, collect_r2), shared by every arm starting from G_1
    r2/<arm>/select                           dynamic | success; static: ``static/select`` (round-1 file, all rounds)
    r2/<arm>/{sft,sft_dev,grpo,grpo_dev}      SFT(init=G_1) -> S_2, flags; FA-GRPO(S_2) -> G_2, accept gate
    r3/<arm>/collect ...                      collect(G_2^arm, collect_r3) (static: none unless static_shadow_collect)

Stages are de-duplicated by content: two arms asking for the same kind with the same
symbolic parameters share one stage (round 1, ``r2/collect`` of the arms starting from the
shared G_1, and ``static/select``). ``dynamic_sft`` starts round 2 from S_1, not G_1, so it has
its own collection, selection and SFT. The pilot phase replaces ``r1/grpo`` by a
learning-rate sweep (``r1/grpo_lr<lr>`` + ``_dev``) and ``r1/grpo_select``, which picks
the accepted candidate with the best dev accuracy (ties: fewer calls, then smaller lr);
its lr is used by the later rounds.

Execution: one subprocess per stage (``python -m src.manager.mcq_rsi stage``), output
validated, then ``.mcq_rsi_complete.json`` written with the stage spec's sha. The run
signature (``rsi_run.json``: config, plan, sha256 of every file of this package
including prompts and of every ``src`` file it imports, directly or not (``imported_sources``), the
split manifest, the content of the import manifest (files and digests, not their
download status), git HEAD and ``harness_identity``) must match on every restart and in
every stage subprocess. The wall-clock deadline is persisted at the first start
(``budget.json``, cap 72 h).

A GRPO candidate whose dev eval fails its eval gate (invalid answers, malformed tool
calls) is rejected like any other candidate (``G_k := S_k``); advisor failures stay a
hard failure. SIGTERM/SIGHUP stop the controller and the stage's process group; a
restart refuses to run while a stage process of an earlier controller is alive, and a
stage subprocess holds ``.<stage>.stage.lock`` (next to its directory) while it runs. The locked test runs once
per benchmark: pilot runs are refused and every registration is recorded in
``<advisor_cache>/locked_test/<bench>.json``; another run directory of the same
benchmark needs ``reuse_test`` (a recorded reason).
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..marginal_value import ADVISOR_KINDS
from . import benchmarks as registry

CONTROLLER_VERSION = "mcq_rsi_controller/1"
MAX_HOURS = 72.0
MARKER = ".mcq_rsi_complete.json"
RUN_FILE = "rsi_run.json"
PHASES = ("main", "pilot")
MAX_ROUNDS = 3  # the split manifests hold collect_r2/r3 and grpo_r1..r3
# selection: label rule; grpo: whether the arm runs FA-GRPO each round.
ARM_SPECS = {
    "dynamic": {"selection": "dynamic", "grpo": True},
    "static": {"selection": "static", "grpo": True},
    "success": {"selection": "success", "grpo": True},
    "dynamic_sft": {"selection": "dynamic", "grpo": False},  # optional ablation: no GRPO (G_k := S_k)
}
LANES = ("inference", "train", "cpu")
PACKAGE = Path(__file__).resolve().parent
SRC = PACKAGE.parents[1]  # agent_routing/src
# Operational settings: they never change a stage's result, so they stay out of the run signature.
OPERATIONAL_KEYS = ("advisor_url", "advisor_workers", "gpus", "preflight", "wandb", "heartbeat_seconds")

DEFAULTS: Dict[str, Any] = {
    "bench": None,
    "base_model": registry.BASE_MODEL,
    "base_revision": registry.BASE_REVISION,
    "import_dir": "outputs/mcq_rsi/import",
    "split_manifest": None,  # default: the registry manifest
    "advisor_cache": "outputs/mcq_rsi/advisor_cache",
    "advisor_url": None,
    # "base": advisors are the base model with each role's prompt (the paper-era behaviour, D14; signed);
    # "lora": the benchmarks' trained adapters, served with renamed keys.
    "advisor_mode": "lora",
    "advisor_workers": 32,
    "gpus": {"inference": "0", "train": "1"},
    "rounds": 3,
    "arms": ["dynamic", "static", "success"],
    "dev_pool": "dev",
    "limits": {"dev": 0, "test": 0, "collect": 0, "grpo": 0},
    # recheck_n: cached dev outputs per kind regenerated one at a time after the concurrent prefetch.
    "prefetch": {"enabled": True, "verifier_pools": ["dev", "test"], "recheck_n": 20},
    "collect": {"max_depth": 2, "seed": 42, "batch_size": 1, "root_mode": "policy"},
    "select": {"rho": None, "seed": None},
    "sft": {},
    "grpo": {},
    "eval": {"speculative": True},
    "gates": {"acc_tolerance": 0.01, "extra_calls": 0.15, "gap_fraction": 0.5,
              "sft_max_calls": 1.5, "sft_max_acc_drop": 0.03},
    "parity": {"enforce": True, "accuracy_tolerance": 0.02, "calls_tolerance": 0.05},
    "pilot": {"learning_rates": [2e-6, 5e-6, 1e-5], "arms": ["dynamic"], "rounds": 2},
    # Single-role forced evals feed the matched-budget replay (every first role matched); forced-all is Table 3.
    "final": {"test_pools": ["test"], "forced": ["extractor", "reasoner", "verifier", "extractor,reasoner,verifier"]},
    "static_shadow_collect": False,
    # min_match: exact replay; else LoRA closer to the recorded output than the base on >= min_closer of the
    # items with median similarity >= min_similarity (bf16 greedy drifts on other GPUs/kernels; preflight.py).
    "preflight": {"required": True, "n_per_kind": 20, "min_match": 0.9, "min_lora_effect": 0.5, "min_closer": 0.75,
                  "min_similarity": 0.5, "seed": 0},
    "wandb": {"enabled": False, "entity": "madisonlijingxuan-ucla", "project": "MCQ_rsi"},
    "heartbeat_seconds": 15,
}
FREE_SECTIONS = ("sft", "grpo")  # validated by RoundSFTConfig / FAGRPOConfig instead


# ------------------------------------------------------------------------------ config

def _merge(base: Dict[str, Any], over: Dict[str, Any], where: str = "") -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if key.startswith("_"):  # comments
            continue
        if key not in base and where not in FREE_SECTIONS:
            raise ValueError(f"unknown config key {where + '.' if where else ''}{key}")
        if isinstance(base.get(key), dict) and isinstance(value, dict) and key not in FREE_SECTIONS:
            out[key] = _merge(base[key], value, key)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = Path(os.path.expanduser(str(path)))
    return str(p if p.is_absolute() else registry.PACKAGE_ROOT / p)


def load_config(source) -> Dict[str, Any]:
    """Config file (or dict) merged over ``DEFAULTS``; unknown keys and bad values are refused."""
    raw = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    cfg = _merge(DEFAULTS, raw)
    registry.get(cfg["bench"] or "")
    for key in ("import_dir", "advisor_cache", "split_manifest"):
        cfg[key] = _abs(cfg[key])
    if cfg["base_model"] and not Path(cfg["base_model"]).is_absolute() and (registry.PACKAGE_ROOT / cfg["base_model"]).is_dir():
        cfg["base_model"] = str(registry.PACKAGE_ROOT / cfg["base_model"])
    from .grpo import FAGRPOConfig
    from .sft import RoundSFTConfig
    FAGRPOConfig.from_dict({**cfg["grpo"], "bench": cfg["bench"]}).validate()
    RoundSFTConfig(**cfg["sft"]).validate()
    for name in ("bench", "base_model", "base_revision"):
        if name in cfg["grpo"]:
            raise ValueError(f"grpo.{name} is set by the controller")
    from .advisors import ADVISOR_MODES
    if cfg["advisor_mode"] not in ADVISOR_MODES:
        raise ValueError(f"advisor_mode must be one of {ADVISOR_MODES}")
    validate_arms_rounds(cfg["arms"], cfg["rounds"])
    for lr in cfg["pilot"]["learning_rates"]:
        if not (isinstance(lr, (int, float)) and math.isfinite(lr) and lr > 0):
            raise ValueError("pilot learning rates must be positive")
    if len(set(cfg["pilot"]["learning_rates"])) != len(cfg["pilot"]["learning_rates"]):
        raise ValueError("duplicate pilot learning rates")
    return cfg


def validate_arms_rounds(arms: Sequence[str], rounds: int) -> None:
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARM_SPECS):
        raise ValueError(f"arms must be unique members of {sorted(ARM_SPECS)}")
    if not isinstance(rounds, int) or not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"rounds must be in [1, {MAX_ROUNDS}]")


def signed_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in OPERATIONAL_KEYS}


# ------------------------------------------------------------------------------ planning

def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def lr_tag(lr: float) -> str:
    return f"{lr:g}"


class Planner:
    """Ordered stages, de-duplicated by (kind, params)."""

    def __init__(self):
        self.stages: List[Dict[str, Any]] = []
        self._by_key: Dict[str, str] = {}
        self._names: Dict[str, str] = {}

    def add(self, name: str, kind: str, lane: str, **params) -> str:
        if lane not in LANES:
            raise ValueError(f"lane {lane}")
        key = _sha({"kind": kind, "params": params})
        if key in self._by_key:
            return self._by_key[key]
        if name in self._names:
            raise ValueError(f"stage name {name} reused for different content")
        self._by_key[key] = name
        self._names[name] = key
        self.stages.append({"name": name, "kind": kind, "lane": lane, "params": params})
        return name


def build_plan(cfg: Dict[str, Any], phase: str = "main", arms: Optional[Sequence[str]] = None,
               rounds: Optional[int] = None) -> Dict[str, Any]:
    """The static stage list and, per arm and round, which stages produce its S_k / G_k."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if phase == "pilot":
        arms = list(arms or cfg["pilot"]["arms"])
        rounds = int(rounds or cfg["pilot"]["rounds"])
    else:
        arms = list(arms or cfg["arms"])
        rounds = int(rounds or cfg["rounds"])
    validate_arms_rounds(arms, rounds)
    P = Planner()
    dev = cfg["dev_pool"]
    if cfg["prefetch"]["enabled"]:
        pools = [dev] + [f"collect_r{k}" for k in range(2, rounds + 1)] + [f"grpo_r{k}" for k in range(1, rounds + 1)]
        pools += [p for p in cfg["final"]["test_pools"] if p not in pools]
        P.add("prefetch/advisors", "prefetch", "inference", pools=pools, checkpoint="import:S_1",
              verifier_pools=[p for p in cfg["prefetch"]["verifier_pools"] if p in pools])
    s1_dev = P.add("r1/S1_dev", "eval", "inference", checkpoint="import:S_1", pool=dev, role="s1")
    any_grpo = any(ARM_SPECS[a]["grpo"] for a in arms)
    lr_ref = None
    shared_g1 = f"decision:{s1_dev}"
    if any_grpo and phase == "pilot":
        candidates = []
        for lr in cfg["pilot"]["learning_rates"]:
            g = P.add(f"r1/grpo_lr{lr_tag(lr)}", "grpo", "train", checkpoint="import:S_1", pool="grpo_r1",
                      anchor="import:labels", learning_rate=float(lr))
            d = P.add(f"r1/grpo_lr{lr_tag(lr)}_dev", "eval", "inference", checkpoint=f"out:{g}::final", pool=dev,
                      role="grpo", baseline=s1_dev, grpo_stage=g, fallback="import:S_1")
            candidates.append({"learning_rate": float(lr), "grpo": g, "dev": d})
        sel = P.add("r1/grpo_select", "grpo_select", "cpu", candidates=candidates, baseline=s1_dev,
                    default_learning_rate=None)
        shared_g1, lr_ref = f"decision:{sel}", f"decision:{sel}#learning_rate"
    elif any_grpo:
        g = P.add("r1/grpo", "grpo", "train", checkpoint="import:S_1", pool="grpo_r1", anchor="import:labels")
        d = P.add("r1/grpo_dev", "eval", "inference", checkpoint=f"out:{g}::final", pool=dev, role="grpo",
                  baseline=s1_dev, grpo_stage=g, fallback="import:S_1")
        shared_g1 = f"decision:{d}"
    G = {arm: (shared_g1 if ARM_SPECS[arm]["grpo"] else f"decision:{s1_dev}") for arm in arms}
    by_arm: Dict[str, Dict[str, Dict[str, Any]]] = {arm: {"1": {"G": G[arm], "dev": G[arm]}} for arm in arms}
    for k in range(2, rounds + 1):
        for arm in arms:
            spec = ARM_SPECS[arm]
            stages: Dict[str, Any] = {}
            if spec["selection"] == "static":
                if cfg["static_shadow_collect"] and k >= 3:
                    stages["collect"] = P.add(f"r{k}/{arm}/collect", "collect", "inference", checkpoint=G[arm],
                                              pool=f"collect_r{k}", round=k)
                stages["select"] = P.add("static/select", "select", "cpu", selection="static", records=None)
            else:
                shared = k == 2 and G[arm] == shared_g1
                stages["collect"] = P.add(f"r{k}/collect" if shared else f"r{k}/{arm}/collect", "collect", "inference",
                                          checkpoint=G[arm], pool=f"collect_r{k}", round=k)
                stages["select"] = P.add(f"r{k}/{arm}/select", "select", "cpu", selection=spec["selection"],
                                         records=f"out:{stages['collect']}::counterfactual_records.jsonl")
            labels = f"out:{stages['select']}::labels.jsonl"
            stages["sft"] = P.add(f"r{k}/{arm}/sft", "sft", "train", labels=labels, init=G[arm])
            s_k = f"out:{stages['sft']}::model"
            stages["sft_dev"] = P.add(f"r{k}/{arm}/sft_dev", "eval", "inference", checkpoint=s_k, pool=dev, role="sft",
                                      previous=G[arm])
            if spec["grpo"]:
                extra = {"learning_rate_ref": lr_ref} if lr_ref else {}
                stages["grpo"] = P.add(f"r{k}/{arm}/grpo", "grpo", "train", checkpoint=s_k, pool=f"grpo_r{k}",
                                       anchor=labels, **extra)
                stages["grpo_dev"] = P.add(f"r{k}/{arm}/grpo_dev", "eval", "inference",
                                           checkpoint=f"out:{stages['grpo']}::final", pool=dev, role="grpo",
                                           baseline=stages["sft_dev"], grpo_stage=stages["grpo"], fallback=s_k)
                G[arm] = f"decision:{stages['grpo_dev']}"
            else:
                G[arm] = f"decision:{stages['sft_dev']}"
            stages["G"] = G[arm]
            by_arm[arm][str(k)] = stages
    return {"version": CONTROLLER_VERSION, "bench": cfg["bench"], "phase": phase, "arms": arms, "rounds": rounds,
            "stages": P.stages, "by_arm": by_arm, "finals": {"S_1": "import:S_1", **{a: G[a] for a in arms}}}


# ------------------------------------------------------------------------------ identity

def _file_sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_file(parts: Sequence[str]) -> Optional[Path]:
    path = SRC.joinpath(*parts) if parts else SRC
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    return path / "__init__.py" if (path / "__init__.py").is_file() else None


def imported_sources(roots: Iterable[Path]) -> List[Path]:
    """Every ``src`` file reachable from ``roots`` through static imports (relative or ``src.``),
    including the ``__init__.py`` of each package on the way. Imports inside functions count too."""
    seen: set = set()
    todo = [Path(r).resolve() for r in roots]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        here = list(f.relative_to(SRC).parts[:-1])
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"), filename=str(f))):
            mods: List[List[str]] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = here[:len(here) - (node.level - 1)]
                    mod = base + (node.module.split(".") if node.module else [])
                elif node.module and node.module.split(".")[0] == "src":
                    mod = node.module.split(".")[1:]
                else:
                    continue
                mods += [mod] + [mod + [a.name] for a in node.names]
            elif isinstance(node, ast.Import):
                mods += [a.name.split(".")[1:] for a in node.names if a.name.split(".")[0] == "src"]
            for mod in mods:
                for i in range(1, len(mod) + 1):
                    hit = _module_file(mod[:i])
                    if hit is not None and hit.resolve() not in seen:
                        todo.append(hit.resolve())
    return sorted(seen)


def code_files() -> List[Path]:
    files = [f for f in PACKAGE.rglob("*") if f.is_file() and "__pycache__" not in f.parts
             and f.suffix in (".py", ".txt", ".jinja", ".json")]
    own = {f.resolve() for f in files}
    extra = [f for f in imported_sources(f for f in files if f.suffix == ".py") if f not in own]
    return sorted(files) + extra


def git_head() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=registry.PACKAGE_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def code_identity() -> Dict[str, Any]:
    from ...verifiable.provenance import harness_identity
    files = {str(f.relative_to(SRC)): _file_sha(f) for f in code_files()}
    return {"files": files, "sha256": _sha(files), "git_head": git_head(), "harness": harness_identity()}


def import_content(path) -> Optional[Dict[str, Any]]:
    """What an import manifest says landed where: version, base and (dest, sha256) of every file.

    A re-import of the same files rewrites each entry's ``status`` (``downloaded``/``extracted``
    -> ``present``), so the manifest bytes change while the imported content does not.
    """
    path = Path(path)
    if not path.is_file():
        return None
    m = _read_json(path)
    return {"import_version": m.get("import_version"), "benchmark": m.get("benchmark"),
            "base_model": m.get("base_model"), "base_revision": m.get("base_revision"),
            "lora_names": m.get("lora_names"),
            "files": sorted([str(f["dest"]), str(f.get("sha256"))] for f in m.get("files", []))}


def manifest_identity(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bench = registry.get(cfg["bench"])
    split = Path(cfg["split_manifest"]) if cfg["split_manifest"] else bench.path(bench.split_manifest)
    content = import_content(Path(cfg["import_dir"]) / bench.name / "import_manifest.json")
    return {"split_manifest": str(split), "split_manifest_sha256": _file_sha(split) if split.is_file() else None,
            "import_content_sha256": _sha(content) if content is not None else None}


def run_signature(cfg: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps({
        "version": CONTROLLER_VERSION, "config": signed_config(cfg), "plan": plan, "code": code_identity(),
        "manifests": manifest_identity(cfg)}, sort_keys=True))


def signature_diff(old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> List[str]:
    keys = sorted(set(old) | set(new))
    out = []
    for k in keys:
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict) and len(out) < 20:
            out += signature_diff(a, b, f"{prefix}{k}.")
        else:
            out.append(f"{prefix}{k}")
    return out


def _write_json(path, value) -> None:
    from .collect import _atomic_write
    _atomic_write(Path(path), json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def _read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run(root) -> Dict[str, Any]:
    path = Path(root) / RUN_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path}: not an MCQ RSI run directory")
    return _read_json(path)


def check_code_unchanged(run: Dict[str, Any]) -> None:
    """Refuse to run any stage under code, packages or manifests other than the run started with."""
    cfg = run["config_full"]
    now = {"code": code_identity(), "manifests": manifest_identity(cfg)}
    old = {"code": run["signature"]["code"], "manifests": run["signature"]["manifests"]}
    if json.loads(json.dumps(now, sort_keys=True)) != old:
        raise RuntimeError(f"code/manifests changed since the run started ({signature_diff(old, now)[:10]}); "
                           "restore them or start a new run directory")


CODE_OVERRIDE = Path("final") / "code_override.json"


def _manifest_content(m: Dict[str, Any]) -> Dict[str, Any]:
    """The manifests' content hashes (not the checkout path they were read from)."""
    return {k: v for k, v in m.items() if k != "split_manifest"}


def code_override(root, run_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The recorded final-test code override, if it covers exactly the code running now (same content
    manifests). Final stages only: the RSI stages always ran under the run's own code."""
    path = Path(root) / CODE_OVERRIDE
    if not path.is_file():
        return None
    ov = _read_json(path)
    now_code = json.loads(json.dumps(code_identity(), sort_keys=True))
    if (_sha(now_code) == ov.get("new_code_identity_sha256")
            and _manifest_content(manifest_identity(run_info["config_full"]))
            == _manifest_content(run_info["signature"]["manifests"])):
        return ov
    return None


def check_code_or_override(root, run_info: Dict[str, Any], stage_name: Optional[str] = None) -> None:
    """``check_code_unchanged``, except that final-test stages may run under a recorded code override."""
    try:
        check_code_unchanged(run_info)
    except RuntimeError:
        if (stage_name is None or stage_name.startswith("final/")) and code_override(root, run_info) is not None:
            return
        raise


def code_override_preconditions(run_info: Dict[str, Any], reason: Optional[str]) -> None:
    if not (reason or "").strip():
        raise ValueError("--allow-code-change needs a reason")
    if (_manifest_content(manifest_identity(run_info["config_full"]))
            != _manifest_content(run_info["signature"]["manifests"])):
        raise RuntimeError("the split/import manifests changed; a code override never covers data changes")


def record_code_override(root, run_info: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Allow the remaining final-test stages to run under the current code (``final-test --allow-code-change``).

    Only the code may differ (the manifests' content must be unchanged). The record names the reason and both
    code identities and is bound to the current code's hash; completed stages are never redone."""
    code_override_preconditions(run_info, reason)
    now_code = json.loads(json.dumps(code_identity(), sort_keys=True))
    old_code = run_info["signature"]["code"]
    path = Path(root) / CODE_OVERRIDE
    history = _read_json(path).get("history", []) if path.is_file() else []
    if path.is_file():
        history.append({k: v for k, v in _read_json(path).items() if k != "history"})
    record = {"reason": reason.strip(), "unix": time.time(),
              # files-only hashes (comparable) and the full identity the override is bound to
              "old_code_sha256": old_code.get("sha256"), "new_code_sha256": now_code.get("sha256"),
              "new_code_identity_sha256": _sha(now_code),
              "old_git_head": old_code.get("git_head"), "new_git_head": now_code.get("git_head"),
              "changed": signature_diff({"code": old_code}, {"code": now_code})[:50], "history": history}
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, record)
    return record


# ------------------------------------------------------------------------------ deadline

def persistent_deadline(root, hours: float, now: Optional[float] = None) -> Dict[str, Any]:
    """The run's wall-clock deadline, fixed at the first start (a restart never buys more time)."""
    if not (isinstance(hours, (int, float)) and 0 < hours <= MAX_HOURS):
        raise ValueError(f"--hours must be in (0, {MAX_HOURS:g}]")
    path = Path(root) / "budget.json"
    if path.exists():
        budget = _read_json(path)
        if float(hours) != float(budget["hours"]):
            print(f"[MCQ_RSI] persisted deadline kept: {budget['hours']} h from "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(budget['started_unix']))} "
                  f"(ignoring --hours {hours})", flush=True)
        return budget
    started = time.time() if now is None else now
    budget = {"hours": float(hours), "started_unix": started, "deadline_unix": started + float(hours) * 3600}
    _write_json(path, budget)
    return budget


# ------------------------------------------------------------------------------ runtime

class Runtime:
    """Everything a stage touches outside the run directory (tests replace methods)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.bench = registry.get(cfg["bench"])
        self._pool = None
        self._manifest = None
        self._base = None

    def pool(self):
        if self._pool is None:
            from .advisors import CachedAdvisorPool
            self._pool = CachedAdvisorPool(self.bench.name, self.cfg["advisor_cache"], self.cfg["advisor_url"],
                                           workers=int(self.cfg["advisor_workers"]), mode=self.cfg["advisor_mode"])
            if self.cfg["advisor_url"]:
                self._pool.check_server()
                if self.cfg["preflight"]["required"]:
                    from .preflight import require_passed
                    require_passed(self.cfg["advisor_cache"], self._pool)
        return self._pool

    def manifest(self):
        if self._manifest is None:
            from . import splits
            path = self.cfg["split_manifest"] or self.bench.path(self.bench.split_manifest)
            self._manifest = splits.read_manifest(path)
        return self._manifest

    def limit(self, pool: str) -> int:
        lim = self.cfg["limits"]
        group = "collect" if pool.startswith("collect_") else "grpo" if pool.startswith("grpo_") else \
            "test" if pool.startswith("test") else "dev"
        return int(lim.get(group) or 0)

    def rows(self, pool: str) -> List[Dict[str, Any]]:
        from . import splits
        rows = splits.pool_rows(self.manifest(), pool)
        n = self.limit(pool)
        return rows[:n] if n else rows

    def base(self) -> str:
        if self._base is None:
            from .evaluate import resolve_base
            self._base = resolve_base(self.cfg["base_model"], self.cfg["base_revision"] or None)
        return self._base

    def manager(self, checkpoint: str):
        from . import protocol
        backend = protocol.load_hf_manager(checkpoint, self.base(), self.cfg["base_revision"] or None,
                                           int(self.cfg["collect"]["batch_size"]))
        return protocol.Manager(backend)

    def first_turns(self, checkpoint: str, rows, out_dir):
        from . import evaluate
        srows = evaluate.standard_rows(rows)
        return srows, evaluate.first_turns(evaluate.stage_context(self.base(), out_dir), checkpoint, srows, self.bench)

    def import_path(self, what: str) -> Path:
        from . import importer
        paths = importer.default_paths(self.bench, self.cfg["import_dir"])
        if what == "S_1":
            return Path(self.cfg["import_dir"]) / self.bench.name / "round1" / "sft"
        if what == "labels":
            return Path(paths["labels"])
        raise ValueError(f"unknown import reference {what}")


def resolve(root, rt: Runtime, ref: Optional[str]):
    if ref is None:
        return None
    if ref.startswith("import:"):
        return str(rt.import_path(ref[len("import:"):]))
    if ref.startswith("out:"):
        stage, sep, rel = ref[len("out:"):].partition("::")
        return str(Path(root) / stage / rel) if sep else str(Path(root) / stage)
    if ref.startswith("decision:"):
        stage, _, field = ref[len("decision:"):].partition("#")
        path = Path(root) / stage / "decision.json"
        if not path.is_file():
            raise RuntimeError(f"{ref}: {path} does not exist (stage {stage} not complete)")
        return _read_json(path)[field or "checkpoint"]
    raise ValueError(f"unknown reference {ref!r}")


# ------------------------------------------------------------------------------ gates

def parity_check(bench: registry.Benchmark, metrics: Dict[str, Any], cfg: Dict[str, Any], split: str = "dev",
                 subset: bool = False) -> Dict[str, Any]:
    """S_1 vs the registry paper targets: accuracy within +-2 pt, calls within +-0.05 (call gap reported only).

    ``*_paper_calls`` (AQuA) is the paper's ambiguous "Calls" column: it passes if either
    ``calls_per_example`` or ``call_rate`` is within tolerance.
    """
    p = cfg["parity"]
    targets = {k[len(split) + 1:]: v for k, v in bench.paper_targets if k.startswith(split + "_")}
    if not targets:
        return {"status": "not_applicable", "reason": f"no paper {split} targets for {bench.name}", "checks": []}
    checks = []
    for name, target in targets.items():
        if name == "accuracy":
            cands, tol, gated = {"accuracy": metrics["accuracy"]}, p["accuracy_tolerance"], True
        elif name == "avg_tool_calls":
            cands, tol, gated = {"calls_per_example": metrics["calls_per_example"]}, p["calls_tolerance"], True
        elif name == "call_rate":
            cands, tol, gated = {"call_rate": metrics["call_rate"]}, p["calls_tolerance"], True
        elif name == "paper_calls":
            cands = {"calls_per_example": metrics["calls_per_example"], "call_rate": metrics["call_rate"]}
            tol, gated = p["calls_tolerance"], True
        elif name == "call_gap":
            cands, tol, gated = {"call_gap": metrics["call_gap"]}, None, False
        else:
            continue
        deltas = {m: v - target for m, v in cands.items()}
        ok = (not gated) or any(abs(d) <= tol + 1e-12 for d in deltas.values())
        checks.append({"target": f"{split}_{name}", "value": target, "observed": cands, "deltas": deltas,
                       "tolerance": tol, "gated": gated, "passed": ok})
    if subset:
        return {"status": "subset", "reason": "eval limited to a subset; not comparable to the paper", "checks": checks}
    return {"status": "pass" if all(c["passed"] for c in checks) else "fail", "checks": checks}


def acks(root) -> Dict[str, Any]:
    path = Path(root) / "acks.json"
    return _read_json(path) if path.is_file() else {}


def ack_gate(root, stage: str, reason: str) -> Dict[str, Any]:
    """Operator acknowledgement that a gate failure of ``stage`` may be passed (recorded, not in the signature)."""
    if not reason.strip():
        raise ValueError("an acknowledgement needs a reason")
    data = acks(root)
    data[stage] = {"reason": reason, "unix": time.time()}
    _write_json(Path(root) / "acks.json", data)
    return data


# ------------------------------------------------------------------------------ stages

def _read_jsonl(path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _eval_result(stage_dir) -> Dict[str, Any]:
    return _read_json(Path(stage_dir) / "mcq_rsi_eval.json")


def stage_prefetch(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    from .advisors import AdvisorRequest
    p = spec["params"]
    pool = rt.pool()
    counts: Dict[str, Any] = {}
    draft_free = [k for k in ADVISOR_KINDS if k != "verifier"]
    for name in p["pools"]:
        rows = rt.rows(name)
        counts[name] = pool.prefetch([AdvisorRequest.for_row(k, r) for r in rows for k in draft_free])
    s1 = resolve(root, rt, p["checkpoint"])
    dev_verifier: List[Any] = []
    for name in p["verifier_pools"]:
        srows, turns = rt.first_turns(s1, rt.rows(name), out)
        by_id = {r.example_id: r.to_dict() for r in srows}
        requests = [AdvisorRequest.for_row("verifier", by_id[t["example_id"]], t["draft"]) for t in turns if t["draft"]]
        counts[f"{name}:verifier@S_1"] = {**pool.prefetch(requests), "roots_without_draft": sum(not t["draft"] for t in turns)}
        if name == rt.cfg["dev_pool"]:
            dev_verifier = requests
    result = {"counts": counts, "advisor_stats": dict(getattr(pool, "stats", {}))}
    # The cache was filled ``advisor_workers`` requests at a time; preflight and the paper servers generated
    # one at a time, and vLLM is not batch-invariant: regenerate a sample of the dev entries sequentially.
    n = int(rt.cfg["prefetch"].get("recheck_n") or 0)
    if n and rt.cfg["dev_pool"] in p["pools"]:
        from .preflight import sequential_recheck
        dev_rows = rt.rows(rt.cfg["dev_pool"])
        requests = [AdvisorRequest.for_row(k, r) for k in draft_free for r in dev_rows] + dev_verifier
        result["sequential_recheck"] = sequential_recheck(pool, requests, n, seed=int(rt.cfg["preflight"]["seed"]))
    _write_json(out / "prefetch.json", result)
    return result


INFRA_GATE_PREFIXES = ("advisor failures",)  # an eval-gate failure that is not the checkpoint's behaviour
FORCED_MIN_VALID = 0.98  # a forced (analysis) eval tolerates a few invalid answers, recorded; fewer valid ones stop


def stage_eval(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    from . import evaluate
    p, cfg = spec["params"], rt.cfg
    role = p.get("role")
    checkpoint = resolve(root, rt, p["checkpoint"])
    rows = rt.rows(p["pool"])
    # A GRPO candidate's own eval-gate failure (invalid answers, malformed tool calls) rejects the
    # candidate (G_k := S_k, ``grpo_accept``) instead of stopping the run; S_1/S_k evals stay hard gates.
    lenient = role == "grpo"
    result = evaluate.evaluate(checkpoint, rows, rt.pool(), out, bench=rt.bench.name, base_model=cfg["base_model"],
                               base_revision=cfg["base_revision"] or None, speculative=bool(cfg["eval"]["speculative"]),
                               require_gate=not lenient)
    if result.get("gate") and any(g.startswith(INFRA_GATE_PREFIXES) for g in result["gate"]):
        raise RuntimeError(f"eval gate failed (advisor infrastructure): {result['gate']}")
    own = str(out / "mcq_rsi_eval.json")
    decision: Dict[str, Any] = {"role": role, "checkpoint": checkpoint, "dev_result": own,
                                "metrics": result["metrics"]}
    if role == "s1":
        parity = parity_check(rt.bench, result["metrics"], cfg, "dev", subset=bool(rt.limit(p["pool"])))
        ack = acks(root).get(spec["name"])
        parity["acknowledged"] = ack
        recheck = Path(root) / "prefetch" / "advisors" / "prefetch.json"
        if recheck.is_file():  # batched-vs-sequential agreement of the advisor outputs this parity rests on
            parity["advisor_sequential_recheck"] = _read_json(recheck).get("sequential_recheck")
        _write_json(out / "parity.json", parity)
        decision["parity"] = parity["status"]
        if parity["status"] == "fail" and cfg["parity"]["enforce"] and not ack:
            raise RuntimeError(f"round-1 parity gate failed: {parity['checks']}; inspect {out}/parity.json, fix the "
                               f"cause or acknowledge with `ack-gate --stage {spec['name']} --reason ...`")
    elif role == "sft":  # never selected on dev, only flagged against the previous round's resolved G_(k-1)
        prev_dec = _decision_of(root, rt, p["previous"])
        prev_metrics = _read_json(prev_dec["dev_result"])["metrics"]
        g = cfg["gates"]
        flags = evaluate.sft_flags(result, {"metrics": prev_metrics}, max_calls=g["sft_max_calls"],
                                   max_acc_drop=g["sft_max_acc_drop"])
        decision.update(flags=flags, previous_checkpoint=prev_dec["checkpoint"], previous_metrics=prev_metrics)
    elif role == "grpo":
        base = _eval_result(Path(root) / p["baseline"])
        summary = _read_json(Path(root) / p["grpo_stage"] / "summary.json")
        g = cfg["gates"]
        accept = evaluate.grpo_accept(result, base, bool(summary["informativeness"]["passed"]),
                                      acc_tolerance=g["acc_tolerance"], extra_calls=g["extra_calls"],
                                      gap_fraction=g["gap_fraction"])
        accept.update(grpo_steps=summary["steps"], selected_step=summary["selected_step"],
                      accepted_by_guard=summary["accepted_by_guard"], candidate_gate=list(result.get("gate") or []))
        decision["accept"] = accept
        if not accept["accepted"]:  # G_k := S_k
            base_dec = _read_json(Path(root) / p["baseline"] / "decision.json")
            decision.update(checkpoint=resolve(root, rt, p["fallback"]), dev_result=base_dec["dev_result"],
                            metrics=base["metrics"], rejected_checkpoint=checkpoint)
    _write_json(out / "decision.json", decision)
    return decision


def _decision_of(root, rt, ref: str) -> Dict[str, Any]:
    if not ref.startswith("decision:"):
        raise ValueError(f"{ref}: expected a decision reference")
    stage = ref[len("decision:"):].partition("#")[0]
    return _read_json(Path(root) / stage / "decision.json")


def stage_grpo_select(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    """Pilot: the accepted lr candidate with the best dev accuracy (ties: fewer calls, smaller lr); none -> S_1."""
    p = spec["params"]
    rows = []
    for c in p["candidates"]:
        dec = _read_json(Path(root) / c["dev"] / "decision.json")
        own = _eval_result(Path(root) / c["dev"])["metrics"]
        rows.append({**c, "accepted": dec["accept"]["accepted"], "reasons": dec["accept"]["reasons"],
                     "accuracy": own["accuracy"], "calls_per_example": own["calls_per_example"],
                     "call_gap": own["call_gap"], "checkpoint": dec["checkpoint"], "dev_result": dec["dev_result"]})
    accepted = [r for r in rows if r["accepted"]]
    default_lr = p.get("default_learning_rate") or rt.cfg["grpo"].get("learning_rate")
    if default_lr is None:
        from .grpo import FAGRPOConfig
        default_lr = FAGRPOConfig().learning_rate
    if accepted:
        best = max(accepted, key=lambda r: (r["accuracy"], -r["calls_per_example"], -r["learning_rate"]))
        decision = {"checkpoint": best["checkpoint"], "dev_result": best["dev_result"],
                    "learning_rate": best["learning_rate"], "selected": best["grpo"], "rule": "accepted, max dev accuracy"}
    else:
        base = _read_json(Path(root) / p["baseline"] / "decision.json")
        decision = {"checkpoint": base["checkpoint"], "dev_result": base["dev_result"], "learning_rate": default_lr,
                    "selected": None, "rule": "no candidate accepted: G_1 := S_1, later rounds use the default lr"}
    decision.update(role="grpo_select", candidates=rows,
                    metrics=_read_json(decision["dev_result"])["metrics"])
    _write_json(out / "decision.json", decision)
    return decision


def stage_grpo(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    from . import grpo
    p, cfg = spec["params"], rt.cfg
    checkpoint = resolve(root, rt, p["checkpoint"])
    config = dict(cfg["grpo"])
    config.update(bench=rt.bench.name, base_model=rt.base(), base_revision=cfg["base_revision"] or None)
    recorded = grpo.recorded_sft_context(checkpoint)
    if recorded:  # the anchor renders S_k's tool block as S_k's SFT did (as ``cmd_grpo``)
        config["anchor_context"] = recorded
    if p.get("learning_rate") is not None:
        config["learning_rate"] = float(p["learning_rate"])
    if p.get("learning_rate_ref"):
        config["learning_rate"] = float(resolve(root, rt, p["learning_rate_ref"]))
    anchor = _read_jsonl(resolve(root, rt, p["anchor"]))
    started = time.time()
    summary = grpo.train_fa_grpo(config, checkpoint, rt.rows(p["pool"]), anchor, rt.pool(), out, resume=True)
    info = {"steps": summary["steps"], "selected_step": summary["selected_step"],
            "informative": summary["informativeness"]["passed"], "learning_rate": config.get("learning_rate"),
            "wall_seconds_this_attempt": time.time() - started, "peak_memory_gb": _peak_memory_gb()}
    _write_json(out / "controller_stage.json", info)  # smoke check: peak memory < 60 GB
    return info


def _peak_memory_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2 ** 30
    except Exception:  # noqa: BLE001 - informational only
        pass
    return None


def stage_collect(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    from . import collect
    p, c = spec["params"], rt.cfg["collect"]
    checkpoint = resolve(root, rt, p["checkpoint"])
    result = collect.collect(rt.rows(p["pool"]), rt.bench, rt.manager(checkpoint), rt.pool(), out,
                             pool_name=p["pool"], round_index=int(p["round"]), root_mode=c["root_mode"],
                             max_depth=int(c["max_depth"]), seed=int(c["seed"]),
                             resume=(out / "collect_manifest.json").exists())
    return {"n": result["report"]["n_examples"], "direct_accuracy": result["report"]["direct_accuracy"]}


def stage_select(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    from . import select
    p, s = spec["params"], rt.cfg["select"]
    records = resolve(root, rt, p["records"])
    report = select.write_selection(rt.bench, p["selection"], records, out / "labels.jsonl", rho=s["rho"],
                                    seed=s["seed"], import_dir=rt.cfg["import_dir"])
    return {"rows": report["n_sft_turns"], "sha256": report["sha256"]}


def stage_sft(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    from . import sft
    p, cfg = spec["params"], rt.cfg
    report = sft.train_round_sft(resolve(root, rt, p["labels"]), resolve(root, rt, p["init"]), out,
                                 base_model=cfg["base_model"], base_revision=cfg["base_revision"] or None,
                                 bench=rt.bench.name, config=cfg["sft"])
    return {"adapter_sha256": report["adapter_sha256"], "rows": report["labels"]["rows"]}


def stage_test(root, rt: Runtime, spec, out: Path) -> Dict[str, Any]:
    from . import evaluate
    p, cfg = spec["params"], rt.cfg
    rows = rt.rows(p["pool"])
    if p.get("forced"):
        # Forced-delegation dev evals are analysis only (matched-budget replay, Tables 4-5): an invalid answer is
        # recorded in final.json, not a stop; the locked test evals below stay hard gates.
        result = evaluate.evaluate_forced(p["checkpoint"], rows, rt.pool(), out, p["forced"].split(","),
                                          bench=rt.bench.name, base_model=cfg["base_model"],
                                          base_revision=cfg["base_revision"] or None, require_gate=False)
        if result.get("gate") and any(g.startswith(INFRA_GATE_PREFIXES) for g in result["gate"]):
            raise RuntimeError(f"eval gate failed (advisor infrastructure): {result['gate']}")
        valid = (result.get("metrics") or {}).get("valid_answer_rate")
        if result.get("gate") and not (isinstance(valid, (int, float)) and valid >= FORCED_MIN_VALID):
            raise RuntimeError(f"forced eval broken beyond the tolerated invalid answers (valid_answer_rate {valid} "
                               f"< {FORCED_MIN_VALID}): {result['gate']}")
    else:
        result = evaluate.evaluate(p["checkpoint"], rows, rt.pool(), out, bench=rt.bench.name,
                                   base_model=cfg["base_model"], base_revision=cfg["base_revision"] or None,
                                   speculative=bool(cfg["eval"]["speculative"]))
    out_metrics = {"metrics": result["metrics"], "label": p["label"], "pool": p["pool"], "forced": p.get("forced"),
                   "gate": list(result.get("gate") or [])}
    if p["label"] == "S_1" and not p.get("forced"):
        out_metrics["parity"] = parity_check(rt.bench, result["metrics"], cfg, "test", subset=bool(rt.limit(p["pool"])))
    _write_json(out / "final.json", out_metrics)
    return out_metrics


STAGE_FUNCS: Dict[str, Callable] = {
    "prefetch": stage_prefetch, "eval": stage_eval, "grpo_select": stage_grpo_select, "grpo": stage_grpo,
    "collect": stage_collect, "select": stage_select, "sft": stage_sft, "test": stage_test,
}


def validate_stage(spec: Dict[str, Any], out: Path) -> None:
    kind = spec["kind"]
    need = {
        "prefetch": ["prefetch.json"], "eval": ["mcq_rsi_eval.json", "decision.json"],
        "grpo_select": ["decision.json"], "grpo": ["summary.json", "final/adapter_model.safetensors"],
        "collect": ["counterfactual_records.jsonl", "marginal_value_report.json"],
        "select": ["labels.jsonl", "labels.report.json"], "sft": ["sft_report.json", "model/adapter_model.safetensors"],
        "test": ["final.json"],
    }[kind]
    missing = [n for n in need if not (out / n).exists()]
    if missing:
        raise RuntimeError(f"stage {spec['name']} is missing {missing}")
    if kind == "eval" and not _eval_result(out)["passed"] and not _gate_failure_rejected(spec, out):
        raise RuntimeError(f"stage {spec['name']}: eval gate failed")
    if kind == "select":
        report = _read_json(out / "labels.report.json")
        if report["sha256"] != _file_sha(out / "labels.jsonl"):
            raise RuntimeError(f"stage {spec['name']}: labels do not match their report")


def _gate_failure_rejected(spec: Dict[str, Any], out: Path) -> bool:
    """A GRPO candidate eval that failed its gate is complete once its decision rejected it (G_k := S_k)."""
    if spec["params"].get("role") != "grpo" or not (out / "decision.json").is_file():
        return False
    accept = _read_json(out / "decision.json").get("accept") or {}
    return accept.get("accepted") is False and bool(accept.get("candidate_gate"))


@contextlib.contextmanager
def stage_lock(out: Path):
    """Held by whoever executes a stage: a second process on the same stage directory is refused.

    The lock file sits next to the stage directory (``.<name>.stage.lock``): stages such as FA-GRPO
    refuse a non-empty output directory they did not write."""
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    handle = open(stage_lock_path(out), "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"{out} is being executed by another process (an orphaned stage of an earlier "
                           "controller?); see `status` (stage_pid)") from None
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def stage_lock_path(out: Path) -> Path:
    return Path(out).parent / f".{Path(out).name}.stage.lock"


def run_stage(root, run: Dict[str, Any], spec: Dict[str, Any], rt: Optional[Runtime] = None) -> Dict[str, Any]:
    """Execute one stage in this process (what the ``stage`` subprocess runs)."""
    rt = rt or Runtime(run["config_full"])
    out = Path(root) / spec["name"]
    with stage_lock(out):
        return STAGE_FUNCS[spec["kind"]](Path(root), rt, spec, out)


def find_spec(root, run: Dict[str, Any], name: str) -> Dict[str, Any]:
    for spec in run["plan"]["stages"]:
        if spec["name"] == name:
            return spec
    finals = Path(root) / "final" / "finals.json"
    if finals.is_file():
        for spec in _read_json(finals)["stages"]:
            if spec["name"] == name:
                return spec
    raise KeyError(f"no stage {name} in {root}")


def is_complete(root, spec) -> bool:
    return (Path(root) / spec["name"] / MARKER).is_file()


# ------------------------------------------------------------------------------ execution

class Status:
    def __init__(self, root, planned: int, deadline: float, heartbeat: float = 15.0):
        self.path = Path(root) / "status.json"
        self.state = {"controller_pid": os.getpid(), "planned_stages": planned, "deadline_unix": deadline,
                      "started_unix": time.time()}
        self.heartbeat, self._last = heartbeat, 0.0

    def update(self, force: bool = True, **fields) -> None:
        self.state.update(fields)
        now = time.time()
        if force or now - self._last >= self.heartbeat:
            self.state["heartbeat_unix"] = now
            _write_json(self.path, self.state)
            self._last = now


def _kill_group(process) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_subprocess(command: List[str], logfile: Path, env: Dict[str, str], deadline: float,
                   status: Optional[Status] = None, poll: float = 1.0) -> float:
    """Run a stage command in its own process group; kill it at the deadline. Returns wall seconds."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with logfile.open("a") as log:
        log.write(f"\n[MCQ_RSI] {time.strftime('%Y-%m-%d %H:%M:%S')} $ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=registry.PACKAGE_ROOT,
                                   start_new_session=True, env=env)
        if status is not None:  # recorded at once: a restart refuses to run while this process group lives
            status.update(force=True, stage_pid=process.pid, stage_command=shlex.join(command))
        try:
            while process.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError("wall-clock deadline reached; completed stages are kept")
                if status is not None:
                    status.update(force=False, stage_pid=process.pid)
                time.sleep(poll)
            if process.returncode:
                raise RuntimeError(f"stage failed (exit {process.returncode}); see {logfile}")
        except BaseException:
            _kill_group(process)
            raise
        finally:
            if status is not None and process.poll() is not None:
                status.update(force=True, stage_pid=None, stage_command=None)
    return time.monotonic() - start


class ControllerSignal(BaseException):
    """SIGTERM/SIGHUP reached the controller: unwind so the running stage's process group is killed."""


@contextlib.contextmanager
def stop_on_signals():
    """Turn SIGTERM/SIGHUP into ``ControllerSignal`` (main thread only; later signals are ignored while unwinding)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    fired: List[int] = []

    def handler(signum, frame):
        if fired:
            return
        fired.append(signum)
        raise ControllerSignal(f"controller received {signal.Signals(signum).name}")

    sigs = [sig for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if sig is not None]
    old = {sig: signal.signal(sig, handler) for sig in sigs}
    try:
        yield
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)


def _cmdline(pid: int) -> Optional[str]:
    proc = Path(f"/proc/{pid}/cmdline")
    try:
        if proc.exists():
            return proc.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def live_stage_process(pid) -> Optional[str]:
    """The command line of the stage process group ``pid`` if it is still alive, else None.

    A stage runs in its own session, so its group id is its pid. A live group whose leader
    runs something other than this package is a reused pid, not an orphaned stage.
    """
    if not pid:
        return None
    try:
        os.killpg(int(pid), 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass
    cmd = _cmdline(int(pid))
    if cmd is not None and "src.manager.mcq_rsi" not in cmd:
        return None
    return cmd or "(leader exited; group still alive)"


def refuse_orphaned_stage(root) -> None:
    path = Path(root) / "status.json"
    if not path.is_file():
        return
    st = _read_json(path)
    cmd = live_stage_process(st.get("stage_pid"))
    if cmd:
        raise RuntimeError(f"a stage of an earlier controller is still running (process group {st['stage_pid']}: "
                           f"{cmd[:200]}); wait for it or stop it with `kill -TERM -- -{st['stage_pid']}`, then resume")


def stage_command(root, name: str) -> List[str]:
    return [sys.executable, "-m", "src.manager.mcq_rsi", "stage", "--run-dir", str(root), "--name", name]


def stage_env(cfg: Dict[str, Any], lane: str) -> Dict[str, str]:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if lane in ("inference", "train"):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"][lane])
    return env


def execute(root, run: Dict[str, Any], spec: Dict[str, Any], deadline: float, *, executor: str = "subprocess",
            rt: Optional[Runtime] = None, status: Optional[Status] = None) -> str:
    """Run one stage unless complete; validate; write its marker. Returns 'done' or 'ran'."""
    root = Path(root)
    out = root / spec["name"]
    marker = out / MARKER
    spec_sha = _sha(spec)
    if marker.is_file():
        done = _read_json(marker)
        if done["spec_sha256"] != spec_sha:
            raise RuntimeError(f"completed stage {spec['name']} was produced by a different spec")
        validate_stage(spec, out)
        return "done"
    if time.time() >= deadline:
        raise TimeoutError("wall-clock deadline reached; completed stages are kept")
    start = time.monotonic()
    if executor == "inprocess":
        run_stage(root, run, spec, rt)
    elif executor == "subprocess":
        cfg = run["config_full"]
        run_subprocess(stage_command(root, spec["name"]), root / "logs" / (spec["name"].replace("/", "__") + ".log"),
                       stage_env(cfg, spec["lane"]), deadline, status)
    else:
        raise ValueError(f"executor {executor}")
    validate_stage(spec, out)
    _write_json(marker, {"stage": spec["name"], "kind": spec["kind"], "spec_sha256": spec_sha,
                         "signature_sha256": run["signature_sha256"], "wall_seconds": time.monotonic() - start,
                         "finished_unix": time.time()})
    return "ran"


@contextlib.contextmanager
def run_lock(root):
    """One controller per run directory (a second ``run`` on the same directory is refused)."""
    import fcntl
    path = Path(root) / "controller.lock"
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"{root} is in use by another controller (lock {path})") from None
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def prepare_run(config_path, out, *, phase: str = "main", arms=None, rounds=None) -> Tuple[Dict[str, Any], Path]:
    cfg = load_config(config_path)
    plan = build_plan(cfg, phase, arms, rounds)
    root = Path(out).resolve()
    signature = run_signature(cfg, plan)
    run = {"signature": signature, "signature_sha256": _sha(signature), "config_full": cfg, "plan": plan,
           "config_path": str(config_path) if not isinstance(config_path, dict) else None}
    return run, root


def start(run: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Write or check ``rsi_run.json``: an existing run directory only resumes the identical run."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / RUN_FILE
    if path.exists():
        old = _read_json(path)
        if old["signature"] != run["signature"]:
            diff = signature_diff(old["signature"], run["signature"])
            raise ValueError(f"{root}: run settings changed ({diff[:12]}); choose a new output directory")
        # Operational settings (advisor URL, GPUs, ...) may change between restarts.
        old["config_full"] = run["config_full"]
        _write_json(path, old)
        return old
    if any(p.name not in ("controller.lock",) for p in root.iterdir()):
        raise ValueError(f"{root}: non-empty directory without {RUN_FILE}")
    _write_json(path, run)
    return run


def run(config_path, out, *, phase: str = "main", arms=None, rounds=None, hours: float = MAX_HOURS,
        dry_run: bool = False, executor: str = "subprocess", rt: Optional[Runtime] = None) -> Dict[str, Any]:
    run_info, root = prepare_run(config_path, out, phase=phase, arms=arms, rounds=rounds)
    if dry_run:
        for spec in run_info["plan"]["stages"]:
            print(f"{spec['name']:<34} {spec['kind']:<12} {spec['lane']:<9} {json.dumps(spec['params'], sort_keys=True)}")
        return run_info["plan"]
    if not (isinstance(hours, (int, float)) and 0 < hours <= MAX_HOURS):
        raise ValueError(f"--hours must be in (0, {MAX_HOURS:g}]")
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root), stop_on_signals():
        refuse_orphaned_stage(root)
        run_info = start(run_info, root)
        budget = persistent_deadline(root, hours)
        deadline = budget["deadline_unix"]
        stages = run_info["plan"]["stages"]
        status = Status(root, len(stages), deadline, float(run_info["config_full"]["heartbeat_seconds"]))
        if executor == "subprocess" and run_info["config_full"]["advisor_url"]:
            Runtime(run_info["config_full"]).pool()  # server identity before any stage
        try:
            for index, spec in enumerate(stages):
                status.update(controller="running", current_stage=spec["name"], current_kind=spec["kind"],
                              completed_stages=sum(is_complete(root, s) for s in stages),
                              log=str(root / "logs" / (spec["name"].replace("/", "__") + ".log")))
                if execute(root, run_info, spec, deadline, executor=executor, rt=rt, status=status) == "ran":
                    print(f"[MCQ_RSI] completed {spec['name']} ({index + 1}/{len(stages)})", flush=True)
                    report(root)
        except BaseException as exc:
            state = "deadline" if isinstance(exc, TimeoutError) else (
                "interrupted" if isinstance(exc, (KeyboardInterrupt, ControllerSignal)) else "failed")
            status.update(controller=state, error=f"{type(exc).__name__}: {exc}")
            with contextlib.suppress(Exception):
                report(root)
            raise
        result = report(root)
        status.update(controller="completed", current_stage=None, completed_stages=len(stages))
        return result


# ------------------------------------------------------------------------------ final test

def resolve_finals(root, run: Dict[str, Any], rt: Runtime, allow_incomplete: bool) -> Dict[str, Dict[str, Any]]:
    """Pre-registered finals: S_1 and the last-round G_R (or S_R if rejected) of each arm.

    With ``allow_incomplete`` an arm whose last round did not finish falls back to the
    latest round whose G_k resolved, flagged ``truncated_at_round``.
    """
    plan = run["plan"]
    finals = {"S_1": {"ref": "import:S_1", "checkpoint": resolve(root, rt, "import:S_1"), "round": 1}}
    for arm in plan["arms"]:
        rounds = sorted(plan["by_arm"][arm], key=int)
        chosen = None
        for k in reversed(rounds):
            ref = plan["by_arm"][arm][k]["G"]
            stage = ref[len("decision:"):].partition("#")[0]
            if is_complete(root, find_spec(root, run, stage)):
                chosen = (k, ref)
                break
            if not allow_incomplete:
                raise RuntimeError(f"{arm}: round {k} not complete")
        if chosen is None:
            raise RuntimeError(f"{arm}: no round resolved")
        k, ref = chosen
        finals[arm] = {"ref": ref, "checkpoint": resolve(root, rt, ref), "round": int(k),
                       **({"truncated_at_round": int(k)} if int(k) != plan["rounds"] else {})}
    return finals


def final_stages(run: Dict[str, Any], finals: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    cfg = run["config_full"]
    stages, by_ckpt = [], {}
    for label, f in finals.items():
        owner = by_ckpt.setdefault(f["checkpoint"], label)
        if owner != label:
            continue  # identical checkpoint: evaluated once under the first label
        for pool in cfg["final"]["test_pools"]:
            stages.append({"name": f"final/{label}/{pool}", "kind": "test", "lane": "inference",
                           "params": {"checkpoint": f["checkpoint"], "pool": pool, "label": label}})
        for tools in cfg["final"]["forced"]:
            tag = tools.replace(",", "+")
            stages.append({"name": f"final/{label}/{cfg['dev_pool']}_forced_{tag}", "kind": "test", "lane": "inference",
                           "params": {"checkpoint": f["checkpoint"], "pool": cfg["dev_pool"], "label": label,
                                      "forced": tools}})
    return stages


def locked_test_registry(cfg: Dict[str, Any]) -> Path:
    """Per-benchmark record of every run directory that registered the locked test (outside any run directory)."""
    return Path(cfg["advisor_cache"]) / "locked_test" / f"{cfg['bench']}.json"


def register_locked_test(root: Path, run_info: Dict[str, Any], finals: Dict[str, Dict[str, Any]],
                         reuse_test: Optional[str]) -> Optional[Dict[str, Any]]:
    """Each locked test set runs once per benchmark: a second run directory needs a recorded ``reuse_test`` reason."""
    cfg = run_info["config_full"]
    pools = list(cfg["final"]["test_pools"])
    if not pools:
        return None  # dev-only finals (smoke): nothing locked is touched
    path = locked_test_registry(cfg)
    reg = _read_json(path) if path.is_file() else {"bench": cfg["bench"], "registrations": []}
    mine = [r for r in reg["registrations"] if r["run_dir"] == str(root)]
    if mine:
        return mine[0]
    others = [r for r in reg["registrations"] if set(r["test_pools"]) & set(pools)]
    if others and not (reuse_test or "").strip():
        raise RuntimeError(f"the {cfg['bench']} locked test ({sorted(set(pools))}) was already registered by "
                           f"{[r['run_dir'] for r in others]} ({path}); each locked test set runs once. "
                           "A deliberate second use needs --reuse-test REASON (recorded)")
    entry = {"run_dir": str(root), "phase": run_info["plan"]["phase"], "signature_sha256": run_info["signature_sha256"],
             "test_pools": pools, "finals": {k: v["checkpoint"] for k, v in finals.items()}, "unix": time.time(),
             "reuse_reason": reuse_test if others or run_info["plan"]["phase"] == "pilot" else None,
             "previous_registrations": len(others)}
    reg["registrations"].append(entry)
    _write_json(path, reg)
    return entry


def final_test(out, *, accept_incomplete: Optional[str] = None, executor: str = "subprocess",
               rt: Optional[Runtime] = None, dry_run: bool = False, reuse_test: Optional[str] = None,
               allow_code_change: Optional[str] = None) -> Dict[str, Any]:
    """Locked test, once per pre-registered final; refused before every planned stage is complete
    unless ``accept_incomplete`` gives a reason (recorded in ``final/incomplete.json``).

    Pilot runs and a second run directory of the same benchmark are refused unless ``reuse_test``
    gives a reason (recorded in the per-benchmark registry ``locked_test_registry``).

    ``allow_code_change`` (a reason) lets the remaining final stages run under code other than the run's
    (``final/code_override.json``, bound to the current code; completed stages are kept)."""
    root = Path(out).resolve()
    run_info = load_run(root)
    if allow_code_change is None:
        check_code_or_override(root, run_info)
    else:
        code_override_preconditions(run_info, allow_code_change)  # also for --dry-run
    rt = rt or Runtime(run_info["config_full"])
    if run_info["plan"]["phase"] == "pilot" and not dry_run and not (reuse_test or "").strip():
        raise RuntimeError("a pilot run never runs the locked test (its checkpoints are not pre-registered finals); "
                           "pass --reuse-test REASON to override (recorded)")
    pending = [s["name"] for s in run_info["plan"]["stages"] if not is_complete(root, s)]
    if pending and not accept_incomplete:
        raise RuntimeError(f"{len(pending)} planned stages are not complete ({pending[:5]}...); the locked test "
                           "runs only after the plan, or pass --accept-incomplete REASON")
    with run_lock(root), stop_on_signals():
        refuse_orphaned_stage(root)
        if allow_code_change is not None and not dry_run:
            try:
                check_code_unchanged(run_info)
            except RuntimeError:
                record_code_override(root, run_info, allow_code_change)
        finals = resolve_finals(root, run_info, rt, bool(pending))
        stages = final_stages(run_info, finals)
        lock = root / "final" / "finals.json"
        registered = {"finals": finals, "stages": stages, "incomplete": pending, "reason": accept_incomplete}
        if lock.exists():
            old = _read_json(lock)
            if old["finals"] != json.loads(json.dumps(finals)) or old["stages"] != json.loads(json.dumps(stages)):
                raise RuntimeError(f"{lock}: the pre-registered finals changed; the locked test is never re-targeted")
        else:
            if dry_run:
                return registered
            registered["locked_test"] = register_locked_test(root, run_info, finals, reuse_test)
            _write_json(lock, registered)
            if pending:
                _write_json(root / "final" / "incomplete.json", {"reason": accept_incomplete, "stages": pending,
                                                                 "unix": time.time()})
        if dry_run:
            return registered
        budget = _read_json(root / "budget.json")
        deadline = max(budget["deadline_unix"], time.time() + 12 * 3600)  # the finals are not cut by the RSI budget
        status = Status(root, len(stages), deadline, float(run_info["config_full"]["heartbeat_seconds"]))
        try:
            for spec in stages:
                status.update(controller="final-test", current_stage=spec["name"], current_kind=spec["kind"])
                execute(root, run_info, spec, deadline, executor=executor, rt=rt, status=status)
        except BaseException as exc:
            status.update(controller="failed" if isinstance(exc, Exception) else "interrupted",
                          error=f"{type(exc).__name__}: {exc}")
            raise
        status.update(controller="final-test complete", current_stage=None)
    return report(root)


# ------------------------------------------------------------------------------ report

def _round_of(name: str) -> Optional[int]:
    head = name.split("/", 1)[0]
    return int(head[1:]) if head.startswith("r") and head[1:].isdigit() else None


def _eval_rows(path) -> Dict[int, Dict[str, Any]]:
    return {int(r["example_id"]): r for r in _read_jsonl(path)}


def drift(base: Dict[int, Dict[str, Any]], rows: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Paired change on the same dev ids (candidate flips, final flips, call-count changes)."""
    ids = sorted(set(base) & set(rows))
    if set(base) != set(rows):
        return {"comparable": False, "n_common": len(ids)}
    cand = lambda r: bool(r.get("initial_draft_correct"))
    return {"comparable": True, "n": len(ids),
            "candidate_new_correct": sum(not cand(base[i]) and cand(rows[i]) for i in ids),
            "candidate_regressed": sum(cand(base[i]) and not cand(rows[i]) for i in ids),
            "final_new_correct": sum(not base[i]["correct"] and rows[i]["correct"] for i in ids),
            "final_regressed": sum(base[i]["correct"] and not rows[i]["correct"] for i in ids),
            "calls_changed": sum(base[i]["tool_calls"] != rows[i]["tool_calls"] for i in ids),
            "call_decision_changed": sum((base[i]["tool_calls"] > 0) != (rows[i]["tool_calls"] > 0) for i in ids)}


def collection_tables(records: Sequence[Dict[str, Any]], report: Dict[str, Any]) -> Dict[str, Any]:
    """Paper Tables 1-2 analogues of one recollection."""
    n = len(records)
    a0 = report["direct_accuracy"]
    a1 = sum(bool(r.get("direct_correct")) or any(b.get("correct") and len(b["sequence"]) == 1 for b in r["branches"])
             for r in records) / max(1, n)
    ad = report["oracle_accuracy"]
    return {"n": n, "commit_A0": a0, "best_one_call_A1": a1, "best_measured_AD": ad,
            "gain_at_1": (a1 - a0) / (ad - a0) if ad > a0 else None, "n_unsolved": report["n_unsolved"],
            "net_marginal_pp": {k: 100 * v["net_marginal_rate"] for k, v in report["by_advisor_one_step"].items()},
            "rescue_rate": {k: v["rescue_rate"] for k, v in report["by_advisor_one_step"].items()},
            "corruption_rate": {k: v["corruption_rate"] for k, v in report["by_advisor_one_step"].items()},
            "preferred_depth_counts": report["preferred_depth_counts"],
            "policy_call_rate": report.get("policy_call_rate"), "policy_action_counts": report.get("policy_action_counts"),
            "unconstrained_argmax": report.get("unconstrained_argmax")}


EVAL_KEYS = ("n", "accuracy", "initial_draft_accuracy", "gain_pp", "calls_per_example", "call_rate", "call_gap",
             "correction_rate", "corruption_rate", "per_advisor_calls", "duplicate_tool_calls")


def report(out) -> Dict[str, Any]:
    root = Path(out)
    run_info = load_run(root)
    plan = run_info["plan"]
    bench = registry.get(plan["bench"])
    K = len(bench.choice_keys)
    stages = plan["stages"]
    complete = {s["name"]: is_complete(root, s) for s in stages}
    base_rows = None
    s1 = root / "r1/S1_dev" / "manager_tool_eval.jsonl"
    if complete.get("r1/S1_dev") and s1.is_file():
        base_rows = _eval_rows(s1)
    dev, collections, labels, grpo_runs, decisions = [], [], [], [], []
    for spec in stages:
        name, out_dir = spec["name"], root / spec["name"]
        if not complete[name]:
            continue
        users = sorted(arm for arm, rounds in plan["by_arm"].items() for k, st in rounds.items()
                       if isinstance(st, dict) and name in st.values())
        if spec["kind"] == "eval":
            res = _eval_result(out_dir)
            m = res["metrics"]
            dec = _read_json(out_dir / "decision.json")
            row = {"stage": name, "round": _round_of(name), "arms": users, "role": spec["params"].get("role"),
                   **{k: m.get(k) for k in EVAL_KEYS}, "margin": m["initial_draft_accuracy"] - 1.0 / K,
                   "gate": res.get("gate") or []}
            own_rows = _eval_rows(out_dir / "manager_tool_eval.jsonl")
            if base_rows is not None:
                row["drift_vs_S1"] = drift(base_rows, own_rows)
            ref = _reference_eval(root, spec)
            if ref is not None:  # round over round: S_k vs G_(k-1), G_k vs S_k (same dev ids)
                row["drift_vs_previous"] = {"reference": ref[0], **drift(_eval_rows(ref[1]), own_rows)}
            dev.append(row)
            entry = {"stage": name, "role": dec.get("role"), "checkpoint": dec.get("checkpoint")}
            for key in ("accept", "flags", "parity"):
                if key in dec:
                    entry[key] = dec[key]
            decisions.append(entry)
        elif spec["kind"] == "collect":
            records = _read_jsonl(out_dir / "counterfactual_records.jsonl")
            collections.append({"stage": name, "round": _round_of(name),
                                **collection_tables(records, _read_json(out_dir / "marginal_value_report.json"))})
        elif spec["kind"] == "select":
            r = _read_json(out_dir / "labels.report.json")
            labels.append({"stage": name, "arm": spec["params"]["selection"], "rows": r.get("n_sft_turns"),
                           "decision_types": r.get("decision_types"),
                           "rescue_decisions": r.get("n_selected_rescue_decisions"),
                           "commit_decisions": r.get("n_selected_commit_decisions"),
                           "depth_counts": r.get("selected_depth_counts"), "sha256": r.get("sha256")})
        elif spec["kind"] == "grpo":
            s = _read_json(out_dir / "summary.json")
            grpo_runs.append({"stage": name, "steps": s["steps"], "selected_step": s["selected_step"],
                              "accepted_by_guard": s["accepted_by_guard"], "guard_failed_step": s["guard_failed_step"],
                              "informative_fraction": s["informativeness"]["fraction"],
                              "informative": s["informativeness"]["passed"],
                              "learning_rate": s["config"]["learning_rate"]})
        elif spec["kind"] == "grpo_select":
            dec = _read_json(out_dir / "decision.json")
            decisions.append({"stage": name, "role": "grpo_select", "checkpoint": dec["checkpoint"],
                              "learning_rate": dec["learning_rate"], "selected": dec["selected"],
                              "candidates": [{k: c[k] for k in ("learning_rate", "accepted", "accuracy",
                                                                  "calls_per_example", "reasons")}
                                             for c in dec["candidates"]]})
    arms = {}
    for arm, rounds in plan["by_arm"].items():
        timeline = []
        for k in sorted(rounds, key=int):
            stage = rounds[k]["G"][len("decision:"):].partition("#")[0]
            path = root / stage / "decision.json"
            if path.is_file() and complete.get(stage):
                dec = _read_json(path)
                if dec.get("role") == "grpo_select":  # pilot: no accepted lr candidate -> G_1 := S_1
                    rejected = dec.get("selected") is None
                else:
                    rejected = bool(dec.get("accept") and not dec["accept"]["accepted"])
                timeline.append({"round": int(k), "G_stage": stage, "checkpoint": dec["checkpoint"],
                                 "grpo_rejected": rejected,
                                 **{m: dec["metrics"].get(m) for m in ("accuracy", "calls_per_example", "call_gap")}})
        arms[arm] = timeline
    finals = []
    for path in sorted((root / "final").glob("*/*/final.json")):
        if (path.parent / MARKER).is_file():
            f = _read_json(path)
            m = f["metrics"]
            finals.append({"stage": str(path.parent.relative_to(root)), "label": f["label"], "pool": f["pool"],
                           "forced": f["forced"], "n": m["n"], "accuracy": m["accuracy"],
                           "candidate": m.get("initial_draft_accuracy"), "gain_pp": m.get("gain_pp"),
                           "calls_per_example": m["calls_per_example"], "call_gap": m.get("call_gap"),
                           "valid_answer_rate": m.get("valid_answer_rate"), "gate": list(f.get("gate") or []),
                           **({"parity": f["parity"]["status"]} if "parity" in f else {})})
    budget_path = root / "budget.json"
    result = {
        "version": CONTROLLER_VERSION, "bench": plan["bench"], "phase": plan["phase"], "arms": plan["arms"],
        "rounds": plan["rounds"], "completed_stages": sum(complete.values()), "planned_stages": len(stages),
        "complete": all(complete.values()), "pending": [n for n, ok in complete.items() if not ok],
        "budget": _read_json(budget_path) if budget_path.is_file() else None,
        "dev": dev, "arms_timeline": arms, "collections": collections, "labels": labels, "grpo": grpo_runs,
        "decisions": decisions, "finals": finals, "test_sets_used": bool(finals),
        "scope": "one seed per cell; dev is used for gates only; the locked test runs once (final-test)",
    }
    _write_json(root / "report.json", result)
    (root / "report.md").write_text(render_markdown(result), encoding="utf-8")
    _write_csv(root / "dev_metrics.csv", dev, ["stage", "round", "role", "n", "accuracy", "initial_draft_accuracy",
                                               "gain_pp", "calls_per_example", "call_rate", "call_gap",
                                               "correction_rate", "corruption_rate", "margin"])
    _wandb_log(run_info["config_full"], result, f"{root.resolve().name}_{run_info['signature_sha256'][:8]}", root)
    return result


def _reference_eval(root: Path, spec: Dict[str, Any]) -> Optional[Tuple[str, Path]]:
    """The eval a dev eval is gated against: G_(k-1)'s for an SFT eval, S_k's for a GRPO eval."""
    p = spec["params"]
    try:
        if p.get("role") == "sft" and p.get("previous", "").startswith("decision:"):
            stage = p["previous"][len("decision:"):].partition("#")[0]
            records = Path(_read_json(root / stage / "decision.json")["dev_result"]).parent / "manager_tool_eval.jsonl"
            return stage, records
        if p.get("role") == "grpo" and p.get("baseline"):
            return p["baseline"], root / p["baseline"] / "manager_tool_eval.jsonl"
    except (OSError, KeyError, ValueError):
        return None
    return None


def _write_csv(path, rows, fields) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(x, digits=3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render_markdown(r: Dict[str, Any]) -> str:
    out = [f"# MCQ RSI report: {r['bench']} ({r['phase']}, arms {', '.join(r['arms'])}, R={r['rounds']})", "",
           f"Stages {r['completed_stages']}/{r['planned_stages']} complete. {r['scope']}.", ""]
    def drift_cells(x):
        if not x or not x.get("comparable"):
            return ["-", "-", "-"]
        return [f"+{x['candidate_new_correct']}/-{x['candidate_regressed']}",
                f"+{x['final_new_correct']}/-{x['final_regressed']}", x["calls_changed"]]

    out += ["## Dev evaluations (per round)", "",
            "Drift vs S_1 and vs the previous checkpoint (S_k vs G_(k-1), G_k vs S_k) on the same dev ids.", "",
            _table(["stage", "arms", "acc", "candidate", "gain pp", "calls", "call rate", "call gap", "correction",
                    "corruption", "S_1 drift cand +/-", "S_1 drift final +/-", "S_1 calls changed",
                    "prev drift final +/-", "prev calls changed", "gate"],
                   [[d["stage"], ",".join(d["arms"]) or "all", d["accuracy"], d["initial_draft_accuracy"], d["gain_pp"],
                     d["calls_per_example"], d["call_rate"], d["call_gap"], d["correction_rate"], d["corruption_rate"],
                     *drift_cells(d.get("drift_vs_S1")), *drift_cells(d.get("drift_vs_previous"))[1:],
                     "; ".join(d.get("gate") or []) or "pass"]
                    for d in r["dev"]]), ""]
    out += ["## Arms: resolved checkpoint per round (G_k, or S_k when GRPO was rejected)", ""]
    for arm, timeline in r["arms_timeline"].items():
        out.append(_table([f"{arm}: round", "G stage", "GRPO rejected", "acc", "calls", "call gap"],
                          [[t["round"], t["G_stage"], t["grpo_rejected"], t["accuracy"], t["calls_per_example"],
                            t["call_gap"]] for t in timeline]))
        out.append("")
    if r["collections"]:
        out += ["## Recollections (Tables 1-2 analogues)", "",
                _table(["stage", "n", "A0 commit", "A1 best 1 call", "AD best", "Gain@1", "unsolved", "E 100Δ",
                        "R 100Δ", "V 100Δ", "policy call rate"],
                       [[c["stage"], c["n"], c["commit_A0"], c["best_one_call_A1"], c["best_measured_AD"], c["gain_at_1"],
                         c["n_unsolved"], *[c["net_marginal_pp"].get(k) for k in ADVISOR_KINDS], c["policy_call_rate"]]
                        for c in r["collections"]]), ""]
    if r["labels"]:
        out += ["## Label mix", "", _table(["stage", "rule", "rows", "commit", "call", "commit after call", "rescue dec",
                                            "commit dec"],
                                           [[x["stage"], x["arm"], x["rows"], *[(x["decision_types"] or {}).get(t) for t in
                                                                               ("commit", "call", "commit_after_call")],
                                             x["rescue_decisions"], x["commit_decisions"]] for x in r["labels"]]), ""]
    if r["grpo"]:
        out += ["## FA-GRPO", "", _table(["stage", "lr", "steps", "selected step", "guard ok", "informative frac"],
                                         [[g["stage"], g["learning_rate"], g["steps"], g["selected_step"],
                                           g["accepted_by_guard"], g["informative_fraction"]] for g in r["grpo"]]), ""]
    if r["decisions"]:
        out += ["## Gates", ""]
        for d in r["decisions"]:
            if "accept" in d:
                out.append(f"- {d['stage']}: {d['accept']['decision']} {d['accept']['reasons'] or ''}")
            if d.get("flags"):
                out.append(f"- {d['stage']}: SFT flags {d['flags']}")
            if "parity" in d:
                out.append(f"- {d['stage']}: parity {d['parity']}")
            if d.get("role") == "grpo_select":
                out.append(f"- {d['stage']}: selected lr {d['learning_rate']} ({d['selected']})")
        out.append("")
    if r["finals"]:
        out += ["## Locked test and forced baselines (Table 3 analogue)", "",
                _table(["stage", "label", "pool", "forced", "n", "candidate", "acc", "gain pp", "calls", "call gap",
                        "gate"],
                       [[f["stage"], f["label"], f["pool"], f["forced"] or "-", f["n"], f["candidate"], f["accuracy"],
                         f["gain_pp"], f["calls_per_example"], f["call_gap"],
                         "; ".join(f.get("gate") or []) or "pass"] for f in r["finals"]]), ""]
    if r["pending"]:
        out += ["## Pending stages", "", ", ".join(r["pending"]), ""]
    return "\n".join(out)


WANDB_DEV = ("stage", "round", "role", "n", "accuracy", "initial_draft_accuracy", "gain_pp", "calls_per_example",
             "call_rate", "call_gap", "correction_rate", "corruption_rate")
WANDB_GRPO = ("loss", "kl_root", "kl_dec", "kl_rev", "pg_loss", "anchor_ce", "grad_norm", "root_accuracy",
              "train_call_rate", "J_mean", "learning_rate", "step_seconds")
WANDB_GRPO_PLOTS = ("loss", "kl_dec", "kl_root", "train_call_rate", "J_mean", "informative_fraction")


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def grpo_step_rows(metrics_path: Path) -> List[Dict[str, Any]]:
    """One row per committed FA-GRPO step (``metrics.jsonl``): scalars, informative fraction, guard scalars."""
    rows = []
    for r in _read_jsonl(metrics_path) if Path(metrics_path).is_file() else []:
        row = {"step": r.get("step"), **{k: r.get(k) for k in WANDB_GRPO if _number(r.get(k))}}
        if r.get("n_states"):
            row["informative_fraction"] = r.get("n_informative", 0) / r["n_states"]
        row.update({f"guard_{k}": v for k, v in (r.get("guard") or {}).items() if _number(v)})
        rows.append(row)
    return rows


def _wandb_log(cfg: Dict[str, Any], result: Dict[str, Any], run_key: str, root: Optional[Path] = None) -> None:
    """After every completed stage: dev metrics (summary + table) and, per FA-GRPO stage, its step table and
    curves. One W&B run per run directory and signature (the smoke run is MedQA/main too). Never fails a run."""
    w = cfg.get("wandb") or {}
    if not w.get("enabled") or os.environ.get("MARGENT_WANDB_MODE") == "disabled":
        return
    try:
        import wandb
    except ImportError:
        return
    try:
        name = f"mcq_rsi_{result['bench']}_{result['phase']}_{run_key}"
        run = wandb.init(entity=w.get("entity"), project=w.get("project"), name=name,
                         id=re.sub(r"[^A-Za-z0-9_-]", "_", name)[:120], resume="allow", reinit=True,
                         config={"bench": result["bench"], "phase": result["phase"], "arms": result["arms"],
                                 "rounds": result["rounds"]})
        log: Dict[str, Any] = {"progress/completed_stages": result["completed_stages"],
                               "progress/planned_stages": result["planned_stages"]}
        run.summary.update({f"dev/{d['stage']}/{k}": d[k] for d in result["dev"] for k in WANDB_DEV[3:] if _number(d.get(k))})
        if result["dev"]:
            log["dev/table"] = wandb.Table(columns=list(WANDB_DEV), data=[[d.get(c) for c in WANDB_DEV] for d in result["dev"]])
        for g in result["grpo"] if root is not None else []:
            rows = grpo_step_rows(Path(root) / g["stage"] / "metrics.jsonl")
            if not rows:
                continue
            cols = ["step"] + sorted({k for r in rows for k in r} - {"step"})
            table = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])
            log[f"grpo/{g['stage']}/steps"] = table
            for m in WANDB_GRPO_PLOTS:
                if m in cols:
                    log[f"grpo/{g['stage']}/{m}"] = wandb.plot.line(table, "step", m, title=f"{g['stage']} {m}")
            run.summary.update({f"grpo/{g['stage']}/{k}": g[k] for k in ("selected_step", "informative_fraction",
                                                                          "learning_rate") if _number(g.get(k))})
        run.log(log)
        run.finish()
    except Exception as e:  # noqa: BLE001 - W&B is monitoring only
        print(f"[MCQ_RSI] W&B logging failed ({type(e).__name__}: {e}); the run continues", flush=True)


# ------------------------------------------------------------------------------ status

def status(out) -> Dict[str, Any]:
    root = Path(out)
    run_info = load_run(root)
    st = _read_json(root / "status.json") if (root / "status.json").is_file() else {}
    rows = []
    for spec in run_info["plan"]["stages"]:
        marker = root / spec["name"] / MARKER
        state = "done" if marker.is_file() else ("running" if st.get("current_stage") == spec["name"]
                                                 and st.get("controller") == "running" else "pending")
        rows.append({"stage": spec["name"], "kind": spec["kind"], "state": state,
                     "wall_seconds": _read_json(marker)["wall_seconds"] if marker.is_file() else None})
    now = time.time()
    return {"status": st, "stages": rows, "stage_process_alive": bool(live_stage_process(st.get("stage_pid"))),
            "heartbeat_age_seconds": now - st["heartbeat_unix"] if st.get("heartbeat_unix") else None,
            "remaining_hours": (st["deadline_unix"] - now) / 3600 if st.get("deadline_unix") else None}


def retry_stage(out, name: str) -> Path:
    """Move a failed (incomplete) stage directory aside so the next ``run`` redoes it (never deletes)."""
    root = Path(out)
    run_info = load_run(root)
    spec = find_spec(root, run_info, name)
    if is_complete(root, spec):
        raise RuntimeError(f"{name} is complete; completed stages are never redone")
    if name.startswith("final/") and spec["params"]["pool"] != run_info["config_full"]["dev_pool"]:
        raise RuntimeError("a locked-test stage is never retried; investigate its evidence instead")
    src = root / name
    if not src.exists():
        raise FileNotFoundError(src)
    dest = src.with_name(f"{src.name}.failed-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(src, dest)
    return dest
