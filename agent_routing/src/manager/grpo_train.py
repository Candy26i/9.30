"""Manager GRPO training loop.

Wires together:
  - SubagentPool (three frozen tools) OR RemoteSubagentPool (vLLM HTTP server)
  - ManagerToolEnvironment OR argument-binding tools
  - GRPOTrainer with the binary correctness reward
  - W&B logging (optional)

For full-parameter GRPO on 8B+ models with 4 GPUs, set subagent_server_url in
ManagerGRPOConfig to point at a vLLM server (GPU 0) serving the three subagents.
The training processes (GPUs 1-3) then carry zero subagent weight, leaving all
VRAM for ZeRO Stage 3 manager sharding. See scripts/start_subagent_server.sh
and scripts/train_manager_grpo_multigpu.sh for the launch workflow.
"""
from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except Exception:
    TRL_AVAILABLE = False

try:
    from trl.chat_template_utils import add_response_schema
    HAS_RESP_SCHEMA = True
except Exception:
    HAS_RESP_SCHEMA = False

from ..benchmarks.base import StandardRow, question_hash as _question_hash
from ..subagents.runtime import FrozenSubagent, RemoteSubagentPool, SubagentPool
from ..utils.io import read_jsonl, write_json
from ..utils.seed import set_seed
from ..subagents.train import load_text_causal_model, training_device
from .prompt import build_manager_system_prompt, build_manager_user_message
from .reward import build_reward_funcs
from .routing_anchor import ANCHOR_MODES, build_anchor_features


def _grpo_supports_environment_factory() -> bool:
    if not TRL_AVAILABLE:
        return False
    try:
        sig = inspect.signature(GRPOTrainer.__init__)
        return "environment_factory" in sig.parameters
    except Exception:
        return False


# ----------------------- Tool environment & registry -----------------------

# Module-level state used by the tool functions and environment. We keep this
# minimal and reset it at the start of every train_manager_grpo() call.
_POOL: Optional[SubagentPool] = None
_ROW_INDEX: Dict[int, StandardRow] = {}


def _ensure_pool() -> SubagentPool:
    if _POOL is None:
        raise RuntimeError("SubagentPool not initialized. Call _init_globals first.")
    return _POOL


def _init_globals(pool: SubagentPool, rows: List[StandardRow]) -> None:
    global _POOL, _ROW_INDEX
    _POOL = pool
    _ROW_INDEX = {int(r.example_id): r for r in rows}


def _run_tool(
    agent_kind: str,
    example_id: int,
    namespace: str = "default",
    candidate_answer: str = "",
) -> str:
    eid = int(example_id)
    row = _ROW_INDEX.get(eid)
    if row is None:
        return '{"error": "example_id not found"}'
    pool = _ensure_pool()
    return pool.call(
        agent_kind=agent_kind,
        example_id=eid,
        question=row.question,
        context=row.context,
        choices=row.choices,
        cache_namespace=namespace,
        candidate_answer=candidate_answer,
    )


# Argument-binding tool functions (used when binding_mode == "argument")
def extractor_tool(example_id: int) -> str:
    """Extract decision-relevant signals for the given example.

    Args:
        example_id: The current example's ID from the user message.

    Returns:
        JSON string with extracted facts and key evidence.
    """
    return _run_tool("extractor", int(example_id))


def reasoner_tool(example_id: int) -> str:
    """Produce a structured reasoning scaffold for the given example.

    Args:
        example_id: The current example's ID from the user message.

    Returns:
        JSON string with sub-questions, knowledge, candidate analysis.
    """
    return _run_tool("reasoner", int(example_id))


def verifier_tool(example_id: int, current_draft: str = "") -> str:
    """Audit reasoning for domain principles and potential errors.

    Args:
        example_id: The current example's ID from the user message.
        current_draft: Your current draft answer key (e.g. "B") to audit.

    Returns:
        JSON string with relevant principles, checks, and potential errors.
    """
    return _run_tool("verifier", int(example_id), candidate_answer=str(current_draft or ""))


class ManagerToolEnvironment:
    """Environment-binding alternative: tools don't take an example_id arg."""

    def reset(self, example_id: int, **kwargs) -> Optional[str]:
        """Bind this rollout to the given example.

        Args:
            example_id: ID of the sampled training example.

        Returns:
            None.
        """
        self.example_id = int(example_id)
        self._called = set()
        return None

    def _guard_repeat(self, kind: str) -> Optional[str]:
        called = getattr(self, "_called", None)
        if called is None:
            called = set()
            self._called = called
        if kind in called:
            return '{"error": "tool_already_called", "detail": "each tool may be used at most once"}'
        called.add(kind)
        return None

    def extractor_tool(self) -> str:
        """Extract decision-relevant signals for the current example.

        Returns:
            JSON string with extracted facts.
        """
        err = self._guard_repeat("extractor")
        if err:
            return err
        return _run_tool("extractor", getattr(self, "example_id", -1))

    def reasoner_tool(self) -> str:
        """Produce a reasoning scaffold for the current example.

        Returns:
            JSON string with reasoning structure.
        """
        err = self._guard_repeat("reasoner")
        if err:
            return err
        return _run_tool("reasoner", getattr(self, "example_id", -1))

    def verifier_tool(self, current_draft: str = "") -> str:
        """Audit reasoning for domain principles and potential errors.

        Args:
            current_draft: Your current draft answer key (e.g. "B") to audit.

        Returns:
            JSON string with relevant principles, checks, and potential errors.
        """
        err = self._guard_repeat("verifier")
        if err:
            return err
        return _run_tool(
            "verifier",
            getattr(self, "example_id", -1),
            candidate_answer=str(current_draft or ""),
        )


# ----------------------- Trainer entry point -----------------------

@dataclass
class ManagerGRPOConfig:
    base_model: str
    rows: List[StandardRow]              # filtered to a training split
    out_dir: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    verifier_adapter: Optional[str]
    manager_adapter: Optional[str] = None
    fail_buffer_jsonl: Optional[str] = None
    raw_trace_jsonl: Optional[str] = None
    seed: int = 42
    per_device_train_batch_size: int = 2
    max_completion_length: int = 2048
    temperature: float = 0.9
    num_generations: int = 6
    gradient_accumulation_steps: Optional[int] = None
    grpo_beta: float = 0.01
    max_steps: int = -1
    routing_efficiency_bonus: float = 0.0
    tool_use_bonus: float = 0.0
    ccr_mode: bool = False               # if true, use log-scoring-rule CCR reward instead of binary
    ccr_p_high: float = 0.9             # CCR implicit confidence when 0 tools called
    ccr_p_low: float = 0.2              # CCR implicit confidence when ccr_k_max tools called
    ccr_k_max: int = 3                  # must match max_tool_calling_iterations
    full_parameter_rl: bool = False      # if true, merge init adapter and train all model weights
    binding_mode: str = "auto"           # auto | environment | argument
    subagent_server_url: Optional[str] = None  # if set, call subagents via vLLM HTTP (no local model load)
    use_wandb: bool = False
    wandb_project: str = "agent_routing"
    wandb_entity: str = ""
    wandb_run_name: str = ""
    wandb_mode: str = "online"
    task_description: str = ""           # optional, passed into manager system prompt
    exploration_hint: str = ""           # START-style hint injected into system prompt during training only
    clip_epsilon_high: float = 0.0       # DAPO Clip-Higher: asymmetric clip upper bound (0 = symmetric/standard)
    # Optional SFT replay anchor. This is an auxiliary supervised loss, not reward shaping.
    sft_anchor_jsonl: Optional[str] = None
    sft_anchor_coef: float = 0.0
    sft_anchor_mode: str = "full"         # full | route_only
    sft_anchor_batch_size: int = 1
    sft_anchor_max_seq_len: int = 4096
    # Adaptive Deliberation Control reward (replaces CCR)
    adc_mode: bool = False               # legacy ADC reward ablation; main method uses binary reward
    adc_cost_per_tool: float = 0.05      # per-tool cost (discourages over-calling without utility)
    adc_draft_bonus: float = 0.2         # bonus per CORRECT draft answer (anytime reward)
    adc_missing_draft_penalty: float = 0.1  # penalty per tool call without an accompanying draft
    adc_final_bonus: float = 1.0         # bonus for final correct answer
    adc_variant: str = "anytime"         # anytime | transition | sum (latter two: ablation arms only)


_GRPOTrainerBase = GRPOTrainer if TRL_AVAILABLE else object


class SFTAnchoredGRPOTrainer(_GRPOTrainerBase):
    """GRPO plus a replayed marginal-SFT loss on every optimization step.

    ``anchor_mode=full`` replays the complete assistant target.  In
    ``route_only`` mode the tokenization masks the prompt and current draft,
    leaving only the post-draft CALL/COMMIT realization supervised.

    Each distributed worker shuffles the replay data with a rank-specific
    seed, while Trainer's normal input preparation moves the auxiliary batch
    to the right device.  The auxiliary forward pass is part of the same loss
    returned to Trainer, so gradient accumulation and DeepSpeed scaling remain
    owned by the normal Trainer/Accelerate path.
    """

    def __init__(
        self,
        *args,
        sft_anchor_dataset=None,
        sft_anchor_collator=None,
        sft_anchor_coef: float = 0.0,
        sft_anchor_batch_size: int = 1,
        sft_anchor_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sft_anchor_coef = float(sft_anchor_coef)
        self._sft_anchor_loader = None
        self._sft_anchor_iter = None
        self._sft_anchor_loss_sum = 0.0
        self._sft_anchor_loss_count = 0

        if self.sft_anchor_coef <= 0.0:
            return
        if sft_anchor_dataset is None or len(sft_anchor_dataset) == 0:
            raise ValueError("sft_anchor_coef > 0 requires a non-empty SFT anchor dataset")
        if sft_anchor_collator is None:
            raise ValueError("sft_anchor_coef > 0 requires an SFT anchor collator")
        if int(sft_anchor_batch_size) <= 0:
            raise ValueError("sft_anchor_batch_size must be positive")

        generator = torch.Generator()
        rank = int(getattr(self.accelerator, "process_index", 0))
        generator.manual_seed(int(sft_anchor_seed) + rank)
        self._sft_anchor_loader = DataLoader(
            sft_anchor_dataset,
            batch_size=int(sft_anchor_batch_size),
            shuffle=True,
            collate_fn=sft_anchor_collator,
            generator=generator,
            drop_last=False,
        )
        self._sft_anchor_iter = iter(self._sft_anchor_loader)

    def _next_sft_anchor_batch(self):
        if self._sft_anchor_loader is None:
            return None
        try:
            return next(self._sft_anchor_iter)
        except StopIteration:
            self._sft_anchor_iter = iter(self._sft_anchor_loader)
            return next(self._sft_anchor_iter)

    def _call_base_compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """Forward only kwargs supported by the installed TRL version."""
        base_compute_loss = super().compute_loss
        params = inspect.signature(base_compute_loss).parameters
        kwargs: Dict[str, Any] = {}
        if "return_outputs" in params:
            kwargs["return_outputs"] = return_outputs
        if "num_items_in_batch" in params:
            kwargs["num_items_in_batch"] = num_items_in_batch
        return base_compute_loss(model, inputs, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        grpo_loss = self._call_base_compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if return_outputs or not model.training or self.sft_anchor_coef <= 0.0:
            return grpo_loss

        anchor_batch = self._next_sft_anchor_batch()
        if anchor_batch is None:
            return grpo_loss
        # Keep model/device placement in Trainer rather than manually calling
        # .cuda(), which also works when the model is managed by DeepSpeed.
        anchor_batch = {k: (v.to(self.args.device) if hasattr(v, "to") else v) for k, v in anchor_batch.items()}
        anchor_outputs = model(
            input_ids=anchor_batch["input_ids"],
            attention_mask=anchor_batch.get("attention_mask"),
            labels=anchor_batch["labels"],
            use_cache=False,
        )
        anchor_loss = anchor_outputs.loss
        if anchor_loss is None or not torch.isfinite(anchor_loss.detach()).all():
            raise RuntimeError(f"Non-finite SFT anchor loss: {anchor_loss}")

        # Gather only the detached scalar for logging. The differentiable loss
        # stays local and is combined with GRPO below.
        logged_loss = anchor_loss.detach().float().reshape(1)
        try:
            logged_loss = self.accelerator.gather(logged_loss).mean()
        except Exception:
            logged_loss = logged_loss.mean()
        self._sft_anchor_loss_sum += float(logged_loss.item())
        self._sft_anchor_loss_count += 1
        # TRL 0.29 returns an already accumulation-normalized GRPO loss;
        # without matching this factor the anchor is amplified by the number
        # of microbatches and changes its meaning across GPU/batch settings.
        accumulation = getattr(self, "current_gradient_accumulation_steps",
                               self.args.gradient_accumulation_steps)
        return grpo_loss + (self.sft_anchor_coef * anchor_loss / accumulation)

    def log(self, logs, *args, **kwargs):
        if self._sft_anchor_loss_count:
            logs = dict(logs)
            logs["sft_anchor/loss"] = (
                self._sft_anchor_loss_sum / self._sft_anchor_loss_count
            )
            logs["sft_anchor/weighted_loss"] = (
                logs["sft_anchor/loss"] * self.sft_anchor_coef
            )
            self._sft_anchor_loss_sum = 0.0
            self._sft_anchor_loss_count = 0
        return super().log(logs, *args, **kwargs)


def _validate_generation_batch(cfg):
    import math
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.num_generations < 2 or cfg.per_device_train_batch_size < 1 or world_size < 1:
        raise ValueError("Positive batch/world size and num_generations >= 2 required")
    per_microbatch = cfg.per_device_train_batch_size * world_size
    accumulation = cfg.gradient_accumulation_steps
    if accumulation is None:
        # Preserve the historical 2 whenever it already forms complete groups.
        # For invalid defaults (single GPU, batch=2, G=6), choose the smallest
        # valid value instead of launching an impossible generation batch.
        accumulation = 2 if (per_microbatch * 2) % cfg.num_generations == 0 else (
            cfg.num_generations // math.gcd(cfg.num_generations, per_microbatch))
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    batch = per_microbatch * accumulation
    if batch % cfg.num_generations:
        raise ValueError(f"Global generation batch {batch} must be divisible by num_generations={cfg.num_generations}")
    return accumulation


def _resolve_checkpoint_source(source):
    # owner/repo is a valid Hub ID, including on POSIX where '/' == os.sep.
    # Only resolve actual local paths or an explicitly relative/absolute path.
    if source and (os.path.exists(source) or source.startswith((".", "~", "/"))):
        return os.path.abspath(os.path.expanduser(source))
    return source


def train_manager_grpo(cfg: ManagerGRPOConfig) -> None:
    if not TRL_AVAILABLE:
        raise RuntimeError("trl is required for manager GRPO training.")

    device = training_device()
    set_seed(cfg.seed)

    if not cfg.rows:
        raise ValueError("Manager GRPO requires nonempty training rows")
    if len({int(r.example_id) for r in cfg.rows}) != len(cfg.rows):
        raise ValueError("Manager GRPO example IDs must be unique for tool binding")
    accumulation = _validate_generation_batch(cfg)

    # ---- Resolve binding mode ----
    binding_mode = cfg.binding_mode
    if binding_mode == "auto":
        binding_mode = "environment" if _grpo_supports_environment_factory() else "argument"
    if binding_mode == "environment" and not _grpo_supports_environment_factory():
        raise RuntimeError(
            "binding_mode=environment requires a TRL version with environment_factory."
        )
    print(f"[MANAGER_GRPO] binding_mode={binding_mode}")

    # ---- Build subagent pool ----
    if cfg.subagent_server_url:
        # Multi-GPU mode: subagents are served by a vLLM process on a dedicated GPU.
        # No model weights are loaded here; all VRAM stays available for ZeRO3.
        pool = RemoteSubagentPool(
            server_url=cfg.subagent_server_url,
            registered_kinds=["extractor", "reasoner", "verifier"],
        )
        print(f"[MANAGER_GRPO] using remote subagent pool -> {cfg.subagent_server_url}")
    else:
        pool = SubagentPool()
        if cfg.extractor_adapter:
            pool.register(FrozenSubagent(cfg.base_model, cfg.extractor_adapter, "extractor", device))
        if cfg.reasoner_adapter:
            pool.register(FrozenSubagent(cfg.base_model, cfg.reasoner_adapter, "reasoner", device))
        if cfg.verifier_adapter:
            pool.register(FrozenSubagent(cfg.base_model, cfg.verifier_adapter, "verifier", device))
        if not pool._agents:
            raise ValueError(
                "At least one subagent adapter must be provided, "
                "or set subagent_server_url to use a vLLM server."
            )
        print(f"[MANAGER_GRPO] subagents loaded: {sorted(pool._agents.keys())}")

    _init_globals(pool, cfg.rows)

    # ---- Side-channel files ----
    fail_buffer = cfg.fail_buffer_jsonl or os.path.join(cfg.out_dir, "fail_buffer.jsonl")
    raw_trace = cfg.raw_trace_jsonl or os.path.join(cfg.out_dir, "train_raw_trace.jsonl")
    is_main = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0

    if is_main:
        os.makedirs(os.path.dirname(fail_buffer) or ".", exist_ok=True)
        if os.environ.get("FAIL_BUFFER_APPEND", "0") == "0":
            open(fail_buffer, "w", encoding="utf-8").close()
        os.makedirs(os.path.dirname(raw_trace) or ".", exist_ok=True)
        if os.environ.get("RAW_TRACE_APPEND", "0") == "0":
            open(raw_trace, "w", encoding="utf-8").close()
        print(f"[MANAGER_GRPO] fail_buffer -> {fail_buffer}")
        print(f"[MANAGER_GRPO] raw_trace   -> {raw_trace}")

    # ---- W&B ----
    if cfg.use_wandb:
        try:
            import wandb  # noqa: F401
        except Exception as e:
            raise RuntimeError("`wandb` not installed but use_wandb=True.") from e
        if cfg.wandb_project:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if cfg.wandb_entity:
            os.environ["WANDB_ENTITY"] = cfg.wandb_entity
        os.environ.pop("WANDB_DISABLED", None)
        os.environ["WANDB_MODE"] = cfg.wandb_mode
        run_name = cfg.wandb_run_name or f"grpo_{os.path.basename(cfg.out_dir.rstrip('/'))}_{int(time.time())}"
        os.environ["WANDB_NAME"] = run_name
        print(f"[WANDB] {cfg.wandb_project} / {run_name}")
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    # ---- Build training dataset ----
    label_keys_per_row: List[List[str]] = [list(r.choices.keys()) for r in cfg.rows]

    def _to_record(r: StandardRow, keys: List[str]) -> Dict[str, Any]:
        sys_prompt = build_manager_system_prompt(
            label_keys=keys,
            task_description=cfg.task_description,
            exploration_hint=cfg.exploration_hint,
        )
        user_msg = build_manager_user_message(
            example_id=r.example_id,
            question=r.question,
            context=r.context,
            choices=r.choices,
            binding_mode=binding_mode,
        )
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            "ground_truth": r.ground_truth,
            "example_id": int(r.example_id),
            "question_hash": _question_hash(r.question),
            "choice_keys": list(r.choices.keys()),
        }

    train_records = [_to_record(r, k) for r, k in zip(cfg.rows, label_keys_per_row)]
    train_dataset = Dataset.from_list(train_records)

    # ---- Manager model + tokenizer ----
    cfg.manager_adapter = _resolve_checkpoint_source(cfg.manager_adapter)

    manager_tok = AutoTokenizer.from_pretrained(
        cfg.manager_adapter or cfg.base_model, trust_remote_code=True
    )
    manager_tok.padding_side = "left"
    if manager_tok.pad_token_id is None and manager_tok.eos_token_id is not None:
        manager_tok.pad_token_id = manager_tok.eos_token_id
    if HAS_RESP_SCHEMA:
        try:
            manager_tok = add_response_schema(manager_tok)
        except Exception:
            pass

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    is_full_init = (
        cfg.manager_adapter
        and os.path.isdir(cfg.manager_adapter)
        and os.path.exists(os.path.join(cfg.manager_adapter, "config.json"))
        and not os.path.exists(os.path.join(cfg.manager_adapter, "adapter_config.json"))
    )
    if cfg.full_parameter_rl and is_full_init:
        manager_model = load_text_causal_model(
            cfg.manager_adapter, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        print(f"[MANAGER_GRPO] full-parameter init model -> {cfg.manager_adapter}")
    else:
        manager_model = load_text_causal_model(
            cfg.base_model, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        if cfg.manager_adapter:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft is required to load --mgr_init_adapter.")
            manager_model = PeftModel.from_pretrained(
                manager_model,
                cfg.manager_adapter,
                is_trainable=(not cfg.full_parameter_rl),
            ).to(device)
            if cfg.full_parameter_rl:
                manager_model = manager_model.merge_and_unload().to(device)
                print(f"[MANAGER_GRPO] merged init adapter for full-parameter RL -> {cfg.manager_adapter}")
            else:
                print(f"[MANAGER_GRPO] manager init adapter -> {cfg.manager_adapter}")
    if cfg.full_parameter_rl:
        for p in manager_model.parameters():
            p.requires_grad_(True)
        trainable = sum(p.numel() for p in manager_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in manager_model.parameters())
        print(f"[MANAGER_GRPO] full_parameter_rl=True trainable_params={trainable}/{total}")
    manager_model.config.use_cache = False
    if not hasattr(manager_model, "warnings_issued") or manager_model.warnings_issued is None:
        manager_model.warnings_issued = {}

    # ---- Optional marginal-SFT replay anchor ----
    sft_anchor_dataset = None
    sft_anchor_collator = None
    sft_anchor_stats: Dict[str, float] = {}
    if cfg.sft_anchor_coef < 0:
        raise ValueError("sft_anchor_coef must be >= 0")
    if cfg.sft_anchor_coef > 0:
        if not cfg.sft_anchor_jsonl:
            raise ValueError("sft_anchor_coef > 0 requires --mgr_sft_anchor_jsonl")
        if not os.path.exists(cfg.sft_anchor_jsonl):
            raise FileNotFoundError(f"SFT anchor JSONL not found: {cfg.sft_anchor_jsonl}")
        if cfg.sft_anchor_mode not in ANCHOR_MODES:
            raise ValueError(
                f"Unknown sft_anchor_mode={cfg.sft_anchor_mode!r}; "
                f"expected one of {ANCHOR_MODES}"
            )
        if cfg.sft_anchor_batch_size <= 0:
            raise ValueError("sft_anchor_batch_size must be positive")
        if cfg.sft_anchor_max_seq_len <= 0:
            raise ValueError("sft_anchor_max_seq_len must be positive")

        anchor_rows = read_jsonl(cfg.sft_anchor_jsonl)
        if not anchor_rows:
            raise ValueError(f"No rows in SFT anchor JSONL: {cfg.sft_anchor_jsonl}")
        # Render the replay targets with the same native tool schema used by
        # this GRPO run.  The anchor never changes the reward function.
        from .marginal_value import _tool_schemas

        anchor_tools = _tool_schemas(binding_mode)
        anchor_features, sft_anchor_stats = build_anchor_features(
            rows=anchor_rows,
            tokenizer=manager_tok,
            max_seq_len=int(cfg.sft_anchor_max_seq_len),
            mode=cfg.sft_anchor_mode,
            tools=anchor_tools,
        )
        if not anchor_features:
            raise ValueError(
                "SFT anchor tokenization produced no supervised targets. "
                "Increase --mgr_sft_anchor_max_seq_len or inspect the chat template."
            )
        sft_anchor_dataset = Dataset.from_list(anchor_features)
        sft_anchor_collator = DataCollatorForSeq2Seq(
            manager_tok,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )
        print(
            "[MANAGER_GRPO/SFT_ANCHOR] "
            f"mode={cfg.sft_anchor_mode} coef={cfg.sft_anchor_coef} "
            f"rows={int(sft_anchor_stats['n_rows'])} "
            f"kept={int(sft_anchor_stats['n_kept'])} "
            f"dropped={int(sft_anchor_stats['n_dropped_no_target'])} "
            f"mean_supervised_tokens={sft_anchor_stats['mean_supervised_tokens']:.1f}"
        )

    # ---- GRPO config ----
    _grpo_extra: Dict[str, Any] = {}
    if cfg.clip_epsilon_high > 0:
        # DAPO Clip-Higher: asymmetric ratio clipping [1-epsilon, 1+epsilon_high]
        # Allows larger updates when increasing probability of under-explored actions
        # without symmetrically loosening the lower bound that guards against collapse.
        _grpo_extra["epsilon_high"] = float(cfg.clip_epsilon_high)
    grpo_args = GRPOConfig(
        output_dir=cfg.out_dir,
        remove_unused_columns=False,
        max_completion_length=int(cfg.max_completion_length),
        temperature=float(cfg.temperature),
        num_generations=int(cfg.num_generations),
        bf16=(device == "cuda"),
        beta=float(cfg.grpo_beta),
        scale_rewards="group",
        report_to=(["wandb"] if cfg.use_wandb else []),
        use_vllm=False,
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        max_tool_calling_iterations=3,           # we allow up to 3 tools
        chat_template_kwargs={"enable_thinking": False},
        gradient_accumulation_steps=accumulation,
        seed=cfg.seed,
        data_seed=cfg.seed,
        use_cpu=(device == "cpu"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=None,
        log_unique_prompts=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
        **_grpo_extra,
    )

    reward_funcs = build_reward_funcs(
        fail_buffer_jsonl=fail_buffer,
        raw_trace_jsonl=raw_trace,
        routing_efficiency_bonus=cfg.routing_efficiency_bonus,
        tool_use_bonus=cfg.tool_use_bonus,
        ccr_mode=cfg.ccr_mode,
        ccr_p_high=cfg.ccr_p_high,
        ccr_p_low=cfg.ccr_p_low,
        ccr_k_max=cfg.ccr_k_max,
        adc_mode=cfg.adc_mode,
        adc_cost_per_tool=cfg.adc_cost_per_tool,
        adc_draft_bonus=cfg.adc_draft_bonus,
        adc_missing_draft_penalty=cfg.adc_missing_draft_penalty,
        adc_final_bonus=cfg.adc_final_bonus,
        adc_variant=cfg.adc_variant,
        is_main_process=is_main,
    )
    if cfg.adc_mode:
        print(
            f"[MANAGER_GRPO] ADC mode ON  variant={cfg.adc_variant}  "
            f"draft_bonus={cfg.adc_draft_bonus}  "
            f"missing_draft_penalty={cfg.adc_missing_draft_penalty}  "
            f"final_bonus={cfg.adc_final_bonus}  "
            f"cost_per_tool={cfg.adc_cost_per_tool}"
        )
        if cfg.adc_variant != "anytime":
            print(
                f"[MANAGER_GRPO] WARNING: adc_variant={cfg.adc_variant} is a provably "
                f"exploitable reward — use only as an RQ3 ablation arm."
            )
    elif cfg.ccr_mode:
        print(
            f"[MANAGER_GRPO] CCR mode ON  p_high={cfg.ccr_p_high} "
            f"p_low={cfg.ccr_p_low} k_max={cfg.ccr_k_max}"
        )
    else:
        print(
            f"[MANAGER_GRPO] binary reward  "
            f"routing_efficiency_bonus={cfg.routing_efficiency_bonus} "
            f"tool_use_bonus={cfg.tool_use_bonus}"
        )

    trainer_cls = SFTAnchoredGRPOTrainer if cfg.sft_anchor_coef > 0 else GRPOTrainer
    anchor_trainer_kwargs: Dict[str, Any] = {}
    if cfg.sft_anchor_coef > 0:
        anchor_trainer_kwargs = {
            "sft_anchor_dataset": sft_anchor_dataset,
            "sft_anchor_collator": sft_anchor_collator,
            "sft_anchor_coef": cfg.sft_anchor_coef,
            "sft_anchor_batch_size": cfg.sft_anchor_batch_size,
            "sft_anchor_seed": cfg.seed + 1701,
        }

    if binding_mode == "environment":
        trainer = trainer_cls(
            model=manager_model,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=manager_tok,
            reward_funcs=reward_funcs,
            rollout_func=None,
            environment_factory=ManagerToolEnvironment,
            **anchor_trainer_kwargs,
        )
    else:
        trainer = trainer_cls(
            model=manager_model,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=manager_tok,
            reward_funcs=reward_funcs,
            rollout_func=None,
            tools=[extractor_tool, reasoner_tool, verifier_tool],
            **anchor_trainer_kwargs,
        )

    trainer.train()
    trainer.model.save_pretrained(cfg.out_dir)
    manager_tok.save_pretrained(cfg.out_dir)

    subagent_keys = (
        sorted(pool._kinds)
        if isinstance(pool, RemoteSubagentPool)
        else sorted(pool._agents.keys())
    )
    write_json(os.path.join(cfg.out_dir, "manager_run_config.json"), {
        "base_model": cfg.base_model,
        "seed": cfg.seed,
        "num_generations": cfg.num_generations,
        "gradient_accumulation_steps": accumulation,
        "requested_gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "temperature": cfg.temperature,
        "grpo_beta": cfg.grpo_beta,
        "binding_mode": binding_mode,
        "n_train_rows": len(cfg.rows),
        "subagents": subagent_keys,
        "manager_adapter": cfg.manager_adapter,
        "full_parameter_rl": bool(cfg.full_parameter_rl),
        "subagent_server_url": cfg.subagent_server_url or "",
        "routing_efficiency_bonus": cfg.routing_efficiency_bonus,
        "tool_use_bonus": cfg.tool_use_bonus,
        "ccr_mode": bool(cfg.ccr_mode),
        "ccr_p_high": float(cfg.ccr_p_high),
        "ccr_p_low": float(cfg.ccr_p_low),
        "ccr_k_max": int(cfg.ccr_k_max),
        "adc_mode": bool(cfg.adc_mode),
        "adc_cost_per_tool": float(cfg.adc_cost_per_tool),
        "adc_draft_bonus": float(cfg.adc_draft_bonus),
        "adc_missing_draft_penalty": float(cfg.adc_missing_draft_penalty),
        "adc_final_bonus": float(cfg.adc_final_bonus),
        "adc_variant": str(cfg.adc_variant),
        "sft_anchor_jsonl": cfg.sft_anchor_jsonl or "",
        "sft_anchor_coef": float(cfg.sft_anchor_coef),
        "sft_anchor_mode": str(cfg.sft_anchor_mode),
        "sft_anchor_batch_size": int(cfg.sft_anchor_batch_size),
        "sft_anchor_max_seq_len": int(cfg.sft_anchor_max_seq_len),
        "sft_anchor_stats": sft_anchor_stats,
    })
    print(f"[MANAGER_GRPO] saved -> {cfg.out_dir}")
