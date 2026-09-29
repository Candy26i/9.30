"""Subagent LoRA SFT training. Runtime classes (FrozenSubagent, SubagentPool)
live in runtime.py — do not duplicate them here."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from ..utils.io import read_jsonl
from ..utils.seed import set_seed

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def _render_chat(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


@dataclass
class SFTConfig:
    base_model: str
    train_jsonl: str
    out_dir: str
    dev_jsonl: Optional[str] = None
    seed: int = 42
    max_seq_len: int = 4096
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    bf16: bool = True
    base_model_revision: Optional[str] = None


def _mask_prefix_len(prompt_ids: List[int], full_ids: List[int]) -> int:
    """Length of the common token prefix between the prompt-only render and the
    full (prompt+response) render.

    Using len(prompt_ids) directly is WRONG for templates where the generation
    prompt is not a strict prefix of the full render — e.g. Qwen3 with
    enable_thinking=False appends an empty <think></think> block to the
    generation prompt that does not appear before the assistant content in the
    full render. That off-by-N would mask the first response tokens.
    """
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    return i


def _tokenize_subagent_sft(rows: List[Dict[str, Any]], tok, max_seq_len: int) -> Any:
    from datasets import Dataset
    from ..manager.routing_anchor import build_anchor_features
    features, stats = build_anchor_features(rows, tok, max_seq_len, "full")
    if len(features) != len(rows):
        raise ValueError(f"Subagent SFT contains empty or overlength targets; no silent truncation: {stats}")
    return Dataset.from_list(features)


def training_device():
    """Select the worker's CUDA device before any model weights are allocated."""
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
    return "cuda"


def load_text_causal_model(source, **kwargs):
    """Dispatch Qwen3.5's enclosing multimodal config to its text-only class.

    Preserve the native template here: historical MCQ tools are not the
    immutable-COMMIT math protocol and must not receive its fixed template.
    """
    import transformers as tr
    config = tr.AutoConfig.from_pretrained(source, **{k: kwargs[k] for k in
        ("revision", "trust_remote_code") if k in kwargs})
    cls = tr.Qwen3_5ForCausalLM if config.model_type == "qwen3_5" else tr.AutoModelForCausalLM
    return cls.from_pretrained(source, **kwargs)


def validate_sft_splits(train_rows, dev_rows=()):
    """Reject held-out training and question overlap across draft variants.

    Historical exports omit split/hash metadata; keep those usable while
    checking identical prompts. New exports are checked by question identity.
    """
    import json
    if not train_rows:
        raise ValueError("Subagent SFT requires nonempty training rows")
    if any(r.get("split") not in (None, "", "train") for r in train_rows):
        raise ValueError("SFT training rows must be train-only; held-out split found")
    if any(r.get("split") not in (None, "", "dev", "validation", "val") for r in dev_rows):
        raise ValueError("SFT development rows must use a development split")
    train_hashes = {r["question_hash"] for r in train_rows if r.get("question_hash")}
    dev_hashes = {r["question_hash"] for r in dev_rows if r.get("question_hash")}
    train_prompts = {json.dumps(r["prompt"], sort_keys=True, ensure_ascii=False) for r in train_rows}
    if train_hashes & dev_hashes or any(
            json.dumps(r["prompt"], sort_keys=True, ensure_ascii=False) in train_prompts for r in dev_rows):
        raise ValueError("SFT train/dev questions overlap (including alternate drafts)")


def _lora_target_modules(model) -> List[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                  "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [name for name in candidates if name in present]
    if not targets:
        raise ValueError("No supported LoRA projection modules in this model")
    return targets


def train_subagent_sft(cfg: SFTConfig) -> None:
    import hashlib
    import json
    from pathlib import Path
    from transformers.trainer_utils import get_last_checkpoint
    from ..verifiable.telemetry import Monitor, atomic_json
    from ..verifiable.provenance import harness_identity
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Subagent SFT currently supports one training process")
    root = Path(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    signature = {"stage": "subagent_sft", "config": asdict(cfg), "harness": harness_identity(),
                 "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes() +
                     Path(__file__).parent.parent.joinpath("manager/routing_anchor.py").read_bytes()).hexdigest(),
                 "data_sha256": hashlib.sha256(Path(cfg.train_jsonl).read_bytes()).hexdigest(),
                 "dev_sha256": hashlib.sha256(Path(cfg.dev_jsonl).read_bytes()).hexdigest() if cfg.dev_jsonl else None}
    manifest = root / "training_run.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError("Subagent training inputs changed; use a new output directory")
    elif any(root.iterdir()):
        raise ValueError("Subagent output has no matching training manifest")
    else:
        atomic_json(manifest, signature)
    if (root / "training_metrics.json").exists():
        weights = ("adapter_model.safetensors", "adapter_model.bin") if cfg.use_lora else ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
        if not any((root / name).is_file() for name in weights):
            raise ValueError("Completed subagent training is missing saved model weights")
        return
    resume = get_last_checkpoint(str(root))
    with Monitor(cfg.out_dir, "subagent_sft") as monitor:
        _train_subagent_sft(cfg, monitor, resume)


def _train_subagent_sft(cfg: SFTConfig, monitor, resume) -> None:
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from ..verifiable.telemetry import training_callback, metrics, usage, atomic_json
    from pathlib import Path
    import hashlib
    import json
    import time

    if cfg.use_lora and not PEFT_AVAILABLE:
        raise RuntimeError("peft is required for LoRA subagent SFT training.")

    train_rows = read_jsonl(cfg.train_jsonl)
    dev_rows = read_jsonl(cfg.dev_jsonl) if cfg.dev_jsonl else []
    validate_sft_splits(train_rows, dev_rows)
    device = training_device()
    set_seed(cfg.seed)

    revision = {"revision": cfg.base_model_revision} if cfg.base_model_revision else {}
    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True, **revision)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
    model = load_text_causal_model(
        cfg.base_model, dtype=dtype, trust_remote_code=True, **revision
    ).to(device)
    model.config.use_cache = False

    if cfg.use_lora:
        target = _lora_target_modules(model)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            revision=cfg.base_model_revision,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target,
        )
        model = get_peft_model(model, lora_cfg)
        print(f"[SUBAGENT_SFT/LoRA] r={cfg.lora_r} alpha={cfg.lora_alpha} target_modules={target}")

    train_ds = _tokenize_subagent_sft(train_rows, tok, cfg.max_seq_len)

    eval_ds = None
    if cfg.dev_jsonl:
        eval_ds = _tokenize_subagent_sft(dev_rows, tok, cfg.max_seq_len) if dev_rows else None

    monitor.summary({"training_algorithm": "subagent_response_only_sft", "base_model": cfg.base_model,
        "requested_model_revision": cfg.base_model_revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "template_sha256": hashlib.sha256(str(tok.chat_template).encode()).hexdigest(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "resumed_checkpoint": resume, "train_examples": len(train_ds), "dev_examples": len(eval_ds) if eval_ds else 0})
    report = {"input_turns": len(train_ds), "kept_turns": len(train_ds), "dropped_turns": 0,
              "input_tokens_per_epoch": sum(len(r["input_ids"]) for r in train_ds),
              "supervised_tokens_per_epoch": sum(sum(x != -100 for x in r["labels"]) for r in train_ds)}
    atomic_json(Path(cfg.out_dir) / "sft_data_report.json", report)
    metrics(report, "sft_data")
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")
    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=(cfg.bf16 and device == "cuda"),
        use_cpu=(device == "cpu"),
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
    )
    class LoggedTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            usage("subagent_sft_train", {}, input_tokens=int(inputs["attention_mask"].sum().item()),
                  supervised_tokens=int((inputs["labels"] != -100).sum().item()), step=self.state.global_step)
            return loss
    trainer = LoggedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[training_callback(cfg.out_dir)],
    )
    started = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume)
    metrics(result.metrics, "train", trainer_step=trainer.state.global_step)
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    atomic_json(Path(cfg.out_dir) / "training_metrics.json", {**result.metrics,
        "optimizer_steps": trainer.state.global_step, "wall_seconds_this_attempt": time.monotonic() - started,
        "config": asdict(cfg)})
    monitor.summary({"optimizer_steps": trainer.state.global_step, "training_complete": True})
    print(f"[SUBAGENT_SFT] saved -> {cfg.out_dir}")
