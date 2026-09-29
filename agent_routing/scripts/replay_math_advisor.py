"""Replay one saved advisor failure, without collecting labels or training.

Compare fixed/native non-thinking templates crossed with greedy/sampled decoding.
The sampled diagnostic uses the Qwen3.5 model card's non-thinking general preset:
https://huggingface.co/Qwen/Qwen3.5-9B#best-practices
This is a diagnostic, not an automatic change to the experiment protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verifiable.backend import HFBackend, strip_generation_endings
from src.verifiable.telemetry import Monitor, atomic_json
from src.verifiable.sampling import presence_penalty, generation_kwargs


def load_failure(stage):
    stage = Path(stage).resolve()
    path = stage / "generations.jsonl"
    payload = path.read_bytes()
    # Read only the saved model messages. Never reconstruct prompts from gold.
    failures = [r for line in payload.splitlines() if line.strip()
                for r in [json.loads(line)]
                if r.get("role") == "advisor" and r.get("truncated") is True]
    if not failures:
        raise ValueError("No truncated advisor generation in the supplied stage")
    record = failures[-1]
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get("role") not in {"system", "user"}
        or not isinstance(m.get("content"), str) for m in messages
    ):
        raise ValueError("Expected saved text-only system/user advisor messages")
    cfg = json.loads((stage / "run.json").read_text())["config"]
    if not cfg.get("base_model_revision"):
        raise ValueError("Source run must pin a model revision")
    return record, cfg, hashlib.sha256(payload).hexdigest()


def variants():
    return [(template + "_" + decode, template, decode == "sampled")
            for template in ("fixed", "native") for decode in ("greedy", "sampled")]


def draw(model, tokenizer, messages, max_tokens, max_context, seed, sampled):
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n = inputs["input_ids"].shape[1]
    if n + max_tokens > max_context:
        raise ValueError("Replay exceeds the original context budget")
    options = {"do_sample": sampled, "max_new_tokens": max_tokens,
               "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    if sampled:
        options.update(generation_kwargs({"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                       "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}, n))
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    if devices:
        torch.cuda.synchronize(model.device)
    started = time.monotonic()
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(seed)
        ids = model.generate(**inputs, **options)[0, n:]
        if devices:
            torch.cuda.synchronize(model.device)
    seconds = time.monotonic() - started
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    return {"text": strip_generation_endings(raw, tokenizer), "raw_text": raw,
            "prompt_tokens": n, "completion_tokens": len(ids), "seconds": seconds,
            "truncated": bool(len(ids) >= max_tokens and int(ids[-1]) != tokenizer.eos_token_id),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "rendered_prompt": prompt}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Existing stage with generations.jsonl")
    p.add_argument("--out", required=True, help="New directory outside experiment loops")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+", choices=[v[0] for v in variants()],
                   default=[v[0] for v in variants()], help="Run only these conditions")
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        p.error("--max-tokens must be between 1 and 4096")
    stage, output = Path(args.run_dir).resolve(), Path(args.out).resolve()
    if stage == output or stage in output.parents or any(
        (parent / "loop.json").exists() for parent in (output, *output.parents)
    ):
        p.error("Replay output must be outside existing experiment loops")
    record, cfg, digest = load_failure(stage)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "run.json", {"config": {
        "purpose": "single_advisor_diagnostic_not_training_or_evaluation",
        "source_stage": str(stage), "source_sha256": digest, "original_config": cfg,
        "source_attempt": record.get("attempt"), "source_question_hash": record.get("question_hash"),
        "max_tokens": args.max_tokens, "seed": args.seed,
        "variants": list(dict.fromkeys(args.variants)),
        "sampled_parameters": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                               "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
        "note": "One seed and one saved input cannot establish general correctness; cap differs from source.",
    }})
    with Monitor(output, "advisor_replay") as monitor:
        backend = HFBackend(cfg["base_model"], max_context=cfg["max_context"],
                            revision=cfg["base_model_revision"])
        from transformers import AutoTokenizer
        native = AutoTokenizer.from_pretrained(cfg["base_model"], revision=cfg["base_model_revision"])
        if not native.chat_template:
            raise ValueError("Pinned model has no native tokenizer chat template")
        if native.pad_token_id is None:
            native.pad_token_id = native.eos_token_id
        rendered = [tok.apply_chat_template(record["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                    for tok in (backend.tokenizer, native)]
        identical = (rendered[0] == rendered[1]
                     and backend.tokenizer.eos_token_id == native.eos_token_id
                     and backend.tokenizer.pad_token_id == native.pad_token_id)
        skipped = [name for name in args.variants if identical and name.startswith("native_")
                   and name.replace("native_", "fixed_", 1) in args.variants]
        atomic_json(output / "template_comparison.json", {
            "identical_rendering_and_endings": identical,
            "fixed_prompt": rendered[0], "native_prompt": rendered[1],
            "skipped_equivalent_variants": skipped,
        })
        print("Fixed/native prompt rendering and ending IDs identical:", identical, flush=True)
        monitor.set_question(SimpleNamespace(question=record.get("question", ""),
                             context=record.get("context", ""), ground_truth=record.get("ground_truth"),
                             benchmark_name=record.get("benchmark_name"), split=record.get("split"),
                             example_id=record.get("example_id")))
        results = []
        for name, template, sampled in variants():
            if name not in args.variants or name in skipped:
                continue
            monitor.update(phase=name, sequence=record.get("sequence", []))
            tok = backend.tokenizer if template == "fixed" else native
            result = draw(backend.model, tok, record["messages"], args.max_tokens,
                          cfg["max_context"], args.seed, sampled)
            result.update(variant=name, seed=args.seed)
            # Preserve the full decoded output, actual rendered prompt and per-mode settings.
            atomic_json(output / (name + ".json"), result)
            monitor.usage("advisor_replay", result)
            monitor.generation("advisor", result, messages=record["messages"],
                               advisor=record["advisor"], max_tokens=args.max_tokens,
                               operation="diagnostic_replay",
                               error="diagnostic_cap_reached" if result["truncated"] else None)
            monitor.flush_tables(force=True)
            results.append({k: result[k] for k in ("variant", "completion_tokens", "seconds", "truncated", "prompt_sha256")})
            print(json.dumps(results[-1]), flush=True)
            print(result["text"], flush=True)
        atomic_json(output / "replay_summary.json", results)
        print("Replay outputs:", output, flush=True)


if __name__ == "__main__":
    main()
