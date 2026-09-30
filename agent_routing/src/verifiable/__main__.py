from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path

from ..utils.io import read_jsonl, write_json
from .data import prepare
from .experiment import compare
from .runner import SFT_ARMS, evaluate_suite, load_config, run_data, run_loop


def doctor(config, output):
    import requests
    import torch
    import transformers
    from .answers import correct
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Run training preflight inside the RunPod GPU container")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This first math runner uses one training process; do not launch it with torchrun")
    assert correct(r"FINAL_ANSWER: \boxed{\frac{1}{2}}", "0.5")
    assert not correct("My working contains 42", "42")
    revision = config.get("base_model_revision")
    if not revision and not Path(config["base_model"]).is_dir():
        raise RuntimeError("Freeze base_model_revision with freeze-config before a GPU run")
    model_cfg = transformers.AutoConfig.from_pretrained(config["base_model"], revision=revision)
    if model_cfg.model_type == "qwen3_5" and not hasattr(transformers, "Qwen3_5ForCausalLM"):
        raise RuntimeError("Transformers lacks Qwen3.5 text-only support")
    from .backend import configure_tokenizer, render
    tok = configure_tokenizer(transformers.AutoTokenizer.from_pretrained(config["base_model"], revision=revision))
    from .protocol import SYSTEM, tool_schemas
    render(tok, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Compute 1+1."}], tool_schemas())
    # TRL needs a response parser in addition to a generation chat template.
    from trl.chat_template_utils import is_chat_template_prefix_preserving
    if not is_chat_template_prefix_preserving(tok):
        raise RuntimeError("Math chat template failed prefix-preservation check")
    parsed = tok.parse_response('<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call><|im_end|>')
    if parsed["tool_calls"][0]["function"]["name"] != "reasoner_tool":
        raise RuntimeError("Native tool parsing failed")
    grammar = None
    if config.get("decision_constraint") == "finite_actions_v1":
        # Check the real tokenizer before any GPU budget starts: every Manager SFT
        # decision target must be a legal constrained action.
        from .actions import decision_paths
        from .protocol import COMMIT, DECIDE, KINDS, call_message
        from .training import tokenize_turn
        history = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Compute 1+1."},
                   {"role": "assistant", "content": "CANDIDATE_ANSWER: 2"}, {"role": "user", "content": DECIDE}]
        paths = decision_paths(tok, history, tool_schemas(), config["decision_max_tokens"])
        for response in [{"role": "assistant", "content": COMMIT}] + [call_message(k, None, "call_" + k) for k in KINDS]:
            row = {"prompt": history, "response": [response],
                   "decision_type": "call" if response.get("tool_calls") else "commit"}
            encoded = tokenize_turn(row, tok, config["max_seq_len"])
            target = [t for t, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
            if not any(target[:len(path)] == path for path in paths):
                raise RuntimeError("Manager SFT decision target is outside the finite action grammar")
        grammar = {"paths": len(paths), "max_path_tokens": max(map(len, paths))}
    health = requests.get(config["advisor_url"].rstrip("/") + "/health", timeout=15)
    health.raise_for_status()
    try:
        advisor = health.json()
    except ValueError:
        advisor = {"external_advisor_identity": config.get("external_advisor_identity")}
    if advisor.get("status") != "ready" and not config.get("external_advisor_identity"):
        raise RuntimeError("Frozen advisor server is not ready")
    result = {"packages": {name: importlib.metadata.version(name) for name in
              ("torch", "transformers", "trl", "peft", "datasets", "math-verify")},
              "gpu": torch.cuda.get_device_name(0),
              "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2 ** 30,
              "model_type": model_cfg.model_type,
              "decision_grammar": grammar,
              "advisor": advisor,
              "note": "API and template preflight only; run the GPU smoke test before the main experiment"}
    write_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description="MARGENT free-response math / iterative self-training")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--train-size", type=int, default=1024)
    prep.add_argument("--dev-size", type=int, default=256)
    prep.add_argument("--scan-limit", type=int, default=30000)
    prep.add_argument("--seed", type=int, default=42)
    for name in ("numina", "aime2026", "beyondaime"):
        prep.add_argument("--local-" + name, help="Optional local source JSONL, mainly for offline tests")
    freeze = sub.add_parser("freeze-config", help="Resolve an immutable model revision before all runs")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--out", required=True)
    check = sub.add_parser("doctor")
    check.add_argument("--config", required=True)
    check.add_argument("--out", default="environment_report.json")
    wb = sub.add_parser("wandb-check", help="Check W&B login/logging without a GPU or model")
    wb.add_argument("--out", required=True)
    review = sub.add_parser("wandb-upload-records", help="Review saved stage outputs in W&B without a model or GPU")
    review.add_argument("--run-dir", required=True, help="Stage directory containing records.jsonl or generations.jsonl")
    review.add_argument("--out", required=True, help="New review directory outside the source experiment")
    for name in ("collect", "diagnose", "evaluate", "sft", "rl"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--data", required=True)
        sp.add_argument("--checkpoint")
        sp.add_argument("--out", required=True)
        if name in ("collect", "diagnose", "evaluate"):
            sp.add_argument("--resume", action="store_true")
            sp.add_argument("--limit", type=int, default=0)
            sp.add_argument("--selection", choices=("counterfactual", "success"), default="counterfactual")
    loop = sub.add_parser("loop")
    loop.add_argument("--config", required=True)
    loop.add_argument("--data-dir", required=True)
    loop.add_argument("--out", required=True)
    loop.add_argument("--arm", choices=SFT_ARMS, default="dynamic_sft")
    loop.add_argument("--rounds", type=int, default=2)
    loop.add_argument("--checkpoint")
    loop.add_argument("--resume", action="store_true")
    loop.add_argument("--dry-run", action="store_true")
    cmp = sub.add_parser("compare")
    cmp.add_argument("--before", required=True)
    cmp.add_argument("--after", required=True)
    cmp.add_argument("--out", required=True)
    status = sub.add_parser("status", help="Read phase/heartbeat/progress/failure state without loading a model")
    status.add_argument("--run-dir", required=True)
    suite = sub.add_parser("evaluate-suite", help="Locked initial/final checkpoints on both external test sets")
    suite.add_argument("--run-dir", required=True)
    suite.add_argument("--data-dir", required=True)
    suite.add_argument("--dry-run", action="store_true")
    report = sub.add_parser("report", help="Recompute CSV/LaTeX tables and PDF/PNG figures from observed records")
    report.add_argument("--runs", nargs="+", required=True)
    report.add_argument("--out", required=True)
    ready = sub.add_parser("paper-check", help="Validate completed matched SFT experiments and regenerate the report")
    ready.add_argument("--runs", nargs="+", required=True)
    ready.add_argument("--out", required=True)
    ready.add_argument("--min-seeds", type=int, default=2)
    estimate = sub.add_parser("pilot-cost", help="Estimate collection/dev runtime from observed pilot timing")
    estimate.add_argument("--collect-dir", required=True)
    estimate.add_argument("--diagnose-dir", required=True)
    for name, default in (("train-size", 128), ("dev-size", 64), ("arms", 2), ("seeds", 2), ("rounds", 2)):
        estimate.add_argument("--" + name, type=int, default=default)
    args = p.parse_args()
    if args.command == "freeze-config":
        cfg = load_config(args.config)
        if not Path(cfg["base_model"]).is_dir():
            from huggingface_hub import HfApi
            cfg["base_model_revision"] = HfApi().model_info(cfg["base_model"], revision=cfg.get("base_model_revision")).sha
        target = Path(args.out)
        if target.exists() and json.loads(target.read_text()) != cfg:
            raise ValueError("Frozen config already differs; use a fresh output")
        write_json(str(target), cfg)
        result = cfg
    elif args.command == "wandb-check":
        from .telemetry import Monitor, metrics
        from .wandb_tracking import tracking_mode
        if tracking_mode() == "disabled":
            raise ValueError("Set MARGENT_WANDB_MODE=online or offline before wandb-check")
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=True)
        manifest = {"config": {"purpose": "logging_check_only_no_model_or_training"}}
        target = root / "run.json"
        if target.exists():
            if json.loads(target.read_text()) != manifest:
                raise ValueError("Choose a separate directory for the W&B check")
        elif any(root.iterdir()):
            raise ValueError("Choose an empty directory for the W&B check")
        else:
            write_json(str(target), manifest)
        with Monitor(root, "tracking_check"):
            metrics({"logging_check": 1}, "check")
        result = json.loads((root / "wandb_link.json").read_text())
    elif args.command == "wandb-upload-records":
        from .debug_records import upload_records
        result = upload_records(args.run_dir, args.out)
    elif args.command == "status":
        from .telemetry import status_snapshot
        result = status_snapshot(args.run_dir)
    elif args.command == "evaluate-suite":
        result = evaluate_suite(args.run_dir, args.data_dir, args.dry_run)
        if args.dry_run:
            return
    elif args.command == "pilot-cost":
        from .analysis import pilot_cost
        result = pilot_cost(args.collect_dir, args.diagnose_dir, args.train_size, args.dev_size, args.arms, args.seeds, args.rounds)
    elif args.command == "paper-check":
        from .readiness import paper_check
        result = paper_check(args.runs, args.out, args.min_seeds)
        if not result["complete"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(2)
    elif args.command == "report":
        from .reporting import generate_report
        result = generate_report(args.runs, args.out)
    elif args.command == "prepare":
        sources = {name: getattr(args, "local_" + name) for name in ("numina", "aime2026", "beyondaime")
                   if getattr(args, "local_" + name)}
        result = prepare(args.data_dir, args.train_size, args.dev_size, args.seed, args.scan_limit, sources)
    elif args.command == "compare":
        result = compare(read_jsonl(args.before), read_jsonl(args.after))
        write_json(args.out, result)
    elif args.command == "loop":
        result = run_loop(args.config, args.data_dir, args.out, args.arm, args.rounds,
                          args.checkpoint, args.resume, args.dry_run)
        if args.dry_run:
            return
    else:
        cfg = load_config(args.config)
        if args.command == "doctor":
            result = doctor(cfg, args.out)
        else:
            checkpoint = args.checkpoint or cfg["base_model"]
            if args.command in ("sft", "rl"):
                from .training import train_sft, train_rl
                fn = train_sft if args.command == "sft" else train_rl
                result = fn(cfg, checkpoint, args.data, args.out)
            else:
                result = run_data(cfg, args.data, checkpoint, args.out, args.command,
                                  args.resume, args.limit, args.selection)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
