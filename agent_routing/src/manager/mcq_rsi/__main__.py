"""CLI: python -m src.manager.mcq_rsi <command> ..."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import List, Optional

from . import benchmarks as registry
from . import importer, prompts, splits

ADVISOR_CACHE = "outputs/mcq_rsi/advisor_cache"


def _benches(name: str) -> List[registry.Benchmark]:
    return list(registry.BENCHMARKS.values()) if name == "all" else [registry.get(name)]


def cmd_show_registry(args) -> int:
    registry.validate()
    prompts.validate()
    out = {}
    for b in _benches(args.bench):
        info = dataclasses.asdict(b)
        info["lora_names"] = {kind: b.lora_name(kind) for kind, _ in b.advisors}
        info["prompts"] = prompts.PROMPTS[b.name]
        out[b.name] = info
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_import(args) -> int:
    registry.validate()
    out = args.out or str(registry.PACKAGE_ROOT / importer.DEFAULT_OUT)
    parts = tuple(x for x in args.parts.split(",") if x)
    if not parts:
        raise SystemExit("--parts must name at least one of " + ",".join(importer.PARTS))
    listing = {}
    for b in _benches(args.bench):
        result = importer.run_import(b, out, dry_run=args.dry_run, cache_dir=args.cache_dir, parts=parts)
        if args.dry_run:
            listing[b.name] = result
        else:
            counts = {}
            for f in result["files"]:
                counts[f["status"]] = counts.get(f["status"], 0) + 1
            print(f"[MCQ_RSI/IMPORT] {b.name}: {counts} -> {out}/{b.name}/import_manifest.json")
    if args.dry_run:
        print(json.dumps(listing, indent=2))
    return 0


def cmd_prepare_splits(args) -> int:
    registry.validate()
    benches = _benches(args.bench)
    if len(benches) > 1 and (args.records or args.advisor_sft or args.cache or args.out):
        raise SystemExit("--records/--advisor-sft/--cache/--out need a single --bench")
    import_dir = args.import_dir or str(registry.PACKAGE_ROOT / importer.DEFAULT_OUT)
    for b in benches:
        paths = importer.default_paths(b, import_dir)
        manifest = splits.prepare_benchmark(
            b.name,
            args.records or str(paths["records"]),
            args.advisor_sft or [str(p) for p in paths["advisor_sft"]],
            cache=args.cache,
            out=args.out,
            hf_cache_dir=args.hf_cache_dir,
            seed=args.seed,
            force=args.force,
        )
        print(f"[MCQ_RSI/SPLITS] {b.name}: {manifest['counts']} -> {args.out or b.split_manifest}")
    return 0


def _load_manager(args, bench):
    from . import protocol
    backend = protocol.load_hf_manager(args.checkpoint, args.base_model, args.base_revision, args.batch_size)
    return protocol.Manager(backend)


def _make_pool(args, bench):
    from .advisors import CachedAdvisorPool
    return CachedAdvisorPool(bench.name, args.advisor_cache or str(registry.PACKAGE_ROOT / ADVISOR_CACHE),
                             args.advisor_url, workers=args.workers)


def cmd_collect(args) -> int:
    from . import collect
    bench = registry.get(args.bench)
    manifest = splits.read_manifest(args.manifest or bench.path(bench.split_manifest))
    rows = splits.pool_rows(manifest, args.pool)
    if args.limit:
        rows = rows[:args.limit]
    round_index = args.round if args.round is not None else int(args.pool.rsplit("_r", 1)[-1])
    pool = _make_pool(args, bench)
    if args.advisor_url:
        pool.check_server()
    result = collect.collect(rows, bench, _load_manager(args, bench), pool, args.out, pool_name=args.pool,
                             round_index=round_index, root_mode=args.root_mode,
                             max_depth=args.max_depth or bench.depth, seed=args.seed, resume=args.resume)
    rep, checks = result["report"], result["report"]["unconstrained_argmax"]
    print(f"[MCQ_RSI/COLLECT] {bench.name}/{args.pool}: direct={rep['direct_accuracy']:.3f} "
          f"oracle={rep['oracle_accuracy']:.3f} policy_call_rate={rep['policy_call_rate']:.3f} "
          f"argmax_mismatch={checks['mismatch']} revision_would_call={checks['would_call']} "
          f"answer_draft_mismatch={rep['revision_answer_draft_mismatch']} advisor_stats={pool.stats} "
          f"-> {result['records_jsonl']}")
    return 0


def cmd_select(args) -> int:
    from . import select
    bench = registry.get(args.bench)
    if args.arm != "static" and not args.records:
        raise SystemExit(f"--records is required for arm {args.arm}")
    report = select.write_selection(bench, args.arm, args.records, args.out, rho=args.rho, seed=args.seed,
                                    max_depth=args.max_depth, tie_break_seed=args.tie_break_seed,
                                    import_dir=args.import_dir)
    print(f"[MCQ_RSI/SELECT] {bench.name}/{args.arm}: {report['n_sft_turns']} rows sha256={report['sha256']} -> {args.out}")
    return 0


def _rows(args, bench):
    manifest = splits.read_manifest(args.manifest or bench.path(bench.split_manifest))
    rows = splits.pool_rows(manifest, args.pool)
    return rows[:args.limit] if args.limit else rows


def _read_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _base(args) -> str:
    """``--base-model`` pinned: a local snapshot of ``--base-model@--base-revision`` (local directories as given),
    so evaluate, SFT and FA-GRPO all load the same base commit."""
    from .evaluate import resolve_base
    return resolve_base(args.base_model, args.base_revision or None)


def cmd_grpo(args) -> int:
    from . import grpo
    bench = registry.get(args.bench)
    config = json.loads(open(args.config, encoding="utf-8").read()) if args.config else {}
    config.update(bench=bench.name, base_model=_base(args), base_revision=args.base_revision or None)
    # The anchor renders S_k's tool block as S_k's SFT did: default from its sft_report.json.
    context = args.anchor_context or grpo.recorded_sft_context(args.checkpoint)
    if context:
        config["anchor_context"] = context
    for key in ("steps", "estimator", "seed", "learning_rate"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.decrl:
        config.update(decision_pg=True, decision_pg_epsilon=args.decrl_epsilon)
    pool = _make_pool(args, bench)
    if args.advisor_url:
        pool.check_server()
    summary = grpo.train_fa_grpo(config, args.checkpoint, _rows(args, bench), _read_jsonl(args.anchor), pool,
                                 args.out, resume=not args.no_resume)
    info = summary["informativeness"]
    print(f"[MCQ_RSI/GRPO] {bench.name}/{args.pool}: steps={summary['steps']} selected_step={summary['selected_step']} "
          f"accepted_by_guard={summary['accepted_by_guard']} informative={info['fraction']:.3f} "
          f"(passed={info['passed']}) -> {summary['final_dir']}")
    return 0


def cmd_sft(args) -> int:
    from . import sft
    bench = registry.get(args.bench)
    config = {"num_train_epochs": args.epochs, "learning_rate": args.lr, "gradient_accumulation_steps": args.grad_accum,
              "max_seq_len": args.max_seq_len, "seed": args.seed, "max_steps": args.max_steps, "context": args.context}
    if args.parity_only:
        from transformers import AutoTokenizer
        labels = sft.validate_labels(args.labels, bench)
        tok = AutoTokenizer.from_pretrained(args.tokenizer or args.init, trust_remote_code=True)
        parity = sft.tokenization_parity(args.labels, tok, args.max_seq_len, args.context)
        print(json.dumps({"labels": labels, "parity": parity}, indent=2))
        return 0 if not parity["supervised_mismatch"] and (args.context != "paper" or not parity["input_mismatch"]) else 1
    report = sft.train_round_sft(args.labels, args.init, args.out, base_model=_base(args),
                                 base_revision=args.base_revision or None, bench=bench.name,
                                 config=config)
    print(f"[MCQ_RSI/SFT] {bench.name}: {report['labels']['rows']} rows, parity input_mismatch="
          f"{report['tokenization_parity']['input_mismatch']} -> {report['model_dir']}")
    return 0


def cmd_evaluate(args) -> int:
    from . import evaluate
    bench = registry.get(args.bench)
    pool = _make_pool(args, bench)
    if args.advisor_url:
        pool.check_server()
    rows = _rows(args, bench)
    if args.forced:
        result = evaluate.evaluate_forced(args.checkpoint, rows, pool, args.out, args.forced.split(","),
                                          bench=bench.name, base_model=_base(args),
                                          base_revision=args.base_revision or None,
                                          require_gate=not args.no_require_gate)
    else:
        result = evaluate.evaluate(args.checkpoint, rows, pool, args.out, bench=bench.name, base_model=_base(args),
                                   base_revision=args.base_revision or None, speculative=not args.no_speculative,
                                   require_gate=not args.no_require_gate)
    m = result["metrics"]
    print(f"[MCQ_RSI/EVAL] {bench.name}/{args.pool} {args.forced or 'tools'}: acc={m['accuracy']:.4f} "
          f"calls={m['calls_per_example']:.3f}" + (f" call_gap={m['call_gap']:.4f}" if "call_gap" in m else "")
          + f" gate={result['gate'] or 'pass'} -> {args.out}")
    return 0 if result["passed"] else 1


def _advisor_args(p) -> None:
    p.add_argument("--advisor-url", default=None, help="vLLM server; omit to run from the advisor cache only")
    p.add_argument("--advisor-cache", default=None, help=f"default: agent_routing/{ADVISOR_CACHE}")
    p.add_argument("--workers", type=int, default=32, help="concurrent advisor requests")
    p.add_argument("--manifest", default=None, help="default: registry split manifest")
    p.add_argument("--limit", type=int, default=0, help="first N rows only (smoke runs)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.manager.mcq_rsi")
    sub = parser.add_subparsers(dest="command", required=True)
    bench_choices = ["all", *registry.BENCHMARKS]

    p = sub.add_parser("show-registry", help="print the frozen per-benchmark registry")
    p.add_argument("--bench", default="all", choices=bench_choices)
    p.set_defaults(func=cmd_show_registry)

    p = sub.add_parser("import", help="fetch pinned advisors, S_1 and round-1 data from HF")
    p.add_argument("--bench", default="all", choices=bench_choices)
    p.add_argument("--out", default=None, help=f"default: agent_routing/{importer.DEFAULT_OUT}")
    p.add_argument("--cache-dir", default=None, help="huggingface_hub cache directory")
    p.add_argument("--dry-run", action="store_true", help="list what would be fetched")
    p.add_argument("--parts", default=",".join(importer.PARTS), help="comma list of advisors,manager,data")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("prepare-splits", help="build frozen question-hash split manifests")
    p.add_argument("--bench", default="all", choices=bench_choices)
    p.add_argument("--import-dir", default=None, help="where `import` put round-1 records and advisor SFT data")
    p.add_argument("--records", default=None, help="round-1 counterfactual_records.jsonl (overrides --import-dir)")
    p.add_argument("--advisor-sft", nargs=3, default=None, metavar=("EXTRACTOR", "REASONER", "VERIFIER"))
    p.add_argument("--cache", default=None, help="normalized cache (default: registry path)")
    p.add_argument("--out", default=None, help="manifest path (default: registry split_manifest)")
    p.add_argument("--hf-cache-dir", default=None, help="huggingface_hub cache for the pinned AQuA raw files")
    p.add_argument("--seed", type=int, default=splits.PAPER_SEED, help="non-default seeds need --out")
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest that differs")
    p.set_defaults(func=cmd_prepare_splits)

    p = sub.add_parser("collect", help="on-policy counterfactual collection of one root pool (GPU manager)")
    p.add_argument("--bench", required=True, choices=list(registry.BENCHMARKS))
    p.add_argument("--pool", required=True, help="split-manifest pool, e.g. collect_r2")
    p.add_argument("--round", type=int, default=None, help="default: the pool's _r<k> suffix")
    p.add_argument("--checkpoint", required=True, help="manager LoRA adapter (with tokenizer + template)")
    p.add_argument("--out", required=True)
    p.add_argument("--advisor-url", default=None, help="vLLM server; omit to run from the advisor cache only")
    p.add_argument("--advisor-cache", default=None, help=f"default: agent_routing/{ADVISOR_CACHE}")
    p.add_argument("--root-mode", default="policy", choices=["policy", "probe"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--manifest", default=None, help="default: registry split manifest")
    p.add_argument("--base-model", default=registry.BASE_MODEL)
    p.add_argument("--base-revision", default=registry.BASE_REVISION)
    p.add_argument("--max-depth", type=int, default=None, help="default: registry depth (2)")
    p.add_argument("--seed", type=int, default=42, help="tie-break Random(seed + example_id)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="revisions per generate call; >1 left-pads (not eval's batch-1 numerics)")
    p.add_argument("--workers", type=int, default=32, help="concurrent advisor requests")
    p.add_argument("--limit", type=int, default=0, help="first N roots only (smoke runs)")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("select", help="records -> Manager SFT rows for one arm")
    p.add_argument("--bench", required=True, choices=list(registry.BENCHMARKS))
    p.add_argument("--arm", required=True, choices=list(registry.ARMS))
    p.add_argument("--records", default=None, help="counterfactual_records.jsonl (not used by static)")
    p.add_argument("--out", required=True)
    p.add_argument("--rho", type=float, default=None, help="default: registry rho")
    p.add_argument("--seed", type=int, default=None,
                   help="_balance_records / success-draw seed (default: registry balance_seed, the round-1 one)")
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--tie-break-seed", type=int, default=42, help="collection seed, used when --max-depth cuts")
    p.add_argument("--import-dir", default=None, help="static arm: where `import` put round1/labels.jsonl")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("grpo", help="FA-GRPO from S_k on a grpo_rk pool (one manager GPU)")
    p.add_argument("--bench", required=True, choices=list(registry.BENCHMARKS))
    p.add_argument("--pool", required=True, help="split-manifest pool, e.g. grpo_r1")
    p.add_argument("--checkpoint", required=True, help="S_k: the round's SFT adapter (also the frozen KL reference)")
    p.add_argument("--anchor", required=True, help="this round's Manager SFT label file (route-only anchor rows)")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None, help="JSON file of FAGRPOConfig overrides")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--estimator", default=None, choices=["exact", "sampled"])
    p.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decrl", action="store_true", help="decision-PG ablation (MedQA only; guard logged, not enforced)")
    p.add_argument("--decrl-epsilon", type=float, default=0.0, help="lexicographic call tie-break (0.05)")
    p.add_argument("--anchor-context", default=None, choices=["paper", "evolve"],
                   help="tool-block rendering of the anchor; default: S_k's sft_report.json context, else paper")
    p.add_argument("--base-model", default=registry.BASE_MODEL)
    p.add_argument("--base-revision", default=registry.BASE_REVISION, help="'' = unpinned")
    p.add_argument("--no-resume", action="store_true")
    _advisor_args(p)
    p.set_defaults(func=cmd_grpo)

    p = sub.add_parser("sft", help="round-k Manager SFT continuation from G_(k-1)")
    p.add_argument("--bench", required=True, choices=list(registry.BENCHMARKS))
    p.add_argument("--labels", required=True, help="round-k Manager SFT jsonl (select output)")
    p.add_argument("--init", required=True, help="G_(k-1) adapter directory")
    p.add_argument("--out", default=None, help="stage directory (adapter in <out>/model); not needed with --parity-only")
    p.add_argument("--base-model", default=registry.BASE_MODEL)
    p.add_argument("--base-revision", default=registry.BASE_REVISION, help="'' = unpinned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--context", default="paper", choices=["paper", "evolve"],
                   help="paper: tool block rendered as paper-era SFT/deployment; evolve: 9.30 train_manager_sft as-is")
    p.add_argument("--parity-only", action="store_true", help="validate labels + tokenisation parity, no training")
    p.add_argument("--tokenizer", default=None, help="--parity-only: tokenizer directory (default --init)")
    p.set_defaults(func=cmd_sft)

    p = sub.add_parser("evaluate", help="paper-protocol eval of a checkpoint with cached fail-stop advisors + gates")
    p.add_argument("--bench", required=True, choices=list(registry.BENCHMARKS))
    p.add_argument("--pool", required=True, help="split-manifest pool, e.g. dev or test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="per-stage output directory")
    p.add_argument("--base-model", default=registry.BASE_MODEL)
    p.add_argument("--base-revision", default=registry.BASE_REVISION, help="'' = unpinned")
    p.add_argument("--forced", default=None, help="forced delegation, e.g. verifier or extractor,reasoner,verifier")
    p.add_argument("--no-speculative", action="store_true", help="skip the pass-1 Verifier prefetch")
    p.add_argument("--no-require-gate", action="store_true", help="write the result even if the eval gate fails")
    _advisor_args(p)
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args(argv)
    if args.command == "sft" and not args.parity_only and not args.out:
        parser.error("sft needs --out (or --parity-only)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
