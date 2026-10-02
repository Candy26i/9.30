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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
