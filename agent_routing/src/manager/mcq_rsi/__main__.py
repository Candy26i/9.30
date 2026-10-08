"""CLI: python -m src.manager.mcq_rsi <command> ..."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import List, Optional

from . import benchmarks as registry
from . import importer, prompts, splits


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
