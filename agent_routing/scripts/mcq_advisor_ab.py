#!/usr/bin/env python3
"""Do the trained advisor LoRAs help more than the base model in the same role? (A/B on a dev pool)

A fixed solver (the base model, greedy, thinking off) answers each question (the draft), reads one
advisor's output and answers again. The two arms differ only in who wrote the advisor output:

- ``lora``: the benchmark's trained adapter, served as ``<bench>_<kind>`` (renamed keys, so vLLM applies it);
- ``base``: the base model with the same system prompt and request (what the paper-era server returned, D14);
- extra arms (``--extra-arms ep1,ep2``): other served adapters of the same advisor, ``<bench>_<kind>_<arm>``
  (e.g. intermediate SFT checkpoints, to test over-fitting); each arm is compared with ``base``.

Advisor requests are built by ``CachedAdvisorPool.request_payload`` (the production request; the Verifier
audits the solver's draft). Reported per benchmark, kind and arm: final accuracy, gain over the draft,
corrections (draft wrong -> final right), corruptions (draft right -> final wrong), invalid final answers,
and the advisor outputs' format compliance (valid JSON, schema keys, every choice covered by the Reasoner).
A paired sign test compares the arms' final correctness per question.

    python scripts/mcq_advisor_ab.py --url http://127.0.0.1:18002 --out /workspace/mcq_rsi/logs/advisor_ab
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KINDS = ("extractor", "reasoner", "verifier")
ARMS = ("lora", "base")
SCHEMA = {
    "extractor": ("key_evidence", "extracted_facts", "missing_info", "context_summary", "confidence"),
    "reasoner": ("case_facts", "task_type", "decision_factors", "knowledge_slots", "candidate_considerations",
                 "missing_information", "format_confidence"),
    "verifier": ("relevant_principles", "checks", "potential_errors", "candidate_answer_audit", "uncertainty_notes",
                 "confidence"),
}
ANSWER = re.compile(r"ANSWER\s*[:：]\s*\(?\s*([A-J])\b")
SOLVER_SYSTEM = "You are an expert at answering multiple-choice questions accurately."


def letter(text: str, keys) -> Optional[str]:
    """The last ``ANSWER: X`` with X among the choice keys, else None."""
    hits = [m.group(1) for m in ANSWER.finditer(text or "") if m.group(1) in keys]
    return hits[-1] if hits else None


def question_block(row: Dict[str, Any]) -> str:
    choices = "\n".join(f"{k}. {v}" for k, v in row["choices"].items())
    ctx = (row.get("context") or "").strip()
    return (f"Context:\n{ctx}\n\n" if ctx else "") + f"Question:\n{row['question']}\n\nChoices:\n{choices}"


def draft_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": question_block(row) + "\n\nThink briefly, then give your final answer on "
                                                               "the last line as 'ANSWER: <letter>'."}]


def revise_messages(row: Dict[str, Any], draft: str, kind: str, advisor_output: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": question_block(row) + f"\n\nYour initial answer: {draft or 'none'}\n\n"
             f"A {kind} sub-agent returned this analysis:\n{advisor_output}\n\nUse the analysis where it helps, "
             "then give your final answer on the last line as 'ANSWER: <letter>'."}]


def parse_json(text: str) -> Optional[Any]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("\n") + 1:] if "\n" in t else t
    try:
        return json.loads(t)
    except (json.JSONDecodeError, ValueError):
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                return None
        return None


def format_checks(kind: str, text: str, keys) -> Dict[str, bool]:
    obj = parse_json(text)
    ok = isinstance(obj, dict)
    out = {"json": ok, "schema": ok and all(k in obj for k in SCHEMA[kind])}
    if kind == "reasoner":
        covered = set()
        if ok and isinstance(obj.get("candidate_considerations"), list):
            covered = {str(c.get("choice_key", "")).strip() for c in obj["candidate_considerations"] if isinstance(c, dict)}
        out["all_choices"] = set(keys) <= covered
    return out


def sign_test(a_only: int, b_only: int) -> float:
    """Two-sided exact sign test on the discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


class Store:
    """Resumable response store: one JSON line per (stage, kind, arm, example_id)."""

    def __init__(self, path: Path):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:  # a line cut short by a kill is skipped (and refetched)
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.data[rec["key"]] = rec["text"]

    def put(self, key: str, text: str) -> None:
        with self.lock:
            self.data[key] = text
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")


def run_bench(bench: str, url: str, out_dir: Path, workers: int, limit: Optional[int], kinds,
              solver_max_tokens: int = 1024, advisor_max_tokens: Optional[int] = None,
              arms: Tuple[str, ...] = ARMS) -> Dict[str, Any]:
    import requests
    from src.manager.mcq_rsi import benchmarks as registry
    from src.manager.mcq_rsi import controller
    from src.manager.mcq_rsi.advisors import CachedAdvisorPool
    root = Path(__file__).resolve().parents[1]
    cfg = controller.load_config(json.loads((root / "configs" / f"mcq_rsi_{bench}.json").read_text()))
    cfg.update(import_dir=os.environ.get("MCQ_IMPORT_DIR", "/workspace/mcq_rsi/import"))
    rows = controller.Runtime(cfg).rows(cfg["dev_pool"])[:limit]
    pool = CachedAdvisorPool(bench, out_dir / "unused_cache", None)
    decode = dict(pool.decode)
    store = Store(out_dir / f"{bench}_responses.jsonl")
    session = requests.Session()

    def chat(key: str, payload: Dict[str, Any]) -> str:
        if key in store.data:
            return store.data[key]
        for attempt in range(4):
            try:
                r = session.post(f"{url}/v1/chat/completions", json=payload, timeout=600)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                store.put(key, text)
                return text
            except Exception:  # noqa: BLE001 - retried, then raised
                if attempt == 3:
                    raise
        raise AssertionError

    # The solver may need more room than the advisors (whose decode is the production one): a draft cut off
    # before its ANSWER line counts as wrong in both arms and hides the advisors' effect.
    solver_decode = {**decode, "max_tokens": solver_max_tokens}
    tag = "" if solver_max_tokens == decode.get("max_tokens") else f"s{solver_max_tokens}/"
    advisor_max_tokens = advisor_max_tokens or decode.get("max_tokens")
    atag = "" if advisor_max_tokens == decode.get("max_tokens") else f"a{advisor_max_tokens}/"

    def solver(key, messages):
        return chat(tag + key, {"model": registry.BASE_MODEL, "messages": messages, **solver_decode})

    with ThreadPoolExecutor(workers) as ex:
        drafts = list(ex.map(lambda r: solver(f"draft/{r['example_id']}", draft_messages(r)), rows))
    draft_letter = [letter(t, r["choices"]) for t, r in zip(drafts, rows)]

    def advisor(job):
        i, kind, arm = job
        r = rows[i]
        cand = (draft_letter[i] or "") if kind == "verifier" else ""
        model = {"lora": pool.spec.lora_name(kind), "base": registry.BASE_MODEL}.get(arm, f"{pool.spec.lora_name(kind)}_{arm}")
        p = pool.request_payload(kind, r["question"], r.get("context") or "", r["choices"], cand, model=model)
        p["max_tokens"] = advisor_max_tokens
        # The Verifier audits the draft, so its request (and key) depends on the solver budget too.
        key = (tag if kind == "verifier" else "") + atag + f"advisor/{kind}/{arm}/{r['example_id']}"
        return chat(key, p)

    jobs = [(i, k, a) for k in kinds for a in arms for i in range(len(rows))]
    with ThreadPoolExecutor(workers) as ex:
        adv = dict(zip(jobs, ex.map(advisor, jobs)))

    def revise(job):
        i, kind, arm = job
        r = rows[i]
        # A Verifier revision follows the audit of this budget's draft ("cand/" keeps earlier, stale keys apart).
        cand = "cand/" if kind == "verifier" and tag else ""
        return solver(atag + cand + f"revise/{kind}/{arm}/{r['example_id']}",
                      revise_messages(r, draft_letter[i] or "", kind, adv[job]))

    with ThreadPoolExecutor(workers) as ex:
        final = dict(zip(jobs, ex.map(revise, jobs)))

    gold = [r["ground_truth"] for r in rows]
    n = len(rows)
    d_ok = [draft_letter[i] == gold[i] for i in range(n)]
    result: Dict[str, Any] = {"bench": bench, "pool": cfg["dev_pool"], "n": n, "solver_max_tokens": solver_max_tokens,
                              "advisor_max_tokens": advisor_max_tokens, "arms": list(arms),
                              "draft_accuracy": sum(d_ok) / n,
                              "draft_invalid": sum(x is None for x in draft_letter), "kinds": {}}
    for kind in kinds:
        per_arm, correct = {}, {}
        for arm in arms:
            f_letter = [letter(final[(i, kind, arm)], rows[i]["choices"]) for i in range(n)]
            ok = [f_letter[i] == gold[i] for i in range(n)]
            correct[arm] = ok
            fmt = [format_checks(kind, adv[(i, kind, arm)], rows[i]["choices"]) for i in range(n)]
            per_arm[arm] = {
                "accuracy": sum(ok) / n,
                "gain_pp": 100 * (sum(ok) - sum(d_ok)) / n,
                "corrections": sum((not d) and o for d, o in zip(d_ok, ok)),
                "corruptions": sum(d and (not o) for d, o in zip(d_ok, ok)),
                "final_invalid": sum(x is None for x in f_letter),
                "advisor_valid_json": sum(f["json"] for f in fmt) / n,
                "advisor_schema": sum(f["schema"] for f in fmt) / n,
                **({"reasoner_all_choices": sum(f["all_choices"] for f in fmt) / n} if kind == "reasoner" else {}),
                "advisor_median_chars": sorted(len(adv[(i, kind, arm)]) for i in range(n))[n // 2],
            }
        vs_base = {}
        for arm in arms:
            if arm == "base":
                continue
            only = sum(a and not b for a, b in zip(correct[arm], correct["base"]))
            base_only = sum(b and not a for a, b in zip(correct[arm], correct["base"]))
            vs_base[arm] = {"minus_base_pp": 100 * (only - base_only) / n, "arm_only_correct": only,
                            "base_only_correct": base_only, "sign_test_p": sign_test(only, base_only)}
        result["kinds"][kind] = {**per_arm, "vs_base": vs_base}
        if "lora" in vs_base:  # the original field names
            v = vs_base["lora"]
            result["kinds"][kind].update(lora_minus_base_pp=v["minus_base_pp"], lora_only_correct=v["arm_only_correct"],
                                         base_only_correct=v["base_only_correct"], sign_test_p=v["sign_test_p"])
    extra = [a for a in arms if a not in ARMS]
    suffix = "".join("_" + t.rstrip("/") for t in (tag, atag) if t) + ("_arms-" + "-".join(extra) if extra else "")
    (out_dir / f"{bench}_summary{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def render(results: List[Dict[str, Any]]) -> str:
    lines = []
    for r in results:
        lines.append(f"\n{r['bench']} ({r['pool']}, n={r['n']}, solver/advisor max_tokens "
                     f"{r['solver_max_tokens']}/{r['advisor_max_tokens']}): "
                     f"draft accuracy {r['draft_accuracy']:.3f} (invalid {r['draft_invalid']})")
        lines.append(f"  {'kind':9} {'arm':5} {'final':>6} {'gain':>6} {'fix':>4} {'break':>5} {'inval':>5} {'json':>5} "
                     f"{'schema':>6} {'chars':>6}")
        for kind, k in r["kinds"].items():
            for arm in r.get("arms", ARMS):
                a = k[arm]
                lines.append(f"  {kind:9} {arm:5} {a['accuracy']:6.3f} {a['gain_pp']:+6.1f} {a['corrections']:4d} "
                             f"{a['corruptions']:5d} {a['final_invalid']:5d} {a['advisor_valid_json']:5.2f} {a['advisor_schema']:6.2f} "
                             f"{a['advisor_median_chars']:6d}")
            for arm, v in k.get("vs_base", {}).items():
                lines.append(f"  {kind:9} {arm} - base: {v['minus_base_pp']:+.1f} pt ({arm}-only correct "
                             f"{v['arm_only_correct']}, base-only {v['base_only_correct']}, sign test p={v['sign_test_p']:.3f})")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benches", default="medqa,mmlu_pro,gpqa,aqua")
    p.add_argument("--kinds", default=",".join(KINDS))
    p.add_argument("--url", default="http://127.0.0.1:18002")
    p.add_argument("--out", default="/workspace/mcq_rsi/logs/advisor_ab")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit", type=int, default=None, help="first N dev questions (default: all)")
    p.add_argument("--solver-max-tokens", type=int, default=1024, help="solver draft/revision budget")
    p.add_argument("--advisor-max-tokens", type=int, default=None, help="advisor budget (default: production decode)")
    p.add_argument("--extra-arms", default="", help="extra served adapters <bench>_<kind>_<arm>, e.g. ep1,ep2")
    p.add_argument("--no-lora", action="store_true", help="drop the final-adapter arm (only base + extra arms)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for bench in a.benches.split(","):
        results.append(run_bench(bench, a.url.rstrip("/"), out, a.workers, a.limit, tuple(a.kinds.split(",")),
                                 a.solver_max_tokens, a.advisor_max_tokens,
                                 tuple(x for x in ARMS if not (a.no_lora and x == "lora"))
                                 + tuple(x for x in a.extra_arms.split(",") if x)))
        print(render(results[-1:]), flush=True)
    text = render(results)
    (out / f"summary_{a.benches.replace(',', '_')}_s{a.solver_max_tokens}_a{a.advisor_max_tokens or 'prod'}.txt").write_text(text + "\n")
    print("\n==== all benchmarks ====" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
