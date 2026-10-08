# MCQ multi-round RSI: results record (2026-10-03 … 2026-10-08)

The shareable results page and the material behind it. The page covers the paper's four MCQ benchmarks (MedQA, MMLU-Pro,
GPQA Diamond, AQuA) with the manager trained for three to five self-improvement rounds:

- sections 0–7: the paper's label-selection rule (`draft_supervision = all`) with FA-GRPO (runs `<bench>_main`, `_abl`, `_r5`);
- section 7.1: the draft feedback loop found in the dynamic arm (rescue rows supervise the manager's own wrong draft);
- section 8: the reruns with the draft tokens masked, v2 (`commit_rows`) and v3 (`none`), no GRPO (runs `<bench>_v2`, `<bench>_v3`);
- section 9: the v3 continuation to round 5 (runs `<bench>_v3r5`; GPQA has no spare questions).

| File | What it is |
|---|---|
| `margent_rsi_results.html` | the page (published as a private claude.ai artifact; the HTML is self-contained apart from Google Fonts) |
| `page_v1_base.html` | the page as of 2026-10-06, sections 0–7 only; `scripts/patch_page_v3.py` builds the current page from it |
| `scripts/patch_page_v3.py`, `scripts/mmlu_r5_numbers.py` | page build: inserts sections 8–9, the summary update, caveats, data links and the v3 figure |
| `scripts/compare_v2.py`, `compare_v3.py`, `compare_v3r5.py` | paired bootstrap / McNemar comparisons on the locked tests (`scripts/mcq_rsi_analysis.py compare`); `python compare_v3.py <bench> [test\|test_paper]` |
| `results/compare_all_v2_v3_v3r5.txt` | their output for every benchmark and pool |
| `results/pod_stats_v2_v3_v3r5.txt` | per-run draft letters and advisor mix (locked tests), dev metrics per round, matched-budget replay rows |
| `results/clean104.json` | comparisons on the 104 `test_paper` questions that overlap no training pool (v1 runs; v2/v3/v3r5 values are in the page) |
| `results/locked_test_outputs_v2_v3_v3r5.tar.gz` | per-question locked-test outputs (`final/<label>/<pool>/manager_tool_eval.jsonl`, `final.json`) of the 11 v2/v3/v3r5 runs, plus their round 4–5 dev decisions; extract to `results/runs/` before running the compare scripts |

Every run directory (configs, per-round adapters, dev and locked-test outputs, analysis tables) is on Hugging Face:
`MaliDDD/margent-mcq-rsi` (v1 runs; `medqa_r5` inside `archives/`) and `MaliDDD/margent-mcq-rsi-<run>` for each v2/v3/v3r5 run.
Pod-side operation scripts are in `scripts/pod_ops/`; the procedure is `docs/MCQ_RSI_RUNBOOK.md`.
