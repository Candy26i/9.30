# MCQ RSI runbook (MedQA, MMLU-Pro, GPQA, AQuA-RAT)

Operator guide for multi-round MARGENT RSI on the four MCQ benchmarks:
`python -m src.manager.mcq_rsi` plus the scripts in `scripts/`.

Every command below runs from `agent_routing/` on the pod. In every new shell, paste this
block once (the wrapper scripts set the same values; manual commands need them too):

```bash
export PY=/workspace/mcq-venv/bin/python HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp
cd /workspace/9.30/agent_routing
```

`HF_HOME` matters beyond downloads: without it the base model resolves to a fresh
snapshot under `~/.cache/huggingface` on the container disk (an 18 GB re-download), and
FA-GRPO's run signature records the resolved snapshot path, so a GRPO stage resumed under
another `HF_HOME` is refused ("FA-GRPO inputs changed").

Long-running commands (`main`, `pilot`, `final-test`) must survive a dropped SSH or web
terminal: start them with `bash scripts/runpod_mcq_rsi.sh bg <step>` (a tmux session per
step) or inside your own `tmux new -s <name>`. Never run them bare in a terminal.

## 0. What runs

Round 1 for each benchmark is the paper's locked checkpoint `S_1`, together with the
label file it was trained on. Nothing is retrained in round 1.

Each later round `k` does the following:

1. Re-collect the counterfactual tree with the current manager `G_{k-1}`. It uses 400
   new roots, policy roots and depth 2. GPQA reuses its 200 roots.
2. Select labels:
   - **dynamic**: shortest successful sequence, capped by ρ.
   - **success**: a random successful option.
   - **static**: the round-1 file, every round.
3. Continue SFT from `G_{k-1}` (3 epochs, lr 1e-5).
4. Run FA-GRPO from `S_k`. Policy gradients reach only the revision; routing gets an exact KL penalty plus a route-only anchor and a guard.
5. Evaluate on dev and apply the gates. If GRPO is rejected, `G_k := S_k`.

The plan runs round by round. Identical stages run once: round 1 is shared by all arms,
`r2/collect` is shared by dynamic and success, and `static/select` serves every round. The
optional `dynamic_sft` arm starts round 2 from `S_1`, so it has its own collection and SFT.
Run `run --dry-run` to print the plan.

| Decision (design §8) | Default chosen |
|---|---|
| D1 GRPO objective | FA-GRPO: revision-only PG, exact root/decision KL (β 0.1/0.1, revision β 0.02), route-only anchor λ 0.03, call-rate/KL guard |
| D2 roots per round | 400 new per round; GPQA reuses its 200 (memorisation caveat) |
| D3 MedQA S_1 | `d2_400/sft_3.0` (seed 0, on HF), standing in for the locked `sft_3.0_s1` |
| D4 GPQA | locked `gpqa_marginal_9b-sft_2.0`; new dev-100; paper parity checked on Diamond-100 at final-test (reported, not gated) |
| D5 MMLU-Pro test | new clean 500 (`test`) + paper-reconstructed 500 (`test_paper`, flagged: overlaps dev and collection) |
| D6 depth | 2 everywhere |
| D7 ρ | fixed at the locked values (3.0 / 2.0 / 2.0 / 1.0) |
| D8 SFT continuation | 3 epochs, lr 1e-5, round-k rows only |
| D9 GRPO lr, size | 5e-6, 64 steps × 4 questions; the MedQA pilot sweeps {2e-6, 5e-6, 1e-5} first |
| D10 kernels | off (fla / causal-conv1d not installed), for paper parity |
| D11 arms / budget | Phase A: MedQA pilot. Phase B: all four benchmarks, dynamic/static/success, R=3. Optional `dynamic_sft` arm |
| D12 final checkpoint | last round per arm (`G_3`, or `S_3` if rejected), pre-registered; dev-best is not used |
| D13 schemas | deployment schema for collect/GRPO/eval; the paper SFT schema for SFT |
| **D14 (new)** advisor serving | Serve LoRAs vLLM actually applies (renamed copies, §5). If preflight shows the **paper's** recorded outputs came from the plain base, stop and decide (§6). |

## 1. Pod

- 2× A100-80GB or H100-80GB. GPU 0 runs the vLLM advisors (0.45 of its memory, about 36 GB) plus one inference manager (about 20 GB). GPU 1 runs SFT and FA-GRPO.
- A persistent network volume of **at least 200 GB mounted at `/workspace`**. It holds the HF cache (base model about 18 GB, adapters, tarballs), the advisor cache, the run directories and the backups. Without one, `/workspace` is on the container disk and is lost when the pod stops: run the hourly HF backup (§15) from the start and never stop the pod mid-run.
- A RunPod PyTorch CUDA image. Use driver ≥ 580 if the vLLM wheel is a CUDA 13 build (the previous pod had 580.126). Older drivers need a CUDA 12.x vLLM build (§2).
- Ports: on the previous pod, RunPod's nginx held `0.0.0.0:8001`, `3001`, `7270`, `7861`, `8081` and `9091`. The advisor server uses **18002**, bound to 127.0.0.1. The start script refuses a port that is in use or on that list.

## 2. Setup (once per pod; reruns are idempotent)

```bash
cd /workspace && git clone git@github.com:Candy26i/9.30.git && cd 9.30
git checkout feat/mcq-rsi-controller        # or the merged branch; record the commit: git rev-parse HEAD
cd agent_routing
bash scripts/setup_mcq_rsi_pod.sh
```

Pin the commit for the whole experiment: the run signature includes git HEAD and the
hash of every source file the controller uses, so a different checkout cannot resume a run.

The script does the following:
- Checks the volume and the GPUs (it needs 2).
- Creates `/workspace/mcq-venv` (experiment, `requirements-math.txt`) and a **separate** `/workspace/vllm-venv` with `vllm==0.26.0`, the version the paper's advisor servers ran. vLLM pins its own torch; v0.30.0, for example, pins torch 2.13, so it can never share the experiment venv. Qwen3.5 is supported from vLLM 0.17.
- Downloads `Qwen/Qwen3.5-9B@c202236…` into `HF_HOME=/workspace/hf-cache`.
- Runs the CPU test files on the CPU (`CUDA_VISIBLE_DEVICES=`), one process per file, each under `timeout 900` with `-rs` (every skip is printed). The SFT tokenisation-parity test on the real round-1 label files needs the imported S_1 tokenizer, so rerun the setup script once after §3's import; it then fails if that test is skipped.

The script checks the vLLM venv's CUDA build against the driver before anything touches the
GPU. If the build is CUDA ≥ 13 and the driver is below 580, rerun with a CUDA 12.x build of
**both** vLLM and torch:

```bash
VLLM_EXTRA_INDEX_URL=<index of a cu12x vLLM build> VLLM_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  bash scripts/setup_mcq_rsi_pod.sh
```

The venv is rebuilt automatically whenever the requested build (version and indexes)
differs from the installed one; pip alone would keep the wrong wheels ("Requirement already
satisfied"). To force a rebuild by hand: `rm -rf /workspace/vllm-venv`. To use another vLLM
version, set `VLLM_VERSION=…`; you must then start the server with `ALLOW_VLLM_VERSION=1`,
and preflight must pass again (the controller refuses a report from another vLLM version).

Credentials are never written by the scripts:
- **HF:** every MaliDDD artifact is public, so no login is needed to download. To upload backups, run `HF_HOME=/workspace/hf-cache /workspace/mcq-venv/bin/hf auth login` (`huggingface-cli` no longer works with huggingface_hub 1.x).
- **W&B:** run `/workspace/mcq-venv/bin/wandb login` once. The configs enable W&B (`"wandb": {"enabled": true}`, an operational setting outside the run signature; `MARGENT_WANDB_MODE=disabled` turns it off). Logging goes to entity `madisonlijingxuan-ucla`, project `MCQ_rsi`, one W&B run per run directory. It updates after every completed stage:
  - progress counters;
  - dev metrics, as summary values and a `dev/table`;
  - for each FA-GRPO stage, a per-step table and curves of loss, KL_dec, KL_root, train call rate, mean J and informative fraction. Guard values appear on guarded steps.

  A W&B failure is printed and never stops the run.

## 3. Import and the split check

```bash
$PY -m src.manager.mcq_rsi import --bench all --out /workspace/mcq_rsi/import --cache-dir $HF_HOME/hub
$PY -m src.manager.mcq_rsi prepare-splits --bench all --import-dir /workspace/mcq_rsi/import --hf-cache-dir $HF_HOME/hub
```

**Import.**
- It fetches only top-level adapter files, never `checkpoint-*` (about 0.4 GB per advisor set instead of 11 GB).
- It also fetches the named tarball members, verifies every digest and writes `<bench>/import_manifest.json`.
- Expect lines like `[MCQ_RSI/IMPORT] medqa: {'downloaded': 17, 'extracted': 5} -> …` (later imports: `{'present': 22}`).
- Re-importing is safe at any time: runs bind to the import manifest's content (files and digests), not to its per-file download status.

**Split check.**
- `prepare-splits` rebuilds every manifest from the caches and the imported round-1 data. AQuA's cache is rebuilt from the pinned `deepmind/aqua_rat` parquet files and checked by sha.
- It **refuses** any difference from the tracked `data/mcq_rsi/*_splits.json`, so success means the frozen splits were reproduced.
- Expect `[MCQ_RSI/SPLITS] medqa: {...counts...}` for all four benchmarks and no `git diff` under `data/mcq_rsi`.

## 4. tmux pipeline (recommended)

```bash
bash scripts/runpod_mcq_rsi.sh start     # tmux session mcq_rsi: import -> splits -> advisors -> preflight (BENCH) -> smoke -> smoke-check, then STOPS
bash scripts/runpod_mcq_rsi.sh attach    # watch; detach with Ctrl-b d
bash scripts/runpod_mcq_rsi.sh bg pilot  # after reading the smoke results (§7): the pilot in its own tmux session
```

- The pipeline stops after the smoke check. The pilot is started explicitly and refuses to start without a passing `<smoke run>/smoke_check.json` (`SKIP_SMOKE_CHECK=1` overrides; do not).
- Each step can also be run alone, in this shell (`import | splits | advisors | preflight | smoke | smoke-check | pilot | main | final-test`) or in its own tmux session (`bg <step>`; use this for `pilot`, `main` and `final-test`).
- Environment: `BENCH` (default `medqa`), `PREFLIGHT_BENCHES` (default `$BENCH`; set `"medqa mmlu_pro gpqa aqua"` before Phase B), `ARMS`, `ROUNDS`, `HOURS`, `SMOKE_RUN` (default `/workspace/mcq_rsi/runs/smoke`), `RUN_DIR` (pilot / main / final-test directory), `REUSE_TEST` (final-test), `MCQ_ADVISOR_PORT` (18002).
- Before smoke, pilot and main, the wrapper checks that GPU 1 has no compute process and at least 60 GB free, and that GPU 0 has at least 20 GB free. The controller itself refuses to start while a stage process of an earlier controller is still alive (§12).
- Logs go to `/workspace/mcq_rsi/logs` (`pipeline.log` plus one log per step). Stage logs go to `<run>/logs/`.

## 5. Advisor server

```bash
bash scripts/start_mcq_advisors.sh start      # stop | status | evidence
```

It serves all 12 LoRAs, named `<bench>_<kind>`, on `Qwen/Qwen3.5-9B@c202236…` with these flags:
- `--enable-lora --max-loras 12 --max-cpu-loras 12 --max-lora-rank 16`
- `--gpu-memory-utilization 0.45 --max-model-len 16384 --dtype bfloat16`
- `--served-model-name Qwen/Qwen3.5-9B`
- port 18002 on GPU 0

Thinking is disabled per request (`chat_template_kwargs.enable_thinking=false`), as in the paper-era client.

The script also writes `served_loras/server_flags_<port>.json` (vLLM version, LoRA mode, GPU
name, the full argument list). Preflight records it, and the controller refuses a passing
preflight once the server runs with other flags or another vLLM version. The pid file is
checked against the process's command line (a stale pid after a pod restart is removed,
never killed), and every start writes a fresh `vllm_advisors_<port>.log` (older logs are
renamed with a timestamp), so `evidence` shows only the current server.

**The LoRA naming trap and the fix.**
- The paper logs show vLLM 0.26.0 serving `Resolved architecture: Qwen3_5ForConditionalGeneration` with no override.
- The advisors were trained on `Qwen3_5ForCausalLM`, so their PEFT keys are `base_model.model.model.layers.N.*`. On the multimodal model, vLLM maps LoRA names with the Qwen3-VL mapper (`model.language_model.` → `language_model.model.`). `model.layers.N.*` therefore matches no module: the adapter "loads", but requests are served by the **plain base**, and vLLM says so only at debug level.
- The script keeps the paper's architecture. It serves **renamed copies** (`LORA_MODE=multimodal`, made by `python -m src.manager.mcq_rsi serve-loras`) whose keys read `base_model.model.model.language_model.layers.N.*`.
- The tensor bytes are identical. `advisors.check_server` accepts a served copy only if its data section hashes equal to the pinned adapter's.
- `LORA_MODE=as_is` serves the original files, which reproduces the paper-era behaviour. Use it for diagnosis only.

**Expected evidence printed by the script:**
```
version 0.26.0
Resolved architecture: Qwen3_5ForConditionalGeneration
non-default args: {... 'enable_lora': True, 'max_loras': 12, 'max_lora_rank': 16, 'gpu_memory_utilization': 0.45, ...}
Loaded new LoRA adapter: name 'medqa_extractor', path '/workspace/mcq_rsi/served_loras/medqa_extractor'   (x12)
/v1/models: medqa_extractor parent=Qwen/Qwen3.5-9B root=/workspace/mcq_rsi/served_loras/medqa_extractor ...
[MCQ_RSI/SERVE] medqa_extractor mode=multimodal ... keys={'base_model.model.model.language_model.layers.': N}
```

## 6. Preflight: identity and the LoRA-applied replay gate (required)

```bash
bash scripts/runpod_mcq_rsi.sh preflight                                   # BENCH only (Phase A: medqa)
PREFLIGHT_BENCHES="medqa mmlu_pro gpqa aqua" bash scripts/runpod_mcq_rsi.sh preflight   # before Phase B
# = $PY -m src.manager.mcq_rsi preflight --config configs/mcq_rsi_$b.json --advisor-url http://127.0.0.1:18002 --skip-if-passed
```

`--skip-if-passed` keeps a report that already passed for this server, its served adapters,
vLLM version, server flags and preflight settings, so restarts do not replay again. Each
benchmark takes up to about 30 min (20 items × (LoRA + base) per kind, plus the runtime-prompt
reasoner variant; up to 1,024 tokens each at batch 1).

Preflight does three things for each benchmark:
- Records `/version`, the GPU and the server flags, and checks `/v1/models`: every LoRA must be the pinned adapter or a verified renamed copy, with parent `Qwen/Qwen3.5-9B`.
- Regenerates 20 recorded paper advisor outputs per kind from the round-1 tree, greedily and sequentially, with exactly the request the cache sends. The Verifier gets its recorded candidate.
- Sends the same requests to the base model.

**This is not the paper's hardware or server.** The paper's servers ran on a ~96 GB
Blackwell GPU (`Free memory on device (94.43/94.97 GiB)`, `SM 12.x` in the logs), one server
and three LoRAs per benchmark, with vLLM defaults (`max_loras` 1, rank 16; GPQA rank 64 with
`trust_remote_code`), `gpu_memory_utilization` 0.85 (GPQA 0.92), an unpinned base, and one
request at a time. Here: A100/H100-80GB (SM80/SM90), all 12 LoRAs in one server
(`max_loras` 12), utilisation 0.45, the pinned revision, and up to 32 concurrent cache
fills. Greedy bf16 decoding on other kernels usually diverges somewhere in a long output, so
exact match is not required.

**Pass**, for every kind:
- LoRA ≠ base on ≥ 50% of the requests (`min_lora_effect`; the adapters are really applied), **and**
- either exact match ≥ 0.9 (`min_match`), or the LoRA output is closer to the recorded output than the base output (word-level similarity) on ≥ 75% of the items (`min_closer`) with median LoRA similarity ≥ 0.5 (`min_similarity`).

Exact-match rates, similarities and first-divergence (common-prefix) ratios are all printed
and stored. The report goes to `<advisor_cache>/preflight/<bench>.json`. The controller
refuses to fetch anything until a passing report exists for this server URL, these served
adapters, this vLLM version and these server flags.

Expected output:
```
[MCQ_RSI/PREFLIGHT] medqa/extractor: n=20 match=0.85 base_match=0.00 lora_closer=1.00 sim(lora)=0.97 sim(base)=0.31 prefix(lora)=0.80 lora_effect=1.00 -> PASS (closer_than_base)
...
[MCQ_RSI/PREFLIGHT] vLLM {'version': '0.26.0'} gpu [...] flags recorded -> PASS (/workspace/mcq_rsi/advisor_cache/preflight/medqa.json)
```

When preflight fails, act on the diagnosis it prints:

| Diagnosis | Meaning | Action |
|---|---|---|
| `LoRA not applied` (lora_effect ≈ 0) | the server ignores the adapters | restart with `LORA_MODE=multimodal`; check the evidence lines |
| `recorded paper outputs are closer to the plain base` (base exact ≥ 0.9, or base closer on ≥ 75%) | the paper's server hit the trap, so the paper's advisors were effectively the base model plus the trained prompt | **Decision D14. Do not run.** Choose between (a) keeping real LoRAs, which means round-1 parity no longer holds and S_1 must be re-evaluated as the new baseline, and (b) reproducing the paper's plain-base serving, which needs a code change to the advisor identity. Record the choice. |
| `LoRA applied but … not reproduced` | neither exact nor closer than the base: prompt, version or decoding differ | compare `runtime_prompt_match` / `median_variant_similarity` for the reasoner (the open question of which Reasoner prompt was live for GPQA, MMLU-Pro and AQuA); check vLLM == 0.26.0, bf16 and thinking off; look at the first-divergence ratios and `examples` in the report; the hardware and flag differences above cannot be removed on this pod |

## 7. GPU smoke (design §7.2, about 1.5 h; must pass before any pilot)

```bash
bash scripts/runpod_mcq_rsi.sh smoke
# = $PY -m src.manager.mcq_rsi run --config configs/mcq_rsi_smoke.json --run-dir /workspace/mcq_rsi/runs/smoke --hours 3 --advisor-url http://127.0.0.1:18002
bash scripts/runpod_mcq_rsi.sh smoke-check     # automatic pass checks -> <smoke run>/smoke_check.json (exit 1 on failure)
```

The smoke budget (3 h) is persisted at the first start. If a smoke run fails and debugging
takes longer than that, it can no longer be resumed: move it aside and start again,
`mv /workspace/mcq_rsi/runs/smoke /workspace/mcq_rsi/runs/smoke.$(date +%Y%m%d_%H%M)`
(or set `SMOKE_RUN=<new dir>`).

The smoke config runs MedQA with dynamic R=2: 50 dev questions, 8 collection roots, SFT for 10 steps and FA-GRPO for 4 steps with B=2. It never touches the test set. Expected results by stage:

| Step | Stage / check | Expected |
|---|---|---|
| 1 vLLM | §5 evidence, preflight | 12 LoRAs, PASS |
| 2 replay | preflight report | PASS per kind: exact match ≥ 0.9, or the LoRA closer to the recorded outputs than the base (§6) |
| 3 round-0 parity | `r1/S1_dev/parity.json` reports `subset` (not gated) and the batched-vs-sequential advisor recheck. `smoke-check` extracts the recorded paper eval (`outputs/eval/medqa_9b_d2400_ev_r3/manager_tool_eval.jsonl` from `assets_0814.tgz`, into `/workspace/mcq_rsi/recorded`) and runs the per-example agreement | `agree ≥ 48` of `n_common = 50` (automatic) |
| 4 collector | `r2/collect/marginal_value_report.json` | 8 roots; `unconstrained_argmax.mismatch` 0; shards resume on rerun |
| 5 SFT | `r2/dynamic/sft/sft_report.json` | `tokenization_parity.input_mismatch` 0; adapter reloads in `sft_dev` |
| 6 FA-GRPO | `r1/grpo/metrics.jsonl`, `summary.json`, `controller_stage.json` | step 1 `kl_*` = 0; finite loss; `peak_memory_gb` < 60 (automatic); seconds per step recorded |
| 7 eval and gates | `*/decision.json` | `malformed_tool_calls` 0; an `accept` block on each `grpo_dev` (automatic) |

`smoke-check` also prints the remaining **manual** items: the §5 evidence and preflight
PASS; the forced-rollback test (rerun one GRPO stage by hand with `--config` setting
`"guard_max_kl_dec": -1`; it must end at step `guard_every` with `selected_step` 0); and
copying the stage wall times (`status`, `wall_seconds`) into §16.

## 8. MedQA pilot (Phase A, about 6–7 h)

```bash
bash scripts/runpod_mcq_rsi.sh bg pilot    # tmux session mcq_pilot_medqa; refuses without a passing smoke_check.json
# = $PY -m src.manager.mcq_rsi run --config configs/mcq_rsi_medqa.json --phase pilot --run-dir /workspace/mcq_rsi/runs/medqa_pilot --advisor-url http://127.0.0.1:18002
```

- Round 1 runs FA-GRPO at lr 2e-6, 5e-6 and 1e-5, each followed by a dev eval and the accept rule.
- A candidate whose dev eval fails its eval gate (invalid answers, malformed tool calls; typical of a too-large lr) is rejected like any other, not a run failure.
- `r1/grpo_select` picks the accepted candidate with the best dev accuracy (ties go to fewer calls, then the smaller lr). If none is accepted, `G_1 := S_1` (shown as "GRPO rejected" in the arm timeline) and later rounds use the config lr.
- A pilot never runs the locked test (`final-test` refuses pilot runs).
- Then dynamic runs for round 2 with that lr.

Read `report.md`, which contains:
- per-round dev metrics, drift against `S_1` on the same ids, and the arm timeline;
- Tables 1–2 analogues of the recollection, the label mix, and the gates.

Before Phase B, set `grpo.learning_rate` in all four configs to the lr the pilot selected. **Freeze the code at this point.** The run signature includes git HEAD and the hashes of every MCQ RSI file, so later edits make resumes refuse.

## 9. Main runs (Phase B)

```bash
PREFLIGHT_BENCHES="medqa mmlu_pro gpqa aqua" bash scripts/runpod_mcq_rsi.sh preflight   # once, before Phase B
BENCH=medqa bash scripts/runpod_mcq_rsi.sh bg main   # tmux session mcq_main_medqa; then mmlu_pro, aqua, gpqa
# = $PY -m src.manager.mcq_rsi run --config configs/mcq_rsi_$BENCH.json --run-dir /workspace/mcq_rsi/runs/${BENCH}_main --hours 72 --advisor-url http://127.0.0.1:18002
```

- `--arms dynamic,static` (or `ARMS=dynamic,static`) is the cheaper variant. `dynamic_sft` adds the no-GRPO ablation.
- `--hours` is persisted at the first start (`budget.json`) with a 72 h cap. A restart never buys more time. When the deadline hits, the running stage is killed and the completed stages are kept.

## 10. Monitoring

```bash
$PY -m src.manager.mcq_rsi status --run-dir /workspace/mcq_rsi/runs/medqa_main    # controller state, heartbeat age, remaining h, per-stage state and minutes
bash scripts/runpod_mcq_rsi.sh status                                              # GPUs, advisor health, every run
tail -f /workspace/mcq_rsi/runs/medqa_main/logs/r2__dynamic__grpo.log
```

- `status.json` is refreshed every 15 s while a stage runs (`heartbeat_unix`). `status` prints `stage_pid` and `(ALIVE)` while that stage's process group exists.
- **Orphan check.** A heartbeat older than a few minutes while `controller=running` means the controller died. Before resuming, check `status`: if `stage_pid` is `(ALIVE)`, the stage of the dead controller is still running. Wait for it to finish, or stop it with `kill -TERM -- -<stage_pid>`; the controller refuses to resume while it lives, and a stage directory is locked (`.<stage>.stage.lock` next to it) by the process executing it. Then resume as in §12.
- SIGTERM or SIGHUP (closing the terminal or the tmux session) stops the controller cleanly: the running stage's process group is killed and `status` shows `interrupted`.
- FA-GRPO prints one line per step: `loss`, `kl_dec`, `J` and informative states. Every 8 steps it adds `guard=pass|FAIL`.

## 11. Gates and what to do when they fail

| Gate | Where | Rule | On failure |
|---|---|---|---|
| eval gate | every eval | `malformed_tool_calls == 0`, `valid_answer_rate == 1.0`, no advisor failure (fail-stop) | S_1, SFT and final evals: fix the cause (server down? `status` shows the error), then `retry-stage --name <stage>` (moves the directory aside, never deletes) and resume. A **GRPO candidate's** own gate failure is not an error: the candidate is rejected (`G_k := S_k`) and the gate is recorded in its `accept` block. Advisor failures always stop the run. |
| parity | `r1/S1_dev` | dev accuracy within ±2 pt and calls within ±0.05 of the registry targets. AQuA accepts call rate **or** calls/example; GPQA has no dev target. | Investigate (advisor cache identity, prompts, base revision). If the shift is understood and accepted: `ack-gate --run-dir R --stage r1/S1_dev --reason "..."`, then resume. |
| FA-GRPO guard | inside each GRPO stage | \|Δ call rate\| ≤ 0.10 and `KL_dec` ≤ 0.05 every 8 steps | Automatic rollback to the last passing step (or `S_k`). The stage ends normally. |
| informativeness | GRPO summary | ≥ 15% of revision states with J in (0.02, 0.98) | The GRPO is rejected by the accept gate (`G_k := S_k`) |
| GRPO accept | `*/grpo_dev` | acc ≥ S_k − 1 pt, calls ≤ S_k + 0.15, gap ≥ 0.5·S_k gap, informative | Automatic `G_k := S_k`, logged as `grpo_rejected`. No action needed. |
| SFT flags | `*/sft_dev` | calls > 1.5 or accuracy down more than 3 pt against `G_{k-1}` | Flag only (SFT is never selected on dev). Inspect the label mix. |
| signature | every start and stage | config, plan, code, manifests, git HEAD and packages unchanged | Restore the code or config, or start a new run directory |

## 12. Resume

Run the same `run` command again (in tmux: `bash scripts/runpod_mcq_rsi.sh bg main`). The rules:
- Completed stages (those with `.mcq_rsi_complete.json`) are validated and skipped.
- Collection resumes per question shard. FA-GRPO resumes from its last committed step. SFT and eval resume from their own signatures.
- A second controller on the same directory is refused (`controller.lock`), and so is a restart while a stage process of an earlier controller is still alive (§10 orphan check).
- Operational settings (advisor URL, GPUs, workers, preflight, W&B) may change between restarts; nothing else may. Re-running `import` (as the pipeline does) does not change a run's identity.

## 13. Final test (locked test, once per pre-registered final)

```bash
BENCH=medqa bash scripts/runpod_mcq_rsi.sh bg final-test
# = $PY -m src.manager.mcq_rsi final-test --run-dir /workspace/mcq_rsi/runs/medqa_main
```

- The finals are `S_1` (re-evaluated) and each arm's last-round `G_R`, or `S_R` if its GRPO was rejected. They are registered in `final/finals.json` on the first call and can never be retargeted.
- The command is refused until every planned stage is complete. If a run must be cut short, `--accept-incomplete "<reason>"` records the reason and uses each arm's last resolved round, flagged `truncated_at_round`.
- Each test eval runs once: a completed one is never rerun, and `retry-stage` refuses locked-test stages.
- **Once per benchmark, not per directory.** Every registration is recorded in `/workspace/mcq_rsi/advisor_cache/locked_test/<bench>.json`. A second run directory of the same benchmark (for example after starting a new directory) and any pilot run are refused unless `--reuse-test "<reason>"` (`REUSE_TEST=...` in the wrapper) is given; the reason is recorded and must be reported.
- The test pools are `test`, plus `test_paper` for MMLU-Pro. Forced Extractor, Reasoner and Verifier (one role each, for the matched-budget replay) and forced-all run on dev for every final.

## 14. Analysis

```bash
$PY scripts/mcq_rsi_analysis.py tables --run-dir /workspace/mcq_rsi/runs/medqa_main    # -> <run>/analysis/{tables.md,*.csv,analysis.json}
$PY scripts/mcq_rsi_analysis.py compare --a <evalA> --b <evalB>                        # paired bootstrap (10k) + McNemar
$PY scripts/mcq_rsi_analysis.py replay --policy <dev eval> --forced verifier=<forced eval>   # matched-budget replay (2000)
```

- `tables` produces per-round versions of paper Tables 1, 2, 3, 7 and 8.
- It also runs paired tests of each final against `S_1`, and of dynamic against each control, on identical test ids.
- It runs the matched-budget replay of every final on dev, using its single-role forced evals (Extractor, Reasoner, Verifier), so every first call is matched by role. Roles without a forced eval would be held fixed; the `fixed` column reports how many examples were (0 with the default configs).
- One seed per cell. The comparisons are pre-registered, not adjusted for multiplicity.

## 15. Backup

On a pod **without** a persistent volume (`/workspace` on the container disk), stopping the pod
deletes everything, so the Hugging Face backup is the only copy. Even with a volume, the backup
is the copy that survives losing the volume. Log in once (the token stays under `HF_HOME`; never
paste it into scripts), then keep the hourly loop running in its own tmux session for the whole experiment:

```bash
HF_HOME=/workspace/hf-cache /workspace/mcq-venv/bin/hf auth login
BACKUP_EVERY_MIN=60 bash scripts/runpod_mcq_rsi.sh bg backup     # tmux session mcq_backup_<BENCH>; log: logs/backup.log
bash scripts/runpod_mcq_rsi.sh backup                            # one extra pass, e.g. right after a phase ends
```

`scripts/backup_mcq_rsi_hf.py` mirrors the following into `/workspace/mcq_rsi/hf_backup_stage`, then uploads that copy with `upload_large_folder` (resumable; unchanged files are skipped):
- `runs/`;
- `logs/`;
- `advisor_cache/{preflight,locked_test}`;
- the import manifests;
- the advisor output cache, packed into `advisor_cache.tar.gz`.

It leaves out:
- per-step FA-GRPO weights and optimizer states (`step-*/`, about 370 MB per step). Every `final/` adapter, `step.json` and `metrics.jsonl` is kept;
- trainer `checkpoint-*` directories;
- temporaries and lock files.

If `hf auth login` ran without `HF_HOME` set, the token is stored elsewhere, for example `/workspace/.cache/huggingface/token`. Point the backup at that file with `HF_TOKEN_PATH=<file>` (passed on by `bg`) instead of moving it. The repo is `MaliDDD/margent-mcq-rsi` (`HF_BACKUP_REPO` overrides). It is created **private**; pass `--public` or flip it on the Hub. A file copied while a stage was writing it is copied again on the next pass, so backups taken after a stage completes are consistent. Files deleted locally stay in the repo.

A restore needs:
- the **same paths** (`/workspace/mcq_rsi/...`): stage signatures record absolute checkpoint paths;
- the **same git commit** of this repository (the run signature includes git HEAD and source hashes);
- the same `HF_HOME=/workspace/hf-cache` (FA-GRPO records the resolved base snapshot path);
- a re-import to the same `import` directory (`runpod_mcq_rsi.sh import`; digests are verified).

```bash
hf download MaliDDD/margent-mcq-rsi --local-dir /workspace/mcq_rsi_restore
rsync -a /workspace/mcq_rsi_restore/{runs,logs,advisor_cache} /workspace/mcq_rsi/
tar -xzf /workspace/mcq_rsi_restore/advisor_cache.tar.gz -C /workspace/mcq_rsi
```

Completed stages come back whole. A stage that was in progress has no step weights in the backup,
so move it aside before resuming:
`$PY -m src.manager.mcq_rsi retry-stage --run-dir <run> --name <stage>`. The next `run` redoes it.

## 16. Cost and time (design §6 estimates; replace with smoke timings)

| Stage | Manager GPU | Notes |
|---|---|---|
| advisor preflight | – | up to about 30 min per benchmark (batch 1); kept on restarts (`--skip-if-passed`) |
| prefetch E/R for all pools (about 2,700 q × 2) | – | about 1–1.5 h per benchmark, once; 32 concurrent advisor requests, then a 20-per-kind sequential recheck |
| collect 400 roots, D=2 | about 50 min | overlapped with advisors, about 55–60 min wall |
| SFT 3 epochs, 300–450 rows | 28–42 min | GPU 1 |
| FA-GRPO 64 steps × 4 questions | about 35 min | GPU 1 |
| dev eval 200 q | 10–12 min | two-pass Verifier prefetch |
| test eval 500 q | 25–30 min | once per final (MMLU-Pro: twice, `test` and `test_paper`) |
| forced dev evals 200 q × 4 (E, R, V, all) | about 40 min | per final, at final-test |
| **final test per benchmark** | **about 4–5 h** | 4 finals (S_1 + 3 arms) × (test + 4 forced dev evals); MMLU-Pro about 6 h |
| **MedQA pilot** | **about 6–7 h** | 3 GRPO + 4 evals in round 1, then dynamic round 2 |
| per benchmark, R=3, 3 arms | about 16 h | MMLU-Pro similar; AQuA about 15 h; GPQA about 7 h |
| full grid | about 55 manager-GPU h, plus about 18 h of final tests and about 2 h of preflight | stages run sequentially, so plan for about 75 h wall time across the four runs (run benchmarks in parallel only on separate pods) |
| dynamic + static only | about 2/3 of the above | |

At about US$2–3 per GPU-hour for A100/H100-80GB, 2 GPUs cost about $4–6 per hour of wall time. The pilot costs about $30–40; the full grid, with final tests and preflight, about $300–450.
