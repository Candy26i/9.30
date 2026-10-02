#!/usr/bin/env bash
# MCQ RSI on RunPod (docs/MCQ_RSI_RUNBOOK.md). Everything logs under /workspace/mcq_rsi.
#
#   bash scripts/runpod_mcq_rsi.sh start            # tmux session: import -> splits check -> advisors -> preflight -> smoke -> smoke-check (stops)
#   bash scripts/runpod_mcq_rsi.sh attach|status|report|stop-advisors
#   bash scripts/runpod_mcq_rsi.sh import|splits|advisors|preflight|smoke|smoke-check|pilot|main|final-test   (single steps, this shell)
#   bash scripts/runpod_mcq_rsi.sh bg <step>        # a single step in its own tmux session (survives a dropped SSH/web terminal)
#   BACKUP_EVERY_MIN=60 bash scripts/runpod_mcq_rsi.sh bg backup   # hourly HF backup (scripts/backup_mcq_rsi_hf.py)
#
# The pipeline stops after the smoke check; the pilot needs `pilot` (or `bg pilot`) and refuses without a passing
# <smoke run>/smoke_check.json (SKIP_SMOKE_CHECK=1 overrides).
#
# Environment: BENCH (preflight/pilot/main/final-test, default medqa), ARMS, ROUNDS, HOURS, MCQ_ADVISOR_PORT (18002),
#   PREFLIGHT_BENCHES (default: $BENCH; "medqa mmlu_pro gpqa aqua" before Phase B), SMOKE_RUN (default $WORK/runs/smoke),
#   RUN_DIR (pilot/main/final-test run directory), REUSE_TEST (final-test: reason for a second locked-test use),
#   MARGENT_WANDB_MODE (unset: W&B follows the config's "wandb.enabled"; "disabled" turns it off).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

WORK="${MCQ_WORK:-/workspace/mcq_rsi}"
PY="${MCQ_PYTHON:-/workspace/mcq-venv/bin/python}"
SESSION="${MCQ_TMUX_SESSION:-mcq_rsi}"
BENCH="${BENCH:-medqa}"
HOURS="${HOURS:-72}"
PORT="${MCQ_ADVISOR_PORT:-18002}"
ADVISOR_GPU="${MCQ_ADVISOR_GPU:-0}"
TRAIN_GPU="${MCQ_TRAIN_GPU:-1}"
export MCQ_ADVISOR_PORT="$PORT"
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
SMOKE_RUN="${SMOKE_RUN:-$WORK/runs/smoke}"
LOGS="$WORK/logs"
mkdir -p "$LOGS" "$WORK/runs" "$TMPDIR"

log() { printf '[mcq-rsi %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOGS/pipeline.log"; }
die() { log "ERROR: $*"; exit 1; }
cli() { "$PY" -m src.manager.mcq_rsi "$@"; }
url() { printf 'http://127.0.0.1:%s' "$PORT"; }

gpu_check() {
  # The training GPU must be idle (no duplicate experiment); the advisor GPU only holds our vLLM server.
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
  local uuid busy
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$TRAIN_GPU" | tr -d ' ')"
  busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid" || true)"
  (( busy == 0 )) || die "GPU ${TRAIN_GPU} already runs ${busy} compute process(es); refusing to start a duplicate experiment"
  local free
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$TRAIN_GPU" | tr -d ' ')"
  (( free >= 60000 )) || die "GPU ${TRAIN_GPU} has ${free} MiB free; need >= 60000"
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$ADVISOR_GPU" | tr -d ' ')"
  (( free >= 20000 )) || die "GPU ${ADVISOR_GPU} has ${free} MiB free; the eval/collection manager needs >= 20000 next to vLLM"
  log "GPU check ok"
}

advisor_up() { curl -fsS "$(url)/health" >/dev/null 2>&1; }

step_backup() {
  # One pass, or a loop with BACKUP_EVERY_MIN (use: BACKUP_EVERY_MIN=60 bash scripts/runpod_mcq_rsi.sh bg backup).
  log "HF backup of ${WORK} -> ${HF_BACKUP_REPO:-MaliDDD/margent-mcq-rsi}"
  "$PY" scripts/backup_mcq_rsi_hf.py --work "$WORK" --repo "${HF_BACKUP_REPO:-MaliDDD/margent-mcq-rsi}" \
    ${BACKUP_EVERY_MIN:+--every-minutes "$BACKUP_EVERY_MIN"} 2>&1 | tee -a "$LOGS/backup.log"
}

step_import() { log "import (all benchmarks)"; cli import --bench all --out "$WORK/import" --cache-dir "$HF_HOME/hub" 2>&1 | tee -a "$LOGS/import.log"; }
step_splits() {
  # Rebuilds every manifest from the caches and the imported round-1 data; write_frozen refuses any
  # difference from the tracked manifests, so a pass means the frozen splits are reproduced exactly.
  log "prepare-splits check"
  cli prepare-splits --bench all --import-dir "$WORK/import" --hf-cache-dir "$HF_HOME/hub" 2>&1 | tee -a "$LOGS/splits.log"
  git -C "$REPO" diff --quiet -- data/mcq_rsi || die "split manifests changed on disk; investigate before running"
}
step_advisors() {
  if advisor_up; then log "advisor server already healthy on port ${PORT}"; return 0; fi
  log "starting advisor server on port ${PORT}"
  MCQ_IMPORT_DIR="$WORK/import" MCQ_LOG_DIR="$LOGS" bash scripts/start_mcq_advisors.sh start 2>&1 | tee -a "$LOGS/advisors.log"
}
step_preflight() {
  advisor_up || die "advisor server is not running (step advisors)"
  # Phase A needs only the benchmark being run; a report that already passes for this server, its adapters,
  # vLLM version, flags and settings is kept (--skip-if-passed), so restarts do not replay it again.
  for b in ${PREFLIGHT_BENCHES:-$BENCH}; do
    log "preflight ${b}"
    cli preflight --config "configs/mcq_rsi_${b}.json" --advisor-url "$(url)" --skip-if-passed 2>&1 | tee -a "$LOGS/preflight_${b}.log"
  done
}
step_smoke() {
  gpu_check
  log "GPU smoke (configs/mcq_rsi_smoke.json) -> ${SMOKE_RUN}"
  cli run --config configs/mcq_rsi_smoke.json --run-dir "$SMOKE_RUN" --hours "${SMOKE_HOURS:-3}" \
    --advisor-url "$(url)" 2>&1 | tee -a "$LOGS/smoke.log"
  cli status --run-dir "$SMOKE_RUN" | tee -a "$LOGS/smoke.log"
}
step_smoke_check() {
  # Design §7.2 step 3 (round-0 agreement >= 48/50 against the recorded paper eval), peak memory, accept blocks.
  local tgz member="outputs/eval/medqa_9b_d2400_ev_r3/manager_tool_eval.jsonl"
  tgz="$(find "$HF_HOME/hub" -path '*snapshots*' -name assets_0814.tgz 2>/dev/null | head -n 1)"
  [[ -n "$tgz" ]] || die "assets_0814.tgz not found under $HF_HOME/hub (run step import)"
  mkdir -p "$WORK/recorded"
  "$PY" scripts/mcq_rsi_analysis.py smoke-check --run-dir "$SMOKE_RUN" --recorded "$tgz" --member "$member" \
    --extract-to "$WORK/recorded" 2>&1 | tee -a "$LOGS/smoke.log"
}
smoke_passed() {
  [[ "${SKIP_SMOKE_CHECK:-0}" == 1 ]] && { log "WARNING: SKIP_SMOKE_CHECK=1"; return 0; }
  [[ -f "$SMOKE_RUN/smoke_check.json" ]] || die "no $SMOKE_RUN/smoke_check.json: run smoke and smoke-check first (runbook §7)"
  "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$SMOKE_RUN/smoke_check.json" \
    || die "the smoke check failed ($SMOKE_RUN/smoke_check.json); fix it before the pilot"
}
step_pilot() {
  smoke_passed
  gpu_check
  local run="${RUN_DIR:-$WORK/runs/${BENCH}_pilot}"
  log "pilot ${BENCH} (lr sweep at round 1, dynamic R=2) -> ${run}"
  cli run --config "configs/mcq_rsi_${BENCH}.json" --phase pilot --run-dir "$run" \
    --hours "${HOURS}" --advisor-url "$(url)" 2>&1 | tee -a "$LOGS/${BENCH}_pilot.log"
}
step_main() {
  gpu_check
  local run="${RUN_DIR:-$WORK/runs/${BENCH}_main}"
  extra=()
  [[ -z "${ARMS:-}" ]] || extra+=(--arms "$ARMS")
  [[ -z "${ROUNDS:-}" ]] || extra+=(--rounds "$ROUNDS")
  log "main ${BENCH} ${extra[*]:-} -> ${run}"
  cli run --config "configs/mcq_rsi_${BENCH}.json" --run-dir "$run" --hours "$HOURS" \
    --advisor-url "$(url)" "${extra[@]}" 2>&1 | tee -a "$LOGS/${BENCH}_main.log"
}
step_final_test() {
  local run="${RUN_DIR:-$WORK/runs/${BENCH}_main}"
  extra=()
  [[ -z "${REUSE_TEST:-}" ]] || extra+=(--reuse-test "$REUSE_TEST")
  log "final-test ${BENCH} (locked test, once) -> ${run}"
  cli final-test --run-dir "$run" "${extra[@]}" 2>&1 | tee -a "$LOGS/${BENCH}_final_test.log"
}

pipeline() {
  log "pipeline start (repo $(git -C "$REPO" rev-parse --short HEAD))"
  step_import
  step_splits
  step_advisors
  step_preflight
  step_smoke
  step_smoke_check
  log "pipeline done. Read ${SMOKE_RUN}/report.md and smoke_check.json, do the MANUAL items above (runbook §7),"
  log "then start the pilot: bash scripts/runpod_mcq_rsi.sh bg pilot"
}

case "${1:-}" in
  start)
    command -v tmux >/dev/null || die "tmux not installed (apt-get install -y tmux)"
    tmux has-session -t "$SESSION" 2>/dev/null && die "tmux session ${SESSION} exists (attach: $0 attach)"
    tmux new-session -d -s "$SESSION" -n pipeline "bash '$REPO/scripts/runpod_mcq_rsi.sh' pipeline; exec bash"
    tmux new-window -t "$SESSION" -n monitor "watch -n 60 'bash $REPO/scripts/runpod_mcq_rsi.sh status 2>&1 | tail -n 60'"
    log "started tmux session ${SESSION} (attach: tmux attach -t ${SESSION}; logs: ${LOGS})" ;;
  attach) exec tmux attach -t "$SESSION" ;;
  pipeline) pipeline ;;
  bg)
    step="${2:-}"
    [[ "$step" =~ ^(import|splits|advisors|preflight|smoke|smoke-check|pilot|main|final-test|backup)$ ]] || die "usage: $0 bg <step>"
    command -v tmux >/dev/null || die "tmux not installed (apt-get install -y tmux)"
    name="mcq_${step}_${BENCH}"
    tmux has-session -t "$name" 2>/dev/null && die "tmux session ${name} exists (tmux attach -t ${name})"
    # The environment of this shell (BENCH, ARMS, RUN_DIR, ...) is passed on explicitly.
    tmux new-session -d -s "$name" -n "$step" \
      "$(printf '%q ' env BENCH="$BENCH" ARMS="${ARMS:-}" ROUNDS="${ROUNDS:-}" HOURS="$HOURS" RUN_DIR="${RUN_DIR:-}" \
         SMOKE_RUN="$SMOKE_RUN" REUSE_TEST="${REUSE_TEST:-}" PREFLIGHT_BENCHES="${PREFLIGHT_BENCHES:-}" \
         SKIP_SMOKE_CHECK="${SKIP_SMOKE_CHECK:-0}" MCQ_ADVISOR_PORT="$PORT" \
         MCQ_WORK="$WORK" MCQ_PYTHON="$PY" MCQ_TMUX_SESSION="$SESSION" \
         MCQ_ADVISOR_GPU="$ADVISOR_GPU" MCQ_TRAIN_GPU="$TRAIN_GPU" SMOKE_HOURS="${SMOKE_HOURS:-}" \
         ${MARGENT_WANDB_MODE+"MARGENT_WANDB_MODE=$MARGENT_WANDB_MODE"} HF_HOME="$HF_HOME" HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
         TMPDIR="$TMPDIR" HF_BACKUP_REPO="${HF_BACKUP_REPO:-}" BACKUP_EVERY_MIN="${BACKUP_EVERY_MIN:-}" \
         ${HF_TOKEN_PATH+"HF_TOKEN_PATH=$HF_TOKEN_PATH"} \
         bash "$REPO/scripts/runpod_mcq_rsi.sh" "$step"); exec bash"
    log "started ${step} in tmux session ${name} (tmux attach -t ${name})" ;;
  import) step_import ;;
  splits) step_splits ;;
  advisors) step_advisors ;;
  preflight) step_preflight ;;
  smoke) step_smoke ;;
  smoke-check) step_smoke_check ;;
  pilot) step_pilot ;;
  main) step_main ;;
  final-test) step_final_test ;;
  backup) step_backup ;;
  status)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    if advisor_up; then echo "advisors: healthy on ${PORT}"; else echo "advisors: DOWN on ${PORT}"; fi
    for d in "$WORK"/runs/*/; do [[ -f "$d/rsi_run.json" ]] && { echo "== ${d}"; cli status --run-dir "$d"; }; done ;;
  report) cli report --run-dir "${RUN_DIR:-$WORK/runs/${BENCH}_pilot}" ;;
  stop-advisors) bash scripts/start_mcq_advisors.sh stop ;;
  *) echo "usage: $0 start|attach|status|report|stop-advisors|bg <step>|import|splits|advisors|preflight|smoke|smoke-check|pilot|main|final-test|backup" >&2; exit 2 ;;
esac
