#!/usr/bin/env bash
# AIME2026-only held-out matrix: M0 plus each predeclared arm's round_2/grpo Manager.
# No `set -e`: a failed, timed-out or budget-exhausted cell is recorded and later cells still run.
# Each cell keeps one absolute deadline from its first launch; rerunning never extends it.
set -uo pipefail
source "${MARGENT_ENV:-/workspace/margent-luna-env.sh}"
cd "$MARGENT_CODE"
BENCH=aime2026; EXPECTED_N=30; CELL_MINUTES="${EVAL_CELL_MINUTES:-120}"; MIN_START_SECONDS=120
DATA="$LUNA_DATA/manager/$BENCH.jsonl"
mkdir -p "$EVAL_ROOT/budgets" "$EVAL_ROOT/logs"
exec 9>"$EVAL_ROOT/.matrix.lock"
flock -n 9 || { echo 'This test matrix already has a running process' >&2; exit 1; }

# Fatal precheck: the RSI controller has exited (or every stage has its marker), the subset in the env file
# is the data that RSI run actually recorded, and every RSI train/dev question is a BeyondAIME question
# disjoint from AIME2026 (normalized identity check, not a filename check). Explicit checks, not asserts.
"$RSI_PYTHON" - <<'PY' || exit 1
import hashlib, json, os
from pathlib import Path
from src.verifiable.data import identity, load_rows
out, sub, pool = Path(os.environ['RSI_OUTPUT']), Path(os.environ['RSI_SUBSET']), Path(os.environ['LUNA_DATA'])/'manager'
def require(ok, message):
    if not ok:
        raise SystemExit(message)
status = json.loads((out/'run_summary.json').read_text()).get('controller_status')
report = out/'pilot_report.json'
all_markers = report.exists() and json.loads(report.read_text()).get('complete') is True
require(status in {'completed', 'failed', 'interrupted'} or all_markers, f'RSI controller not finished: {status}')
recorded = json.loads((out/'rsi_run.json').read_text())['data']['sha256']
for name in ('train.jsonl', 'dev.jsonl'):
    require(hashlib.sha256((sub/name).read_bytes()).hexdigest() == recorded[name],
            f'RSI_SUBSET/{name} is not the data recorded in RSI_OUTPUT/rsi_run.json')
ids = lambda f, s: {identity(r.question) for r in load_rows(f, required_split=s)}
aime, beyond = ids(pool/'aime2026.jsonl', 'test'), ids(pool/'beyondaime.jsonl', 'test')
used = ids(sub/'train.jsonl', 'train') | ids(sub/'dev.jsonl', 'dev')
require(len(aime) == 30, 'AIME2026 must have 30 questions')
require(not used & aime, 'RSI train/dev overlaps AIME2026')
require(used <= beyond, 'RSI train/dev contains non-BeyondAIME questions')
print(json.dumps({'rsi_controller_status': status, 'all_stage_markers': all_markers,
                  'rsi_questions': len(used), 'aime_overlap': 0}))
PY

cell_done() {  # exit 0 only for a completed cell with exactly EXPECTED_N records
  "$RSI_PYTHON" - "$1" "$EXPECTED_N" <<'PY'
import json, sys
from pathlib import Path
p, n = Path(sys.argv[1]), int(sys.argv[2])
try:
    ok = (json.loads((p/'status.json').read_text())['status'] == 'completed'
          and json.loads((p/'summary.json').read_text())['n'] == n
          and sum(1 for x in (p/'records.jsonl').read_text().splitlines() if x.strip()) == n)
except (OSError, ValueError, KeyError):
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

budget_left() {  # prints whole seconds left (0 = exhausted); nonzero exit only if inputs changed
  "$RSI_PYTHON" - "$EVAL_ROOT/budgets/$1-$BENCH.json" "$RSI_CONFIG" "$DATA" "$2" "$CELL_MINUTES" <<'PY'
import hashlib, json, math, sys, time
from pathlib import Path
target, config, data, checkpoint, minutes = sys.argv[1:]
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
signature = {'config_sha256': digest(config), 'data_sha256': digest(data), 'checkpoint': checkpoint, 'minutes': int(minutes)}
p = Path(target)
if p.exists():
    budget = json.loads(p.read_text())
    if budget['signature'] != signature:
        sys.exit('Budget inputs changed; use a new EVAL_ROOT')
else:
    budget = {'signature': signature, 'deadline_unix': time.time() + int(minutes) * 60}
    with p.open('x') as f:
        json.dump(budget, f, indent=2)
print(max(0, math.floor(budget['deadline_unix'] - time.time())))
PY
}

note() { printf '{"cell":"%s","state":"%s","rc":%s,"unix":%s}\n' "$1" "$2" "$3" "$(date +%s)" >> "$EVAL_ROOT/matrix_attempts.jsonl"; echo "$1: $2 (rc=$3)"; }

incomplete=0
for arm in base ${RSI_ARMS:-dynamic static success}; do
  if [[ "$arm" == base ]]; then
    label=base; checkpoint=Qwen/Qwen3.5-9B
  else
    label="${arm}_final"; checkpoint="$RSI_OUTPUT/$arm/round_2/grpo"
  fi
  out="$EVAL_ROOT/$label/$BENCH"
  if cell_done "$out"; then note "$label" already_complete 0; continue; fi
  # An arm is final only when its last dev assessment ran; a round_2/grpo stage that failed the
  # controller's mixed-reward gate still has a marker but no grpo_dev. Never substitute a checkpoint.
  if [[ "$arm" != base && ! ( -f "$checkpoint/.rsi_complete.json" && -f "$checkpoint/adapter_config.json"
        && -f "$RSI_OUTPUT/$arm/round_2/grpo_dev/.rsi_complete.json" ) ]]; then
    note "$label" arm_incomplete_not_evaluated 0; incomplete=1; continue
  fi
  if ! left=$(budget_left "$label" "$checkpoint"); then note "$label" budget_inputs_changed 2; incomplete=1; continue; fi
  if (( left < MIN_START_SECONDS )); then
    # Never run `timeout 0s`: GNU timeout treats a zero duration as "no time limit".
    note "$label" budget_exhausted_not_started 0; incomplete=1; continue
  fi
  rc=0
  timeout --signal=INT --kill-after=60s "${left}s" \
    env CUDA_VISIBLE_DEVICES="$RSI_MANAGER_GPU" "$RSI_PYTHON" -u -m src.verifiable evaluate \
    --config "$RSI_CONFIG" --checkpoint "$checkpoint" --data "$DATA" \
    --out "$out" --resume >> "$EVAL_ROOT/logs/$label-$BENCH.log" 2>&1 || rc=$?
  case "$rc" in 0) state=exited_ok ;; 124) state=timed_out ;; 137) state=killed_after_timeout ;; *) state=failed ;; esac
  note "$label" "$state" "$rc"
  cell_done "$out" || incomplete=1
done
exit "$incomplete"
