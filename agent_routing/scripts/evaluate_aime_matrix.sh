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

# Fatal precheck: the RSI controller has exited, and every RSI train/dev question is a BeyondAIME
# question disjoint from AIME2026 (normalized identity check, not a filename check).
"$RSI_PYTHON" - <<'PY' || exit 1
import json, os
from pathlib import Path
from src.verifiable.data import identity, load_rows
out, sub, pool = Path(os.environ['RSI_OUTPUT']), Path(os.environ['RSI_SUBSET']), Path(os.environ['LUNA_DATA'])/'manager'
status = json.loads((out/'run_summary.json').read_text()).get('controller_status')
assert status in {'completed', 'failed', 'interrupted'}, f'RSI controller not finished: {status}'
ids = lambda f, s: {identity(r.question) for r in load_rows(f, required_split=s)}
aime, beyond = ids(pool/'aime2026.jsonl', 'test'), ids(pool/'beyondaime.jsonl', 'test')
used = ids(sub/'train.jsonl', 'train') | ids(sub/'dev.jsonl', 'dev')
assert len(aime) == 30 and not used & aime and used <= beyond, 'RSI data not disjoint from AIME2026 / not BeyondAIME'
print(json.dumps({'rsi_controller_status': status, 'rsi_questions': len(used), 'aime_overlap': 0}))
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
for label in base ${RSI_ARMS:-dynamic static success}; do
  case "$label" in
    base) checkpoint=Qwen/Qwen3.5-9B ;;
    *) checkpoint="$RSI_OUTPUT/$label/round_2/grpo"; label="${label}_final" ;;
  esac
  out="$EVAL_ROOT/$label/$BENCH"
  if cell_done "$out"; then note "$label" already_complete 0; continue; fi
  if [[ "$label" != base && ! ( -f "$checkpoint/.rsi_complete.json" && -f "$checkpoint/adapter_config.json" ) ]]; then
    note "$label" arm_incomplete_not_evaluated 0; incomplete=1; continue   # never substitute an earlier checkpoint
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
