#!/usr/bin/env bash
# Rebuttal pod (2026-11): the equal-information delegation grid (scripts/mcq_verifier_grid.py) for the paper's manager
# S_1, the RSI managers and the untrained base manager, on the locked test and dev pools of the four benchmarks.
#   clone -> setup (experiment venv + vLLM venv) -> import + splits -> advisor cache from the HF backups -> manager
#   checkpoints from the HF run backups -> advisor server (GPU 0) -> preflight per benchmark -> 6-question smoke ->
#   tmux vgrid_queue (scripts/pod_ops/verifier_grid_queue.py) + tmux vgrid_backup (HF, every 30 min).
# Steps leave markers in $WORK/logs/bootstrap/ (idempotent: run it again after a failure). PREPARE_ONLY=1 stops
# before the smoke run (environment, checkpoints, advisor server and preflight only). Credentials are the
# operator's (HF_HOME=/workspace/hf-cache); nothing here reads a token. Log: $WORK/logs/verifier_grid_bootstrap.log
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1
BRANCH="${BRANCH:-feat/mcq-rsi-controller}"
WORK=/workspace/mcq_rsi; REPO=/workspace/9.30; TREE=$REPO/agent_routing; PY=/workspace/mcq-venv/bin/python
CKPT=$WORK/ckpt/verifier_grid; M=$WORK/logs/bootstrap; BACKUP_REPO="${BACKUP_REPO:-MaliDDD/margent-mcq-rsi-verifier_grid}"
mkdir -p $WORK/logs $M /workspace/tmp $CKPT
log() { printf '[vgrid %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
step() { local name=$1; shift; if [ -f "$M/$name" ]; then log "$name: done earlier"; return 0; fi; log "$name: start"
  if "$@"; then touch "$M/$name"; log "$name: ok"; else log "$name: FAILED (rc=$?)"; return 1; fi; }

clone() { if [ ! -d $REPO/.git ]; then git clone https://github.com/Candy26i/9.30.git $REPO || return 1; fi
  cd $REPO && git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH" && git rev-parse HEAD; }
setup() { cd $TREE && bash scripts/setup_mcq_rsi_pod.sh; }
# vLLM 0.26.0 does not pin transformers: a venv built on 2026-11-25 got transformers 5.24.1, whose stricter multimodal
# input check makes the server fail at start-up ("Found 1 <|image_pad|> tokens in the text but no images were passed").
# The lock file is the pip freeze of the vLLM venv of the 2026-10 pods (transformers 5.18.0), installed as is.
pin_vllm() { /workspace/vllm-venv/bin/python -m pip install --no-deps -q -r $TREE/scripts/pod_ops/vllm_env_2026-10.lock.txt \
    && /workspace/vllm-venv/bin/python -m pip freeze > $WORK/logs/environment_vllm.lock.txt \
    && /workspace/vllm-venv/bin/python -c 'import transformers, vllm; print("vllm", vllm.__version__, "transformers", transformers.__version__)'; }
import_assets() { cd $TREE && $PY -m src.manager.mcq_rsi import --bench all --out $WORK/import --cache-dir $HF_HOME/hub \
    && $PY -m src.manager.mcq_rsi prepare-splits --bench all --import-dir $WORK/import --hf-cache-dir $HF_HOME/hub; }
# The advisor cache is one file per request, keyed by content. Entries are created only when absent (never
# overwritten), so the union of several backups is safe and a re-run cannot change a file a running job has read.
# Sources: this experiment's own backup first (a replacement pod continues from it), then the paper-era caches, which
# are required: without them every advisor output would be computed afresh and differ from the paper-era ones.
restore_cache() { cd $TREE && BACKUP_REPO="$BACKUP_REPO" $PY - <<'PYEOF'
import os, pathlib, shutil, tarfile, tempfile
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
WORK = pathlib.Path("/workspace/mcq_rsi")
own = os.environ["BACKUP_REPO"]
failed = []
for repo in (own, "MaliDDD/margent-mcq-rsi-base_manager", "MaliDDD/margent-mcq-rsi-mmlu_pro_v3g_dyn", "MaliDDD/margent-mcq-rsi-mmlu_pro_v3r5"):
    try:
        p = hf_hub_download(repo, "advisor_cache.tar.gz", local_dir=f"/workspace/tmp/restore/{repo.split('/')[-1]}")
    except (RepositoryNotFoundError, EntryNotFoundError) as exc:
        if repo == own:
            print("advisor cache: no earlier backup of this experiment (first pod)"); continue
        failed.append(repo); print("advisor cache: MISSING", repo, type(exc).__name__); continue
    except Exception as exc:
        failed.append(repo); print("advisor cache: FAILED", repo, type(exc).__name__, str(exc)[:160]); continue
    new = kept = 0
    with tarfile.open(p) as tar:
        for m in tar:
            parts = pathlib.PurePosixPath(m.name).parts
            if not m.isfile() or not parts or parts[0] != "advisor_cache" or ".." in parts:
                continue
            target = WORK.joinpath(*parts)
            if target.exists():
                kept += 1; continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                shutil.copyfileobj(tar.extractfile(m), tmp)
            try:
                os.link(tmp.name, target); new += 1
            except FileExistsError:
                kept += 1
            finally:
                os.unlink(tmp.name)
    print(f"advisor cache: {repo}: {new} new entries, {kept} already present")
# finished or interrupted grids of an earlier pod (shards, records, tables): only files that are not here yet
try:
    local = pathlib.Path(snapshot_download(own, allow_patterns=["runs/verifier_grid/**"], local_dir="/workspace/tmp/restore/grid"))
    n = 0
    for f in (local / "runs" / "verifier_grid").rglob("*"):
        target = WORK / f.relative_to(local)
        if f.is_file() and f.name != "queue.lock" and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, target); n += 1
    print(f"grid outputs restored from {own}: {n} files")
except RepositoryNotFoundError:
    print("grid outputs: nothing to restore (first pod)")
except Exception as exc:
    failed.append(own + " (grid outputs)"); print("grid outputs: FAILED", type(exc).__name__, str(exc)[:160])
raise SystemExit(1 if failed else 0)
PYEOF
}
# RSI managers recorded as the final checkpoints of their runs (final/<arm>/test/mcq_rsi_eval.json -> signature.checkpoint).
# One failed download does not block the others; the step stays unmarked, so a re-run fetches only what is missing.
fetch_ckpts() { cd $TREE && $PY - <<'PYEOF'
import pathlib, shutil
from huggingface_hub import snapshot_download
from src.manager.mcq_rsi.evaluate import resolve_base
CK = pathlib.Path("/workspace/mcq_rsi/ckpt/verifier_grid")
CK.mkdir(parents=True, exist_ok=True)
(CK / "base_model_path.txt").write_text(resolve_base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a") + "\n")
print("base", (CK / "base_model_path.txt").read_text().strip())
want = [(f"{b}_v3", f"runs/{b}_v3/r3/dynamic_sft/sft/model", f"{b}_dyn3") for b in ("medqa", "mmlu_pro", "gpqa", "aqua")]
want.append(("mmlu_pro_v3g_dyn", "runs/mmlu_pro_v3g_dyn/r3/dynamic/grpo/final", "mmlu_pro_dyn3g"))
failed = []
for run, sub, name in want:
    dst = CK / name
    if (dst / "adapter_model.safetensors").is_file() and (dst / "adapter_config.json").is_file():
        print(name, "present"); continue
    try:
        local = snapshot_download(f"MaliDDD/margent-mcq-rsi-{run}", allow_patterns=[f"{sub}/*"], local_dir=f"/workspace/tmp/ckpt_dl/{run}")
        files = [f for f in (pathlib.Path(local) / sub).iterdir() if f.is_file()]
        assert {"adapter_model.safetensors", "adapter_config.json"} <= {f.name for f in files}, f"{name}: adapter files missing"
        tmp = CK / f".{name}.part"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        for f in files:
            shutil.copyfile(f, tmp / f.name)
        shutil.rmtree(dst, ignore_errors=True); tmp.rename(dst)
        print(name, "ok", sorted(f.name for f in files))
    except Exception as exc:
        failed.append(name); print(name, "FAILED", type(exc).__name__, str(exc)[:200])
raise SystemExit(1 if failed else 0)
PYEOF
}
# Not marker-gated: after a pod restart (or a dead vLLM) the server must be started again, whatever ran before.
advisor_up() { curl -fsS http://127.0.0.1:18002/health >/dev/null 2>&1; }
advisors() { cd $TREE || return 1
  if ! advisor_up; then
    log "advisors: starting the server"
    MCQ_CONFIG_SUFFIX=_v3 BENCH=medqa bash scripts/runpod_mcq_rsi.sh stop-advisors >/dev/null 2>&1 || true
    tmux kill-session -t mcq_advisors_medqa 2>/dev/null || true
    MCQ_CONFIG_SUFFIX=_v3 BENCH=medqa bash scripts/runpod_mcq_rsi.sh bg advisors || return 1
  fi
  for i in $(seq 1 120); do advisor_up && return 0; sleep 10; done
  log "advisors: not healthy after 20 min"; return 1; }
# Not marker-gated either: --skip-if-passed keeps a report that matches the live server and replays it otherwise.
preflight_one() { cd $TREE && MCQ_CONFIG_SUFFIX=_v3 BENCH=$1 PREFLIGHT_BENCHES=$1 bash scripts/runpod_mcq_rsi.sh preflight 2>&1 | tail -2
  $PY -c "import json,sys; sys.exit(0 if json.load(open('$WORK/advisor_cache/preflight/$1.json')).get('passed') else 1)"; }
smoke() { cd $TREE && rm -rf $WORK/runs/verifier_grid_smoke && CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" MARGENT_WANDB_MODE=disabled \
    $PY scripts/mcq_verifier_grid.py --config configs/mcq_rsi_medqa_v3.json --pool dev --checkpoint $WORK/import/medqa/round1/sft \
    --out $WORK/runs/verifier_grid_smoke --limit 6 2>&1 | grep -v "Warning\|warn(\|Loading weights" | tail -3
  $PY - <<'PYEOF'
import json
rows = [json.loads(l) for l in open("/workspace/mcq_rsi/runs/verifier_grid_smoke/grid_records.jsonl")]
assert len(rows) == 6, len(rows)
for r in rows:
    assert [b["sequence"] for b in r["branches"]] == [["extractor"], ["reasoner"], ["verifier"]], r["example_id"]
    assert [b["kind"] for b in r["extra_branches"]] == ["verifier_nocand", "null"], r["example_id"]
    assert r["policy_action"]["key_logprobs"], r["example_id"]
print("smoke ok:", json.load(open("/workspace/mcq_rsi/runs/verifier_grid_smoke/grid_summary.json"))["always"])
PYEOF
}
queue() {
  mkdir -p $WORK/runs/verifier_grid $WORK/logs/verifier_grid
  local qlog=$WORK/logs/verifier_grid/queue.log blog=$WORK/logs/verifier_grid/backup.log n0 i a b
  if tmux has-session -t vgrid_queue 2>/dev/null && pgrep -f '[v]erifier_grid_queue\.py' >/dev/null && [ "${RESTART_QUEUE:-0}" != 1 ]; then
    log "queue: already running (RESTART_QUEUE=1 stops it and its jobs, then starts it again)"
  else
    tmux kill-session -t vgrid_queue 2>/dev/null
    pkill -f '[v]erifier_grid_queue\.py' 2>/dev/null
    for i in $(seq 1 60); do pgrep -f '[m]cq_verifier_grid\.py --config' >/dev/null || break; sleep 2; done
    pkill -9 -f '[m]cq_verifier_grid\.py --config' 2>/dev/null
    a=x; for i in $(seq 1 30); do b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
      [ "$a" = "$b" ] && break; a=$b; sleep 3; done   # GPU memory of the stopped jobs is released before slots are counted
    n0=0; [ -f $qlog ] && n0=$(wc -l < $qlog)
    tmux new-session -d -s vgrid_queue "cd $TREE && HF_HOME=$HF_HOME HF_HUB_DISABLE_XET=1 TMPDIR=$TMPDIR MCQ_ADVISOR_PORT=$MCQ_ADVISOR_PORT PYTHONUNBUFFERED=1 python3 scripts/pod_ops/verifier_grid_queue.py ${QUEUE_ARGS:-} 2>&1 | tee -a $qlog; exec bash"
    for i in $(seq 1 30); do tail -n +$((n0 + 1)) $qlog 2>/dev/null | grep -q "jobs to run" && break; sleep 2; done
    if ! tail -n +$((n0 + 1)) $qlog 2>/dev/null | grep -q "jobs to run"; then
      log "queue: did NOT start:"; tail -n +$((n0 + 1)) $qlog 2>/dev/null | tail -5; return 1; fi
    log "queue: $(tail -n +$((n0 + 1)) $qlog | grep "jobs to run" | tail -1)"
  fi
  # One pass every 30 minutes in a shell loop: an exit of the backup script (no login yet, a network error) is retried.
  if ! tmux has-session -t vgrid_backup 2>/dev/null; then
    tmux new-session -d -s vgrid_backup "cd $TREE; while true; do if [ -f $HF_HOME/token ]; then HF_HOME=$HF_HOME HF_HUB_DISABLE_XET=1 TMPDIR=$TMPDIR $PY scripts/backup_mcq_rsi_hf.py --work $WORK --run verifier_grid --repo $BACKUP_REPO --public 2>&1 | tail -3 | cut -c1-300 | tee -a $blog; sleep 1800; else echo \"[vgrid-backup \$(date -u +%FT%TZ)] waiting for the HF login ($HF_HOME/token)\" | tee -a $blog; sleep 120; fi; done"
    log "backup: loop started (every 30 min -> $BACKUP_REPO; confirmed once $WORK/logs/hf_backup_last__verifier_grid.json exists)"
  fi
}

{
  mkdir -p $WORK/logs/verifier_grid
  step clone clone || exit 1
  step setup setup || exit 1
  step pin_vllm pin_vllm || exit 1
  step import import_assets || exit 1
  step restore_cache restore_cache || { [ "${ALLOW_COLD_CACHE:-0}" = 1 ] || { log "restore_cache FAILED: run again, or ALLOW_COLD_CACHE=1 to compute every advisor output afresh"; exit 1; }
    log "restore_cache FAILED, ALLOW_COLD_CACHE=1: advisor outputs will NOT be the paper-era ones"; }
  step fetch_ckpts fetch_ckpts || log "fetch_ckpts: some checkpoints are missing (their jobs will fail to start); run the chain again to retry"
  advisors || exit 1
  for b in medqa mmlu_pro aqua gpqa; do preflight_one $b || log "preflight $b FAILED: its jobs will be refused"; done
  if [ "${PREPARE_ONLY:-0}" = 1 ]; then log "PREPARE_ONLY=1: environment ready, stopping before the smoke run"; exit 0; fi
  step vgrid_smoke smoke || exit 1
  queue || exit 1
  log "bootstrap complete: tmux vgrid_queue (queue log $WORK/logs/verifier_grid/queue.log), vgrid_backup, mcq_advisors_medqa"
} 2>&1 | tee -a $WORK/logs/verifier_grid_bootstrap.log
