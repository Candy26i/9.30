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
import_assets() { cd $TREE && $PY -m src.manager.mcq_rsi import --bench all --out $WORK/import --cache-dir $HF_HOME/hub \
    && $PY -m src.manager.mcq_rsi prepare-splits --bench all --import-dir $WORK/import --hf-cache-dir $HF_HOME/hub; }
# The advisor cache is one file per request, keyed by content: the union of several backups is safe.
restore_cache() { cd $TREE && $PY - <<'PYEOF'
import tarfile
from huggingface_hub import hf_hub_download
ok = 0
for repo in ("MaliDDD/margent-mcq-rsi-base_manager", "MaliDDD/margent-mcq-rsi-mmlu_pro_v3g_dyn", "MaliDDD/margent-mcq-rsi-mmlu_pro_v3r5"):
    try:
        p = hf_hub_download(repo, "advisor_cache.tar.gz", local_dir=f"/workspace/tmp/restore/{repo.split('/')[-1]}")
        with tarfile.open(p) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith("advisor_cache/") and ".." not in m.name and m.isfile()]
            tar.extractall("/workspace/mcq_rsi", members=members)
        print("advisor cache restored from", repo, len(members), "files"); ok += 1
    except Exception as exc:
        print("advisor cache: skipped", repo, type(exc).__name__, str(exc)[:120])
raise SystemExit(0 if ok else 1)
PYEOF
}
# RSI managers recorded as the final checkpoints of their runs (final/<arm>/test/mcq_rsi_eval.json -> signature.checkpoint).
fetch_ckpts() { cd $TREE && $PY - <<'PYEOF'
import pathlib, shutil
from huggingface_hub import snapshot_download
CK = pathlib.Path("/workspace/mcq_rsi/ckpt/verifier_grid")
want = [(f"{b}_v3", f"runs/{b}_v3/r3/dynamic_sft/sft/model", f"{b}_dyn3") for b in ("medqa", "mmlu_pro", "gpqa", "aqua")]
want.append(("mmlu_pro_v3g_dyn", "runs/mmlu_pro_v3g_dyn/r3/dynamic/grpo/final", "mmlu_pro_dyn3g"))
for run, sub, name in want:
    local = snapshot_download(f"MaliDDD/margent-mcq-rsi-{run}", allow_patterns=[f"{sub}/*"], local_dir=f"/workspace/tmp/ckpt_dl/{run}")
    src, dst = pathlib.Path(local) / sub, CK / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copyfile(f, dst / f.name)
    assert (dst / "adapter_model.safetensors").is_file() and (dst / "adapter_config.json").is_file(), name
    print(name, "ok", sorted(p.name for p in dst.iterdir()))
from src.manager.mcq_rsi.evaluate import resolve_base
(CK / "base_model_path.txt").write_text(resolve_base("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a") + "\n")
print("base", (CK / "base_model_path.txt").read_text().strip())
PYEOF
}
advisors() { cd $TREE && MCQ_CONFIG_SUFFIX=_v3 BENCH=medqa bash scripts/runpod_mcq_rsi.sh bg advisors
  for i in $(seq 1 120); do curl -fsS http://127.0.0.1:18002/health >/dev/null 2>&1 && return 0; sleep 10; done
  log "advisors: not healthy after 20 min"; return 1; }
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
  for v in HF_HOME HF_HUB_DISABLE_XET TMPDIR MCQ_ADVISOR_PORT; do tmux set-environment -g $v "${!v}" 2>/dev/null; done
  tmux kill-session -t vgrid_queue 2>/dev/null
  tmux new-session -d -s vgrid_queue "cd $TREE && HF_HOME=$HF_HOME HF_HUB_DISABLE_XET=1 TMPDIR=$TMPDIR PYTHONUNBUFFERED=1 python3 scripts/pod_ops/verifier_grid_queue.py ${QUEUE_ARGS:-} 2>&1 | tee -a $WORK/logs/verifier_grid/queue.log; exec bash"
  tmux kill-session -t vgrid_backup 2>/dev/null
  tmux new-session -d -s vgrid_backup "until [ -f $HF_HOME/token ]; do echo waiting for the HF login; sleep 120; done; cd $TREE && HF_HOME=$HF_HOME HF_HUB_DISABLE_XET=1 TMPDIR=$TMPDIR $PY scripts/backup_mcq_rsi_hf.py --work $WORK --run verifier_grid --repo $BACKUP_REPO --public --every-minutes 30 2>&1 | tee -a $WORK/logs/verifier_grid/backup.log; exec bash"
}

{
  mkdir -p $WORK/logs/verifier_grid
  step clone clone || exit 1
  step setup setup || exit 1
  step import import_assets || exit 1
  step restore_cache restore_cache || log "restore_cache FAILED: every advisor request will be computed"
  step fetch_ckpts fetch_ckpts || log "fetch_ckpts FAILED: the dyn3 / dyn3g / base jobs will fail to start"
  step advisors advisors || exit 1
  for b in medqa mmlu_pro aqua gpqa; do step vgrid_preflight_$b preflight_one $b || log "preflight $b FAILED: its jobs will be refused"; done
  if [ "${PREPARE_ONLY:-0}" = 1 ]; then log "PREPARE_ONLY=1: environment ready, stopping before the smoke run"; exit 0; fi
  step vgrid_smoke smoke || exit 1
  queue
  log "bootstrap complete: tmux vgrid_queue (queue log $WORK/logs/verifier_grid/queue.log), vgrid_backup, mcq_advisors_medqa"
} 2>&1 | tee -a $WORK/logs/verifier_grid_bootstrap.log
