#!/usr/bin/env bash
# Strict single-agent baselines on one GPU (2026-10-11): clone + setup + import, then on every locked test
#   1. the base model with a plain MCQ prompt (no DRAFT/tool protocol),
#   2. S_1 with its manager prompt but no tools,
#   3. the v3 dynamic round-3 manager (from its HF backup) with no tools,
# then a backup of runs/single_agent to MaliDDD/margent-mcq-rsi-single_agent. Markers in logs/bootstrap/. Log: logs/single_agent.log
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
BRANCH="${BRANCH:-feat/mcq-rsi-controller}"
WORK=/workspace/mcq_rsi; REPO=/workspace/9.30; TREE=$REPO/agent_routing; PY=/workspace/mcq-venv/bin/python
M=$WORK/logs/bootstrap; OUT=$WORK/runs/single_agent
mkdir -p $WORK/logs $M /workspace/tmp $OUT
log() { printf '[single-agent %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
step() { local name=$1; shift; if [ -f "$M/$name" ]; then log "$name: done earlier"; return 0; fi; log "$name: start"
  if "$@"; then touch "$M/$name"; log "$name: ok"; else log "$name: FAILED (rc=$?)"; return 1; fi; }
clone() { if [ ! -d $REPO/.git ]; then git clone https://github.com/Candy26i/9.30.git $REPO || return 1; fi
  cd $REPO && git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH" && git rev-parse HEAD; }
setup() { bash $TREE/scripts/pod_ops/setup_lite.sh; }
import_assets() { cd $TREE && $PY -m src.manager.mcq_rsi import --bench all --out $WORK/import --cache-dir $HF_HOME/hub \
    && $PY -m src.manager.mcq_rsi prepare-splits --bench all --import-dir $WORK/import --hf-cache-dir $HF_HOME/hub; }
fetch_dyn() { cd $TREE && $PY - <<'PYEOF'
from huggingface_hub import snapshot_download
import pathlib, shutil
for b in ("medqa", "mmlu_pro", "gpqa", "aqua"):
    run = f"{b}_v3"; sub = f"runs/{run}/r3/dynamic_sft/sft/model"
    local = snapshot_download(f"MaliDDD/margent-mcq-rsi-{run}", allow_patterns=[f"{sub}/*"], local_dir="/workspace/tmp/dyn_dl")
    src = pathlib.Path(local) / sub; dst = pathlib.Path(f"/workspace/mcq_rsi/runs/single_agent/adapters/{b}_dynamic_r3"); dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file(): shutil.copyfile(f, dst / f.name)
    assert (dst / "adapter_model.safetensors").is_file(), b
    print(b, "dynamic r3 adapter ok")
PYEOF
}
evals() {
  cd $TREE
  BASE=$($PY -c "from src.manager.mcq_rsi.evaluate import resolve_base; print(resolve_base('Qwen/Qwen3.5-9B'))"); echo "$BASE" > $OUT/base_model_path.txt
  for bp in medqa:test gpqa:test aqua:test mmlu_pro:test mmlu_pro:test_paper; do
    b=${bp%%:*}; pool=${bp##*:}
    for spec in "base_plain:plain:$BASE" "s1_manager_notools:manager:$WORK/import/$b/round1/sft" "dyn3_manager_notools:manager:$OUT/adapters/${b}_dynamic_r3"; do
      name=${spec%%:*}; rest=${spec#*:}; mode=${rest%%:*}; ckpt=${rest#*:}
      d=$OUT/$name/$b/$pool
      [ -f $d/summary.json ] && continue
      $PY scripts/mcq_single_agent.py --config configs/mcq_rsi_${b}_v3.json --pool $pool --mode $mode --checkpoint $ckpt --out $d \
        2>&1 | grep -v "Warning\|warn(\|Loading weights" | tail -1 | tee -a $WORK/logs/single_agent.log || log "$name $b $pool FAILED"
    done
  done
}
backup() { cd $TREE && $PY scripts/backup_mcq_rsi_hf.py --work $WORK --run single_agent --repo MaliDDD/margent-mcq-rsi-single_agent --public 2>&1 | tail -1 | cut -c1-160; }
{
  step clone clone || exit 1
  step setup setup || exit 1
  step import import_assets || exit 1
  step fetch_dyn fetch_dyn || log "fetch_dyn FAILED (the dyn3 rows will be skipped)"
  step evals evals || log "some evals FAILED (see logs/single_agent.log)"
  step backup backup || log "backup FAILED"
  touch $WORK/logs/single_agent_done
  log "single-agent chain complete"
} 2>&1 | tee -a $WORK/logs/single_agent_bootstrap.log
