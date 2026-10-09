#!/usr/bin/env bash
# Experiment D: does the GRPO step produce better SFT data than the SFT model it started from?
# For v1 rounds whose GRPO candidate was accepted (G_k != S_k; mmlu_pro_main and aqua_main at k = 2, dynamic arm), the
# next round's pool (collect_r3) is collected twice, with S_k and with G_k, each collection is labelled with the dynamic
# rule, and each label set is trained from S_1 (v3 config: draft_supervision none) and evaluated once on dev
# (scripts/mcq_label_quality.py). Round-1 pairs need no collection: where G_1 was accepted (aqua_main, gpqa_main) the
# v1 run's r2 labels were collected by G_1 and the v3 run's r2 labels by S_1.
# GPU 1, right after the bootstrap (the label-quality chain waits for logs/grpoq_done); writes logs/grpoq_done at the end.
set -o pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
WORK=/workspace/mcq_rsi; TREE=/workspace/9.30/agent_routing; PY=/workspace/mcq-venv/bin/python; URL=http://127.0.0.1:18002
Q=$WORK/grpoq; LOG=$WORK/logs/grpoq.log
log() { printf '[grpoq %s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a $LOG; }
until grep -q "bootstrap complete" $WORK/logs/bootstrap.log 2>/dev/null; do sleep 60; done   # advisors, cache, labels, preflight
log "bootstrap finished; starting (GPU 1, before the label-quality chain: user priority 2026-10-09)"
cd $TREE
mkdir -p $Q

# 1. the v1 checkpoints S_2 / G_2 (mmlu_pro, aqua) and the G_1-collected round-2 labels (aqua, gpqa)
if [ ! -f $Q/downloads_done ]; then
$PY - <<'PYEOF' 2>&1 | tee -a $LOG && touch $Q/downloads_done
from huggingface_hub import snapshot_download, hf_hub_download
import pathlib, shutil
repo = "MaliDDD/margent-mcq-rsi"
out = pathlib.Path("/workspace/mcq_rsi/grpoq")
for run, k in (("mmlu_pro_main", 2), ("aqua_main", 2)):
    b = run.replace("_main", "")
    for kind, sub in (("S", f"r{k}/dynamic/sft/model"), ("G", f"r{k}/dynamic/grpo/final")):
        local = snapshot_download(repo, allow_patterns=[f"runs/{run}/{sub}/*"], local_dir="/workspace/tmp/grpoq_dl")
        src = pathlib.Path(local) / "runs" / run / sub
        dst = out / b / f"{kind}{k}"; dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copyfile(f, dst / f.name)
        need = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
        have = {p.name for p in dst.iterdir()}
        assert need <= have, (run, kind, k, need - have)
        print(run, f"{kind}{k}", "ok", sorted(have))
for b in ("aqua", "gpqa"):
    p = hf_hub_download(repo, f"runs/{b}_main/r2/dynamic/select/labels.jsonl", local_dir="/workspace/tmp/grpoq_dl")
    dst = out / b / "labels_G1_r2.jsonl"; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dst)
    r = hf_hub_download(repo, f"runs/{b}_main/r2/dynamic/select/labels.report.json", local_dir="/workspace/tmp/grpoq_dl")
    shutil.copyfile(r, out / b / "labels_G1_r2.report.json")
    print(b, "G1 labels ok")
PYEOF
fi
[ -f $Q/downloads_done ] || { log "downloads FAILED"; touch $WORK/logs/grpoq_done; exit 1; }

# 2. collect_r3 with S_2 and with G_2, then the dynamic selection of each (the run's own settings: depth 2, seed 42, policy roots)
for b in mmlu_pro aqua; do
  for kind in S G; do
    C=$Q/$b/collect_r3_${kind}2
    if [ ! -f $C/counterfactual_records.jsonl ] || [ ! -f $C/collect_manifest.json ]; then
      log "collect $b collect_r3 with ${kind}2"
      $PY -m src.manager.mcq_rsi collect --bench $b --pool collect_r3 --checkpoint $Q/$b/${kind}2 --out $C \
        --advisor-url $URL --advisor-cache $WORK/advisor_cache --advisor-mode base --resume 2>&1 | tee -a $WORK/logs/grpoq_collect_${b}_${kind}2.log \
        || log "collect $b ${kind}2 FAILED"
    fi
    if [ -f $C/counterfactual_records.jsonl ] && [ ! -f $Q/$b/labels_${kind}2_r3.jsonl ]; then
      $PY -m src.manager.mcq_rsi select --bench $b --arm dynamic --records $C/counterfactual_records.jsonl \
        --out $Q/$b/labels_${kind}2_r3.jsonl --import-dir $WORK/import 2>&1 | tee -a $LOG || log "select $b ${kind}2 FAILED"
    fi
  done
done

# 3. SFT from S_1 on each label set + one dev eval each (round-1 pair where G_1 was accepted; round-2 pair where collected)
units_mmlu="S2=$Q/mmlu_pro/labels_S2_r3.jsonl G2=$Q/mmlu_pro/labels_G2_r3.jsonl"
units_aqua="S1=$WORK/labels/aqua/r2.jsonl G1=$Q/aqua/labels_G1_r2.jsonl S2=$Q/aqua/labels_S2_r3.jsonl G2=$Q/aqua/labels_G2_r3.jsonl"
units_gpqa="S1=$WORK/labels/gpqa/r2.jsonl G1=$Q/gpqa/labels_G1_r2.jsonl"
for b in mmlu_pro aqua gpqa; do
  var="units_$b"; units=""
  for u in ${!var}; do [ -f "${u#*=}" ] && units="$units $u" || log "missing label set $u (skipped)"; done
  [ -n "$units" ] || continue
  log "label quality $b: $units"
  $PY scripts/mcq_label_quality.py --config configs/mcq_rsi_${b}_v3.json --out $WORK/runs/${b}_v3gq --labels $units \
    --advisor-url $URL 2>&1 | tee -a $WORK/logs/${b}_v3gq.log || log "label quality $b FAILED"
done
log "done"
touch $WORK/logs/grpoq_done
