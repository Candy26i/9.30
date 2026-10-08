# Per-run backup check: every staged file (hf_backup_stage__<run>, minus the uploader's .cache bookkeeping) is on HF with the same size.
import sys
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi(); work = Path("/workspace/mcq_rsi")
ok_all = True
for run in sys.argv[1:]:
    repo = f"MaliDDD/margent-mcq-rsi-{run}"
    remote = {f.path: f.size for f in api.list_repo_tree(repo, recursive=True) if hasattr(f, "size")}
    stage = work / f"hf_backup_stage__{run}"
    local = {str(p.relative_to(stage)): p.stat().st_size for p in stage.rglob("*") if p.is_file() and ".cache/huggingface" not in str(p)}
    missing = [p for p in local if p not in remote]
    diff = [p for p in local if p in remote and remote[p] != local[p]]
    extra = [p for p in remote if p not in local and not p.startswith(".git")]
    status = "OK" if not missing and not diff else "PROBLEM"
    ok_all = ok_all and status == "OK"
    print(f"{run}: staged {len(local)} | on HF {len(remote)} | missing {len(missing)} | size-diff {len(diff)} | extra-on-HF {len(extra)} -> {status}")
    if missing or diff:
        print("   ", missing[:5], diff[:5])
print("ALL OK" if ok_all else "CHECK FAILED")
