# Paired comparisons (locked test) of a v3 rounds 4-5 continuation against its v3 round-3 source, S_1 and the v1 r5 run.
import json, os, pathlib, subprocess, sys
R = os.environ.get("MCQ_RESULTS_RUNS", str(pathlib.Path(__file__).resolve().parent.parent / "results" / "runs"))
A = str(pathlib.Path(__file__).resolve().parents[3] / "scripts" / "mcq_rsi_analysis.py")
b_ = sys.argv[1]; pool = sys.argv[2] if len(sys.argv) > 2 else "test"
pairs = [(f"{b_}_v3/dynamic_sft", f"{b_}_v3r5/dynamic_sft"), (f"{b_}_v3/static_sft", f"{b_}_v3r5/static_sft"),
         (f"{b_}_v3/S_1", f"{b_}_v3r5/dynamic_sft"), (f"{b_}_v3/S_1", f"{b_}_v3r5/static_sft"),
         (f"{b_}_v3r5/static_sft", f"{b_}_v3r5/dynamic_sft"),
         (f"{b_}_r5/dynamic", f"{b_}_v3r5/dynamic_sft"), (f"{b_}_r5/static", f"{b_}_v3r5/static_sft"),
         (f"{b_}_main/success", f"{b_}_v3r5/static_sft"), (f"{b_}_abl/success", f"{b_}_v3r5/static_sft")]
for a, b in pairs:
    pa = f"{R}/{a.split('/')[0]}/final/{a.split('/')[1]}/{pool}"; pb = f"{R}/{b.split('/')[0]}/final/{b.split('/')[1]}/{pool}"
    try:
        d = json.loads(subprocess.run([sys.executable, A, "compare", "--a", pa, "--b", pb], capture_output=True, text=True, check=True).stdout)
    except Exception as e:
        print(f"{a:24s} -> {b:24s} (missing)"); continue
    acc, c, m = d["accuracy"], d["calls"], d["mcnemar"]
    print(f"{a:24s} -> {b:24s} acc {acc['mean_a']*100:.1f} -> {acc['mean_b']*100:.1f} (d={acc['diff']*100:+.1f} p={acc['p_two_sided']:.3f} mcn={m['exact_p']:.3f})  calls {c['mean_a']:.3f} -> {c['mean_b']:.3f} (p={c['p_two_sided']:.3f})")
