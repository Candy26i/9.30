# Paired comparisons (locked test) of a v3-with-GRPO run against its own S_1, the no-GRPO v3 arms, and v1.
import json, os, subprocess, sys
R = os.environ.get("MCQ_RESULTS_RUNS", "/private/tmp/claude-501/-Users-madili-Desktop-7-98/17a9c165-9b16-40c6-bd20-a405349d1016/scratchpad/results/runs")
A = "/Users/madili/Desktop/7.98/agent_routing/scripts/mcq_rsi_analysis.py"
b_ = sys.argv[1]; pool = sys.argv[2] if len(sys.argv) > 2 else "test"
pairs = [(f"{b_}_v3g/S_1", f"{b_}_v3g/dynamic"), (f"{b_}_v3g/S_1", f"{b_}_v3g/static"), (f"{b_}_v3g/S_1", f"{b_}_v3g/success"),
         (f"{b_}_v3g/static", f"{b_}_v3g/dynamic"), (f"{b_}_v3g/dynamic", f"{b_}_v3g/success"),
         (f"{b_}_v3/dynamic_sft", f"{b_}_v3g/dynamic"), (f"{b_}_v3/static_sft", f"{b_}_v3g/static"),
         (f"{b_}_main/dynamic", f"{b_}_v3g/dynamic"), (f"{b_}_main/static", f"{b_}_v3g/static"),
         (f"{b_}_main/success", f"{b_}_v3g/success"), (f"{b_}_abl/success", f"{b_}_v3g/success"), (f"{b_}_v3/S_1", f"{b_}_v3g/S_1")]
for a, b in pairs:
    pa = f"{R}/{a.split('/')[0]}/final/{a.split('/')[1]}/{pool}"; pb = f"{R}/{b.split('/')[0]}/final/{b.split('/')[1]}/{pool}"
    if not (os.path.isdir(pa) and os.path.isdir(pb)):
        continue
    d = json.loads(subprocess.run([sys.executable, A, "compare", "--a", pa, "--b", pb], capture_output=True, text=True, check=True).stdout)
    acc, c, m = d["accuracy"], d["calls"], d["mcnemar"]
    print(f"{a:24s} -> {b:24s} acc {acc['mean_a']*100:.1f} -> {acc['mean_b']*100:.1f} (d={acc['diff']*100:+.1f} p={acc['p_two_sided']:.3f} mcn={m['exact_p']:.3f})  calls {c['mean_a']:.3f} -> {c['mean_b']:.3f} (p={c['p_two_sided']:.3f})")
