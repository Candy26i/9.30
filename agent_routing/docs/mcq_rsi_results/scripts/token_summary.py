# Final token table per benchmark from the mcq_token_cost.py outputs (results/tokcost/out/<bench>_<pool>/summary.json,
# plus results/tokcost/base/<bench>_<pool>/summary.json once the untrained-manager baseline is in).
import json, os, sys
R = "/private/tmp/claude-501/-Users-madili-Desktop-7-98/17a9c165-9b16-40c6-bd20-a405349d1016/scratchpad/results/tokcost"
ROWS = [("只写草稿", "draft_only"), ("未训练 base manager", "base/manager"), ("S₁（论文一轮 SFT）", "v3/S_1"),
        ("RSI dynamic 第 3 轮", "v3/dynamic_sft"), ("RSI static 第 3 轮", "v3/static_sft"), ("RSI dynamic 第 5 轮", "v3r5/dynamic_sft"),
        ("RSI static 第 5 轮", "v3r5/static_sft"), ("success（每题都问）", "abl/success"), ("success（MedQA 主实验）", "main/success")]
for g in sys.argv[1:] or ("medqa_test", "mmlu_pro_test", "gpqa_test", "aqua_test"):
    labels = {}
    for d in (f"{R}/out/{g}", f"{R}/base/{g}"):
        p = f"{d}/summary.json"
        if os.path.isfile(p):
            labels.update(json.load(open(p))["labels"])
    s1 = labels.get("v3/S_1")
    print(f"\n### {g}  (tokens per question; Δ vs S₁)")
    print("| policy | acc | calls | total | Δtotal | decode | Δdecode | advisor tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for name, key in ROWS:
        x = labels.get(key)
        if not x:
            continue
        acc = "–" if x["accuracy"] is None else f"{x['accuracy']*100:.1f}"
        dt = f"{(x['total']/s1['total']-1)*100:+.0f}%" if s1 else ""
        dd = f"{(x['decode']/s1['decode']-1)*100:+.0f}%" if s1 else ""
        adv = ", ".join(f"{k[:3]} {v:.0f}" for k, v in sorted(x.get("tokens_per_advisor_per_q", {}).items()))
        print(f"| {name} | {acc} | {x['calls']:.2f} | {x['total']:.0f} | {dt} | {x['decode']:.0f} | {dd} | {adv} |")
