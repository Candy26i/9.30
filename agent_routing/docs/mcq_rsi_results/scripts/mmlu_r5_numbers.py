# MMLU-Pro v3 rounds 4-5 (mmlu_pro_v3r5), locked tests 2026-10-08 11:03-12:40 UTC. Filled by hand from compare_v3r5.py / letters / tables.
MMLU = {
    "dyn5_test": 61.4, "dyn5_test_calls": 0.298, "sta5_test": 58.0, "sta5_test_calls": 0.252,
    "dyn5_paper": 58.6, "dyn5_paper_calls": 0.294, "sta5_paper": 53.8, "sta5_paper_calls": 0.262,
    "cmp_test": {"dyn53": (-3.0, 0.043), "sta53": (0.0, 1.0), "dyn5s1": (-2.6, 0.18), "sta5s1": (-6.0, 0.0004),
                 "dynsta5": (3.4, 0.049), "dyn5v1": (-5.6, 0.0004), "sta5v1": (-0.2, 0.94)},
    "cmp_paper": {"dyn53": (-1.2, 0.45), "sta53": (-5.2, 0.0004), "dyn5s1": (-1.0, 0.59), "sta5s1": (-5.8, 0.0004),
                  "dynsta5": (4.8, 0.006), "dyn5v1": (-8.0, 0.0004), "sta5v1": (-3.2, 0.021)},
    "dev": {"dyn": [61.5, 61.5], "sta": [60.0, 56.0]},
    "note": ('<li><strong>MMLU-Pro：</strong>dynamic 第 5 轮在 <code>test</code> 上 61.4（每题调用 0.30），比自己的第 3 轮低 3.0 个点（p = 0.04），对 S₁ −2.6（p = 0.18）；'
             '<code>test_paper</code> 58.6，对 S₁ −1.0（不显著）。static 第 5 轮 <code>test</code> 58.0，和第 3 轮一样（对 S₁ −6.0），<code>test_paper</code> 继续掉到 53.8（对第 3 轮 −5.2，p &lt; 0.001）。'
             '第 5 轮 dynamic 仍高于 static：<code>test</code> +3.4（p = 0.049，McNemar 0.057），<code>test_paper</code> +4.8（p = 0.006）；在 104 道干净题上 dynamic 51.9、static 48.1、S₁ 53.8，都不显著。'
             '固定调用次数的回放里 dynamic 第 5 轮高于随机（60.5，区间 53.0–59.0，p = 0.003），static 不高于随机。'
             'dynamic 的初稿准确率不变（51.2，S₁ 50.8），B 的比例从 23% 升到 30%，调用从几乎只用 Verifier 变成 Reasoner / Verifier = 38 / 110。'
             'v1 里 dynamic 第 5 轮涨到 67.0（每题调用 0.44，含 GRPO）的结果在 v3 里没有复现：v3 的 dynamic 第 5 轮比它低 5.6 个点（p &lt; 0.001），调用也少了三分之一。</li>'),
}
