#!/usr/bin/env python3
"""Add the v2/v3 draft-supervision rerun (section 8) and the v3 rounds 4-5 (section 9) to the results page.

Applies to the v1 page (page_v1_base.html) and writes margent_rsi_results.html, so it can be re-run
after MMLU-Pro's rounds 4-5 finish (fill MMLU in the dict below). All numbers: locked tests (paired bootstrap / McNemar
from scripts/mcq_rsi_analysis.py compare), dev metrics from each round's sft_dev/decision.json.
"""
import re, sys
from pathlib import Path

SP = Path(__file__).resolve().parent.parent  # docs/mcq_rsi_results/
src = (SP / "page_v1_base.html").read_text()  # the page as of 2026-10-06 (sections 0-9 of the v1 results)

# ---- MMLU-Pro rounds 4-5 (v3): None while running -------------------------------------------------------------------
MMLU = {
    "dyn5_test": None, "dyn5_test_calls": None, "sta5_test": None, "sta5_test_calls": None,
    "dyn5_paper": None, "dyn5_paper_calls": None, "sta5_paper": None, "sta5_paper_calls": None,
    # paired diffs (pp, p): dyn r5-r3, sta r5-r3, dyn r5-S1, sta r5-S1, r5 dyn-sta, dyn r5 - v1 r5 dyn, sta r5 - v1 r5 sta
    "cmp_test": None, "cmp_paper": None,
    "dev": None,  # {"dyn": [r4, r5], "sta": [r4, r5]} dev accuracies (%)
    "note": None,  # bullet text for section 9
}
cfg_path = SP / "scripts" / "mmlu_r5_numbers.py"
if cfg_path.is_file():
    exec(cfg_path.read_text())  # defines MMLU

PENDING = '<td class="num pending">进行中</td>'


def cell(acc, calls=None):
    if acc is None:
        return PENDING
    s = f"{acc:.1f}"
    if calls is not None:
        s += f' <span class="calls">{calls:.2f}</span>'
    return f'<td class="num">{s}</td>'


def dcell(d, p, sig=None):
    """A difference cell: d in pp, p two-sided (None -> pending)."""
    if d is None:
        return PENDING
    sgn = "+" if d > 0 else ("−" if d < 0 else "")
    ptxt = "&lt;0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"
    txt = f"{sgn}{abs(d):.1f} ({ptxt})"
    if sig is None:
        sig = p < 0.05
    return f'<td class="num"><span class="sig">{txt}</span></td>' if sig else f'<td class="num">{txt}</td>'


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


# ---- nav ----------------------------------------------------------------------------------------------------------------
rep('<a href="#rounds">多跑两轮</a><a href="#caveats">局限</a><a href="#data">数据与代码</a>',
    '<a href="#rounds">多跑两轮</a><a href="#fix">修掉草稿反馈环（v3）</a><a href="#rounds-v3">v3 的第 4–5 轮</a>'
    '<a href="#caveats">局限</a><a href="#data">数据与代码</a>')

# ---- summary: an update block at the top of section 0 ----------------------------------------------------------------
mm_r5_summary = ("MMLU-Pro：dynamic 第 5 轮 61.4（对 S₁ −2.6，不显著；对自己的第 3 轮 −3.0，p = 0.04），static 58.0（不变，对 S₁ −6.0）；第 5 轮 dynamic 仍比 static 高 3.4 个点（p = 0.049）。v1 里 dynamic 第 5 轮涨到 67.0 的结果在 v3 里没有复现。"
                 if MMLU["note"] else "MMLU-Pro 的第 4–5 轮还在跑，跑完会补到第 9 节。")
summary_update = f'''
  <div class="callout">
    <p><strong>10 月 8 日更新（第 8、9 节）。</strong>7.1 节找到的问题已经修掉并重跑：SFT 不再对 manager 自己的草稿字母计算损失，只训练"调用谁 / 不调用"的路由决策（<code>draft_supervision = none</code>，下文叫 v3）。修掉之后的结论比原来干净，也比原来保守：</p>
    <ul class="plain">
      <li><strong>dynamic 第 3 轮在四个基准上都和 S₁ 持平</strong>（差 0.0 到 +0.4 个点，p 全部大于 0.8），同时调用显著减少：MedQA 每题 0.36 → 0.15 次（−58%），MMLU-Pro 0.35 → 0.27 次（−24%）；GPQA 和 AQuA 的调用不变。草稿准确率和字母分布不再漂移（8.3 节）。</li>
      <li><strong>重新收集仍然有用，但只在 MMLU-Pro 上看得到。</strong>v3 的 static 第 3 轮在 MMLU-Pro <code>test</code> 上比 S₁ 低 6.0 个点（p &lt; 0.001），dynamic 比 static 高 6.4 个点（p &lt; 0.001）。其他三个基准 dynamic 和 static 差别不显著。</li>
      <li><strong>v1 里 dynamic 的两处"优势"没有保住。</strong>AQuA 第 3 轮的 +3.1（原 +4.3）不再显著；MMLU-Pro <code>test_paper</code> 上 dynamic 不再高于 S₁（+0.2），v1 的 +3.2 里混着训练题重叠（4.3 节）。</li>
      <li><strong>继续跑到第 5 轮，dynamic 变差、static 不变或变好。</strong>MedQA：dynamic 第 5 轮 77.4（对 S₁ −1.0，不显著），static 81.0（对 S₁ +2.6，p = 0.02，和 v1 的 80.4 一致，用 37% 的调用量追平 success 的 80.2）。AQuA：dynamic 第 5 轮 71.7（对 S₁ −5.9，p = 0.009），static 77.2（持平）。这次 dynamic 的下滑和草稿无关（草稿准确率、字母分布都不变），变化在路由上：MedQA 的调用从 Reasoner 转向 Extractor，AQuA 在初稿错的题上少调用了 Verifier；固定调用次数的回放里，两者第 5 轮的挑题都不再高于随机，static 则高于随机（第 9 节）。{mm_r5_summary}</li>
      <li><strong>中间版本 v2</strong>（只在 call 行掩码草稿，commit 行照旧）会让模型偏向直接作答：MMLU-Pro 第 2 轮的 dev 调用率掉到 0.07，AQuA 的 static 比 S₁ 低 8.7 个点。v2 在 AQuA 上的 dynamic 82.7（+5.1，p = 0.012）是全部实验里 success 之外最高的数，但 v3 没有复现（78.0），我们不把它当结论（8.4 节）。</li>
    </ul>
  </div>
'''
rep('<section id="summary">\n  <h2><span class="secno">0</span>结论速览</h2>\n',
    '<section id="summary">\n  <h2><span class="secno">0</span>结论速览</h2>\n' + summary_update +
    '  <p class="caption">下面的速览是 10 月 3 日的版本，对应第 1–7 节（论文的选标签规则，含 GRPO）。</p>\n')

# ---- 7.1: the fix has been run ------------------------------------------------------------------------------------------
rep('代码里 GRPO 的锚定损失已经有这种 <code>route_only</code> 掩码。这个修法还没有跑过。</li>',
    '代码里 GRPO 的锚定损失已经有这种 <code>route_only</code> 掩码。这个修法已经跑过，见第 8 节。</li>')

# ---- section 8: v2 / v3 --------------------------------------------------------------------------------------------------
sec8 = '''
<section id="fix">
  <h2><span class="secno">8</span>修掉草稿反馈环之后：v2 和 v3</h2>
  <p class="prose">7.1 节找到的问题出在 SFT 的训练目标：dynamic 的救回示范（call 行）把 manager 自己的错误草稿 <code>DRAFT_ANSWER_X</code> 原样写进响应，损失同时覆盖草稿和后面的调用。我们给 SFT 加了一个开关 <code>sft.draft_supervision</code>，用两种掩码各把前三轮重跑了一遍：</p>
  <ul class="plain">
    <li><strong>v2（<code>commit_rows</code>）：</strong>call 行只对调用决策计算损失，草稿字母不计；直接作答的 commit 行照旧（草稿就是最终答案，仍然监督）。</li>
    <li><strong>v3（<code>none</code>）：</strong>所有行都不对草稿计算损失，SFT 只训练"调用谁 / 不调用"这个路由决策，也就是代码里 GRPO 锚定损失已有的 <code>route_only</code> 掩码。第 0 节的更新和下面的结论都以 v3 为准，v2 只作为中间版本记录（8.4 节）。</li>
  </ul>
  <p class="prose">两次重跑都只做 SFT、不做 GRPO（第 6 节：GRPO 没有可靠贡献），两组各跑 3 轮：dynamic_sft 每轮用当前模型重新收集、重新选标签，static_sft 每轮复用 S₁ 的那份标签。重跑在一台新机器上进行，子模型的输出缓存从主实验恢复；S₁ 在这台机器上重新做了一次锁定测试：MedQA 78.4（主实验 78.6）、MMLU-Pro <code>test</code> 64.0（63.4）、<code>test_paper</code> 59.6（59.6）、GPQA 54.0（54.0）、AQuA 77.6（77.2），差别都不显著。本节和第 9 节所有"对 S₁"的比较都用这次重测。</p>

  <figure class="figure" id="fig-cost-v3" aria-label="成本与准确率：v3 重跑，四个基准">
    <div class="legend"></div>
    <p class="caption" style="margin:0 0 0.6rem">v3（不监督草稿、无 GRPO）。纵轴：锁定测试准确率（%）；横轴：每题调用次数。实心是第 3 轮，空心是第 5 轮；success 是第 4 节的 v1 结果，作为上限参考。</p>
    <div class="panels"></div>
    <div class="tooltip" hidden></div>
  </figure>
  <p class="caption">MMLU-Pro 用 <code>test</code>。GPQA 没有第 4–5 轮。</p>

  <h3>8.1 第 3 轮锁定测试</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>准确率 <span class="calls">每题调用</span></th><th class="num">MedQA</th><th class="num">MMLU-Pro test</th><th class="num">MMLU-Pro test_paper</th><th class="num">GPQA</th><th class="num">AQuA</th></tr></thead>
      <tbody>
        <tr><th>S₁（重测）</th><td class="num">78.4 <span class="calls">0.36</span></td><td class="num">64.0 <span class="calls">0.35</span></td><td class="num">59.6 <span class="calls">0.35</span></td><td class="num">54.0 <span class="calls">0.52</span></td><td class="num">77.6 <span class="calls">0.78</span></td></tr>
        <tr class="group"><td colspan="6">v1：论文的训练目标（监督草稿），含 GRPO（第 4 节）</td></tr>
        <tr><th>dynamic 第 3 轮</th><td class="num">77.6 <span class="calls">0.24</span></td><td class="num">64.0 <span class="calls">0.28</span></td><td class="num">62.8 <span class="calls">0.28</span></td><td class="num">50.0 <span class="calls">0.65</span></td><td class="num">80.3 <span class="calls">0.87</span></td></tr>
        <tr><th>static 第 3 轮</th><td class="num">79.0 <span class="calls">0.35</span></td><td class="num">58.4 <span class="calls">0.24</span></td><td class="num">56.2 <span class="calls">0.24</span></td><td class="num">48.0 <span class="calls">0.60</span></td><td class="num">76.0 <span class="calls">0.72</span></td></tr>
        <tr><th>success 第 3 轮</th><td class="num">80.2 <span class="calls">1.00</span></td><td class="num">71.0 <span class="calls">0.93</span></td><td class="num">69.0</td><td class="num">56.0 <span class="calls">1.00</span></td><td class="num">83.9 <span class="calls">1.00</span></td></tr>
        <tr class="group"><td colspan="6">v2：call 行不监督草稿，无 GRPO</td></tr>
        <tr><th>dynamic_sft 第 3 轮</th><td class="num">77.8 <span class="calls">0.25</span></td><td class="num">60.6 <span class="calls">0.24</span></td><td class="num">57.2 <span class="calls">0.23</span></td><td class="num">48.0 <span class="calls">0.58</span></td><td class="num">82.7 <span class="calls">0.89</span></td></tr>
        <tr><th>static_sft 第 3 轮</th><td class="num">79.8 <span class="calls">0.30</span></td><td class="num">59.0 <span class="calls">0.28</span></td><td class="num">57.0 <span class="calls">0.31</span></td><td class="num">51.0 <span class="calls">0.38</span></td><td class="num">68.9 <span class="calls">0.65</span></td></tr>
        <tr class="group"><td colspan="6">v3：完全不监督草稿，无 GRPO</td></tr>
        <tr><th>dynamic_sft 第 3 轮</th><td class="num">78.6 <span class="calls">0.15</span></td><td class="num">64.4 <span class="calls">0.27</span></td><td class="num">59.8 <span class="calls">0.27</span></td><td class="num">54.0 <span class="calls">0.58</span></td><td class="num">78.0 <span class="calls">0.79</span></td></tr>
        <tr><th>static_sft 第 3 轮</th><td class="num">78.6 <span class="calls">0.27</span></td><td class="num">58.0 <span class="calls">0.26</span></td><td class="num">59.0 <span class="calls">0.29</span></td><td class="num">55.0 <span class="calls">0.49</span></td><td class="num">74.8 <span class="calls">0.73</span></td></tr>
      </tbody>
    </table>
  </div>
  <p class="caption">MedQA 500 题，MMLU-Pro 两个测试集各 500 题，GPQA 100 题，AQuA 254 题。v1 的数字取自第 4 节（主实验在另一台机器上测，S₁ 两次测试差 0.0–0.6 个点）。</p>

  <h3>8.2 配对比较</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>准确率差（个点），括号里是 p</th><th class="num">MedQA</th><th class="num">MMLU-Pro test</th><th class="num">MMLU-Pro test_paper</th><th class="num">GPQA</th><th class="num">AQuA</th></tr></thead>
      <tbody>
        <tr class="group"><td colspan="6">v3</td></tr>
        <tr><th>dynamic − S₁</th><td class="num">+0.2 (0.94)</td><td class="num">+0.4 (0.84)</td><td class="num">+0.2 (0.97)</td><td class="num">0.0 (1.00)</td><td class="num">+0.4 (0.93)</td></tr>
        <tr><th>static − S₁</th><td class="num">+0.2 (0.91)</td><td class="num"><span class="sig">−6.0 (&lt;0.001)</span></td><td class="num">−0.6 (0.65)</td><td class="num">+1.0 (0.79)</td><td class="num">−2.8 (0.10)</td></tr>
        <tr><th>dynamic − static</th><td class="num">0.0 (1.00)</td><td class="num"><span class="sig">+6.4 (&lt;0.001)</span></td><td class="num">+0.8 (0.69)</td><td class="num">−1.0 (0.83)</td><td class="num">+3.1 (0.19)</td></tr>
        <tr><th>dynamic − v1 dynamic</th><td class="num">+1.0 (0.43)</td><td class="num">+0.4 (0.84)</td><td class="num">−3.0 (0.08)</td><td class="num">+4.0 (0.31)</td><td class="num">−2.4 (0.27)</td></tr>
        <tr><th>static − v1 static</th><td class="num">−0.4 (0.75)</td><td class="num">−0.4 (0.77)</td><td class="num"><span class="sig">+2.8 (0.03)</span></td><td class="num"><span class="sig">+7.0 (0.002)</span></td><td class="num">−1.2 (0.59)</td></tr>
        <tr class="group"><td colspan="6">v3：每题调用次数差</td></tr>
        <tr><th>dynamic − S₁</th><td class="num"><span class="sig">−0.21 (&lt;0.001)</span></td><td class="num"><span class="sig">−0.08 (&lt;0.001)</span></td><td class="num"><span class="sig">−0.07 (0.001)</span></td><td class="num">+0.06 (0.10)</td><td class="num">+0.01 (0.73)</td></tr>
        <tr><th>static − S₁</th><td class="num"><span class="sig">−0.09 (&lt;0.001)</span></td><td class="num"><span class="sig">−0.09 (&lt;0.001)</span></td><td class="num"><span class="sig">−0.06 (&lt;0.001)</span></td><td class="num">−0.03 (0.44)</td><td class="num"><span class="sig">−0.05 (0.02)</span></td></tr>
        <tr class="group"><td colspan="6">v2</td></tr>
        <tr><th>dynamic − S₁</th><td class="num">−0.6 (0.71)</td><td class="num"><span class="sig">−3.4 (0.03)</span></td><td class="num">−2.4 (0.17)</td><td class="num">−6.0 (0.13)</td><td class="num"><span class="sig">+5.1 (0.012)</span></td></tr>
        <tr><th>static − S₁</th><td class="num">+1.4 (0.15)</td><td class="num"><span class="sig">−5.0 (&lt;0.001)</span></td><td class="num">−2.6 (0.059)</td><td class="num">−3.0 (0.33)</td><td class="num"><span class="sig">−8.7 (&lt;0.001)</span></td></tr>
        <tr><th>v3 dynamic − v2 dynamic</th><td class="num">+0.8 (0.60)</td><td class="num"><span class="sig">+3.8 (0.009)</span></td><td class="num">+2.6 (0.10)</td><td class="num">+6.0 (0.067)</td><td class="num"><span class="sig">−4.7 (0.027)</span></td></tr>
      </tbody>
    </table>
  </div>
  <ul class="plain">
    <li><strong>v3 的 dynamic 没有一处低于 S₁，</strong>MedQA 和 MMLU-Pro 上调用显著减少，GPQA 和 AQuA 上调用不变。这个"同样的准确率、更少的调用"是修掉反馈环之后最稳的结论。</li>
    <li><strong>重新收集的价值只在 MMLU-Pro 上成立。</strong>static 复用 S₁ 的标签再训练两轮，在 MMLU-Pro <code>test</code> 上掉了 6.0 个点（v1 的 static 也掉了 5.0，这一点 v1 和 v3 一致）；dynamic 不掉。<code>test_paper</code> 上 static 不掉（−0.6），这和 <code>test_paper</code> 里 200 道题就是 dev、S₁ 的标签里有这些题有关（4.3 节）。在 104 道干净的 <code>test_paper</code> 题上，v3 的 dynamic 52.9、static 55.8、S₁ 53.8，三者都不显著；v1 的 dynamic 在这 104 题上是 60.6，比 v3 的 dynamic 高 7.7 个点（p = 0.04），但在 500 道 <code>test</code> 题上两者相同（64.0 对 64.4）。104 题太少，我们以 <code>test</code> 为准。</li>
    <li><strong>v1 里 dynamic 的两处优势没有保住。</strong>AQuA 第 3 轮 dynamic 比 static 高 3.1 个点（v1：4.3），p = 0.19；MMLU-Pro <code>test_paper</code> 上 dynamic 对 S₁ +0.2（v1：+3.2）。GPQA 上 v3 的两组都比 v1 好（+4.0、+7.0），其中 static 显著，但 100 题的 GPQA 只能作参考（第 10 节）。</li>
    <li><strong>固定调用次数的回放（第 5 节的方法）：</strong>v3 的 dynamic 第 3 轮只在 MedQA 上高于随机（回放 78.5，随机区间 73.5–77.0，p = 0.002），MMLU-Pro（p = 0.10）、GPQA、AQuA（p = 0.08）都落在随机区间内；static 在 MedQA（p = 0.013）和 AQuA（p = 0.011）上高于随机。v1 的 dynamic 在 MMLU-Pro 第 3 轮高于随机，v3 没有。</li>
  </ul>

  <h3 id="drift-fixed">8.3 草稿不再漂移</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>锁定测试</th><th class="num">MedQA 初稿准确率</th><th class="num">MedQA 初稿为 B（正确答案为 B：113 / 500）</th><th class="num">AQuA 初稿准确率</th><th class="num">AQuA 初稿为 B（正确答案为 B：58 / 254）</th></tr></thead>
      <tbody>
        <tr><th>S₁</th><td class="num">73.4</td><td class="num">142 (28%)</td><td class="num">36.2</td><td class="num">112 (44%)</td></tr>
        <tr class="group"><td colspan="5">v1（监督草稿）</td></tr>
        <tr><th>dynamic 第 3 轮</th><td class="num">71.6</td><td class="num">172 (34%)</td><td class="num">35.0</td><td class="num">158 (62%)</td></tr>
        <tr><th>dynamic 第 5 轮</th><td class="num">67.4</td><td class="num">213 (43%)</td><td class="num">35.0</td><td class="num">178 (70%)</td></tr>
        <tr><th>static 第 5 轮</th><td class="num">73.8</td><td class="num">135 (27%)</td><td class="num">39.0</td><td class="num">118 (46%)</td></tr>
        <tr class="group"><td colspan="5">v3（不监督草稿）</td></tr>
        <tr><th>dynamic 第 3 轮</th><td class="num">74.8</td><td class="num">129 (26%)</td><td class="num">35.0</td><td class="num">138 (54%)</td></tr>
        <tr><th>dynamic 第 5 轮</th><td class="num">73.4</td><td class="num">141 (28%)</td><td class="num">37.0</td><td class="num">98 (39%)</td></tr>
        <tr><th>static 第 3 轮</th><td class="num">75.6</td><td class="num">134 (27%)</td><td class="num">38.6</td><td class="num">125 (49%)</td></tr>
        <tr><th>static 第 5 轮</th><td class="num">74.6</td><td class="num">129 (26%)</td><td class="num">37.4</td><td class="num">124 (49%)</td></tr>
      </tbody>
    </table>
  </div>
  <p class="caption">v3 里 MedQA 的 dynamic 初稿准确率和 B 的数量在五轮里都保持在 S₁ 的水平（v1 第 5 轮：67.4、213）。AQuA 第 3 轮 B 的比例仍从 44% 升到 54%，但第 5 轮回到 39%，初稿准确率不变；v1 是一路升到 70%。MMLU-Pro 和 GPQA 在 v1 里就没有明显漂移，v3 也一样（dynamic 第 3 轮初稿准确率 51.0 / 41.0，S₁ 50.8 / 41.0）。</p>

  <h3 id="v2-tilt">8.4 v2 为什么偏向直接作答</h3>
  <p class="prose">v2 只在 call 行掩掉草稿，commit 行的草稿（也就是最终答案）照旧监督。这样每个 call 行只剩下几个 token 的调用损失，commit 行却仍有完整的答案损失，训练信号的重量偏向"直接作答"。效果在 dev 上很直接：MMLU-Pro 第 2 轮 dynamic 的调用率从 S₁ 的 0.31 掉到 0.07，第 3 轮回到 0.21；AQuA 的 static 每题调用从 0.78 降到 0.65，准确率比 S₁ 低 8.7 个点。v2 在 AQuA 上的 dynamic 是全部实验里 success 之外最高的数（82.7，对 S₁ +5.1，p = 0.012，每题调用 0.89），但同样的流程只换成 v3 的掩码就回到 78.0（对 v2 −4.7，p = 0.027），单次运行之间的这种差别我们不当作结论。v3 对所有行一视同仁，没有这个倾斜。</p>
</section>
'''

# ---- section 9: v3 rounds 4-5 -------------------------------------------------------------------------------------------
cm_t = MMLU["cmp_test"] or {}
cm_p = MMLU["cmp_paper"] or {}
dev = MMLU["dev"] or {"dyn": [None, None], "sta": [None, None]}


def dv(x):
    return PENDING if x is None else f'<td class="num">{x:.1f}</td>'


mm_bullet = MMLU["note"] or "<li><strong>MMLU-Pro：</strong>第 4–5 轮还在跑（约 4 小时），跑完补上。</li>"
sec9 = f'''
<section id="rounds-v3">
  <h2><span class="secno">9</span>v3 的第 4–5 轮</h2>
  <p class="prose">从 v3 第 3 轮的模型接着跑到第 5 轮，设置和第 7 节一样（dynamic 每轮 400 道新的收集题，static 复用 S₁ 的标签），没有 GRPO。GPQA 没有剩余的题，没有跑。</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>锁定测试</th><th class="num">MedQA</th><th class="num">AQuA</th><th class="num">MMLU-Pro test</th><th class="num">MMLU-Pro test_paper</th></tr></thead>
      <tbody>
        <tr class="group"><td colspan="5">准确率 / 每题调用</td></tr>
        <tr><th>S₁</th>{cell(78.4, 0.356)}{cell(77.6, 0.780)}{cell(64.0, 0.354)}{cell(59.6, 0.348)}</tr>
        <tr><th>dynamic 第 3 轮</th>{cell(78.6, 0.148)}{cell(78.0, 0.791)}{cell(64.4, 0.270)}{cell(59.8, 0.274)}</tr>
        <tr><th>dynamic 第 5 轮</th>{cell(77.4, 0.172)}{cell(71.7, 0.720)}{cell(MMLU["dyn5_test"], MMLU["dyn5_test_calls"])}{cell(MMLU["dyn5_paper"], MMLU["dyn5_paper_calls"])}</tr>
        <tr><th>static 第 3 轮</th>{cell(78.6, 0.270)}{cell(74.8, 0.728)}{cell(58.0, 0.262)}{cell(59.0, 0.286)}</tr>
        <tr><th>static 第 5 轮</th>{cell(81.0, 0.368)}{cell(77.2, 0.764)}{cell(MMLU["sta5_test"], MMLU["sta5_test_calls"])}{cell(MMLU["sta5_paper"], MMLU["sta5_paper_calls"])}</tr>
        <tr class="group"><td colspan="5">准确率差（个点），括号里是 p</td></tr>
        <tr><th>dynamic：第 5 − 第 3 轮</th>{dcell(-1.2, 0.33)}{dcell(-6.3, 0.0004)}{dcell(*cm_t.get("dyn53", (None, None)))}{dcell(*cm_p.get("dyn53", (None, None)))}</tr>
        <tr><th>static：第 5 − 第 3 轮</th>{dcell(2.4, 0.027)}{dcell(2.4, 0.21)}{dcell(*cm_t.get("sta53", (None, None)))}{dcell(*cm_p.get("sta53", (None, None)))}</tr>
        <tr><th>dynamic 第 5 轮 − S₁</th>{dcell(-1.0, 0.49)}{dcell(-5.9, 0.009)}{dcell(*cm_t.get("dyn5s1", (None, None)))}{dcell(*cm_p.get("dyn5s1", (None, None)))}</tr>
        <tr><th>static 第 5 轮 − S₁</th>{dcell(2.6, 0.023)}{dcell(-0.4, 0.93)}{dcell(*cm_t.get("sta5s1", (None, None)))}{dcell(*cm_p.get("sta5s1", (None, None)))}</tr>
        <tr><th>第 5 轮 dynamic − static</th>{dcell(-3.6, 0.008)}{dcell(-5.5, 0.027)}{dcell(*cm_t.get("dynsta5", (None, None)))}{dcell(*cm_p.get("dynsta5", (None, None)))}</tr>
        <tr><th>dynamic 第 5 轮 − v1 dynamic 第 5 轮</th>{dcell(1.0, 0.37)}{dcell(-6.3, 0.008)}{dcell(*cm_t.get("dyn5v1", (None, None)))}{dcell(*cm_p.get("dyn5v1", (None, None)))}</tr>
        <tr><th>static 第 5 轮 − v1 static 第 5 轮</th>{dcell(0.6, 0.54)}{dcell(-0.8, 0.76)}{dcell(*cm_t.get("sta5v1", (None, None)))}{dcell(*cm_p.get("sta5v1", (None, None)))}</tr>
      </tbody>
    </table>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>dev 准确率（每轮 SFT 模型）</th><th class="num">第 2 轮</th><th class="num">第 3 轮</th><th class="num">第 4 轮</th><th class="num">第 5 轮</th></tr></thead>
      <tbody>
        <tr><th>MedQA dynamic（S₁ 79.0）</th><td class="num">79.0</td><td class="num">79.0</td><td class="num">79.5</td><td class="num">76.0</td></tr>
        <tr><th>MedQA static</th><td class="num">78.0</td><td class="num">79.5</td><td class="num">80.5</td><td class="num">80.5</td></tr>
        <tr><th>AQuA dynamic（S₁ 73.2）</th><td class="num">74.8</td><td class="num">77.6</td><td class="num">70.9</td><td class="num">72.4</td></tr>
        <tr><th>AQuA static</th><td class="num">72.4</td><td class="num">73.2</td><td class="num">74.8</td><td class="num">75.6</td></tr>
        <tr><th>MMLU-Pro dynamic（S₁ 62.5）</th><td class="num">56.0</td><td class="num">62.5</td>{dv(dev["dyn"][0])}{dv(dev["dyn"][1])}</tr>
        <tr><th>MMLU-Pro static</th><td class="num">62.0</td><td class="num">61.0</td>{dv(dev["sta"][0])}{dv(dev["sta"][1])}</tr>
      </tbody>
    </table>
  </div>
  <ul class="plain">
    <li><strong>MedQA：</strong>static 第 5 轮 81.0，显著高于 S₁（+2.6）和自己的第 3 轮（+2.4），和 v1 的 80.4 一致，用每题 0.37 次调用追平 success 的 80.2（p = 0.61）。dynamic 第 5 轮 77.4，对 S₁ 不显著（−1.0），比 static 低 3.6 个点（p = 0.008）。这次的差距和草稿无关：dynamic 的初稿准确率（73.4）和 B 的数量（141）都和 S₁ 一样（8.3 节）。变化在路由上：第 5 轮的调用从 Reasoner 转向 Extractor（锁定测试上 Extractor / Reasoner / Verifier = 36 / 48 / 2 次，第 3 轮是 0 / 74 / 0），而第 5 轮重新收集出来的标签里 Extractor 的救回率从第 4 轮的 3% 升到 8%，模型学到了这个信号，但它在 dev 和锁定测试上不成立。固定调用次数的回放里，dynamic 第 5 轮不再高于随机（回放 76.0，区间 73.5–77.5，p = 0.37），static 第 5 轮高于随机（79.0，区间 72.0–78.5，p = 0.015）。</li>
    <li><strong>AQuA：</strong>dynamic 第 4 轮 dev 掉了 6.7 个点，第 5 轮锁定测试 71.7，对 S₁ −5.9（p = 0.009），对自己的第 3 轮 −6.3（p &lt; 0.001）。初稿准确率 37.0（S₁ 36.2），B 的比例 39%（S₁ 44%），也不是草稿的问题。它在初稿错的题上调用 Verifier 的比例从第 3 轮的 0.88 降到 0.79（S₁ 0.85），纠正率从 0.47 降到 0.39；回放里不高于随机（p = 0.15）。static 第 5 轮 77.2，和 S₁ 持平，回放高于随机（p = 0.002）。</li>
    {mm_bullet}
    <li><strong>怎么理解。</strong>修掉草稿反馈环之后，dynamic 在后两轮的退步只剩一个来源：用当前模型重新收集的 400 道题算出来的 one-step 边际价值标签本身有噪声，而每轮都把它学进去。static 的标签固定，在 MedQA 和 AQuA 上不变或变好，在 MMLU-Pro 上停在比 S₁ 低 6 个点的位置。两种规则各有适用的地方：S₁ 的标签在 MMLU-Pro 上不够好，重新收集才有用（dynamic 第 5 轮虽然比第 3 轮低 3 个点，仍高于 static）；在 MedQA 和 AQuA 上 S₁ 的标签已经够好，再收集反而引入噪声。第 7 节末尾关于跨第 3 / 4 轮比较的提醒这里不再适用，因为 v3 没有 GRPO，第 4–5 轮和前三轮的训练题来源相同。</li>
  </ul>
</section>
'''

rep('\n<section id="caveats">', sec8 + sec9 + '\n<section id="caveats">')
rep('<h2><span class="secno">8</span>局限和需要注明的地方</h2>', '<h2><span class="secno">10</span>局限和需要注明的地方</h2>')
rep('<h2><span class="secno">9</span>数据与代码</h2>', '<h2><span class="secno">11</span>数据与代码</h2>')

# ---- caveats additions --------------------------------------------------------------------------------------------------
rep('''    <li><strong>dynamic 的救回示范会复制 manager 自己的错误草稿。</strong>这在 MedQA 和 AQuA 上造成了初稿向 B 坍缩（7.1 节）。dynamic 的所有结果都带着这个效应；修掉它之后 dynamic 的数字可能会变。</li>''',
    '''    <li><strong>dynamic 的救回示范会复制 manager 自己的错误草稿。</strong>这在 MedQA 和 AQuA 上造成了初稿向 B 坍缩（7.1 节）。第 1–7 节 dynamic 的所有结果都带着这个效应；修掉之后的结果在第 8、9 节。</li>
    <li><strong>v2 / v3 是另一台机器上的单次运行，没有 GRPO。</strong>子模型的输出缓存从主实验恢复，所以同一道题的子模型回答和主实验一致；S₁ 重测和主实验差 0.0–0.6 个点。v2 和 v3 各只跑了一次，v2 与 v3 在 AQuA 上 4.7 个点的差别说明单次运行之间的波动可以到这个量级。MMLU-Pro 开跑前的子模型一致性检查里，Reasoner 的相似度 0.49 低于原来 0.5 的门槛，我们把门槛降到 0.45 放行（相似度只是检查，不影响训练）。</li>
    <li><strong>v2 / v3 里按无效答案规则放行的评估：</strong>MedQA v2 第 3 轮 static 的 dev 评估、v3 第 3 轮 static 的 dev 评估和锁定测试、v3 第 4、5 轮 static 的 dev 评估，各 1 道题。确认在运行开始前就按同一条规则（只放行无效答案率不低于 0.99 的情况）预先记录，其他任何失败仍会停下。</li>''')

# ---- data section --------------------------------------------------------------------------------------------------------
rep('''    <li>重新训练的 v2 子模型（未在本文实验中使用）：<code>MaliDDD/agent-routing-advisors-{bench}-9b-v2</code>。</li>''',
    '''    <li>v2 / v3 和第 4–5 轮的运行各自一个公开仓库：<code>MaliDDD/margent-mcq-rsi-&lt;run&gt;</code>，run 为 <code>medqa_v2</code>、<code>mmlu_pro_v2</code>、<code>gpqa_v2</code>、<code>aqua_v2</code>、<code>medqa_v3</code>、<code>mmlu_pro_v3</code>、<code>gpqa_v3</code>、<code>aqua_v3</code>、<code>medqa_v3r5</code>、<code>aqua_v3r5</code>、<code>mmlu_pro_v3r5</code>（原仓库已到 2 万个文件的上限）。配置在仓库的 <code>configs/mcq_rsi_&lt;bench&gt;_v2.json</code> / <code>_v3.json</code>，开关是 <code>sft.draft_supervision</code>；第 4–5 轮按手册第 17 节从 <code>&lt;bench&gt;_v3</code> 续跑。</li>
    <li>重新训练的 v2 子模型（未在本文实验中使用）：<code>MaliDDD/agent-routing-advisors-{bench}-9b-v2</code>。</li>''')
rep('''mmlu_pro_r5、aqua_r5、medqa_r5（第 4–5 轮）。所有数字取自''',
    '''mmlu_pro_r5、aqua_r5、medqa_r5（第 4–5 轮）；medqa_v2 … aqua_v2、medqa_v3 … aqua_v3（第 8 节）；medqa_v3r5、aqua_v3r5、mmlu_pro_v3r5（第 9 节）。所有数字取自''')

# ---- figure script: render two figures ----------------------------------------------------------------------------------
rep('''  var ORDER = ["s1", "dyn3", "dyn5", "dsft", "sta3", "sta5", "suc"];
  var NS = "http://www.w3.org/2000/svg";''',
    '''  var ORDER = ["s1", "dyn3", "dyn5", "dsft", "sta3", "sta5", "suc"];
  // v3 (section 8): draft_supervision none, no GRPO. S_1 re-tested on the same machine; success is the v1 result.
  var DATA_V3 = [
    { name: "MedQA", n: 500, pts: [
      { id: "s1", acc: 78.4, calls: 0.356 }, { id: "dyn3", acc: 78.6, calls: 0.148 }, { id: "sta3", acc: 78.6, calls: 0.270 },
      { id: "suc", acc: 80.2, calls: 1.000 }, { id: "dyn5", acc: 77.4, calls: 0.172 }, { id: "sta5", acc: 81.0, calls: 0.368 } ] },
    { name: "MMLU-Pro", sub: "test", n: 500, pts: [
      { id: "s1", acc: 64.0, calls: 0.354 }, { id: "dyn3", acc: 64.4, calls: 0.270 }, { id: "sta3", acc: 58.0, calls: 0.262 },
      { id: "suc", acc: 71.0, calls: 0.934 }__MMLU_V3_PTS__ ] },
    { name: "GPQA Diamond", n: 100, pts: [
      { id: "s1", acc: 54.0, calls: 0.520 }, { id: "dyn3", acc: 54.0, calls: 0.580 }, { id: "sta3", acc: 55.0, calls: 0.490 },
      { id: "suc", acc: 56.0, calls: 1.000 } ] },
    { name: "AQuA", n: 254, pts: [
      { id: "s1", acc: 77.6, calls: 0.780 }, { id: "dyn3", acc: 78.0, calls: 0.791 }, { id: "sta3", acc: 74.8, calls: 0.728 },
      { id: "suc", acc: 83.9, calls: 1.000 }, { id: "dyn5", acc: 71.7, calls: 0.720 }, { id: "sta5", acc: 77.2, calls: 0.764 } ] }
  ];
  var KIND_V3 = {
    s1:   { label: "S₁（重测）", color: "--neutral-mark", shape: "square", filled: true },
    dyn3: { label: "dynamic 第 3 轮（v3）", color: "--series-1", shape: "circle", filled: true },
    dyn5: { label: "dynamic 第 5 轮（v3）", color: "--series-1", shape: "circle", filled: false },
    sta3: { label: "static 第 3 轮（v3）", color: "--series-2", shape: "circle", filled: true },
    sta5: { label: "static 第 5 轮（v3）", color: "--series-2", shape: "circle", filled: false },
    suc:  { label: "success 第 3 轮（v1，第 4 节）", color: "--series-3", shape: "diamond", filled: true }
  };
  var ORDER_V3 = ["s1", "dyn3", "dyn5", "sta3", "sta5", "suc"];
  var NS = "http://www.w3.org/2000/svg";''')
mm_pts = ""
if MMLU["dyn5_test"] is not None:
    mm_pts = f', {{ id: "dyn5", acc: {MMLU["dyn5_test"]:.1f}, calls: {MMLU["dyn5_test_calls"]:.3f} }}, {{ id: "sta5", acc: {MMLU["sta5_test"]:.1f}, calls: {MMLU["sta5_test_calls"]:.3f} }}'
rep("__MMLU_V3_PTS__", mm_pts)

rep('''  function shape(g, kind, cx, cy, r) {
    var k = KIND[kind], col = "var(" + k.color + ")";''',
    '''  function shape(g, kind, cx, cy, r, KIND) {
    var k = KIND[kind], col = "var(" + k.color + ")";''')
rep('''  var legend = document.getElementById("legend");
  ORDER.forEach(function (id) {
    var span = document.createElement("span");
    var svg = el("svg", { width: 16, height: 16, viewBox: "0 0 16 16", "aria-hidden": "true" });
    shape(svg, id, 8, 8, 5);
    span.appendChild(svg);
    span.appendChild(document.createTextNode(KIND[id].label));
    legend.appendChild(span);
  });

  var W = 340, H = 230, M = { l: 38, r: 14, t: 10, b: 34 };
  var tip = document.getElementById("tip"), fig = document.getElementById("fig-cost");
  function showTip(html, target) {''',
    '''  var W = 340, H = 230, M = { l: 38, r: 14, t: 10, b: 34 };
  function renderFigure(figId, DATA, KIND, ORDER) {
  var fig = document.getElementById(figId);
  var legend = fig.querySelector(".legend"), tip = fig.querySelector(".tooltip"), panels = fig.querySelector(".panels");
  ORDER.forEach(function (id) {
    var span = document.createElement("span");
    var svg = el("svg", { width: 16, height: 16, viewBox: "0 0 16 16", "aria-hidden": "true" });
    shape(svg, id, 8, 8, 5, KIND);
    span.appendChild(svg);
    span.appendChild(document.createTextNode(KIND[id].label));
    legend.appendChild(span);
  });
  function showTip(html, target) {''')
rep('''  var panels = document.getElementById("panels");
  DATA.forEach(function (b) {''', '''  DATA.forEach(function (b) {''')
rep('''      shape(g, id, x(p.calls), y(p.acc), 5);''', '''      shape(g, id, x(p.calls), y(p.acc), 5, KIND);''')
rep('''    wrap.appendChild(svg); panels.appendChild(wrap);
  });
})();''', '''    wrap.appendChild(svg); panels.appendChild(wrap);
  });
  }
  renderFigure("fig-cost", DATA, KIND, ORDER);
  renderFigure("fig-cost-v3", DATA_V3, KIND_V3, ORDER_V3);
})();''')

# pending-cell style
rep('.calls { color: var(--ink-3); font-size: 0.85em; }',
    '.calls { color: var(--ink-3); font-size: 0.85em; }\n.pending { color: var(--ink-3); font-style: italic; }')

# description meta / title date
if '<meta name="description"' in src:
    pass
if "<meta charset" not in src:
    src = "<meta charset=\"utf-8\">\n" + src
(SP / "margent_rsi_results.html").write_text(src)
print("written", len(src), "bytes; sections:", src.count("<section "), "figures:", src.count('class="figure"'))
