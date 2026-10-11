#!/usr/bin/env python3
"""Second page update (2026-10-11): sections 10-13 on top of the page built by patch_page_v3.py.

Reads docs/mcq_rsi_results/margent_rsi_results.html (the v3 page: sections 0-11) and writes it back with, before the
caveats: 10 the v3 grid WITH GRPO, 11 the token accounting with the untrained base manager, 12 the analyses (label
quality by round, G_k vs S_k data, the oracle, the self-check), and the caveats / data sections renumbered to 13 / 14.
Numbers come from the locked tests and dev evals recorded in results/*.tar.gz (paired tests via
scripts/mcq_rsi_analysis.py compare, tokens via scripts/mcq_token_cost.py).
"""
import re
from pathlib import Path

SP = Path(__file__).resolve().parent.parent  # docs/mcq_rsi_results/
page = SP / "margent_rsi_results.html"
src = page.read_text()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


if "<section id=\"grpo-v3\">" in src:
    raise SystemExit("already applied")

# ---- nav --------------------------------------------------------------------------------------------------------------
rep('<a href="#rounds-v3">v3 的第 4–5 轮</a><a href="#caveats">局限</a><a href="#data">数据与代码</a>',
    '<a href="#rounds-v3">v3 的第 4–5 轮</a><a href="#grpo-v3">v3 + GRPO</a><a href="#tokens">token 成本</a><a href="#analyses">数据、天花板、自检</a>'
    '<a href="#caveats">局限</a><a href="#data">数据与代码</a>')

# ---- summary: second update block --------------------------------------------------------------------------------------
summary2 = '''
  <div class="callout">
    <p><strong>10 月 11 日更新（第 10–12 节）。</strong>三件事补齐了：v3 设置下带 GRPO 的完整网格、每道题的 token 账本（含一个没训过的 base manager 基线）、以及三组解释性分析。</p>
    <ul class="plain">
      <li><strong>GRPO 只在 MMLU-Pro 上有效。</strong>掩掉草稿后带 GRPO 的 dynamic 在 MMLU-Pro <code>test</code> 上 68.6，比 S₁ 高 5.0（p = 0.003），比无 GRPO 的 dynamic 高 4.2（p = 0.003），调用不变；MedQA（7 次全拒）、GPQA、AQuA 上有无 GRPO 没有差别。</li>
      <li><strong>训练压掉的是 token，不是准确率。</strong>没训过的 manager 每题把三个子模型全问一遍（MedQA 每题 11,770 token），论文的一轮 SFT 压到 2,191，RSI 再压到 1,506（−31%，解码 token −60%）；MMLU-Pro −14%，GPQA、AQuA 没有可省的。</li>
      <li><strong>重采集的数据不随轮数变好，GRPO 后的模型收集的数据也不更好</strong>（固定起点 S₁ 各训一次的对照）；天花板（逐题最优路径）不随轮数变化，策略也没有向它靠近；manager 自己的置信度只在 MedQA 上能预测"草稿错"（AUROC 0.72），其他基准接近随机。</li>
    </ul>
  </div>
'''
rep('  <p class="caption">下面的速览是 10 月 3 日的版本，对应第 1–7 节（论文的选标签规则，含 GRPO）。</p>\n',
    summary2 + '  <p class="caption">下面的速览是 10 月 3 日的版本，对应第 1–7 节（论文的选标签规则，含 GRPO）。</p>\n')

# ---- section 10: v3 with GRPO ------------------------------------------------------------------------------------------
sec10 = '''
<section id="grpo-v3">
  <h2><span class="secno">10</span>v3 设置下带 GRPO 的网格</h2>
  <p class="prose">第 8 节去掉了 GRPO 来隔离掩码的效果。这里把 GRPO 加回来：掩掉草稿的 SFT 目标，每轮 SFT 之后照论文的规则做一次 FA-GRPO（验收条件不变），dynamic、static、success 三组，3 轮，四个基准。MedQA、GPQA、AQuA 是三组在一个运行里（共用第 1 轮的 GRPO 和第 2 轮的收集）；MMLU-Pro 因为第一台机器中途停机，改为三组各自一个运行并行（各自做第 1 轮 GRPO 和第 2 轮收集），第 1 轮 GRPO 在 MMLU-Pro 上本来就被拒，所以三组都从 S₁ 起步。</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>锁定测试 准确率 <span class="calls">每题调用</span></th><th class="num">MedQA</th><th class="num">MMLU-Pro test</th><th class="num">MMLU-Pro test_paper</th><th class="num">GPQA</th><th class="num">AQuA</th></tr></thead>
      <tbody>
        <tr><th>S₁（重测）</th><td class="num">78.8 <span class="calls">0.37</span></td><td class="num">63.6 <span class="calls">0.35</span></td><td class="num">59.6 <span class="calls">0.35</span></td><td class="num">54.0 <span class="calls">0.52</span></td><td class="num">77.2 <span class="calls">0.78</span></td></tr>
        <tr><th>dynamic + GRPO</th><td class="num">78.6 <span class="calls">0.17</span></td><td class="num"><span class="sig">68.6</span> <span class="calls">0.36</span></td><td class="num"><span class="sig">64.6</span> <span class="calls">0.39</span></td><td class="num">55.0 <span class="calls">0.60</span></td><td class="num">77.6 <span class="calls">0.77</span></td></tr>
        <tr><th>static + GRPO</th><td class="num">79.8 <span class="calls">0.28</span></td><td class="num">61.4 <span class="calls">0.26</span></td><td class="num">58.0 <span class="calls">0.28</span></td><td class="num">50.0 <span class="calls">0.47</span></td><td class="num">73.6 <span class="calls">0.72</span></td></tr>
        <tr><th>success + GRPO</th><td class="num">80.2 <span class="calls">0.99</span></td><td class="num">71.8 <span class="calls">1.00</span></td><td class="num">69.0 <span class="calls">1.00</span></td><td class="num">58.0 <span class="calls">1.00</span></td><td class="num">82.7 <span class="calls">1.00</span></td></tr>
        <tr class="group"><td colspan="6">准确率差（个点），括号里是 p</td></tr>
        <tr><th>dynamic + GRPO − S₁</th><td class="num">−0.2 (0.94)</td><td class="num"><span class="sig">+5.0 (0.003)</span></td><td class="num"><span class="sig">+5.0 (0.002)</span></td><td class="num">+1.0 (0.84)</td><td class="num">+0.4 (0.92)</td></tr>
        <tr><th>dynamic + GRPO − dynamic（第 8 节，无 GRPO）</th><td class="num">0.0 (1.00)</td><td class="num"><span class="sig">+4.2 (0.003)</span></td><td class="num"><span class="sig">+4.8 (&lt;0.001)</span></td><td class="num">+1.0 (0.78)</td><td class="num">−0.4 (0.93)</td></tr>
        <tr><th>static + GRPO − static（无 GRPO）</th><td class="num">+1.2 (0.13)</td><td class="num"><span class="sig">+3.4 (0.02)</span></td><td class="num">−1.0 (0.47)</td><td class="num">−5.0 (0.12)</td><td class="num">−1.2 (0.52)</td></tr>
        <tr><th>success + GRPO − S₁</th><td class="num">+1.4 (0.31)</td><td class="num"><span class="sig">+8.2 (&lt;0.001)</span></td><td class="num"><span class="sig">+9.4 (&lt;0.001)</span></td><td class="num">+4.0 (0.39)</td><td class="num"><span class="sig">+5.5 (0.016)</span></td></tr>
        <tr><th>GRPO 候选被采用</th><td class="num">0 / 7</td><td class="num">dyn 2/3, static 2/3, success 2/3</td><td class="num"></td><td class="num">5 / 7</td><td class="num">6 / 7</td></tr>
      </tbody>
    </table>
  </div>
  <ul class="plain">
    <li><strong>MMLU-Pro 是唯一有收益的基准：</strong>dynamic + GRPO 68.6，比 S₁ 和无 GRPO 的 dynamic 都高 4–5 个点，调用与 S₁ 相同（0.36），每题 token 也相同（2,355 对 2,362，第 11 节）。固定调用次数的回放里它的挑题显著高于随机（0.645，随机区间 0.545–0.615，p &lt; 0.001），这是 v3 里第一次在 MMLU-Pro 上看到真正的路由能力。它和 v1 第 5 轮的 67.0（第 7 节）不显著差（+1.6），在 104 道干净的 <code>test_paper</code> 题上比 S₁ 高 2.9（不显著）。</li>
    <li><strong>MedQA 上 GRPO 训不动：</strong>7 次候选全部因"没有训练信号"被拒（修订后的结果几乎非 0 即 1），和 v1 完全一样，所以这组就是第 8 节的复现（dynamic 78.6 对 78.6）。</li>
    <li><strong>GPQA、AQuA 上候选大多被采用但没有收益：</strong>GPQA 5/7、AQuA 6/7 被采用，锁定测试上与无 GRPO 版本的差都不显著。</li>
    <li>success 带 GRPO 的数字和 v1 的 success 一致（MMLU-Pro 71.8 对 71.0，AQuA 82.7 对 83.9），仍然是每题都调用。</li>
  </ul>
  <p class="caption">MedQA 500 题、MMLU-Pro 两个测试集各 500 题、GPQA 100 题、AQuA 254 题。S₁ 在每个运行里重测一次，与第 8 节的重测差 0.0–0.4 个点。MMLU-Pro 三个运行各自重测了 S₁，数字相同。</p>
</section>
'''

# ---- section 11: tokens ---------------------------------------------------------------------------------------------------
sec11 = '''
<section id="tokens">
  <h2><span class="secno">11</span>每道题的 token：训练压掉的是成本</h2>
  <p class="prose">把每个策略在锁定测试上的完整轨迹按当时的上下文重新渲染，用 Qwen 的 tokenizer 数四项：manager 读入的（prefill）、manager 写出的、子模型读入的、子模型写出的；子模型的完整输出来自缓存。总 token 是四项之和，解码 token 是两项"写出的"之和，决定延迟。另外补了一个没训过的基线：基础模型直接放在 manager 的位置，同样的提示词、工具和子模型，在锁定测试上各评一次（不在任何运行的预注册里）。</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>每题 token（总 / 解码）<br><span class="calls">准确率 · 每题调用</span></th><th class="num">MedQA</th><th class="num">MMLU-Pro test</th><th class="num">GPQA</th><th class="num">AQuA</th></tr></thead>
      <tbody>
        <tr><th>只写草稿（下界）</th><td class="num">1065 / 10</td><td class="num">1132 / 10</td><td class="num">1052 / 10</td><td class="num">956 / 10</td></tr>
        <tr><th>未训练的 base manager<br><span class="calls">准确率 · 调用 · 可解析率</span></th><td class="num">11770 / 2184<br><span class="calls">82.8 · 2.98 · 95%</span></td><td class="num">9545 / 1780<br><span class="calls">69.0 · 2.51 · 85%</span></td><td class="num">10098 / 2184<br><span class="calls">39.0 · 2.53 · 52%</span></td><td class="num">5310 / 1065<br><span class="calls">78.3 · 1.62 · 84%</span></td></tr>
        <tr><th>S₁（论文的一轮 SFT）</th><td class="num">2191 / 294<br><span class="calls">78.4 · 0.36</span></td><td class="num">2373 / 318<br><span class="calls">64.0 · 0.35</span></td><td class="num">2807 / 496<br><span class="calls">54.0 · 0.52</span></td><td class="num">3180 / 587<br><span class="calls">77.6 · 0.78</span></td></tr>
        <tr><th>v3 dynamic 第 3 轮</th><td class="num"><span class="sig">1506 / 117</span><br><span class="calls">78.6 · 0.15</span></td><td class="num"><span class="sig">2040 / 239</span><br><span class="calls">64.4 · 0.27</span></td><td class="num">2983 / 540<br><span class="calls">54.0 · 0.58</span></td><td class="num">3204 / 594<br><span class="calls">78.0 · 0.79</span></td></tr>
        <tr><th>v3 static 第 3 轮</th><td class="num">1961 / 238<br><span class="calls">78.6 · 0.27</span></td><td class="num">2045 / 236<br><span class="calls">58.0 · 0.26</span></td><td class="num">2694 / 462<br><span class="calls">55.0 · 0.49</span></td><td class="num">3061 / 556<br><span class="calls">74.8 · 0.73</span></td></tr>
        <tr><th>v3 dynamic + GRPO（第 10 节）</th><td class="num">1568 / 133<br><span class="calls">78.6 · 0.17</span></td><td class="num">2355 / 322<br><span class="calls">68.6 · 0.36</span></td><td class="num">3053 / 559<br><span class="calls">55.0 · 0.60</span></td><td class="num">3127 / 571<br><span class="calls">77.6 · 0.77</span></td></tr>
        <tr><th>success（每题都问）</th><td class="num">3969 / 700<br><span class="calls">80.2 · 1.00</span></td><td class="num">4137 / 751<br><span class="calls">71.0 · 0.93</span></td><td class="num">4553 / 979<br><span class="calls">56.0 · 1.00</span></td><td class="num">3738 / 721<br><span class="calls">83.9 · 1.00</span></td></tr>
        <tr class="group"><td colspan="5">对 S₁ 的变化（总 token），配对 bootstrap</td></tr>
        <tr><th>base manager</th><td class="num"><span class="sig">+437%</span></td><td class="num"><span class="sig">+302%</span></td><td class="num"><span class="sig">+260%</span></td><td class="num"><span class="sig">+67%</span></td></tr>
        <tr><th>v3 dynamic 第 3 轮</th><td class="num"><span class="sig">−31%（解码 −60%）</span></td><td class="num"><span class="sig">−14%（解码 −25%）</span></td><td class="num">+6% (0.12)</td><td class="num">+1% (0.78)</td></tr>
        <tr><th>v3 static 第 3 轮</th><td class="num"><span class="sig">−11%</span></td><td class="num"><span class="sig">−14%</span></td><td class="num">−4% (0.34)</td><td class="num"><span class="sig">−4% (0.04)</span></td></tr>
        <tr><th>success</th><td class="num"><span class="sig">+76%</span></td><td class="num"><span class="sig">+73%</span></td><td class="num"><span class="sig">+60%</span></td><td class="num"><span class="sig">+18%</span></td></tr>
      </tbody>
    </table>
  </div>
  <ul class="plain">
    <li><strong>三级成本。</strong>没训过的 manager 每题把三个子模型全问一遍（MedQA 493/500 题按 Extractor → Reasoner → Verifier），自己还写得长（每题 179–520 个输出 token，S₁ 是 27–51），GPQA 上一半的题写到长度上限没给出答案。论文的一轮 SFT 把成本压掉 72–81%（AQuA 40%）；RSI 在 MedQA 上再压 31%、MMLU-Pro 14%。准确率的变化只发生在第一级：MedQA 和 MMLU-Pro 让出 4–5 个点（"每题全问"换来的那部分），GPQA 因为格式稳定反而涨 15 个点。</li>
    <li><strong>省 call 和省 token 的比例不同。</strong>每题约 1,000 token 是带工具 schema 的提示词，不调用也要付，所以 MedQA 调用 −58% 对应总 token −31%、解码 token −60%。</li>
    <li><strong>"call"不是统一单位。</strong>一次 Extractor 约 0.55–1.0k token（子模型侧），Reasoner 0.85–1.4k，Verifier 1.1–1.4k，再加 manager 回读子模型输出约 2k 的 prefill。</li>
    <li><strong>买准确率的汇率。</strong>success 每提高 1 个点要多花 MedQA 约 1,100、MMLU-Pro 230、GPQA 850、AQuA 85 token；MMLU-Pro 上 dynamic + GRPO 的 +5 个点不花额外 token。</li>
  </ul>
  <p class="caption">base manager 的"准确率"把无法解析的答案计为错，所以 GPQA 的 39.0 对应可解析题上的 76.0。数字和脚本：<code>docs/mcq_rsi_results/results/token_accounting_*.tar.gz</code>、<code>scripts/mcq_token_cost.py</code>、<code>scripts/mcq_token_compare.py</code>。</p>
</section>
'''

# ---- section 12: analyses --------------------------------------------------------------------------------------------------
sec12 = '''
<section id="analyses">
  <h2><span class="secno">12</span>三组解释性分析：数据、天花板、自检</h2>
  <h3>12.1 重采集的数据不随轮数变好</h3>
  <p class="prose">把 v3 每一轮收集出来的标签单独拿出来，都从同一个起点 S₁ 训一次，在 dev 上比（去掉链式训练带来的模型漂移）：</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>用哪一轮的标签训 S₁</th><th class="num">MedQA</th><th class="num">MMLU-Pro</th><th class="num">GPQA</th><th class="num">AQuA</th></tr></thead>
      <tbody>
        <tr><th>S₁ 本身（不训练）</th><td class="num">79.0 <span class="calls">0.37</span></td><td class="num">62.5 <span class="calls">0.32</span></td><td class="num">53.0 <span class="calls">0.62</span></td><td class="num">73.2 <span class="calls">0.79</span></td></tr>
        <tr><th>第 1 轮标签（static 的数据）</th><td class="num">78.0 <span class="calls">0.25</span></td><td class="num">61.5 <span class="calls">0.26</span></td><td class="num">55.0 <span class="calls">0.69</span></td><td class="num">72.4 <span class="calls">0.74</span></td></tr>
        <tr><th>第 2 轮重采（S₁ 收集）</th><td class="num">79.0 <span class="calls">0.29</span></td><td class="num">56.5 <span class="calls">0.11</span></td><td class="num">53.0 <span class="calls">0.62</span></td><td class="num">72.8 <span class="calls">0.78</span></td></tr>
        <tr><th>第 3 轮重采（S₂ 收集）</th><td class="num">76.5 <span class="calls">0.09</span></td><td class="num">63.0 <span class="calls">0.23</span></td><td class="num">52.0 <span class="calls">0.56</span></td><td class="num">73.6 <span class="calls">0.82</span></td></tr>
        <tr><th>第 4 轮重采（S₃ 收集）</th><td class="num">79.0 <span class="calls">0.19</span></td><td class="num">64.0 <span class="calls">0.23</span></td><td class="num">–</td><td class="num">77.2 <span class="calls">0.80</span></td></tr>
        <tr><th>第 5 轮重采（S₄ 收集）</th><td class="num">76.0 <span class="calls">0.20</span></td><td class="num">63.5 <span class="calls">0.21</span></td><td class="num">–</td><td class="num">78.7 <span class="calls">0.90</span></td></tr>
      </tbody>
    </table>
  </div>
  <ul class="plain">
    <li>三个基准都在 S₁ ±1.5 个点内；MedQA 后几轮的标签让模型少调用、掉 2.5–3 个点。原因在标签成分：收集模型自己答对的越多，救回示范越少（MedQA 88 → 73 → 64 → 64 行），ρ 固定时整份数据跟着变小。MMLU-Pro 第 2 轮那份标签本身就带着调用崩塌（训出来调用率 0.11），说明 v3 链条第 2 轮的下滑是数据带的，不是训练偶然。</li>
    <li>AQuA 是例外：第 4、5 轮标签从 S₁ 出发 +4 到 +5.5，但同一份第 5 轮标签在链条里训出的 S₅ 只有 72.4。AQuA 的退步来自逐轮累积训练，不是数据；"每轮都从 S₁ 重训"在那里反而更好（单次 dev 评测，±3 个点的噪声）。</li>
    <li><strong>GRPO 后的模型收集的数据也不更好。</strong>同一轮、同一批题，G_k 收集的标签对 S_k 收集的标签（都从 S₁ 训一次）：MMLU-Pro 第 2 轮 65.0 对 65.0；AQuA 第 2 轮 74.4 对 75.6，第 1 轮 69.3 对 72.4；GPQA 第 1 轮 58.0 对 54.0（100 题）。方向不一致，幅度在噪声内。</li>
  </ul>
  <h3>12.2 天花板不动，策略也没有向它靠近</h3>
  <p class="prose">用强制调用评测算 dev 上每道题的最优路径（直接作答或调某个子模型，取最好的），和自由策略放在同一批题上：</p>
  <div class="table-wrap">
    <table>
      <thead><tr><th>dev 准确率</th><th class="num">只写草稿</th><th class="num">策略 S₁ → dynamic 第 3 轮 → 第 5 轮</th><th class="num">永远调 Verifier</th><th class="num">逐题最优 oracle（S₁ → 第 3 轮 → 第 5 轮）</th></tr></thead>
      <tbody>
        <tr><th>MedQA (200)</th><td class="num">70.5</td><td class="num">79.0 → 79.0 → 76.0</td><td class="num">86.5</td><td class="num">92.0 → 92.0 → 92.5</td></tr>
        <tr><th>MMLU-Pro (200)</th><td class="num">55.5</td><td class="num">62.5 → 62.5 → 61.5</td><td class="num">67.0</td><td class="num">79.0 → 77.5 → 76.5</td></tr>
        <tr><th>GPQA (100)</th><td class="num">41.0</td><td class="num">52.0 → 53.0</td><td class="num">52.0</td><td class="num">61.0 → 63.0</td></tr>
        <tr><th>AQuA (254)</th><td class="num">39.4</td><td class="num">73.2 → 77.6 → 72.4</td><td class="num">77.2</td><td class="num">81.9 → 82.3 → 81.9</td></tr>
      </tbody>
    </table>
  </div>
  <p class="caption">oracle 由冻结的子模型决定，RSI 只训练路由，所以它不随轮数变；收集池上看似上涨的 MMLU-Pro oracle（79 → 83）是每轮换了一批题的抽样差异，同一批 dev 题上没有。策略离天花板 9–16 个点，RSI 没有缩小这个距离；"永远调 Verifier"这个零训练策略在 MedQA、MMLU-Pro 上比学到的路由高 4.5–7.5 个点。</p>
  <h3>12.3 manager 有没有"我错了"的信号</h3>
  <div class="table-wrap">
    <table>
      <thead><tr><th>S₁ 在 dev 上</th><th class="num">草稿字母置信度预测"草稿错"的 AUROC</th><th class="num">8 次采样不一致的 AUROC</th><th class="num">置信度阈值路由在 S₁ 预算下的准确率（S₁ 自己）</th></tr></thead>
      <tbody>
        <tr><th>MedQA</th><td class="num"><span class="sig">0.72</span></td><td class="num">0.58</td><td class="num"><span class="sig">82.5</span>（79.0）</td></tr>
        <tr><th>MMLU-Pro</th><td class="num">0.47</td><td class="num">0.60</td><td class="num">59.0（62.5）</td></tr>
        <tr><th>GPQA</th><td class="num">0.65</td><td class="num">0.66</td><td class="num">49.0（52.0）</td></tr>
        <tr><th>AQuA</th><td class="num">0.58</td><td class="num">0.56</td><td class="num">70.9（73.2）</td></tr>
      </tbody>
    </table>
  </div>
  <p class="caption">0.5 = 没有信号。只有 MedQA 上 manager 有可用的自知：一个零训练的置信度阈值在 S₁ 的调用预算下比学到的路由高 3.5 个点，说明边际价值目标没有把这个信号用尽；但在 dynamic 压缩后的低预算（0.14）下，学到的路由反而高 3.5 个点。其他三个基准上置信度接近随机，学到的路由已经和任何便宜信号能做到的差不多。采样一致性几乎没用：这个 manager 太"确定"，温度 1.0 下 8 次采样的一致率对错题 0.93、对对题 0.97。这直接支撑第 8、9 节的机制：模型分不清自己什么时候错，RSI 只能学到"什么时候别问"。</p>
  <p class="caption">脚本：<code>scripts/mcq_label_quality.py</code>（12.1）、<code>scripts/pod_ops/grpoq_chain.sh</code>（G_k 对 S_k）、<code>scripts/mcq_self_check.py</code>（12.3）；dev 上的 oracle 来自各运行 <code>final/*/dev_forced_*</code> 的强制调用评测。</p>
</section>
'''

rep('\n<section id="caveats">', sec10 + sec11 + sec12 + '\n<section id="caveats">')
rep('<h2><span class="secno">10</span>局限和需要注明的地方</h2>', '<h2><span class="secno">13</span>局限和需要注明的地方</h2>')
rep('<h2><span class="secno">11</span>数据与代码</h2>', '<h2><span class="secno">14</span>数据与代码</h2>')
rep('GPQA 上 v3 的两组都比 v1 好（+4.0、+7.0），其中 static 显著，但 100 题的 GPQA 只能作参考（第 10 节）。',
    'GPQA 上 v3 的两组都比 v1 好（+4.0、+7.0），其中 static 显著，但 100 题的 GPQA 只能作参考（第 13 节）。')

# ---- caveats additions -----------------------------------------------------------------------------------------------------
rep('''    <li><strong>v2 / v3 里按无效答案规则放行的评估：</strong>''',
    '''    <li><strong>第 10–12 节的运行跨了两台机器。</strong>带 GRPO 的 MedQA、GPQA、AQuA 和全部分析实验在第三台机器上跑；它在 MMLU-Pro 的锁定测试中途停机（信用额度用完），MMLU-Pro 带 GRPO 的三组和 base manager 基线在第四台机器上重跑，MMLU-Pro 的三组因此是三个独立运行而不是共用第 1 轮 GRPO 和第 2 轮收集的一个运行。第三台机器上四个带 GRPO 运行的模型权重和收集记录没有备份成功，只保留了锁定测试的逐题记录、各轮的 dev 结果和 GRPO 验收记录。</li>
    <li><strong>base manager 基线不在预注册里</strong>，是在锁定测试上额外评测一次的参照；它的准确率要连同可解析率一起看。token 是用 tokenizer 离线数的，不含被关闭的思维链。</li>
    <li><strong>v2 / v3 里按无效答案规则放行的评估：</strong>''')

# ---- data section ------------------------------------------------------------------------------------------------------------
rep('''<code>medqa_v3r5</code>、<code>aqua_v3r5</code>、<code>mmlu_pro_v3r5</code>（原仓库已到 2 万个文件的上限）。''',
    '''<code>medqa_v3r5</code>、<code>aqua_v3r5</code>、<code>mmlu_pro_v3r5</code>、<code>mmlu_pro_v3g_dyn</code>、<code>mmlu_pro_v3g_sta</code>、<code>mmlu_pro_v3g_suc</code>、<code>base_manager</code>，以及分析实验 <code>&lt;bench&gt;_v3lq</code>（12.1）、<code>&lt;bench&gt;_v3gq</code>（G_k 对 S_k）、<code>&lt;bench&gt;_v3sc</code>（12.3）（原仓库已到 2 万个文件的上限）。带 GRPO 的 MedQA、GPQA、AQuA 运行只有本仓库 <code>docs/mcq_rsi_results/results/</code> 里的锁定测试记录。''')

page.write_text(src)
print("written", len(src), "bytes; sections:", src.count("<section "), "secno:", re.findall(r'<span class="secno">(\d+)</span>', src))
