# MARGENT Agent RSI：完整实验计划

设计日期：2026-09-26；2026-09-29更新为专家优先执行，参见 [EXPERT_SFT_RUNPOD.md](EXPERT_SFT_RUNPOD.md)。

本文是执行和论文分析计划，不是已完成的实验报告。新增计划不会自动启动 RunPod。数学 subagent 的详细训练规范见 [SUBAGENT_TRAINING_PLAN.md](SUBAGENT_TRAINING_PLAN.md)；已有运行入口见 [RSI_RUNPOD.md](RSI_RUNPOD.md)。

## 1. 要回答的问题

核心假设：Manager 能力变化后，重新计算“调用 advisor 是否有帮助”，再交替做 SFT 与 GRPO，可以促进独立解题能力增长，同时保持有选择的委派。

预先定义三个问题：

1. 更新后的 Manager 生成新训练经验，是否优于一直复用初始经验？
2. MARGENT 的路线选择是否优于普通成功轨迹蒸馏？
3. 改善是否出现在没有 advisor 的独立解题上，而不仅仅来自更多调用？

这里的 RSI 指“更新后的模型参与生成下一轮学习经验”的有限轮次参数自改进。两轮循环不能证明自主改进训练算法、无限递归进步或自加速。

## 2. 已实现、已跑通与待实现

| 项目 | 本次计划制定时的状态 |
|---|---|
| 数学 Manager collect → SFT → GRPO → recollect | 已有两轮三组计划和实现，入口 src.verifiable.rsi |
| Manager LoRA、GRPO 逐步日志、checkpoint 延续 | 有实现；GPU 小测试完成一次 SFT、一次 GRPO、再一次 SFT |
| 当前数学 advisor | 一个冻结 Qwen3.5-9B，通过三个角色提示提供 extractor/reasoner/verifier；不是三个已训练专家 |
| 旧 subagent SFT | src/subagents/train.py 有训练器；不能据此声称数学专家训练与服务集成已完成 |
| 数学三角色独立 adapter、数据质检、角色评测、W&B | 数据/训练/服务/日志已实现，本地小模型验证；9B GPU与人工质量待验证 |
| AIME2026 全量基线 | 已启动但失败，不能算完成 |
| 完整两轮 benchmark 前后提升 | 尚无结果 |
| Manager 与 subagent 同时演化 | 仅作为后续扩展设计，当前无此流水线 |

已保存的小测试：1 个 GRPO group、4 条轨迹、reward_mean=0.75、valid_rate=0.75、1 条 revision_answer_format。它证明存在组内奖励差异，不证明泛化提升。next_sft 复用旧目标，只验证权重能继续训练，没有执行第二轮 recollect → GRPO。

AIME 失败记录：[父运行](https://wandb.ai/yuningyangaillm/MATH_rsi/runs/f97dff0ba7a6)、[AIME 子运行](https://wandb.ai/yuningyangaillm/MATH_rsi/runs/06e7dbcdaf67)。当时父运行失败阶段为 aime2026_all_30，子运行完成题数为 0/30；详细 traceback 仍需从 /workspace/margent-aime-baseline-01/logs/aime2026.log 定位。不能把它解释成模型准确率为零。

## 3. 模型、训练对象与完整顺序

模型固定为 Qwen/Qwen3.5-9B，revision c202236235762e1c871ad0ccb60c8ee5ba337b9a。两张 A100 80GB：通常 GPU 0 服务 advisor，GPU 1 采集、评估及训练 Manager。

### 主实验的训练对象

- Manager：训练 LoRA，包含独立解题 SFT、调用/提交决策 SFT、修订 SFT，以及决策/修订 GRPO。
- 三个 subagent：主实验首先分别做角色 SFT，随后冻结；Manager 训练期间不更新它们。
- 外部数学答案校验器：固定，不训练。verifier subagent 的口头 verdict 不是奖励依据。
- 按用户确认顺序，先做三专家SFT和质量检查，再启动Manager；prompt-only保留为单独消融。两套 advisor 条件分别报告，不能中途替换后拼接学习曲线。

### 完整主线

    准备并锁定数据
        ↓
    [首先执行] 分别训练 extractor / reasoner / verifier → 评估 → 冻结专家
        ↓
    评估初始 Manager M0（独立 + 允许委派）
        ↓
    用 M0 生成首轮反事实树 → 选择 SFT 目标 D0
        ↓
    SFT1 → dev 评估 → GRPO1 → dev 评估
        ↓
    用 GRPO1 后的 M1 重新采集 D1
        ↓
    从 GRPO1 adapter 继续 SFT2 → dev 评估 → GRPO2 → dev 评估
        ↓
    锁定最终 checkpoint → AIME2026 / BeyondAIME 评测与对照分析

三个训练组各有自己的 Manager 权重。SFT2 必须加载本组 GRPO1 权重，不回到 base；advisor 版本在整个比较中固定。

## 4. 用哪些数据和 benchmark

以下数量来自当前已准备的 manifest；不是把 Numina 全库或所有竞赛题都用于训练。

| 数据 | 上游 split | 实验角色 | 已准备数量 | 使用规则 |
|---|---|---|---:|---|
| [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) | train | Manager train / dev | 128 / 64 | dev 不进入采集标签、SFT 或 GRPO |
| Numina pilot 子集 | 来自上述固定划分 | 小规模机制检验 | 16 train / 16 dev | 按题目 hash 固定选择，不按答对情况挑题 |
| Numina 专家数据池 | 同一 pinned source 的剩余合格题 | subagent train / dev | pilot目标128/32，扩展目标1,024/128个唯一题目 | 与 Manager train/dev 和外部 test 做exact/词法近重复排除；实际数量以新manifest为准 |
| [AIME2026](https://huggingface.co/datasets/MathArena/aime_2026) | 上游名为 train | 外部 test | 30 | 在本项目中始终是 test，不能因上游名字含 train 而用于训练 |
| [BeyondAIME](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME) | test | 外部 test | 100 | 主对照完成、配置锁定后再评测 |

当前来源版本：

- NuminaMath-1.5：1b05109f9e5c1ad06c0663519502416c30b300f8。
- AIME2026：d2de22f3c656b4f56cf8981212186377d1e23bc3。
- BeyondAIME：c705198ae1043810b1e1693bd879250b51a7a523。

沿用现有筛选：文本题、可校验最终答案；Numina 问题/解答有效标记均为 Yes，排除证明、选择题和需要图片的题。冻结题目 ID、原始来源、文件 sha256、筛选与排除计数。新增expert_data构建器单独保存solution sidecar、监督依据、排除清单；运行时输入仍只包含白名单字段。

去重先于划分，当前代码覆盖 NFKC、大小写和空白归一化后的完全相同题目；不等于语义去污染。专家构建器已增加词法近重复排查、保留排除清单，但不宣称消除了模型预训练污染。新数据池不能覆盖已有目录。

AIME 已用于运行调试，须披露这次接触；不得按 AIME 对错选训练配置。BeyondAIME 保持最终配置锁定后的评测用途。每轮曲线使用 Numina dev，不反复用外部 test 调参。

旧 MedQA 实验不并入当前数学训练/结果。跨任务 MedQA 复现是后续可选扩展，需单独核对旧权重、split、协议和预算，不能与数学分数求平均。

## 5. Manager 每一轮具体做什么

### 5.1 反事实采集

当前 Manager 先生成独立候选，再在相同题目上比较直接提交、不同 advisor 调用与修订后的结果。最大调用深度 2，每个角色最多使用一次。

当前数学实现会搜索最多 3+6 条调用分支，不是原稿“找到成功层便停止”的完全复现。三组用相同搜索范围，并记录实际计算成本。

MARGENT 目标优先保留可直接答对的 COMMIT；需要帮助时保留最短成功路线。保留无成功路线的计数，不偷偷换掉困难题。收集的工具文本作为上下文，不对其施加 Manager 的监督 loss。

### 5.2 SFT

目标包含选中路线里的 Manager 决策、修订，以及开启 distill_solutions 时的 question-only 独立解答蒸馏。prompt 和工具回复 token 不计入监督目标。答案正确只能验证终局，不能证明每一步推导无误。

pilot 参数：LoRA rank 16，learning rate 2e-5，batch size 1，gradient accumulation 2，每阶段 8 个 optimizer steps，每 2 步保存。最终使用固定末步；dev-selected-best 仅作另行标注的辅助结果。

### 5.3 GRPO

从当前 SFT checkpoint 继续。每道题的一组轨迹共享该组初始候选，采样 Manager 的决策和修订；初始候选不计入该组 RL loss。advisor 输出仅作为条件上下文。

- 每组 4 条轨迹，每阶段 8 个 group/update；这不是 32 个独立问题。
- 奖励：协议有效且最终答案正确为 1，否则为 0；不加调用惩罚。
- Manager RL temperature=0.8；组内标准化优势；同奖组 outcome advantage 为零。
- learning rate 1e-6，clip=0.2，KL beta=0.01，最大梯度范数 1。
- KL reference 固定为该阶段入口的 SFT adapter；到下一轮重新建立 reference。
- 非法输出保留并记零奖励；不为制造奖励差异重新抽样，也不把非零 KL 梯度当作 outcome 学习证据。
- 整个阶段没有任何 mixed-reward group 时，当前 pilot 停止后续阶段并报告信号不足；保留此次实验结果。

固定配置使用 math_rsi_actions.json 中的 finite_actions_v1。它约束决策语法，不保证数学答案格式或正确性。采样与概率打分使用一致的约束，不能只在评测阶段加约束。

### 5.4 生成与评测预算

Manager 标注/评测 temperature=0；独立答案和修订各最多 2,048 tokens，决策 128；advisor 2,048；上下文与训练序列 32,768。advisor 固定 temperature=0.7、top_p=0.8、top_k=20、presence_penalty=1.5、seed=42。

这些是当前候选配置，不是已证明适合完整 AIME 的最优值。先在 dev 检查截断；若改长度，建立新配置/新目录并重做可比 baseline。记录 raw output、截断和失败原因，不把被截断的输出当作可靠数学负例。

## 6. 对照组与实验矩阵

| 组 | 每轮 SFT 数据 | GRPO | 检验的问题 |
|---|---|---|---|
| D：dynamic MARGENT | 当前 Manager 刷新，直接提交/最短成功路线 | 相同设置 | 完整方案 |
| S：static MARGENT | 一直使用首轮标签 | 相同设置 | 刷新经验是否必要 |
| U：success trajectory | 当前 Manager 刷新，在匹配覆盖/配比的题内随机选成功路线 | 相同设置 | 是否只是成功样本蒸馏 |

首轮共享同一搜索树；D 和 S 首轮应接近一致。static 第二轮仍收集 shadow tree 来衡量标签变化，训练只读取旧数据。分别报告含 shadow 的实际成本和部署时省去 shadow 的成本。此设计匹配采集机会和更新数，不宣称等 token/FLOPs。

执行矩阵：

1. A1（先执行）：三个角色SFT后冻结的advisor，从相同M0运行D/S/U。
2. A0（单独消融）：未做角色SFT的冻结advisor，仍从相同M0运行D/S/U。
3. 在 A0/A1 内比较 D−S 与 D−U；跨 A0/A1 比较必须把专家训练成本单独列出。
4. 正式论文增加同一 warm-start 后 RL-only、动态 SFT-only、关闭独立解答蒸馏，以及匹配推理预算的 self-revision / 多次独立采样对照。这些不挤进首次 pilot。

不把“专家变强”和“Manager 学习变好”混成一个数。预先训练专家再冻结，与专家每轮更新，是两个不同实验。

## 7. 执行顺序与预算

硬件以 2×A100 80GB 为准。下面的小时数是停止预算，不是完成时间承诺。

| 阶段 | 规模与工作 | 预算与完成条件 |
|---|---|---|
| 历史评测故障跟踪（不阻挡专家SFT） | 定位已有 AIME traceback；在 dev 复现、修复、恢复测试，再跑锁定30题 baseline | 新任务最多2小时；已有失败运行目录及原时限保留，不抹掉后复跑 |
| 后续A0消融 | 16 train / 16 dev，D/S/U，各2轮，每轮 SFT8步 + GRPO8步，seed42 | 整体最多24小时，含采集与评测；到时保存并报告已完成范围 |
| 第一步：subagent pilot | 每角色128 train题 / 32 dev题，先16步SFT，做质量与服务检查 | 独立持久化预算最多2小时，含数据构建/训练/重载；dev对照另行启动 |
| 第二步：训练专家后的A1 | 通过专家质量检查后，固定专家，跑Manager三组 | 单独申请/锁定计算预算，不包含在P1的24小时内 |
| P4：正式证据 | 128 train / 64 dev，至少3轮，训练seed 42/43/44；AIME30、BeyondAIME100 | 新配置每轮SFT32步/GRPO32步作为预注册起点，先做吞吐测量再确定时限；尚未实现自动总控 |

不能承诺两小时完成“三专家训练 + 三组两轮 Manager + 两个 benchmark”。若吞吐不够，在看测试分数前缩减训练规模或增加预算；不要事后删掉失败组来凑结果。

Manager pilot 每个组共16步SFT、16步GRPO，三组总计48步SFT、48步GRPO；还包括初始及逐阶段dev评测、搜索树和模型加载成本。P4 的增加步数/轮数属于新实验，不覆盖 pilot。

## 8. 评测、统计与论文图表

每个初始、SFT后、GRPO后 checkpoint 在同一 dev 题集测：

- independent_accuracy：禁用 advisor，独立解题正确率。
- policy_accuracy：允许固定 advisor 和固定调用预算的最终正确率。
- invalid/truncated rate、平均调用数、已独立答对题上的不必要调用率。
- 每题 prompt/completion tokens、实际执行 tokens、时间、峰值显存。
- 按题记录 wrong→correct、correct→wrong；训练集 rescued→direct 只作为机制诊断，泛化结论看 dev/test。

主要对比预先固定为最终 GRPO checkpoint 的 D−S、D−U，以及各组相对 M0 的变化。没有提升或发生退化也完整报告。训练 reward 不作为模型选择的替代 benchmark。

外部评测至少包含 M0 与每组最后 checkpoint；如增加 dev-selected-best，单列而不替换末轮。报告准确计数 k/n；AIME 一题约3.33个百分点，16题dev一题6.25个百分点。使用按题配对 bootstrap 区间，并分别报告各训练seed和跨seed离散程度；不能把同题4条rollout当作4个独立评测样本。AIME与BeyondAIME分别报告，不简单合并成一个分数。

建议交付：

1. independent / policy accuracy 对 checkpoint 的双曲线。
2. 准确率—实际生成 token/总计算成本图。
3. D/S/U 的标签更新比例、救回和遗忘的配对转移表。
4. A0/A1 专家条件分表，以及专家预训练成本。
5. 所有失败、预算截止、无奖励差异的运行附表。

## 9. W&B、断点和失败处理

项目：[MATH_rsi](https://wandb.ai/yuningyangaillm/MATH_rsi)。下表区分已有能力与需要补齐的监控。

| 层面 | 必须记录 | 当前情况 |
|---|---|---|
| Manager SFT | step、loss、grad norm、lr、监督tokens、checkpoint | 已有 callback/Monitor；小测试只有少量点 |
| Manager GRPO | reward mean/std、优势绝对值、零优势比例、mixed groups、合法率、loss/KL、调用数 | 已有逐步/轨迹记录；需要多步训练才有曲线 |
| subagent SFT | 每角色train/dev loss、质量指标、step、checkpoint | 新增数学expert_train已接Monitor；自动指标与待人工质量审核分开记录 |
| 总控 | 当前阶段、子进程退出码、最近心跳、完成题数、预算剩余、总完成标记 | AIME及专家父总控已有；正式扩展矩阵仍需单独预算 |
| 失败诊断 | traceback尾部、failed_stage、状态文件、日志artifact | 新代码会上传完整异常/日志artifact；旧缺失日志不会被追溯补造 |
| 通知 | W&B告警与本任务定期检查 | 邮件送达未验证，不承诺自动收到邮件 |

终端任务必须在 tmux 中启动，并保存独立总日志和每阶段日志。浏览器 connection closed 不应结束 tmux 内任务；Pod停止或进程退出仍会终止计算。恢复必须核对数据/配置/代码/adapter指纹，不能重复累加已完成题目；保留原始失败记录和尝试编号。

阶段 Finished 只说明该阶段结束。完整实验完成要同时核对：全部计划阶段、规定训练步数、可重载权重、完整评测ID与样本数、最终报告。AIME baseline 用父运行 baseline_complete=true 且 n=30 确认；整个RSI实验另需明确的总完成标记，不能借用单个SFT的Finished。

定期监控只读且非实时；不能仅凭GPU利用率0判断失败。详细日志上传、通知渠道验证和持久化恢复演练完成后，才启动更长训练。

## 10. 开跑前交付清单

- [ ] 找到并修复 AIME 子进程退出的真实原因，上传可定位的异常记录。
- [ ] 冻结 config、数据manifest、代码commit、模型revision、seed、advisor fingerprint。
- [ ] 验证当前GPU环境的一题端到端路径及中断恢复，不只跑CPU单测。
- [ ] 对齐 finite_actions_v1 配置；pilot shell默认使用专家生成的manager_config.json，保留finite_actions_v1。
- [ ] 生成三组执行计划，确认第二轮加载本组GRPO1权重。
- [ ] 完成专家数据、训练与多adapter服务的实现/验证后才进入A1。
- [ ] 在看外部测试结果前锁定评测预算、比较对象和完成标准。
- [ ] 保存W&B链接、run manifest、逐题结果、配对统计和失败清单。

代码定位：src/verifiable/rsi.py（计划）、rsi_grpo.py（RL）、training.py（Manager SFT）、experiment.py（采集/目标选择）、data.py（数据）、serve.py（当前advisor服务）。更早文献与设计背景见 [RSI_RESEARCH_DESIGN.md](RSI_RESEARCH_DESIGN.md)；本文件中的正式扩展规模和9B GPU训练结果仍需执行验证。
