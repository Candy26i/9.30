# Main 审计修复与论文记录规范

审计基线：`e84ef4652b05802f65868caf91ec74c1b4b97187`，2026-09-29 UTC。本文描述代码行为和验证范围，不是新的实验结果。完整实验设计见 [MARGENT RSI 实验计划](MARGENT_RSI_EXPERIMENT_PLAN.md)，subagent 的数据、训练和冻结方案见 [Subagent 训练计划](SUBAGENT_TRAINING_PLAN.md)。

## 1. 已定位的问题与修复

用户提供的 W&B CSV 中，同题的两次 Manager answer/revision 都在 2048 tokens 截断；之后 reasoner 在 2048 tokens 截断，记录 `advisor_output_truncated`。源码将这种普通生成结束情况作为异常抛出，可能使整批 AIME 评估中止。CSV 不是完整 traceback，不能据此排除其他运行问题。

| 问题 | 修复后的行为 |
| --- | --- |
| advisor 空输出或正常长度截断使整个评估退出 | 保留原文、finish reason、截断与有效性字段，让 Manager 用实际返回内容继续；不额外重采样、不增加 token 预算。网络、身份或响应结构错误仍明确失败。 |
| 超出上下文预算被混为基础设施异常 | 记录 `context_budget_exceeded`，终止当前轨迹，保留在分母中且奖励为 0；没有生成时实际 token 为 0。 |
| 明确的 FINAL 答案被外层 Markdown/数学定界符拒绝 | 允许明确 FINAL 后的等价外层格式；不从任意数字猜答案，不接受冲突的 FINAL，也不把截断答案算作有效答案。 |
| AIME 预检因格式或截断率直接阻止全量评估 | 默认作为预检警告；`--strict-preflight` 恢复严格门槛。3 题恢复验证、预计时长检查和持久化两小时上限继续生效。 |
| 恢复缓存只凭文件存在或最终文本 | 校验题目身份、元数据、评分与成本字段；无效 shard 不覆盖已有有效 ledger。模型、数据、harness 不一致仍不能续跑。 |
| SFT prompt/response 边界可能发生 BPE 合并或模板偏移 | 保留与推理相同的完整 prompt token 前缀，只监督 response；拒绝空 target 或截断 target。修复 Manager 和 subagent 路径。 |
| Qwen3.5 多模态配置在旧训练路径由错误模型类加载 | 按配置选取文本 causal LM 类，保持该路径原有 chat template；加载前选择 worker 的 CUDA local rank。 |
| GRPO 崩溃后的孤立目录被当作已提交训练步骤 | 只读取原子 `resume.json` 指向的提交链；adapter、优化器状态和报告一起提交。 |
| 无 sampled token 的失败组被伪计为优化步骤 | 样本仍参与组内奖励分母，但不捏造梯度；分别记录 `completed_groups` 与 `optimizer_steps`，恢复时也保持区别。 |
| beta=0 时仍计算可能溢出的 KL，及梯度累积缩放问题 | beta=0 不计算该 KL 损失；加入有限值检查，修复 legacy routing anchor 的累积归一化。 |
| legacy GRPO seed 没有实际传递、batch 无法整除生成组 | 传递 seed；保留合法默认有效 batch，不合法默认自动选可整除累积数；显式不合法设置报错。远程 adapter ID 不强制解释为本地路径。 |
| test-only 数据随机拆入训练集、GPQA 排除集加载失败后继续 | test-only 源不再变为训练数据；排除集失败明确报错；修复嵌套 subset 重复。 |
| subagent 同题不同 draft 绕过 train/dev 隔离 | 新合成数据带 question hash/split，训练按题目身份检查交集，并拒绝显式 held-out 训练行；历史无元数据行仍只能做 prompt 级校验。 |
| W&B 子运行缺配置、父运行无失败日志、暂时上传失败后永久停记 | 补充父子 manifest、统一 RSI group/arm/round、错误 summary 和有边界的 evidence artifacts；上传故障保留本地记录并允许后续恢复。 |
| 截断 usage ledger 导致无法恢复或成本被误称完整 | 仅修复末尾未完成行，保留坏行证据；持续标记 `usage_incomplete`，成本是下界。内部损坏仍拒绝。 |
| 已完成 SFT 用字符串路径再次启动时报类型错误 | 完成检查统一规范为 Path，正常校验已保存 adapter 并幂等返回。 |
| RSI 成本只统计最后一次成功尝试 | 汇总所有可观察的失败/中断/成功 attempt；未完成阶段消费也纳入，未知值为 null，并标记完整性。 |

这些修复针对已定位和已测试的问题；不代表穷尽所有环境相关或隐藏错误。

## 2. 论文数据应在哪里找

W&B 是展示和备份层。本地不可变的输入 manifest、逐题 records 和已提交 checkpoint 是复算依据。原始记录与 UI Table 的展示值要分开：Table 可能有行数和字符数上限，不能用一份被裁剪的 CSV 代替全量预测。

| 层次 | 记录内容 | 解释约束 |
| --- | --- | --- |
| 身份与复现 | W&B group、stage/job type、stage path、arm、round、logical stage ID、attempt；父/子 manifest、seed、模型 revision、输入 checkpoint/数据身份；Python/依赖/Git/CUDA 环境 | 同一 RSI 父运行下各 arm 和阶段共用 group；恢复 attempt 不等于新独立 seed。离线恢复分段保留独立 active run ID。 |
| 生成过程 | question hash、benchmark、split、example ID、phase、role/advisor、prompt/output、finish reason、validity、truncation、error、请求/实际 token、耗时和 cache hit | `generation_step` 按实际日志递增；恢复继续计数。请求预算不是实际消费。 |
| SFT | loss、learning rate、gradient norm 等 Trainer 日志；优化步、训练参数量、数据 hash、输入/监督 token、checkpoint 来源和恢复点 | 只监督目标 response；不同样本长度下相同步数不代表相同 FLOPs。 |
| RSI GRPO | reward mean/std、valid/invalid rate、sample count、advantage 绝对均值/零比例、mixed group、policy loss、weighted KL loss、sampled KL、sampled surprisal、importance ratio、clip fraction、梯度范数、学习率、轨迹长度、耗时、真实优化步和提交组数 | sampled surprisal **不是**完整词表 entropy。训练 reward **不是** benchmark accuracy；当前每组一次更新的 ratio/clip 也不能解释为多轮 PPO 优化。 |
| 独立评估 | 固定题目 n、正确数/准确率、格式有效率、截断、错误类型、calls、rescue/harm 与各自分母；逐题 prediction 和 ground truth | 无效答案保留在原分母，不能只报有效子集。训练集的 collection/search rescue 是训练机制诊断。 |
| 总控与故障 | controller status、current/failed stage、完成阶段数、heartbeat/last progress、完整异常 traceback、子进程日志和 deadline | 子运行 Finished 不等于父实验完成。AIME 全量完成要求父运行 `baseline_complete=true` 且 n=30。 |
| 成本与证据 | usage ledger、GPU samples、观测 wall time、实际与逻辑 token、缓存标记、checkpoint 文件大小/hash、evidence 文件 hash/遗漏清单 | 硬中断、坏尾行或未知消费不能称为精确总成本；GPU 利用率 0 不能单独判断运行失败。权重只记录身份，不上传权重。 |

当前数学 RSI 使用 `src.verifiable.rsi` 和自定义 GRPO。历史 MCQ ManagerGRPO 使用 TRL 自带 W&B，字段并不完全相同；legacy 多卡 SFT 的使用量明确标为 rank 0 观测，不等于全卡总量。当前数学 pilot 仍是一个冻结模型配三种 role prompt，不能写成已经训练并部署了三个独立数学专家 adapter。

## 3. W&B 开关和故障证据

沿用已有登录方式，不需要把 key 写进命令、配置或仓库。运行环境示例（这里只说明配置，不会启动实验）：

```bash
export MARGENT_WANDB_MODE=online
export WANDB_ENTITY=yuningyangaillm
export WANDB_PROJECT=MATH_rsi
export MARGENT_WANDB_TEXT=1
```

`MARGENT_WANDB_TEXT=1` 显式启用题目/生成/轨迹 Tables 和原始文本 evidence；关闭时仍记录标量、配置、状态与故障证据。日志可能包含模型打印的内容，因此仍应在实验数据许可范围内启用 W&B。

- 默认每 300 秒及正常/异常退出时提交 `experiment-evidence` artifact。`MARGENT_WANDB_ARTIFACT_INTERVAL` 可调整周期，最小 30 秒。首次 W&B 初始化失败仍明确报错；运行期间暂时上传失败不覆盖原实验错误，也不会阻止本地记录。
- artifacts 只收集明确列出的 manifest、summary、metrics、usage、environment、状态/traceback 和 `logs/*.log`。文本开启时额外收集 records/generations/rollout diagnostics；不递归扫描任意文件、不上传模型权重、不跟随指向其他位置的日志链接。
- 单次默认源文件总预算 100 MiB，可用 `MARGENT_WANDB_ARTIFACT_MAX_BYTES` 调整。过大文件明确列入 `evidence_index.json` 的 `omitted`，不会悄悄截断成完整文件。导出会清理常见凭据字段，同时记录源文件及导出文件的 SHA-256。
- Table 默认最多 10,000 行，每字符串 20,000 字符，可用 `MARGENT_WANDB_TABLE_MAX_ROWS/MAX_CHARS` 配置；裁剪列和遗漏计数保留。artifact 的文本不受 Table 字符上限影响，但受 artifact 文件预算限制。
- `tracking_exports` 与 W&B 内部目录不参与实时状态扫描或论文输入文件扫描。历史 snapshot 不应制造“运行卡住”或重复证据。
- `submitted` 仅表示 SDK 已接收上传请求；断网、硬 kill 或 Pod 丢失仍可能导致远端缺最后一段。暂时失败后不会伪造缺失的在线历史，优先查本地 `metrics.jsonl` 和 artifacts；离线记录需要另行同步。

## 4. 实验汇总与可写出的结论

RSI 的 `report` 子命令根据真实输出生成 `pilot_report.json`、`pilot_timeline.csv`，并记录同一小型 dev 集内的配对变化和探索性 bootstrap 区间。它不是完整 benchmark、多 seed 显著性结论。

其中 `stage_costs.observed_wall_seconds` 汇总已关闭 Monitor attempt 的耗时（包含失败/中断），不含未关闭尝试或 Monitor 外进程启动开销；实际 token 根据所有 attempt 的 ledger 累计。`last_successful_attempt_wall_seconds` 来自成功子进程 marker，旧 `wall_seconds` 仅保留为它的兼容别名。两种时间视角不能相加，也不能把最后成功耗时当累计成本。缺失证据为 null；`accounting_complete=false` 时观测值只能作下界。

`src.verifiable.reporting.generate_report` 的 CSV/TeX/PDF 报告面向历史 `loop.json` 流程，包含 Wilson 区间、逐题配对变化和成本审计；不能把 RSI 根目录直接传入并宣称两者接口通用。未来完整论文应使用冻结的 benchmark、相同题目集合与匹配推理预算，分别比较 base、SFT、GRPO、后续轮次，以及 dynamic/static/success 控制；多 seed 结果单列，不把恢复 attempt 当独立重复。

已知旧 smoke 仅验证小流程：单步 4 条采样、reward mean=0.75、valid rate=0.75、有组内奖励差异；它不能证明 benchmark 提升。单题 after-GRPO 准确率 0 也不能证明总体退化。旧 AIME 失败运行没有完整 30 题基线，修复代码不会把旧运行补成成功或生成不存在的结果。

## 5. 修复后的首次运行

本次修改改变了 harness 和终止行为。**使用新的输出目录与 W&B group**，保留旧失败目录、CSV 和日志。不要删除原 manifest 来强行续跑旧实验，也不要在旧目录中修改持久化 deadline 来增加预算。新的基线和未来训练必须使用同一新版本的评分/生成协议，避免把行为变化误记为模型提升。

先完成 3 题 dev 预检（包括计划内中断恢复），再按原上限评估 30 题 AIME；仅由父运行的完整完成标志确认完成。后续训练的阶段顺序和匹配预算仍按已有实验计划执行。本次代码审计未启动、停止或重跑远端实验。

## 6. 验证与参考

完整回归结果：**222 passed，11 subtests passed**（开启 `MARGENT_CPU_INTEGRATION=1`；W&B SDK 测试使用离线模式）。另通过 Python 编译检查、补丁空白检查和 AIME 启动脚本语法检查。

验证覆盖真实 tiny CPU 模型的 SFT/GRPO 权重更新、中断恢复逐张量一致、next-SFT 继承、错误归因/评分与 split 隔离、checkpoint 提交链，以及真实 W&B SDK 的离线 Table/artifact 写入。9B/CUDA、真实多 GPU、vLLM 长生成及线上 W&B 网络恢复仍需要部署环境实测。

修复时参考以下一手实现，按本仓库 immutable-COMMIT 协议选择适用部分：

- [TRL v0.29 GRPO 文档](https://huggingface.co/docs/trl/v0.29.0/grpo_trainer)：group/batch 约束、KL、mask 与损失归一。
- [TRL GRPOTrainer 实现](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)：交叉核对训练输入和累积行为；本地安装的 0.29 实现用于具体归一验证。
- [Open-R1 rewards](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py) 与 [Math-Verify](https://github.com/huggingface/Math-Verify)：区分答案等价性、格式有效性和真实奖励，不照搬宽松提取来放大准确率。
- [W&B Tables 文档](https://docs.wandb.ai/models/track/log/log-tables)：Table 展示/增量行为与原始实验文件分开保存。
