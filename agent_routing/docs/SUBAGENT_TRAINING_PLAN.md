# 数学 Subagent 训练与冻结计划

日期：2026-09-26。配套 [MARGENT_RSI_EXPERIMENT_PLAN.md](MARGENT_RSI_EXPERIMENT_PLAN.md)。

## 1. 范围和现状

目标是训练三个独立角色 adapter：extractor、reasoner、verifier，并检验它们能否给 Manager 提供更有用、更可靠的帮助。

2026-09-29 实现更新：已新增 expert_data.py、expert_train.py、expert_eval.py、experts.py，支持数据构建、独立角色LoRA、dev对照、重载冻结及一个base切换三套adapter。执行说明与实际标签边界见 [EXPERT_SFT_RUNPOD.md](EXPERT_SFT_RUNPOD.md)。当前是弱监督pilot；代码和本地小模型验证不等于已完成9B GPU训练。

下文保留完整研究目标。当前首版Extractor为题面词法提取、Reasoner为未经独立证明校验的Numina参考、Verifier为可执行的正确/错误局部算术步骤；uncertain=0。完整推导审核、自然错误、人工质量验证仍待补充，不据此声称三类覆盖或能力提升。

## 2. 先训练专家，再冻结训练 Manager

主实验采用：

    同一个 pinned base
       ├─ extractor SFT → E*
       ├─ reasoner SFT  → R*
       └─ verifier SFT  → V*
                ↓
       角色质量验证和版本冻结
                ↓
       Manager: collect → SFT → GRPO → recollect → SFT → GRPO

三套 LoRA 都从相同 base 独立开始，不按 E→R→V 串行继承 adapter。可以按顺序占用同一张 GPU 训练，节省显存。主实验不对专家做 GRPO，不在 Manager 两轮之间改变专家权重。

模型为 Qwen/Qwen3.5-9B，revision c202236235762e1c871ad0ccb60c8ee5ba337b9a。参考解答用于离线构造监督目标；运行时专家只能看到允许的题目/候选，不能看到 gold 或参考解答。

## 3. 数据池、规模和隔离

来源：[NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)，固定版本与主计划一致。新增专家池从未被 Manager train/dev 使用的合格题中确定性选择。

- pilot：128个train题、32个dev题，先做每角色16步更新。
- 扩展：1,024个train题、128个dev题；pilot题按所属split包含其中，不能从dev移入train。
- 三角色可使用相同题目池，但按题目分组切分。一个题目的正确候选、错误候选、多个改写必须留在同一split。
- 全部排除 Manager train128/dev64、AIME2026、BeyondAIME；任何同题或近重复排除都记录。
- 保留原始solution sidecar、来源ID、question hash、教师版本、模板hash、标签依据、人工审核字段。
- 现有 prepare 不导出完整参考solution给专家训练，现由expert_data.py单独构建references.jsonl，不修改旧数据文件。

扩展池1,024题不等于每角色必有1,024条合格监督。若质检后不足，报告实际数量；按预定hash顺序补充新题时保持角色/split规则，不按下游test成绩选题。

## 4. 三角色分别学什么

训练输入必须与 src/verifiable/protocol.py 的 advisor_messages 一致：

| 角色 | 运行时输入 | 监督输出 | 不应训练的行为 |
|---|---|---|---|
| extractor | 题目及允许的context；无Manager草稿 | 已知量、变量、约束、目标、可检验等价表达 | 凭空增加条件、把最终答案伪装成题目条件 |
| reasoner | 题目及允许的context；无Manager草稿 | 解题路径、关键中间推导和必要计算 | 无依据的定理、只报答案、使用不可见gold |
| verifier | 题目 + 当前Manager推导 | Verdict / Evidence / Correction，一次明确结论 | “被要求检查所以一定有错”、反复自我推翻、把终局正确等同于推导全对 |

### 4.1 Extractor 标签

离线从题目和参考解答整理候选事实清单，逐条标注是否可由题面支持；参考解答只能帮助标注者理解题目，不能给运行时prompt注入答案。用变量覆盖、数值/范围一致性、无新增假设和人工抽查过滤。

### 4.2 Reasoner 标签

优先压缩已有有效参考解答，保留可独立理解的关键步骤。可用本地冻结模型辅助压缩，但最终答案校验只证明终局一致，不能认证全部推理；需要针对定理使用、关键等式、边界条件的人工或可执行检查。

长度超限的目标不能截掉后直接训练。记录并重新编写简洁目标，或在dev上确定新预算后统一重建。

### 4.3 Verifier 标签

每个题目构造“正确、明确错误、不确定”三类候选，每题至多各1条；扩展目标最多3,072条train / 384条dev。可不足但须报告，不能伪造类别凑数。

- correct：参考推导或经审核的正确候选，标明检查依据。
- incorrect：对已知正确步骤做可定位的符号、算术、条件或逻辑扰动；记录被修改步骤及其错误证据。错误候选即使偶然得到正确终值，也不能标correct。
- uncertain：关键步骤无法从可见信息验证且审核者也不能确定；不能把“最终答案不匹配”自动改成uncertain，也不能把所有短答案都当不确定。

来自 Manager 的自然错误可补充，但 gold不匹配只用于筛选候选，必须检查具体错误后才能生成推导级标签。至少人工抽查每角色32条train；verifier覆盖三类及正确答案错误推理的陷阱。dev尽量全量审核；未审核部分标明弱监督。审核规则固定后保留排除计数，不能只保留看起来有利的结果。

## 5. 生成教师与训练样本格式

首版不依赖新增付费API。使用数据集参考解答、确定性扰动、本地冻结模型辅助标注，并记录其确切revision。自生成质量不足时停止扩大规模，不能靠同一模型“自信地说正确”放行。若后续用外部教师，单独记录型号、提示、预算、输出来源；本提交不授权调用付费服务。

每条数据至少包含：

| 字段 | 内容 |
|---|---|
| question_hash / source_id / split | 分组、溯源与防泄漏依据 |
| role | extractor、reasoner或verifier |
| prompt | 与实际角色输入一致的system/user消息 |
| response | 该角色assistant目标文本 |
| label_source / teacher_revision | reference、local生成、controlled_corruption或人工标注来源 |
| quality_checks / reviewed | 校验结果、审核状态和排除原因 |
| candidate_hash / verdict / corruption | verifier额外的候选与错误依据 |
| schema_version / template_sha256 | 数据协议和模板指纹 |

gold与参考解答放在独立质检字段/sidecar；构造模型prompt用显式字段白名单。loss只覆盖本角色response token，题目、候选和其他agent回复均mask。

## 6. SFT 参数和选模

下列是提议的数学专家配置，不是旧训练器默认值，也不是已调优结果：

| 参数 | pilot | 扩展 |
|---|---:|---:|
| 每角色optimizer steps | 16 | 128 |
| micro batch / accumulation | 1 / 8 | 1 / 8 |
| learning rate | 2e-5 | 2e-5 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | 同左 |
| precision | BF16 | BF16 |
| 最大训练序列 | 8,192 | 8,192 |
| 保存并评估 | 每8步及末步 | 每32步及末步 |
| 角色训练seed | 42 | 首轮42，正式复现42/43/44 |

使用数学backend兼容的模型加载、固定模板和prefix检查；记录实际可训练模块名称。训练前核对全部样本有目标token且长度合规。启用gradient checkpointing，报告实际输入/监督token和总GPU时长。

训练loss不是专家效果指标。首先报告固定末步checkpoint；若用dev挑选，预先固定排序为“角色约束通过 → 下游帮助收益 → 角色质量指标 → dev loss → 较早step”，并同时保留末步结果。不用AIME/BeyondAIME挑专家。

## 7. 专家怎么评测

在专家dev题目上比较同一base的prompt-only角色与训练角色；使用同一题目、同一固定Manager M0、同一生成预算和seed。

| 角色 | 角色质量指标 | 对Manager的实际帮助 |
|---|---|---|
| extractor | 事实覆盖率、错误事实率、约束遗漏率 | 单次调用后最终正确率的配对变化 |
| reasoner | 关键步骤通过率、答案一致率、无效/截断率 | 同上；并报告原本错题救回率 |
| verifier | 三类macro-F1、正确推导误报率、错误定位正确率、无依据纠正率 | 原本正确题被改错率，以及错误题修复率 |

角色质量的分母、人工rubric、审核者一致性应随报告保存。所有角色再记录响应时间、tokens、异常率。

不能仅因loss下降就晋级。出现不可重载adapter、角色串用、非有限loss、候选/gold泄漏立即停止。pilot中若正向帮助没有改善或误导增加，完整报告，不宣称专家已变强；样本小导致结论不确定时先扩展dev验证，不动外部test。

训练后的专家采用同一既定advisor采样设置与prompt-only对照；若另试greedy或更长输出，归为新条件并重新比较。

## 8. 服务接入与必要实现

建议一个冻结base加载三套adapter，由请求中的角色别名显式选择；GPU0串行处理请求，避免adapter切换的并发竞态。每次调用记录实际adapter，而非只记录别名。无需同时驻留三个完整9B模型。

当前 serve.py 已支持此功能，实现和待GPU验证清单：

- [x] 数学专家数据构建器、solution sidecar和分组去重manifest。
- [x] 兼容Qwen3.5及固定revision/模板的训练入口；拒绝静默截断或零监督目标。
- [x] 支持按role加载/切换adapter，并在health/result中返回各role fingerprint。
- [x] HTTPAdvisors客户端与服务端统一校验三个fingerprint，Manager重启不能接上另一个专家版本。
- [x] 三角色请求交错测试，证明每次响应使用正确adapter且冻结参数没有变化。
- [x] checkpoint+optimizer+scheduler+RNG恢复；恢复前后step计数和训练样本顺序验证。
- [ ] GPU短测覆盖训练、重载、三角色调用和Manager一次修订。
- [x] W&B、traceback artifact、状态和预算总控接入。

数学入口为 scripts/runpod_expert_sft.sh；不要混用旧benchmark默认模板。通用旧训练器已在前次审计补齐Monitor/revision/恢复，本版进一步保证数学专家训练与运行时同模板。

## 9. 日志、产物与算力预算

W&B沿用MATH_rsi，父expert_sft_controller与三个expert_sft子运行、expert_reload共用group，记录角色、train/dev loss、optimizer steps、输入/监督token、梯度/学习率、模型/数据/模板/adapter指纹、恢复、GPU及异常。expert_eval单独输出dev对照和人工审阅材料。完整证明质量、manager_rescue/harm_rate仍需独立质量与Manager dev配对评估。

实际输出 /workspace/margent-expert-sft-01，三个角色在training/<role>，顶层experts.json、manager_config.json、expert_report.json；参考sidecar和排除表在data/。文件格式、W&B文本开关、证据artifact和启动命令以执行指南为准。

本轮专家总控持久化上限2小时，包含数据构建、训练和重载；角色dev对照另行显式启动、最多2小时。新正式规模需先测吞吐，再锁定预算。所有长进程在tmux，不与Manager抢占GPU；截止终止自身子进程，不自动停止Pod计费。

## 10. 后续可选：Manager 与专家共同演化

只有冻结专家主实验完成后才考虑。每轮先固定 E_t/R_t/V_t 训练Manager，再用训练集轨迹与独立质检目标更新专家得到 E_(t+1)/R_(t+1)/V_(t+1)。下一轮开始时冻结新版本；不在同一GRPO group中改变环境。

需要2×2对照：Manager不更新/更新 × 专家冻结/更新，所有组从相同已训练专家和同一Manager起点开始。每个checkpoint交叉评测 M_t+A_0、M_0+A_t、M_t+A_t，并同时报告禁用advisor的M_t。这样才能区分Manager学习、专家学习与联合收益。

更新专家会改变环境、采集成本和奖励分布；需要新的恢复指纹与预算控制。当前没有此训练器/总控，也不把该扩展的预期收益写成已有结果。
