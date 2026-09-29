# 数学 Subagent 训练与冻结计划

设计日期：2026-09-26；teacher 接入更新：2026-09-29。配套 [MARGENT_RSI_EXPERIMENT_PLAN.md](MARGENT_RSI_EXPERIMENT_PLAN.md)。

## 1. 范围和现状

目标是训练三个独立角色 adapter：extractor、reasoner、verifier，并检验它们能否给 Manager 提供更有用、更可靠的帮助。

2026-09-29 实现更新：主线按用户确认沿用 OpenAI / gpt-4o teacher 合成。`expert_synthesis.py` 先锁定 Numina 独立题池，生成三个角色监督、完整候选推导及审查标签，完成后才交给 `expert_train.py` 做三套独立 LoRA。数据、断点、服务、W&B 接入和本地小模型验证不等于已完成真实 API 合成或 9B GPU 训练。执行说明见 [EXPERT_SFT_RUNPOD.md](EXPERT_SFT_RUNPOD.md)。

旧 `expert_data.py` 的题面词法提取、参考压缩和局部算术扰动保留为 `weak_debug` 调试/消融。它不能替代 teacher 主实验，也不能冒充已有完整推导审核。Teacher 标签默认未经人工审核；自动完成不证明专家能力提升。

## 2. 先训练专家，再冻结训练 Manager

主实验采用：

    Numina 独立题池 → GPT-4o 合成 E/R/候选/V 数据 → 格式与来源检查
                ↓
    同一个 pinned base
       ├─ extractor SFT → E*
       ├─ reasoner SFT  → R*
       └─ verifier SFT  → V*
                ↓
       角色质量验证和版本冻结
                ↓
       Manager: collect → SFT → GRPO → recollect → SFT → GRPO

三套 LoRA 都从相同 base 独立开始，不按 E→R→V 串行继承 adapter。可以按顺序占用同一张 GPU 训练，节省显存。主实验不对专家做 GRPO，不在 Manager 两轮之间改变专家权重。

模型为 Qwen/Qwen3.5-9B，revision c202236235762e1c871ad0ccb60c8ee5ba337b9a。teacher 与运行时专家都只能看到允许的题目/候选，不能看到 gold 或参考解答。Numina solution 仅保存到 sidecar，供事后审计和候选终值诊断。

## 3. 数据池、规模和隔离

来源：[NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)，固定版本与主计划一致。新增专家池从未被 Manager train/dev 使用的合格题中确定性选择。

- pilot：128个train题、32个dev题，先做每角色16步更新。
- 扩展：1,024个train题、128个dev题；pilot题按所属split包含其中，不能从dev移入train。
- 三角色可使用相同题目池，但按题目分组切分。一个题目的正确候选、错误候选、多个改写必须留在同一split。
- 全部排除冻结的完整 Manager train/dev、AIME2026、BeyondAIME；当前 Manager 为128/64，但以完整 manifest 和文件为准，不能用 smoke 子集代替。任何同题或近重复排除都记录。
- 保留原始solution sidecar、来源ID、question hash、教师版本、模板hash、标签依据、人工审核字段。
- expert_synthesis 的 prepare 单独构建 references.jsonl；prompt 使用白名单，不修改旧 Manager 数据文件。

扩展规模属于新配置/新题池。当前固定题池内失败生成会保留并报告，不能换题或退回规则答案；所有固定任务有合格响应后才发布。合格仅指工程检查通过，数学质量仍需独立审核。

## 4. 三角色分别学什么

训练输入必须与 src/verifiable/protocol.py 的 advisor_messages 一致：

| 角色 | 运行时输入 | 监督输出 | 不应训练的行为 |
|---|---|---|---|
| extractor | 题目及允许的context；无Manager草稿 | 已知量、变量、约束、目标、可检验等价表达 | 凭空增加条件、把最终答案伪装成题目条件 |
| reasoner | 题目及允许的context；无Manager草稿 | 解题路径、关键中间推导和必要计算 | 无依据的定理、只报答案、使用不可见gold |
| verifier | 题目 + 当前Manager推导 | Verdict / Evidence / Correction，一次明确结论 | “被要求检查所以一定有错”、反复自我推翻、把终局正确等同于推导全对 |

### 4.1 Extractor 标签

GPT-4o 根据题面生成条件、变量、约束和目标，使用与服务端相同的 `advisor_messages`，不接收 reference solution。检查非空、完成状态、长度和协议；事实覆盖、无新增假设仍需独立抽查。

### 4.2 Reasoner 标签

GPT-4o 从题目独立生成解题建议和关键推导，不直接复制 Numina reference。保留 teacher 原始输出和完整调用证据。终局答案匹配不能认证所有推理；定理使用、关键等式和边界条件需要人工或可执行检查。

长度超限不静默裁掉后训练：API 已知截断会作为失败保存并按预算重试；学生 tokenizer 的序列超限会整条排除并报告实际监督数。

### 4.3 Verifier 标签

每个题目由 GPT-4o 独立采样两条完整候选解，再对每条候选单独调用 verifier teacher，输出 `Verdict / Evidence / Correction`。每次审查绑定具体候选响应 hash。候选与标签均由 teacher 生成，不预先指示正确/错误，也不强行按 gold 匹配改 verdict。

pilot 预期 Verifier 为256 train / 64 dev条，但只有128/32个独立题目。允许 correct、incorrect、uncertain，保存实际类别计数和候选重复数；不保证三类都有、不按比例伪造标签。自然错误不够或类别严重偏斜时，先报告，在新版本数据计划中增加独立候选来源或经人工审核的错误，不能修改已冻结数据。

相同 teacher 生成并审核可能共享错误。至少抽查每角色32条train；Verifier 抽查覆盖实际出现的类别与“终值对但推导错”的陷阱。dev尽量全量审核，未审核标签明确标为 teacher pseudo-label。人工审阅集与合成原始集分版本保存。

## 5. 生成教师与训练样本格式

主线 teacher 为用户选定的 OpenAI / `gpt-4o`；复用旧 benchmark 的 TeacherClient 接口，数学 prompt 与旧选择题 schema 分开。默认配置为 [math_expert_teacher.json](../configs/math_expert_teacher.json)。API 模型别名可能变化，因此同时记录请求型号、实际返回型号和 system fingerprint；不把它伪装成固定权重 revision。

每题 E、R、两条候选、两次 V 审核，共6个任务；160题为960任务，允许每任务一次重试，总上限1,920次任务尝试。该限制不是 wire 请求数或美元上限；失败和未知调用保留计数，缺失 token 用量不能写成零。真正生成只在显式执行 `generate` 时发生，预览和 SFT 入口不会调用 API。

每条数据至少包含：

| 字段 | 内容 |
|---|---|
| question_hash / source_id / split | 分组、溯源与防泄漏依据 |
| role / prompt / response | 角色、实际运行时消息、teacher 原始监督文本 |
| label_source | 主线为 teacher_synthetic；旧规则单独标记 |
| teacher_provider / teacher_model / teacher_actual_model / teacher_revision | 请求和实际来源，未知字段保留 null |
| teacher_request_id / teacher_prompt_sha256 / teacher_response_sha256 | 连接完整请求、原始响应及调用记录 |
| reviewed | 默认 false，不自动声称人工审核 |
| candidate_request_id / candidate_response_sha256 / candidate_hash / verdict | Verifier 的具体候选、审查标签和来源绑定 |
| schema_version / template_sha256 | 数据协议和模板指纹 |

原始参考保存在独立 sidecar；loss 只覆盖本角色 response token，题目、候选和其他 agent 回复均 mask。prepare/generate/finalize 的恢复核对数据、请求、模型、配置和代码，已成功任务不重新付费请求。

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

上表是研究目标。当前 teacher dev 的 Verifier 自动结果仅为 `teacher_label_agreement`；Extractor/Reasoner 不伪造自动质量分数。没有独立审核时 `quality_assessed=false`，不能用 teacher 标签一致性代替数学正确率。

角色质量的分母、人工rubric、审核者一致性应随报告保存。所有角色再记录响应时间、tokens、异常率。

不能仅因loss下降就晋级。出现不可重载adapter、角色串用、非有限loss、候选/gold泄漏立即停止。pilot中若正向帮助没有改善或误导增加，完整报告，不宣称专家已变强；样本小导致结论不确定时先扩展dev验证，不动外部test。

训练后的专家采用同一既定advisor采样设置与prompt-only对照；若另试greedy或更长输出，归为新条件并重新比较。

## 8. 服务接入与必要实现

建议一个冻结base加载三套adapter，由请求中的角色别名显式选择；GPU0串行处理请求，避免adapter切换的并发竞态。每次调用记录实际adapter，而非只记录别名。无需同时驻留三个完整9B模型。

当前 serve.py 已支持此功能，实现和待GPU验证清单：

- [x] GPT-4o 数学角色合成、完整候选审查、solution sidecar、逐次调用证据和分组去重 manifest。
- [x] 兼容Qwen3.5及固定revision/模板的训练入口；拒绝静默截断或零监督目标。
- [x] 支持按role加载/切换adapter，并在health/result中返回各role fingerprint。
- [x] HTTPAdvisors客户端与服务端统一校验三个fingerprint，Manager重启不能接上另一个专家版本。
- [x] 三角色请求交错测试，证明每次响应使用正确adapter且冻结参数没有变化。
- [x] checkpoint+optimizer+scheduler+RNG恢复；恢复前后step计数和训练样本顺序验证。
- [ ] GPU短测覆盖训练、重载、三角色调用和Manager一次修订。
- [x] W&B、traceback artifact、状态和预算总控接入。

数学入口为 scripts/runpod_expert_sft.sh；不要混用旧benchmark默认模板。通用旧训练器已在前次审计补齐Monitor/revision/恢复，本版进一步保证数学专家训练与运行时同模板。

## 9. 日志、产物与算力预算

W&B沿用MATH_rsi。Teacher synthesis 单独记录合成来源、接受/拒绝、用量与完整请求/响应证据，通过数据 manifest 指纹关联 SFT。父expert_sft_controller与三个expert_sft子运行、expert_reload共用group，记录角色、train/dev loss、optimizer steps、输入/监督token、梯度/学习率、模型/数据/模板/adapter指纹、恢复、GPU及异常。expert_eval单独输出dev对照和人工审阅材料。完整证明质量、manager_rescue/harm_rate仍需独立质量与Manager dev配对评估。

实际输出 /workspace/margent-expert-teacher-sft-01，三个角色在training/<role>，顶层experts.json、manager_config.json、expert_report.json；参考sidecar和排除表在data/。文件格式、W&B文本开关、证据artifact和启动命令以执行指南为准。

Teacher API 合成先在 CPU 环境单独执行并核算；本轮专家总控持久化上限2小时，包含完成数据复制检查、训练和重载；角色dev对照另行显式启动、最多2小时。新正式规模需先测吞吐，再锁定预算。所有长进程在tmux，不与Manager抢占GPU；截止终止自身子进程，不自动停止Pod计费。

## 10. 后续可选：Manager 与专家共同演化

只有冻结专家主实验完成后才考虑。每轮先固定 E_t/R_t/V_t 训练Manager，再用训练集轨迹与独立质检目标更新专家得到 E_(t+1)/R_(t+1)/V_(t+1)。下一轮开始时冻结新版本；不在同一GRPO group中改变环境。

需要2×2对照：Manager不更新/更新 × 专家冻结/更新，所有组从相同已训练专家和同一Manager起点开始。每个checkpoint交叉评测 M_t+A_0、M_0+A_t、M_t+A_t，并同时报告禁用advisor的M_t。这样才能区分Manager学习、专家学习与联合收益。

更新专家会改变环境、采集成本和奖励分布；需要新的恢复指纹与预算控制。当前没有此训练器/总控，也不把该扩展的预期收益写成已有结果。
