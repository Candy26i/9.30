# Codex / gpt-6-luna 专家数据与 RunPod 接入

本次 teacher 由用户指定为 `gpt-6-luna`，选择在当前 Codex 任务中生成，并保留 Codex 来源。题目来自固定版本 NuminaMath；Extractor、Reasoner、完整候选推导和 Verifier 审查由实际选择了该模型的 Codex 子任务生成。数据属于 **teacher synthetic supervision**，不是把 Numina 参考解直接改名为 teacher 输出。

## 固定数据与生成顺序

原 RunPod 的完整 Manager 数据未在本机提供，因此本实验按用户选择重新冻结 Manager train 128 / dev 64 题及 AIME2026 30 / BeyondAIME 100 题。新 manifest 不代表旧实验文件，不能直接续跑旧 Manager checkpoint。专家 Numina 题池排除上述完整数据的同题和词法近重复，再按题目分成 train 128 / dev 32；同题的所有角色和候选都留在同一个 split。该筛查不证明无语义重叠。

本次 Numina 源是固定 revision 的前30,000条原始记录，经筛选和去重后选题；不是全量 Numina 的均匀随机抽样。具体版本、原始文件 hash、筛选计数和许可保存在数据包的 `source_data/`、`manager_data/manifest.json` 与 `synthesis/synthesis_run.json`。

每题依次需要以下6个逻辑任务：

1. Extractor 生成题目条件、变量、约束和目标。
2. Reasoner 生成解题路径和关键推导。
3. 在不同的新 Codex 子任务中分别生成候选0和候选1。
4. 两次 Verifier 审查，各只接收题目及相应的完整候选；输出 `Verdict`、`Evidence`、`Correction`，各字段单独一行。

共160个题目、960个逻辑任务；最终 E/R 各128 train + 32 dev行，V为256 train + 64 dev行，合计640行 SFT 监督。320条候选用于构建 V 输入和审计，不另算320个独立题目。格式失败保留原文并在预算内重新生成，不改写失败原文、伪造标签或换题。

各批次使用全新子任务上下文；train与dev分批，候选0、候选1与各自审查分开。一个批次中多个题目共享上下文，候选和审查又使用同一模型，因此不能声称各输出在统计上相互独立。数学质量需要单独评估。

## 可以证明哪些来源信息

| 记录 | 解释 |
|---|---|
| `selected_model=gpt-6-luna` | 来自成功的 `collaboration.spawn_agent` 模型选择参数；不是独立的后端模型身份鉴证 |
| `generation_surface=codex_subagent` | 当前 Codex 子任务生成，非 OpenAI API 实验 |
| 请求、输入批次、wrapper、raw output hash | 连接导出的角色请求、实际操作说明和保存的原始结果文件 |
| `isolated_from_parent=true` | 以 `fork_turns=none` 创建；未向子任务复制父任务历史 |
| `shared_batch_context=true` | 同批多条请求共享子任务上下文 |
| `no_gold_disclosed_scope=orchestrator_inputs_only` | 编排器提供的消息和指定输入文件不含参考答案；不是对全部隐藏上下文的证明 |
| `effective_prompt_identical_to_runtime=false` | 角色任务消息复用运行时模板，但 Codex 还包含操作说明和自身系统环境 |

实际 API model、token usage、temperature、top_p、finish_reason、system fingerprint 未被工具返回，均保留 `null`。配置中的温度和输出预算只表达请求意图，不能写成实际采样参数。未独立观测到的工具调用记录也不伪造；文件访问限制作为执行策略保存。不能用缺失用量计算为零费用或声称完全可复现。

根任务负责包装原始结果和校验，不为 teacher 修写数学答案。`reviewed=false` 表示没有人工审核；终值与 Numina 参考一致只是诊断，不能证明推导正确。Verifier 的标签分布按真实输出统计，不强制填齐三类，也不据 gold 强行改判。

## 生成和数据包校验

配置：[math_expert_teacher_codex_luna.json](../configs/math_expert_teacher_codex_luna.json)、[math_expert_sft_codex_luna.json](../configs/math_expert_sft_codex_luna.json)。`generation_surface=codex_subagent` 配置会拒绝 API `generate` 路径，避免误调用 API。

数据包保存 `codex_inputs/`、`codex_wrappers/`、`codex_outputs/`、`codex_dispatches/`、`codex_receipts/`、`codex_imports/`，以及 `synthesis/` 中的题池、完整请求/响应、失败尝试、参考 sidecar、排除记录与最终数据。`orchestration/` 是文件包装和校验工具，不能自行替代模型生成。原始30,000条 Numina 文件可按固定版本重取；上传包不必重复包含这份大文件，但要保留其 hash、获取脚本和数据来源说明。

只有960个任务均有通过工程检查的响应时，`finalize` 才发布 `synthesis/data/`。使用实际训练器的 `read_dataset` 和总控的 `check_teacher_data` 验证六个角色文件、请求与响应 hash、来源、数量、模板及 Manager/test 隔离。发布前保留一份完整校验报告和文件 SHA-256 清单。工程校验通过不是数学正确性认证。

## 在 RunPod 上先做三专家 SFT

使用包含上述 Codex 接入的代码版本与完整数据包，解压到新目录。以下路径是示例，先按实际位置修改。不要使用默认 GPT-4o SFT 配置读取 Luna 数据。

```bash
cd /workspace/7.98/agent_routing
export EXPERT_PYTHON=/workspace/margent-venv/bin/python
export LUNA_DATA_BUNDLE=/workspace/margent-luna-codex-pilot
export EXPERT_CONFIG="$LUNA_DATA_BUNDLE/configs/expert_sft.json"
export EXPERT_DATA="$LUNA_DATA_BUNDLE/synthesis/data"
export EXPERT_MANAGER_DATA="$LUNA_DATA_BUNDLE/manager_data"
export EXPERT_ROOT=/workspace/margent-luna-experts-sft-01
export EXPERT_SESSION=margent-luna-experts
export EXPERT_GPU=0
bash scripts/runpod_expert_sft.sh plan
# 确认 plan 中的文件和来源匹配后，在 GPU 环境启动：
bash scripts/runpod_expert_sft.sh start
bash scripts/runpod_expert_sft.sh status
```

三套 LoRA 分别从固定 Qwen3.5-9B base 开始，每角色16个 optimizer steps，micro batch 1、accumulation 8、lr 2e-5、rank16/alpha32/dropout0.05、BF16、最大序列8192。两小时持久化训练时限不会关闭 Pod 或停止计费。9B CUDA 的吞吐与显存仍需实跑确认。

专家完成后先做角色 dev 对照与质量审核，再冻结三个 adapter。Manager 的初始 SFT 来自**独立 Manager train 题池**上的反事实采集轨迹：外部答案校验器筛选成功路径，监督 CALL/COMMIT、修订与解答蒸馏；不是直接复制这批专家 teacher 回答。随后才进行 Manager GRPO 与下一轮采集/SFT。AIME2026、BeyondAIME 留作外部评估，不能用其结果筛选专家训练数据。

## W&B 与论文记录

本次 Codex 离线导入本身不等于已经上传 W&B。后续 SFT 保存数据 manifest 指纹、模型/模板版本、三角色样本数、train/dev loss、optimizer steps、监督 token、checkpoint、GPU、成本缺失标记和失败日志。训练脚本的 `experiment-evidence` 按其文件与大小限制上传训练目录证据；Codex 原始批次档案应连同校验清单独立归档，并在论文记录中链接实际上传的归档位置，不能假称自动全部上传。

报告需区分四件事：数据格式通过、teacher 标签质量、专家 dev 表现、Manager 独立 benchmark 表现。生成完成和 SFT loss 下降都不证明 AIME 提升；同 teacher verifier agreement 也不是独立数学正确率。不同来源数据的许可与引用见数据包 `source_data/README.md`，不要给组合数据包统一附上不适用的许可。
