# Teacher 合成专家数据，再进行数学专家 SFT

数学实验沿用其他 benchmark 的顺序：**Numina 题池 → teacher 合成三个角色的监督数据 → 独立 LoRA SFT → 专家 dev 检查与冻结 → Manager SFT / GRPO / RSI**。本次用户选择在 Codex 内以 `gpt-6-luna` 生成，请先读 [Codex Luna 数据包与 RunPod 接入](CODEX_LUNA_TEACHER_DATA.md)。下文保留 `gpt-4o` API 路径及其默认配置，二者使用不同来源标记和配置，不能混称同一实验。GPU 入口只读取已完成的 teacher 数据，不自动调用 teacher、不启动 Manager 或 AIME。

## 三个角色的数据如何生成

| 角色 | Teacher 可见输入 | Teacher 生成的 SFT 目标 |
|---|---|---|
| Extractor | Numina 题目、允许的 context | 已知条件、变量、约束、目标和有依据的等价表述 |
| Reasoner | Numina 题目、允许的 context | 解题路径、关键中间推导与计算 |
| Verifier | 题目 + 一条完整候选推导 | `Verdict / Evidence / Correction`，可判 correct、incorrect 或 uncertain |

每题先额外调用 teacher 生成两条完整候选解，再各自交给 verifier teacher 审查。候选也由模型生成，不默认使用参考解答或“右侧加一”的程序错误。所有角色的任务消息由数学运行时 `advisor_messages` 构建，导出请求不含 gold 或 Numina solution，也不含“此候选正确/错误”的暗示。Codex 另有批次操作说明和系统上下文，因此不能声称完整有效 prompt 与运行时相同，或独立证明模型所有上下文均不可见 gold。参考答案只保留在 sidecar，用于候选终值诊断；终值相等不能证明整段推导正确，也不用于强制改写 teacher verdict。

Teacher 标签仍可能出错。相同 teacher 生成候选并审查，会存在相关错误；不保证自然得到均衡的 correct/incorrect/uncertain 三类。实际类别数量和候选重复数会记录，不能伪造标签凑数。专家 dev 的 verifier 指标称为 **teacher 标签一致性**，不是独立数学正确率；Extractor/Reasoner 的质量以及对 Manager 的帮助仍需人工或独立评估。

旧 `expert_data.py` 的规则弱监督保留为显式调试/消融，配置为 `math_expert_sft_weak_debug.json`。它不再是主实验默认数据来源。之前记录的“前30,000条筛出786题”只属于旧规则筛选，不能作为新 teacher 题池的实际数量。

## 冻结题池和生成预算

来源为 [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)，revision `1b05109f9e5c1ad06c0663519502416c30b300f8`。题池排除现有完整 Manager train/dev、AIME2026 和 BeyondAIME 的同题/词法近重复，然后按题目分组划分。三个角色、候选及改写保持同一 split。近重复筛查不等于已证明无语义污染。

配置见 [math_expert_teacher.json](../configs/math_expert_teacher.json)：128 个 train 题、32 个 dev 题，每题 6 个生成任务（Extractor、Reasoner、两条候选、两次 Verifier 审查），计划 **960 个任务**，每个任务允许一次重试，配置上限 1,920 个任务尝试。该上限既不是 provider HTTP 请求数上限，也不是美元预算；SDK 参数回退可能增加实际请求数。另行记录 observed_provider_request_attempts 和未知请求记录。API 合成费用与 GPU 租赁费用分开计算。记录缺失的 token 用量不能补成零；观测到的成本信息可能只是下界。

固定题池后不会因生成困难偷偷换题或退回规则答案。拒绝/截断的响应会保留原因并重试；所有计划任务有合格响应后才发布 SFT 数据。这里的“合格”是格式/完成状态等工程检查，不是数学正确性认证。题池 seed 固定；API 生成仍可能不确定。`gpt-4o` 是服务端别名，保存实际返回的 model 和 system fingerprint，不能假称它是固定的开源权重 revision。

## 先合成数据（无需学生 GPU）

在最新代码的 `agent_routing` 目录，使用现有已配置 API 认证的环境。不将 key 写入配置、命令参数或日志。

```bash
export TEACHER_PYTHON=/workspace/margent-venv/bin/python
export TEACHER_ROOT=/workspace/margent-expert-teacher-01
export EXPERT_MANAGER_DATA=/workspace/margent-data-restart-20260925
bash scripts/prepare_math_expert_teacher.sh plan
bash scripts/prepare_math_expert_teacher.sh prepare
# 下列 generate 会实际调用 teacher API：
bash scripts/prepare_math_expert_teacher.sh generate
bash scripts/prepare_math_expert_teacher.sh finalize
```

`prepare` 必须能读取原有完整 Manager 数据目录及 manifest；不能用几道 smoke 题代替排除池。`plan` 只预览，不发 API 请求、不加载 GPU。需要恢复时在相应 `prepare` / `generate` 后加 `--resume`，沿用相同配置、代码和目录，不覆盖旧结果。数据合成的完成/失败状态见 `synthesis_status.json`。

也可在别的机器使用既有 teacher 批处理服务：`prepare` 导出 `pending_requests.jsonl`；先执行 E/R/两条候选，导入响应后才会产生 `verifier_requests.jsonl`。离线响应必须用导出的 request_id（它已绑定完整请求指纹）匹配，不能按题号匹配。每行 JSON 格式如下；response.model 是请求的模型名称，actual_model 是服务端返回的实际名称。缺失元数据用 null，禁止编造用量或 finish_reason。

```json
{"request_id":"从导出请求复制的完整ID","response":{"text":"teacher原始响应","provider":"openai","model":"gpt-4o","actual_model":null,"usage":null,"finish_reason":null,"request_id":null,"system_fingerprint":null,"latency_seconds":null,"request_attempts":null,"provider_usage":null}}
```

导入使用：

```bash
bash scripts/prepare_math_expert_teacher.sh finalize --responses-jsonl /path/teacher-responses.jsonl
```

在 Verifier 响应未齐时，命令会更新后续请求清单并明确报告未完成，不发布可训练目录。补齐并再次导入后，输出 `"$TEACHER_ROOT/data/manifest.json"` 及三角色 train/dev 文件。外部导入不会调用 API，但其生成费用与缺失元数据仍需如实记录。

## 再开 RunPod 做三专家 SFT

配置 [math_expert_sft_pilot.json](../configs/math_expert_sft_pilot.json) 要求 `teacher_synthetic` 和 OpenAI/gpt-4o 身份。三个角色都从固定的 Qwen3.5-9B base 独立初始化 LoRA，顺序使用同一张 GPU，不把前一个专家的 adapter 接着训练成下一个。

首轮参数：每角色 16 个 optimizer steps，batch 1、accumulation 8、learning rate 2e-5、rank 16、alpha 32、dropout 0.05、BF16、最大序列 8192。Extractor/Reasoner 各 128 train / 32 dev 条，Verifier 各 256 / 64 条；候选不是独立题目。每 8 步和末步保存完整恢复状态并计算 dev loss。过长样本整条排除并报告，不静默截断后训练。

```bash
export EXPERT_PYTHON=/workspace/margent-venv/bin/python
export EXPERT_DATA=/workspace/margent-expert-teacher-01/data
export EXPERT_ROOT=/workspace/margent-expert-teacher-sft-01
export EXPERT_MANAGER_DATA=/workspace/margent-data-restart-20260925
export EXPERT_GPU=0
bash scripts/runpod_expert_sft.sh plan
bash scripts/runpod_expert_sft.sh start
bash scripts/runpod_expert_sft.sh status
```

训练入口在加载 GPU 模型前验证 teacher 数据、来源证据、题数、运行时 prompt 与 Manager/test 排除池，再将不可变数据副本保存到实验目录。建议一张空闲 80 GB GPU；9B CUDA 的实际显存和吞吐仍需实跑确认。使用新目录/group，不能把旧规则数据试验改称 teacher 实验。

SFT 总控保留持久化 120 分钟时限，重新启动不会延长原截止时间。到时只终止自己的进程、保留 checkpoint；**不会关闭 RunPod，也不会停止 Pod 计费**。这段 GPU 时限不包括此前单独执行的 teacher API 合成。

## 证据、质量与完成条件

Teacher 目录保留 `synthesis_run.json`、题池/参考/排除表、完整请求、原始响应、逐次调用记录和 `synthesis_status.json`。发布的 SFT manifest 绑定全部 teacher 请求/响应证据和六个角色文件。每条训练样本保存 requested/actual teacher model、请求/响应 hash、角色、split、候选来源、标签来源与审核状态；人工 `reviewed` 默认为 false。

W&B 沿用 `yuningyangaillm/MATH_rsi`：teacher synthesis 单独成组；SFT 父总控、三个 `expert_sft` 子运行与 `expert_reload` 共用另一组，通过 teacher 数据 manifest 指纹关联。记录生成接受/拒绝数、调用尝试与已知 token 用量，及训练的 train/dev loss、optimizer steps、学习率、梯度、监督 token、checkpoint、GPU 和错误。完整日志及证据进入 `experiment-evidence`；`MARGENT_WANDB_TEXT=1` 才上传原始文本，脚本默认启用，大小限制导致的遗漏会明确列出。模型权重不由这个 artifact 自动上传。

SFT 输出包括 `training/{extractor,reasoner,verifier}`、`experts.json`、`manager_config.json`、`expert_report.json`。`experts_complete=true` 要求三角色步数完成、文件校验通过、权重实际重载和角色切换成功。它证明工程流程完成，不证明专家质量或 benchmark 提升。

训练后单独做 base prompt-only / SFT 专家 dev 对照：

```bash
CUDA_VISIBLE_DEVICES=0 "$EXPERT_PYTHON" -m src.verifiable.expert_eval \
  --bundle "$EXPERT_ROOT/experts.json" --data-dir "$EXPERT_ROOT/data" \
  --out "$EXPERT_ROOT/dev_comparison" --limit 32 --max-tokens 512 --minutes 120
```

审核角色质量后，冻结 `experts.json`，再用生成的 Manager 配置：

```bash
export RSI_EXPERT_ROOT="$EXPERT_ROOT"
bash scripts/runpod_rsi_pilot.sh advisor
# 另一个终端，Manager 使用另一张 GPU：
bash scripts/runpod_rsi_pilot.sh plan
bash scripts/runpod_rsi_pilot.sh run
```

Manager 首轮 SFT 仍从独立 Manager Numina train 池采集反事实轨迹，由外部答案校验器选择成功分支，训练 CALL/COMMIT、成功修订和独立解答蒸馏。不会直接复制专家 teacher 答案或使用专家 dev 训练 Manager。AIME/BeyondAIME 保持锁定外部测试；当前两个入口都不会自动启动它们。

## 代码验证与实际实验边界

完整本地回归：398项测试与11项子测试通过，包含离线 teacher 模拟、Codex 来源校验、真实小模型 CPU 三角色训练/重载/恢复及离线 W&B 证据检查。另已检查入口脚本语法和 Python 编译。API 合成与 Codex 生成分开记录；某个数据包是否完整，以其 `synthesis_status.json` 和校验报告为准。尚未完成9B CUDA训练，软件测试与数据生成都不能计为 benchmark 提升。
