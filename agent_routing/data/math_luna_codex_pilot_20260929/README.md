# Codex Luna 数学三专家 SFT 数据

下载本目录即可使用，无需先上传 ZIP 到 RunPod。配套训练代码已包含在本分支中。

题目来自 NuminaMath-1.5，角色监督由 Codex 中选定 `gpt-6-luna` 的任务生成。原始160题产生640行角色监督；按编码规则排除24道训练题后，本目录的 **136题、544行** 分布如下：

| 专家 | Train | Dev | 学习内容 |
|---|---:|---:|---|
| Extractor | 104 | 32 | 提取条件、变量、约束和目标 |
| Reasoner | 104 | 32 | 解题思路与数学推导 |
| Verifier | 208 | 64 | 审查每题两条候选解答 |

## JSON、JSONL 和 ZIP 分别是什么

- **`sft/`：仓库训练器直接使用。** 六个 `.jsonl` 文件，每行是一条完整 JSON；同时保留 manifest、原始 teacher 证据和来源校验文件。逐行格式便于加载与检查。
- **`json/`：普通 JSON 数组。** 同样按三个角色和 train/dev 分为六个文件，每条含 `messages`（system、user、assistant）及少量来源 metadata，方便查看或导入其他训练框架。保留的 prompt/answer 与 `sft/` 一致，没有再次生成答案。
- **ZIP：完整实验归档容器。** 之前交付的 ZIP 还保留30,000题原始来源前缀、逐批次编排和隔离失败等材料；这些不是每次 SFT 都需要读的输入。完整 ZIP 未放进本次 Git 提交，名称与 SHA-256 见 `PUBLICATION.json`。

当前 `runpod_expert_sft.sh` 使用 **`sft/`**；不要把 `EXPERT_DATA` 指向 `json/`。若使用通用训练框架，可读取 `json/*.json` 的 `messages`，并设置仅 assistant token 计算 loss、相同聊天模板及对应角色划分。普通 JSON 转换不等于已集成另一套训练框架。

`sft/teacher_requests.jsonl` / `teacher_responses.jsonl` 保存全部960个生成任务及导入尝试的证据；它们不是额外训练样本。`manager/` 是新冻结的128题 train、64题 dev 与 AIME2026 30题、BeyondAIME 100题，只用于本次专家数据的隔离检查和后续独立实验。

## 在 RunPod 使用

在 GPU/依赖环境已按 [专家 SFT 环境说明](../../docs/EXPERT_SFT_RUNPOD.md) 准备好后，取得含数据的分支：

```bash
git clone --branch codex/luna-expert-data-20260929 --single-branch \
  https://github.com/Jeremyyny/7.98.git /workspace/7.98-luna-pilot
cd /workspace/7.98-luna-pilot/agent_routing

export EXPERT_PYTHON=/workspace/margent-venv/bin/python
export LUNA_DATA="$PWD/data/math_luna_codex_pilot_20260929"
export EXPERT_DATA="$LUNA_DATA/sft"
export EXPERT_MANAGER_DATA="$LUNA_DATA/manager"
export EXPERT_CONFIG="$LUNA_DATA/configs/expert_sft_text_clean.json"
export EXPERT_ROOT=/workspace/margent-luna-experts-github-01
export EXPERT_SESSION=margent-luna-experts-github
export EXPERT_GPU=0

# 只验证数据，不加载模型，也不调用 teacher。
"$EXPERT_PYTHON" "$LUNA_DATA/validate_data.py"

# plan 展示命令；上面的 validate_data.py 才实际验证数据。
bash scripts/runpod_expert_sft.sh plan

# 以下命令才会启动 GPU 训练，由使用者执行。
bash scripts/runpod_expert_sft.sh start --minutes 120
bash scripts/runpod_expert_sft.sh status
```

`EXPERT_ROOT` 应使用新目录。已有目录属于旧实验时不要混用。三个 LoRA 专家分别从固定 Qwen3.5-9B base 初始化，各16个 optimizer steps；本入口只做专家 SFT、验证与 adapter 重载，不执行 Manager 训练或 AIME 评估。两小时控制器上限不会停止 Pod 计费。

脚本默认 W&B 项目为 `yuningyangaillm/MATH_rsi`，online 模式并记录样本文本。沿用你在 RunPod 的登录；可通过 `MARGENT_WANDB_MODE`、`MARGENT_WANDB_TEXT` 和 W&B 项目环境变量显式设置。

## 质量与来源

- 原始 teacher 内容未改写；训练子集只按 prompt/candidate/response 是否含换行以外的 U+0000..U+001F 控制字符筛选，并在三个角色中同步排除整题。这是生成后的机械筛选，没有按答案正确率挑题。
- 544行均通过实际学生 tokenizer 检查，最长1678 tokens，配置上限8192。数据可被配套训练器读取；9B CUDA/多卡尚未实跑。
- 全部 `reviewed=false`。15条 Reasoner 被文字规则提示可能有未完成推导，仍需审核；该提示不是经过验证的错误数量。Verifier 的184 correct /64 incorrect /24 uncertain 是 teacher 标签，不是独立正确率。
- 来源准确表述为 **Codex subagent synthesis, selected model gpt-6-luna**；实际后端模型、token用量和采样参数未独立观测，保持为 null。每批共享上下文，候选与审查分开生成，可能仍存在相关错误。
- `reports/` 部分报告描述原始完整归档，不能误读为本目录包含其全部源文件。实际上传范围见 `PUBLICATION.json`。不把本数据包当作 benchmark 提升证据。

许可、上游固定版本与引用见 [source_notices/NOTICE.md](source_notices/NOTICE.md)。旧机器的绝对路径是历史元数据，保留原样；训练加载不要求这些路径存在。
