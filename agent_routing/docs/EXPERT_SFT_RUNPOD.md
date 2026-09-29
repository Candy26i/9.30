# 先训练数学专家，再训练 Manager

本入口实现 Numina 数据构建 → Extractor SFT → Reasoner SFT → Verifier SFT → 重载/角色切换检查 → 冻结专家清单。它不会自动启动 Manager、GRPO 或 AIME。三个角色都从同一个 Qwen3.5-9B 固定版本独立初始化 LoRA；三次训练顺序使用同一张 GPU。

## 第一轮训练内容

配置为 [math_expert_sft_pilot.json](../configs/math_expert_sft_pilot.json)：128 个专家 train 题、32 个 dev 题，三个角色分别 16 个 optimizer steps，batch 1、accumulation 8、learning rate 2e-5、rank 16、alpha 32、dropout 0.05、BF16、最大序列 8192。每 8 步和末步保存完整恢复状态并评估 dev loss。Verifier 每题两个候选，因此有 256 条 train 和 64 条 dev；这不等于 256/64 个独立题。

| 角色 | 输入 | 本版监督来源 | 证据边界 |
|---|---|---|---|
| Extractor | 题目、允许的 context | 从题面提取的 givens/目标/变量/数值清单 | 词法弱监督，不是人工验证的等价变形 |
| Reasoner | 题目、允许的 context | Numina 完整参考推导 | 来源标记有效，但未独立验证整段数学证明 |
| Verifier | 题目、一条当前算术步骤 | 精确有理数计算可验证的正确等式，以及右侧加一的已知错误 | 只训练局部算术检查；不冒充完整证明审核，也不伪造 uncertain 标签 |

这是可运行的角色 SFT 起点。当前数据有选择偏差：需题面可分出目标、参考解答包含可执行数值等式。固定版本前30,000条可行性扫描主要来自olympiads，不能当作全Numina分布。不同题目也可能包含同一条算术等式，manifest记录verifier候选在train/dev之间的重复数量；这个dev是工程诊断，不是独立推导泛化benchmark。正式研究要补经过审核的完整候选推导、自然错误与 uncertain 样本，再测对 Manager 的帮助。不能仅凭 loss 下降声称专家能力提升。

数据固定使用 [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) revision `1b05109f9e5c1ad06c0663519502416c30b300f8`。构建器检查原始 validity 字段、题型、参考解答及缺失图片；保留完整 solution sidecar。专家池排除传入 Manager 数据目录的 train、dev、AIME2026、BeyondAIME，并按题目分组做 exact/词法近重复检查。近重复不等于已证明无语义污染。原始 gold/solution 不进入运行时专家输入；reasoner 的参考推导属于监督输出。

2026-09-29 的真实数据可行性检查：前 30,000 条记录经过标准化、三个角色的共同质量筛选和题目去重后，得到 786 题（train 629、dev 157），来源均为 olympiads。此检查尚未排除 RunPod 上的真实 Manager 数据，不能作为正式训练集的最终数量。前 128/32 题对应的 verifier 候选存在 2 条跨 train/dev 重复等式，正式构建时会重新记录该诊断；没有据此声称独立泛化效果。

代码验证：默认完整本地测试 296 项通过、11 项子测试通过；默认跳过的 3 项 CPU 集成测试另行启用后也通过，合计 299 项。包括微型模型三角色训练/恢复/重载、真实 W&B 离线 SDK、Parquet 流式退出与 Qwen3.5 文本权重加载。另已验证微型 Qwen3.5 混合注意力模型的 LoRA 梯度反传。上述结果不包含 9B CUDA 实跑或正式专家质量评估。

## RunPod 启动

使用已安装项目依赖的环境。建议一张空闲 80 GB GPU；本入口只暴露 `EXPERT_GPU` 指定的一张卡，另一张卡无需运行服务。9B/CUDA 的实际显存和吞吐须在此次 GPU pilot 核实，本地小模型测试不替代 GPU 验证。

在最新代码的 `agent_routing` 目录中：

```bash
export EXPERT_PYTHON=/workspace/margent-venv/bin/python
export EXPERT_MANAGER_DATA=/workspace/margent-data-restart-20260925
export EXPERT_ROOT=/workspace/margent-expert-sft-01
export EXPERT_GPU=0
bash scripts/runpod_expert_sft.sh plan
bash scripts/runpod_expert_sft.sh start
bash scripts/runpod_expert_sft.sh status
```

Manager 数据目录必须含原有冻结的 `manifest.json`、`train.jsonl`、`dev.jsonl`、`aime2026.jsonl`、`beyondaime.jsonl`。不能拿只有少数题的 smoke 数据来做排除表。使用环境中已有 W&B 登录；脚本不会读取或打印 API key。默认 project 为 `yuningyangaillm/MATH_rsi`，可通过现有环境变量覆盖。

默认持久化时限 120 分钟，重启沿用原截止时间；到时只终止自身子进程、保留 checkpoint。**这不关闭 RunPod，也不停止 Pod 计费。** 相同代码/数据/配置下重新调用 `start` 会恢复；配置或代码改变必须使用新目录。若原时限已过，需明确建立新预算方案，不能靠反复重启续费。脚本启动 tmux 成功只说明已派发；查看父运行和训练子运行，才能确认 SFT 真正开始。

本地原始 Numina JSONL 可通过 `start --raw-jsonl /path/raw.jsonl` 传入。其内容须保留 problem、solution、answer 和 validity 字段；文件 SHA 被冻结，但本地文件并不自动证明来自所声明上游 revision。

## 输出和完成条件

```text
margent-expert-sft-01/
  expert_run.json, budget.json, expert_status.json
  data/manifest.json, references.jsonl, exclusions.jsonl
  data/{extractor,reasoner,verifier}/{train,dev}.jsonl
  training/{extractor,reasoner,verifier}/
    adapter_model.safetensors, adapter_config.json, tokenizer*
    training_run.json, summary.json, sft_data_report.json, dev_metrics.json
    checkpoint-*/  # optimizer / scheduler / RNG / 完整性标记
  reload_smoke/summary.json
  experts.json, manager_config.json, expert_report.json
  logs/*.log, controller_traceback.txt
```

父级 `experts_complete=true` 仅在三个角色达到各自步数、全部文件指纹通过、权重实际重载并完成角色切换后写入。它证明训练/服务流程完成，**不证明角色质量合格或 benchmark 提升**。单个角色 Finished 不能替代总完成标记。超长样本整条排除，报告保留/排除数量，不静默截断。

W&B 的一个 group 包含父总控、三个 `expert_sft` 子运行及 `expert_reload`。记录角色、数据/模型/模板/代码指纹、train/dev loss、optimizer steps、梯度范数、学习率、实际监督 token、可训练模块、GPU 采样、恢复来源和错误。`experiment-evidence` artifact 保留 manifest、日志、traceback、数据筛除报告、checkpoint 文件清单与 hash；不上传完整权重。`MARGENT_WANDB_TEXT=1` 才上传训练文本/参考 sidecar/生成结果，默认 RunPod 脚本已启用，与此前实验一致；关闭后仍记录计数和指纹。上传超出证据大小限制时会记录遗漏，不冒充完整。

## 专家 dev 对照与后续接入

训练后先检查各角色数据和输出；人工审核字段默认 false，不伪造审核结果。可单独运行固定 base prompt-only 与 SFT 专家的 dev 对照：

```bash
CUDA_VISIBLE_DEVICES=0 "$EXPERT_PYTHON" -m src.verifiable.expert_eval \
  --bundle "$EXPERT_ROOT/experts.json" --data-dir "$EXPERT_ROOT/data" \
  --out "$EXPERT_ROOT/dev_comparison" --limit 32 --max-tokens 512 --minutes 120
```

报告输出有效/截断率、局部算术 verifier 的可解析 verdict 和已出现类别 macro-F1，保存人工审阅材料。Extractor/Reasoner 的事实与推导质量、Verifier 的完整证明审核能力仍需人工/执行器评估；对 Manager 的帮助需在固定 Manager dev 上另做配对实验。这个命令不读 AIME 来选专家。

确认专家质量后，冻结 `experts.json`，Manager 使用生成的 `manager_config.json`。默认 RSI pilot 脚本现在使用这份配置：

```bash
export RSI_EXPERT_ROOT="$EXPERT_ROOT"
bash scripts/runpod_rsi_pilot.sh advisor
# 在另一个终端中运行，Manager 使用另一张 GPU：
bash scripts/runpod_rsi_pilot.sh plan
bash scripts/runpod_rsi_pilot.sh run
```

Manager 首轮 SFT 的目标仍来自 **独立 Manager Numina train 池**上的反事实采集：当前 Manager 独立解答，调用冻结 E*/R*/V*，用外部答案校验器选择成功分支，训练 CALL/COMMIT、成功修订和独立解答蒸馏。它不会直接拿专家的角色答案或专家 dev 做 Manager SFT。

若做预先声明的 prompt-only 消融，显式设置 `RSI_CONFIG=configs/math_rsi_actions.json`；需新输出目录/group。AIME baseline 若要用专家，可向 `scripts/runpod_aime_baseline.py` 传 `--config "$EXPERT_ROOT/manager_config.json" --out /workspace/margent-aime-experts-01`。AIME/BeyondAIME 只作为锁定测试，当前专家入口不会启动它们。
