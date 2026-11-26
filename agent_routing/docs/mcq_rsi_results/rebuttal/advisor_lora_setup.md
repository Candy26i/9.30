# Advisor LoRA adapters: training setup

Compiled on 2026-11-26 from the files that the training runs saved next to the adapters on Hugging Face (`adapter_config.json`, the
safetensors header, `training_args.bin`, `trainer_state.json`; for v2 the run reports), from the repository code as it was when the
adapters were trained (git history), and from the data manifests and data files. Column "Source": **run** = stored by the training run
itself; **code** = training code, documentation or data files only (the run did not store it); **—** = not recorded anywhere.
`training_args.bin` was read with a parser that executes nothing; no adapter weights were downloaded (the layer layout comes from the
safetensors header). Every value was checked a second time by an independent recomputation. Details and file references:
`configuration.md`, section 2.

**What the paper's numbers used.** In every reported number (paper, RSI and rebuttal) the Extractor, Reasoner and Verifier were the
base model Qwen3.5-9B under role-specific prompts: the adapters below were trained, but the paper-era server did not apply them, and the
RSI and rebuttal runs use the base model by configuration (section 5). This document describes the adapters themselves.

## 1. The adapters trained for the paper ("registered", August 2026)

Repositories (public): `MaliDDD/agent-routing-advisors-{medqa,mmlu_pro,gpqa,aqua}-9b`, folders `extractor_adapter`,
`reasoner_adapter`, `verifier_adapter`. Uploaded 2026-08-04 (MedQA), 2026-08-10 (GPQA, MMLU-Pro), 2026-08-17 (AQuA-RAT).
The optimisation rows are recorded for MedQA, MMLU-Pro and GPQA; for AQuA-RAT only the LoRA rows are recorded (see below).

| Group | Parameter | Value | Source |
|:--|:--|:--|:--|
| Model | Base model | Qwen/Qwen3.5-9B (revision not recorded) | run |
| LoRA | Library | PEFT 0.20.0 | run |
| LoRA | Rank r / alpha / dropout | 16 / 32 / 0.05 | run |
| LoRA | Bias, DoRA, rsLoRA, modules_to_save | none, no, no, none | run |
| LoRA | Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | run |
| LoRA | Layers actually adapted | MLP (gate/up/down) in all 32 layers; attention (q/k/v/o) only in the 8 full-attention layers 3, 7, 11, …, 31; the linear-attention projections of the other 24 layers are not adapted | run (weight header) + base model config |
| LoRA | Trainable parameters | 29,097,984 per adapter (saved in fp32, 116 MB) | run (weight header) |
| Optimisation | Optimizer | AdamW (`adamw_torch_fused`), betas 0.9 / 0.999, eps 1e-8 | run |
| Optimisation | Weight decay | 0.0 | run |
| Optimisation | Gradient clipping | max grad norm 1.0 | run |
| Optimisation | Learning rate | 2e-4 | run |
| Optimisation | Schedule / warm-up | linear decay to 0 / no warm-up | run |
| Optimisation | Batch | 1 per device × gradient accumulation 8 = effective 8, one GPU | run |
| Optimisation | Epochs | 10 (scheduled) | run |
| Optimisation | Optimizer steps | 750 (MedQA, MMLU-Pro: 75 per epoch); 200 (GPQA: 20 per epoch) | run |
| Optimisation | Seed | 42 | run |
| Compute | Precision | bf16 | run |
| Compute | Gradient checkpointing | on for GPQA, off for MedQA and MMLU-Pro | run |
| Validation | Validation set, evaluation during training | none | run |
| Selection | Early stopping / best checkpoint | none; the exported adapter is the last stored checkpoint (epoch 10; MedQA Verifier epoch 8, below) | run |
| Data | Maximum sequence length | 4096 (code default; the run does not store it); the August code cut longer rows from the right | code |
| Loss | Supervised tokens | response only: system and user prompt masked, loss on the assistant's answer and its end token | code |
| Format | Chat template | the base model's Qwen3.5 template, saved with the adapters (MedQA Verifier: in its checkpoint folders only); thinking disabled (`enable_thinking=False`) | run (template) + code |

Per benchmark (training rows are Extractor / Reasoner / Verifier):

| Benchmark | Training rows | Steps | Exported checkpoint | Last logged training loss (E / R / V) |
|:--|:--|--:|:--|:--|
| MedQA | 600 / 600 / 600 | 750 | epoch 10; **Verifier: epoch 8 of 10** (step 600 of 750; no later checkpoint stored) | 0.0040 / 0.0031 / 0.0171 |
| MMLU-Pro | 600 / 600 / 598 | 750 | epoch 10 | 0.0023 / 0.0027 / 0.0054 |
| GPQA | 160 / 160 / 160 | 200 | epoch 10 | 0.0077 / 0.0120 / 0.0148 |
| AQuA-RAT | 600 / 598 / 599 | — | — | — |

**AQuA-RAT:** the repository holds the three final adapters only (no checkpoints, no training arguments, no training state). Its
LoRA configuration and layer layout are the same as above (run), but every optimisation and compute setting (optimizer, learning rate
and schedule, batch, epochs, seed, precision) is not recorded, and the written sources conflict: the AQuA run sheet says 3 epochs at
5e-5 on 500 samples per role (`AQUA_ARC_BENCHMARKS.md`), a general docstring says the paper-era advisors were trained for 10 epochs on
600 examples per role (`scripts/train_subagents_v2.py`); 600 AQuA questions are registered.

**Not the documented commands:** every documented advisor-training command for an 8B or 9B base model (AQuA and ARC run sheets for
Qwen3.5-9B; MedQA and GPQA commands written for Qwen3-8B) uses 3 epochs (GPQA 4) at 5e-5 and no validation file; the legacy commands for
Qwen3-0.6B and Qwen3-4B use 2e-4 and 1e-4. The stored records (10 epochs, 2e-4) show that the MedQA and GPQA commands were not the ones
run. No MMLU-Pro command is documented; the AQuA run is not recorded; the ARC sheet belongs to no registered adapter. No committed version
of `src/subagents/train.py`, the trainer of the registered adapters, has early stopping; validation-based selection appears only in the
v2 script of 2 October 2026 (section 3).

## 2. Training data of the registered adapters

| Item | Value | Source |
|:--|:--|:--|
| Teacher | DeepSeek through an offline export/import bridge; model label `deepseek_flash` (Extractor and Verifier of all four benchmarks; all three roles for AQuA-RAT) and `deepseek-v4-flash` (Reasoner of MedQA, MMLU-Pro, GPQA); the AQuA run sheet names `deepseek-chat`; the generator script, endpoint and sampling settings are not in the repository | code (data files) |
| Teacher prompts | built without the gold answer (ground truth stored only as a separate field) | code |
| Targets | the teacher's text as returned (leading and trailing whitespace stripped), paired with the runtime prompt; the importer applies no JSON or schema filter and drops only empty responses (none). Exception: the MMLU-Pro Verifier was trained on a separately filtered file (598 of 600 rows) that drops examples 4266 and 11711, two non-JSON responses of 30,401 and 26,369 characters that stop mid-sentence; the script and criterion behind that file are not recorded | code (data files) |
| Examples per role | about 600 (MedQA, MMLU-Pro, AQuA-RAT), 160 (GPQA); exactly: MMLU-Pro Verifier 598, AQuA-RAT Reasoner 598 and Verifier 599, all others 600 or 160 | run (logged epoch fraction; MedQA, MMLU-Pro, GPQA) + data files (AQuA-RAT) |
| Question source | MedQA: official train split. MMLU-Pro: the training part of the random split (MMLU-Pro has no train split). GPQA: `gpqa_train446` (no Diamond-100 question). AQuA-RAT: official train split. All sampled with seed 42 | code (reproduced exactly) |
| Overlap with evaluation pools | none for MedQA, GPQA, AQuA-RAT; MMLU-Pro: 3 development and 4 `test_paper` questions, none of the rebuttal test pool | data manifests |
| Verifier candidate | about half of the Verifier rows include a candidate answer to audit, drawn uniformly over the choices (MedQA 298 / 600, MMLU-Pro 304 / 598, GPQA 74 / 160, AQuA-RAT 297 / 599) | data files |
| System prompts | one per role; Extractor and Verifier prompts shared by all benchmarks, Reasoner prompt in three variants (medical; academic for MMLU-Pro and GPQA; algebra for AQuA-RAT); `src/manager/mcq_rsi/prompts/` | code |
| User message | `QUESTION`, `CHOICES`, `CONTEXT`; for the Verifier with a candidate also `CANDIDATE ANSWER TO AUDIT: X` and "Specifically assess whether this candidate is well-supported."; then "Produce the JSON object." | code |
| Output | one JSON object per role: Extractor `key_evidence, extracted_facts, missing_info, context_summary, confidence`; Reasoner `case_facts, task_type, decision_factors, knowledge_slots, candidate_considerations, missing_information, format_confidence`; Verifier `relevant_principles, checks, potential_errors, candidate_answer_audit, uncertainty_notes, confidence` | code |

## 3. The retrained adapters with a validation split and best-epoch selection ("v2", 2 October 2026)

Repositories (public): `MaliDDD/agent-routing-advisors-{medqa,mmlu_pro,gpqa,aqua}-9b-v2`; script `scripts/train_subagents_v2.py`.
Used only for the advisor A/B below, not for any paper or rebuttal number.

| Parameter | v2 | Registered (MedQA, MMLU-Pro, GPQA; AQuA-RAT not recorded) |
|:--|:--|:--|
| Base model | Qwen/Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | same model, revision not recorded |
| LoRA r / alpha / dropout / target modules | 16 / 32 / 0.05 / same 7 modules (same layer layout) | same |
| Learning rate | 2e-4 | 2e-4 |
| Batch | 1 × accumulation 8 | same |
| Seed | 42 | 42 |
| Maximum sequence length | 4096 (16384 for the GPQA Extractor and Reasoner) | 4096 (code default) |
| Data cleaning | development and test questions removed; teacher responses that are not one complete JSON object removed | none (MMLU-Pro Verifier: two rows removed, section 2) |
| Validation | 15% of the remaining questions (split by question), loss after every epoch | none |
| Epochs | 3; all 12 runs trained the full 3 epochs (no run was stopped early) | 10 (scheduled) |
| Exported adapter | the epoch with the lowest validation loss, chosen after training: **epoch 1 in all 12 runs** (validation loss rose after every epoch) | last stored checkpoint: epoch 10; MedQA Verifier epoch 8 of 10 |
| Software | PEFT 0.18.1, transformers 5.3.0, torch 2.8.0 | PEFT 0.20.0; transformers and torch not recorded |

v2 rows (train / validation) and validation loss by epoch (1 / 2 / 3):

| Benchmark | Extractor | Reasoner | Verifier |
|:--|:--|:--|:--|
| MedQA | 510 / 90; 0.353 / 0.367 / 0.409 | 510 / 90; 0.475 / 0.490 / 0.535 | 510 / 90; 0.583 / 0.598 / 0.635 |
| MMLU-Pro | 507 / 89; 0.381 / 0.403 / 0.468 | 503 / 88; 0.368 / 0.379 / 0.412 | 505 / 89; 0.573 / 0.584 / 0.620 |
| GPQA | 133 / 23; 0.407 / 0.412 / 0.433 | 130 / 23; 0.530 / 0.534 / 0.544 | 128 / 22; 0.614 / 0.616 / 0.627 |
| AQuA-RAT | 510 / 90; 0.321 / 0.334 / 0.377 | 508 / 90; 0.439 / 0.448 / 0.487 | 509 / 90; 0.626 / 0.628 / 0.657 |

## 4. Do the adapters help? (A/B on the development pools)

754 development questions (MedQA 200, MMLU-Pro 200, GPQA 100, AQuA-RAT 254). A fixed base-model solver writes a draft, one advisor
answers (the Verifier gets the draft as its candidate), and the same solver answers again; only the advisor's weights differ. The
adapters are loaded correctly here (renamed parameter names, section 5). Accuracy (%) after the advisor, pooled; Δ against the base
model with the same role prompt, exact sign test (`paper_assets/tab/tab_app_advisors.md`, per benchmark there).

| Role | Base model | Registered LoRA (loaded correctly) | v2 LoRA (best-epoch selection) |
|:--|--:|--:|--:|
| Extractor | 79.7 | 80.4 (+0.7, p = .54) | 80.8 (+1.1, p = .34) |
| Reasoner | 80.6 | 80.1 (−0.5, p = .70) | 79.6 (−1.1, p = .33) |
| Verifier | 81.4 | 79.8 (−1.6, p = .16) | 78.2 (−3.2, p = .006) |

## 5. How the adapters were served

- Paper era (described from the MedQA, MMLU-Pro and GPQA server logs, which are not in the repository; for AQuA-RAT the run sheet starts
  the legacy launcher): vLLM 0.26.0 with the three adapters registered as LoRA modules. The adapter keys
  (`base_model.model.model.layers.N…`, saved from the text-only model class) match no module of the multimodal model class that vLLM
  serves, so vLLM loads them without error and answers with the base weights. Replay evidence (`configuration.md`, section 2): the
  Verifier adapter as served returned text identical to the base model on 20 of 20 prompts (a copy with renamed keys: 0 of 20); the
  Extractor on 5 of 20 (median similarity 0.924 as served, 0.392 for the renamed copy); no identity check is recorded for the Reasoner.
  A correctly applied copy only renames the keys (the weights are byte-identical); the A/B above uses such copies.
- The RSI runs and every rebuttal experiment set `advisor_mode: "base"`: the advisors are the base model under each role's prompt, with
  no LoRA, and every result records this.

## 6. Not recorded

Training: for all adapters, the training command lines. For the registered runs: the base-model revision, the transformers and torch
versions, the GPU type, the maximum sequence length actually used, and whether the MedQA Verifier run was interrupted after epoch 8 or
its last two checkpoints were not uploaded. For AQuA-RAT: every optimisation and compute setting and which checkpoint was exported (its
LoRA configuration and layer layout are recorded, section 1).

Data: the teacher's generator script, endpoint and sampling settings; which file each registered run trained on (`training_args.bin`
stores only the output folder, so the link to the data files rests on the registry pins and the matching row counts); the script and
criterion that produced the MMLU-Pro Verifier's filtered file.

Serving (paper era): the command lines of the MedQA and MMLU-Pro vLLM servers and whether the legacy launcher was run as written for GPQA
and AQuA-RAT; which Reasoner prompt was live on the paper-era servers for MMLU-Pro, GPQA and AQuA-RAT; a token-for-token reproduction of
the paper-era advisor outputs; an identity check for the Reasoner.

## 7. Suggested text for the response

"In every reported number the Extractor, Reasoner and Verifier are the base model Qwen3.5-9B under role-specific prompts. Role adapters
were trained but were not applied at serving time: LoRA (rank 16, alpha 32, dropout 0.05, on the MLP projections of all layers and the
attention projections of the full-attention layers), trained with response-only supervised fine-tuning on about 600 teacher (DeepSeek)
outputs per role (160 for GPQA), with AdamW, learning rate 2e-4 with linear decay, effective batch 8, a 10-epoch schedule and bf16
(recorded for MedQA, MMLU-Pro and GPQA; not recorded for AQuA-RAT). With the adapters loaded correctly, a controlled A/B on 754
development questions shows no significant gain over the base model in the same role (pooled Δ +0.7 / −0.5 / −1.6 points for Extractor
/ Reasoner / Verifier; exact sign test p = .54 / .70 / .16), and adapters retrained with a 15% validation split and best-epoch selection
show none either (+1.1 / −1.1 / −3.2; the retrained Verifier is significantly lower, p = .006)."

This wording is consistent with the advisor correction in the response (common block, item 1, and change 9). Do not write "each advisor
is a LoRA adapter": the paper's numbers were produced with the advisors as served, which the recorded evidence identifies as the base
model. Do not link the `MaliDDD/…` repositories in the response (double-blind review).
