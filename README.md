# MARGENT — Learning When to Delegate

MARGENT trains a Manager language model to solve a problem independently, decide whether an expert can help, and commit to a final answer. The current math workflow first trains three specialist adapters, freezes them, then alternates Manager counterfactual data collection, supervised fine-tuning (SFT), and group-relative policy optimization (GRPO).

## Start here

| Document | Use it for |
|---|---|
| **[Experiment runbook](agent_routing/docs/MARGENT_END_TO_END_RUNBOOK.md)** | Complete RunPod commands: setup → expert SFT → Manager SFT/GRPO on BeyondAIME → AIME2026 → W&B → recovery and backup |
| **[Architecture](agent_routing/README.md)** | Model roles, decision protocol, training objectives, data flow and source map |
| [Dataset card](agent_routing/data/math_luna_codex_pilot_20260929/README.md) | Published Numina/Codex Luna examples, JSON/JSONL layout, provenance and quality limits |

The runbook is the single operational reference for the current math experiment. It defines the shared environment, exact code/model revisions, output paths, budgets, completion checks and monitoring fields. Execute its setup first; isolated commands copied from historical benchmark guides use different assumptions.

## Current math experiment

- **Student model:** `Qwen/Qwen3.5-9B`, pinned revision; three independent expert LoRAs and a separate Manager LoRA.
- **Experts:** Extractor, Reasoner and Verifier. Train them on the published 544 role examples, evaluate/review them, then freeze them for Manager training.
- **Manager data:** BeyondAIME split 64 train / 36 dev (`agent_routing/data/manager_beyond_rsi_20260930`); the default mechanism pilot uses 16 / 16. Experts still train on the Numina Luna data.
- **Manager cold start:** successful routes collected from the initial Manager with the frozen experts. Expert teacher answers are not copied directly into the Manager SFT set.
- **Iterative comparison:** dynamic, static and success arms; two SFT/GRPO rounds and 31 controller stages in the default pilot.
- **Held-out test:** all 30 AIME2026 questions, with independent and tool-assisted policy accuracy reported separately. BeyondAIME is RSI train/dev data and is never reported as held-out.
- **Tracking:** W&B project `MATH_rsi` under the entity set in the environment file, local question records, configuration/checkpoint identities, heartbeats and evidence artifacts.

This is a bounded iterative parameter-training experiment. A completed smoke test or increasing training reward does not establish benchmark improvement. The default pilot is small, the synthetic expert labels are not mathematically certified, and documentation/CPU checks are not 9B CUDA validation.

## Repository layout

```text
agent_routing/
  README.md                         Architecture and implementation map
  docs/MARGENT_END_TO_END_RUNBOOK.md Current math experiment instructions
  docs/legacy/BENCHMARK_RUNBOOK.md   Historical structured/MCQ walkthroughs
  data/math_luna_codex_pilot_20260929/
                                    Expert data and frozen Manager/test pools
  configs/                          Experiment configurations
  scripts/                          Setup, controllers and utility entry points
  src/verifiable/                   Free-response math, expert SFT and Manager RSI
  src/{benchmarks,subagents,manager,pipeline,teachers,utils}/
                                    Shared and historical benchmark components
  tests/                            Unit, protocol and optional CPU integration tests
  outputs/                          Preserved historical artifacts
```

## Other benchmark workflows

The repository also retains MedQA, LegalBench, MMLU-Pro, GPQA, AQuA-RAT and ARC-Challenge workflows. Their structured outputs and training entry points differ from the current math protocol. Start with the [historical benchmark walkthrough](agent_routing/docs/legacy/BENCHMARK_RUNBOOK.md), [MCQ marginal-value experiments](agent_routing/MARGINAL_VALUE_EXPERIMENTS.md), or [AQuA/ARC instructions](agent_routing/AQUA_ARC_BENCHMARKS.md). Additional historical plans are indexed in the architecture document.

## License and data attribution

See the [code license](agent_routing/LICENSE). Dataset licenses and pinned upstream attribution are listed separately in [source notices](agent_routing/data/math_luna_codex_pilot_20260929/source_notices/NOTICE.md); the code license does not replace their terms.
