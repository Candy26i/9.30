# research_0703 — Learning When to Commit

Training an 8B orchestrator that learns the **delegate-or-commit** decision:
at every step, either delegate a cognitive subtask to one of three frozen
specialist advisors (extractor / reasoner / verifier) or commit to an answer.
The manager exposes a `DRAFT_ANSWER`, learns routing from paired
counterfactual branches, and is then optimized with GRPO using only **binary
final correctness**. Among successful branches, the counterfactual cold start
prefers the shortest one; wrong/no-call trajectories receive no artificial
advantage.

- **System and pipeline**: [`agent_routing/README.md`](agent_routing/README.md)
- **Free-response math and RunPod iteration**: [`agent_routing/MATH_RUNPOD.md`](agent_routing/MATH_RUNPOD.md)
- **Current full experiment plan (data budgets, gates, every command)**:
  [`agent_routing/MARGINAL_VALUE_EXPERIMENTS.md`](agent_routing/MARGINAL_VALUE_EXPERIMENTS.md)
- **Historical ADC ablation plan**:
  [`agent_routing/EXPERIMENTS.md`](agent_routing/EXPERIMENTS.md)

Snapshot lineage: forked from `research_6.8` (rule_applier era); synced to the
2026-07-03 codebase. See the migration table at the top of EXPERIMENTS.md
before reusing any artifacts produced by the old snapshot.

## RSI math pilot

The `main` branch includes a bounded two-round
collect → SFT → GRPO pilot with dynamic MARGENT, static-data, and
successful-trajectory controls. This uses `python -m src.verifiable.rsi`;
the historical SFT-only loop and disabled legacy RL entry point are unchanged.

- [RunPod setup and commands](agent_routing/docs/RSI_RUNPOD.md)
- [Literature review and experiment design](agent_routing/docs/RSI_RESEARCH_DESIGN.md)
- [Validation and GPU limitations](agent_routing/docs/RSI_VALIDATION.md)
- [Main audit fixes and paper logging contract](agent_routing/docs/AUDIT_AND_PAPER_LOGGING.md)

## MARGENT RSI full experiment plan

- [Complete experiment plan (Chinese): training stages, benchmarks, controls, budgets, and current status](agent_routing/docs/MARGENT_RSI_EXPERIMENT_PLAN.md)
- [Subagent training plan (Chinese): role data, LoRA SFT, validation, frozen serving, and implementation gaps](agent_routing/docs/SUBAGENT_TRAINING_PLAN.md)

The math workflow starts with OpenAI/gpt-4o synthesis on an isolated Numina question pool, then trains three independent role SFT adapters and freezes them for Manager training. Teacher generation, data isolation, resumable SFT, role serving and W&B evidence are connected. Real teacher API generation, 9B GPU execution and independent role-quality evaluation still need to be performed. Rule-based labels are an explicit debug ablation only.

- [Expert-first RunPod entry point and supervision limits](agent_routing/docs/EXPERT_SFT_RUNPOD.md)
