### Training-seed replicates of the manager, pool `dev`

Checkpoints: `s1`, `seed1`, `seed2`, `seed3`. `s1` is the paper's checkpoint; `seedK` is the same label file retrained with the recipe printed in the paper-era training logs (LoRA r16 / alpha 32, lr 1e-5, 10 epochs, batch 1 x 8) and training seed K. The original run's seed and maximum sequence length were not logged (command defaults 42 and 4096 are used for the re-trainings), and the re-trainings use the fast linear-attention kernels, so the replicates differ from `s1` in the seed and possibly in those unlogged details. Each cell lists the checkpoints in that order, then mean ± sample SD over the ones present.

**MedQA** (n = 200; seeds present: s1)

| | s1 | mean ± SD |
|:--|--:|--:|
| Candidate only, accuracy (%) | 70.5 | 70.5 |
| Learned calling (first call), accuracy (%) | 79.0 | 79.0 |
| Learned calling, calls per question | 0.37 | 0.37 |
| Random questions, same calls per advisor, accuracy (%) | 75.7 | 75.7 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 80.5 | 80.5 |
| Always Verifier with the candidate, accuracy (%) | 84.5 | 84.5 |
| Always Verifier, candidate removed, accuracy (%) | 85.5 | 85.5 |
| Empty-output revision turn, accuracy (%) | 71.0 | 71.0 |
| Share of random draws (same calls per advisor) ≥ learned calling | .024 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -5.5 (.027) | -5.5 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -1.5 (.664) | -1.5 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | -1.0 (.824) | -1.0 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +13.5 (<.001) | 13.5 |

**MMLU-Pro** (n = 200; seeds present: s1)

| | s1 | mean ± SD |
|:--|--:|--:|
| Candidate only, accuracy (%) | 55.5 | 55.5 |
| Learned calling (first call), accuracy (%) | 62.0 | 62.0 |
| Learned calling, calls per question | 0.30 | 0.30 |
| Random questions, same calls per advisor, accuracy (%) | 60.4 | 60.4 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 65.0 | 65.0 |
| Always Verifier with the candidate, accuracy (%) | 73.5 | 73.5 |
| Always Verifier, candidate removed, accuracy (%) | 69.0 | 69.0 |
| Empty-output revision turn, accuracy (%) | 57.0 | 57.0 |
| Share of random draws (same calls per advisor) ≥ learned calling | .185 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -11.5 (<.001) | -11.5 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -3.0 (.345) | -3.0 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +4.5 (.122) | 4.5 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +16.5 (<.001) | 16.5 |

**GPQA** (n = 100; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 41.0 | 40.0 | 40.0 | 39.0 | 40.0 ± 0.8 |
| Learned calling (first call), accuracy (%) | 53.0 | 53.0 | 54.0 | 55.0 | 53.8 ± 1.0 |
| Learned calling, calls per question | 0.62 | 0.61 | 0.56 | 0.74 | 0.63 ± 0.08 |
| Random questions, same calls per advisor, accuracy (%) | 49.1 | 48.4 | 47.8 | 50.3 | 48.9 ± 1.1 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 57.0 | 54.0 | 53.0 | 58.0 | 55.5 ± 2.4 |
| Always Verifier with the candidate, accuracy (%) | 57.0 | 57.0 | 57.0 | 57.0 | 57.0 ± 0.0 |
| Always Verifier, candidate removed, accuracy (%) | 52.0 | 50.0 | 50.0 | 50.0 | 50.5 ± 1.0 |
| Empty-output revision turn, accuracy (%) | 43.0 | 41.0 | 41.0 | 40.0 | 41.2 ± 1.3 |
| Share of random draws (same calls per advisor) ≥ learned calling | .111 | .054 | .016 | .056 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -4.0 (.388) | -4.0 (.388) | -3.0 (.581) | -2.0 (.727) | -3.2 ± 1.0 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -4.0 (.388) | -1.0 (1.000) | +1.0 (1.000) | -3.0 (.508) | -1.8 ± 2.2 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +5.0 (.424) | +7.0 (.210) | +7.0 (.210) | +7.0 (.210) | 6.5 ± 1.0 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +14.0 (.020) | +16.0 (.005) | +16.0 (.005) | +17.0 (.002) | 15.8 ± 1.3 |

**AQuA** (n = 254; seeds present: s1)

| | s1 | mean ± SD |
|:--|--:|--:|
| Candidate only, accuracy (%) | 39.0 | 39.0 |
| Learned calling (first call), accuracy (%) | 73.2 | 73.2 |
| Learned calling, calls per question | 0.78 | 0.78 |
| Random questions, same calls per advisor, accuracy (%) | 71.8 | 71.8 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 76.0 | 76.0 |
| Always Verifier with the candidate, accuracy (%) | 81.1 | 81.1 |
| Always Verifier, candidate removed, accuracy (%) | 75.6 | 75.6 |
| Empty-output revision turn, accuracy (%) | 38.6 | 38.6 |
| Share of random draws (same calls per advisor) ≥ learned calling | .197 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -7.9 (<.001) | -7.9 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -2.8 (.296) | -2.8 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +5.5 (.044) | 5.5 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +42.5 (<.001) | 42.5 |
