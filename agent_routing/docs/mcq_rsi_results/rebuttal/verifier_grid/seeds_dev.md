### Training-seed replicates of the manager, pool `dev`

Checkpoints: `s1`, `seed1`, `seed2`, `seed3`. `s1` is the paper's checkpoint; `seedK` is the same label file retrained with the recipe printed in the paper-era training logs (LoRA r16 / alpha 32, lr 1e-5, 10 epochs, batch 1 x 8) and training seed K. The original run's seed and maximum sequence length were not logged (command defaults 42 and 4096 are used for the re-trainings), and the re-trainings use the fast linear-attention kernels, so the replicates differ from `s1` in the seed and possibly in those unlogged details. Each cell lists the checkpoints in that order, then mean ± sample SD over the ones present.

**MedQA** (n = 200; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 70.5 | 69.0 | 70.0 | 72.0 | 70.4 ± 1.2 |
| Learned calling (first call), accuracy (%) | 79.0 | 81.0 | 81.0 | 82.0 | 80.8 ± 1.3 |
| Learned calling, calls per question | 0.37 | 0.36 | 0.40 | 0.34 | 0.37 ± 0.02 |
| Random questions, same calls per advisor, accuracy (%) | 75.7 | 74.6 | 75.8 | 76.5 | 75.6 ± 0.8 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 80.5 | 77.5 | 78.0 | 79.5 | 78.9 ± 1.4 |
| Always Verifier with the candidate, accuracy (%) | 84.5 | 83.5 | 84.0 | 85.5 | 84.4 ± 0.9 |
| Always Verifier, candidate removed, accuracy (%) | 85.5 | 85.0 | 85.0 | 85.5 | 85.2 ± 0.3 |
| Empty-output revision turn, accuracy (%) | 71.0 | 69.0 | 70.0 | 72.5 | 70.6 ± 1.5 |
| Share of random draws (same calls per advisor) ≥ learned calling | .024 | <.001 | .001 | <.001 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -5.5 (.027) | -2.5 (.359) | -3.0 (.238) | -3.5 (.189) | -3.6 ± 1.3 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -1.5 (.664) | +3.5 (.167) | +3.0 (.263) | +2.5 (.383) | 1.9 ± 2.3 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | -1.0 (.824) | -1.5 (.664) | -1.0 (.824) | +0.0 (1.000) | -0.9 ± 0.6 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +13.5 (<.001) | +14.5 (<.001) | +14.0 (<.001) | +13.0 (<.001) | 13.8 ± 0.6 |

**MMLU-Pro** (n = 200; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 55.5 | 53.5 | 51.0 | 52.0 | 53.0 ± 2.0 |
| Learned calling (first call), accuracy (%) | 62.0 | 63.0 | 59.5 | 60.5 | 61.2 ± 1.6 |
| Learned calling, calls per question | 0.30 | 0.39 | 0.41 | 0.40 | 0.37 ± 0.05 |
| Random questions, same calls per advisor, accuracy (%) | 60.4 | 60.2 | 58.3 | 58.8 | 59.4 ± 1.0 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 65.0 | 66.5 | 65.5 | 66.0 | 65.8 ± 0.6 |
| Always Verifier with the candidate, accuracy (%) | 73.5 | 72.5 | 71.0 | 72.0 | 72.2 ± 1.0 |
| Always Verifier, candidate removed, accuracy (%) | 69.0 | 69.0 | 68.0 | 68.5 | 68.6 ± 0.5 |
| Empty-output revision turn, accuracy (%) | 57.0 | 55.0 | 52.5 | 53.0 | 54.4 ± 2.1 |
| Share of random draws (same calls per advisor) ≥ learned calling | .185 | .048 | .266 | .178 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -11.5 (<.001) | -9.5 (<.001) | -11.5 (<.001) | -11.5 (<.001) | -11.0 ± 1.0 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -3.0 (.345) | -3.5 (.230) | -6.0 (.036) | -5.5 (.071) | -4.5 ± 1.5 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +4.5 (.122) | +3.5 (.248) | +3.0 (.327) | +3.5 (.248) | 3.6 ± 0.6 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +16.5 (<.001) | +17.5 (<.001) | +18.5 (<.001) | +19.0 (<.001) | 17.9 ± 1.1 |

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

**AQuA** (n = 254; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 39.0 | 37.8 | 37.4 | 37.8 | 38.0 ± 0.7 |
| Learned calling (first call), accuracy (%) | 73.2 | 74.0 | 74.8 | 72.0 | 73.5 ± 1.2 |
| Learned calling, calls per question | 0.78 | 0.77 | 0.79 | 0.75 | 0.77 ± 0.02 |
| Random questions, same calls per advisor, accuracy (%) | 71.8 | 71.4 | 72.3 | 70.2 | 71.4 ± 0.9 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 76.0 | 75.2 | 76.4 | 74.0 | 75.4 ± 1.0 |
| Always Verifier with the candidate, accuracy (%) | 81.1 | 81.5 | 81.9 | 81.1 | 81.4 ± 0.4 |
| Always Verifier, candidate removed, accuracy (%) | 75.6 | 75.6 | 75.6 | 75.6 | 75.6 ± 0.0 |
| Empty-output revision turn, accuracy (%) | 38.6 | 37.8 | 37.0 | 39.4 | 38.2 ± 1.0 |
| Share of random draws (same calls per advisor) ≥ learned calling | .197 | .045 | .043 | .129 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -7.9 (<.001) | -7.5 (<.001) | -7.1 (<.001) | -9.1 (<.001) | -7.9 ± 0.9 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -2.8 (.296) | -1.2 (.728) | -1.6 (.597) | -2.0 (.511) | -1.9 ± 0.7 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +5.5 (.044) | +5.9 (.036) | +6.3 (.023) | +5.5 (.044) | 5.8 ± 0.4 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +42.5 (<.001) | +43.7 (<.001) | +44.9 (<.001) | +41.7 (<.001) | 43.2 ± 1.4 |
