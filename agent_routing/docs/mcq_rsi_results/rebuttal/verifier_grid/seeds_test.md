### Training-seed replicates of the manager, pool `test`

Checkpoints: `s1`, `seed1`, `seed2`, `seed3`. `s1` is the paper's checkpoint; `seedK` is the same label file retrained with the recipe printed in the paper-era training logs (LoRA r16 / alpha 32, lr 1e-5, 10 epochs, batch 1 x 8) and training seed K. The original run's seed and maximum sequence length were not logged (command defaults 42 and 4096 are used for the re-trainings), and the re-trainings use the fast linear-attention kernels, so the replicates differ from `s1` in the seed and possibly in those unlogged details. Each cell lists the checkpoints in that order, then mean ± sample SD over the ones present.

**MedQA** (n = 500; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 73.4 | 74.4 | 74.6 | 73.2 | 73.9 ± 0.7 |
| Learned calling (first call), accuracy (%) | 78.8 | 78.0 | 78.6 | 79.0 | 78.6 ± 0.4 |
| Learned calling, calls per question | 0.37 | 0.35 | 0.38 | 0.36 | 0.36 ± 0.01 |
| Random questions, same calls per advisor, accuracy (%) | 76.6 | 77.0 | 77.4 | 76.5 | 76.9 ± 0.4 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 79.6 | 79.0 | 80.2 | 81.0 | 80.0 ± 0.9 |
| Always Verifier with the candidate, accuracy (%) | 83.4 | 82.8 | 83.2 | 84.8 | 83.5 ± 0.9 |
| Always Verifier, candidate removed, accuracy (%) | 84.2 | 84.4 | 84.2 | 84.4 | 84.3 ± 0.1 |
| Empty-output revision turn, accuracy (%) | 72.8 | 73.2 | 73.0 | 72.6 | 72.9 ± 0.3 |
| Share of random draws (same calls per advisor) ≥ learned calling | .008 | .160 | .139 | .005 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -4.6 (.001) | -4.8 (.002) | -4.6 (.004) | -5.8 (<.001) | -5.0 ± 0.6 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -0.8 (.672) | -1.0 (.576) | -1.6 (.350) | -2.0 (.229) | -1.4 ± 0.6 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | -0.8 (.672) | -1.6 (.322) | -1.0 (.576) | +0.4 (.883) | -0.8 ± 0.8 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +10.6 (<.001) | +9.6 (<.001) | +10.2 (<.001) | +12.2 (<.001) | 10.6 ± 1.1 |

**MMLU-Pro** (n = 500; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 50.8 | 51.4 | 51.4 | 50.8 | 51.1 ± 0.3 |
| Learned calling (first call), accuracy (%) | 63.2 | 65.2 | 65.4 | 64.8 | 64.7 ± 1.0 |
| Learned calling, calls per question | 0.33 | 0.35 | 0.37 | 0.37 | 0.35 ± 0.02 |
| Random questions, same calls per advisor, accuracy (%) | 57.2 | 58.4 | 58.5 | 58.2 | 58.1 ± 0.6 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 61.6 | 63.8 | 63.2 | 64.6 | 63.3 ± 1.3 |
| Always Verifier with the candidate, accuracy (%) | 72.4 | 73.6 | 73.2 | 73.6 | 73.2 ± 0.6 |
| Always Verifier, candidate removed, accuracy (%) | 69.2 | 67.4 | 68.8 | 68.4 | 68.5 ± 0.8 |
| Empty-output revision turn, accuracy (%) | 51.4 | 51.6 | 51.8 | 51.0 | 51.5 ± 0.3 |
| Share of random draws (same calls per advisor) ≥ learned calling | <.001 | <.001 | <.001 | <.001 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -9.2 (<.001) | -8.4 (<.001) | -7.8 (<.001) | -8.8 (<.001) | -8.5 ± 0.6 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +1.6 (.416) | +1.4 (.470) | +2.2 (.228) | +0.2 (1.000) | 1.4 ± 0.8 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +3.2 (.089) | +6.2 (<.001) | +4.4 (.017) | +5.2 (.004) | 4.8 ± 1.3 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +21.0 (<.001) | +22.0 (<.001) | +21.4 (<.001) | +22.6 (<.001) | 21.8 ± 0.7 |

**GPQA** (n = 100; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 41.0 | 40.0 | 43.0 | 41.0 | 41.2 ± 1.3 |
| Learned calling (first call), accuracy (%) | 54.0 | 50.0 | 51.0 | 52.0 | 51.8 ± 1.7 |
| Learned calling, calls per question | 0.52 | 0.55 | 0.44 | 0.73 | 0.56 ± 0.12 |
| Random questions, same calls per advisor, accuracy (%) | 47.7 | 47.6 | 47.9 | 49.5 | 48.2 ± 0.9 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 52.0 | 50.0 | 50.0 | 53.0 | 51.2 ± 1.5 |
| Always Verifier with the candidate, accuracy (%) | 56.0 | 56.0 | 56.0 | 55.0 | 55.8 ± 0.5 |
| Always Verifier, candidate removed, accuracy (%) | 61.0 | 59.0 | 59.0 | 59.0 | 59.5 ± 1.0 |
| Empty-output revision turn, accuracy (%) | 41.0 | 40.0 | 43.0 | 41.0 | 41.2 ± 1.3 |
| Share of random draws (same calls per advisor) ≥ learned calling | .005 | .233 | .141 | .196 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -2.0 (.774) | -6.0 (.180) | -5.0 (.267) | -3.0 (.375) | -4.0 ± 1.8 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +2.0 (.754) | +0.0 (1.000) | +1.0 (1.000) | -1.0 (1.000) | 0.5 ± 1.3 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | -5.0 (.332) | -3.0 (.607) | -3.0 (.607) | -4.0 (.424) | -3.8 ± 1.0 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +15.0 (.003) | +16.0 (.002) | +13.0 (.011) | +14.0 (.004) | 14.5 ± 1.3 |

**AQuA** (n = 254; seeds present: s1, seed1, seed2, seed3)

| | s1 | seed1 | seed2 | seed3 | mean ± SD |
|:--|--:|--:|--:|--:|--:|
| Candidate only, accuracy (%) | 36.2 | 36.2 | 35.8 | 37.4 | 36.4 ± 0.7 |
| Learned calling (first call), accuracy (%) | 77.2 | 76.0 | 76.0 | 75.6 | 76.2 ± 0.7 |
| Learned calling, calls per question | 0.77 | 0.74 | 0.75 | 0.72 | 0.75 ± 0.02 |
| Random questions, same calls per advisor, accuracy (%) | 73.9 | 72.1 | 71.7 | 71.7 | 72.3 ± 1.1 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 76.8 | 73.2 | 73.6 | 74.0 | 74.4 ± 1.6 |
| Always Verifier with the candidate, accuracy (%) | 85.0 | 84.6 | 83.9 | 85.0 | 84.6 ± 0.6 |
| Always Verifier, candidate removed, accuracy (%) | 78.3 | 79.9 | 79.5 | 78.7 | 79.1 ± 0.7 |
| Empty-output revision turn, accuracy (%) | 32.7 | 36.2 | 35.8 | 35.8 | 35.1 ± 1.7 |
| Share of random draws (same calls per advisor) ≥ learned calling | .017 | .007 | .005 | .009 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -7.9 (<.001) | -8.7 (<.001) | -7.9 (<.001) | -9.4 (<.001) | -8.5 ± 0.8 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +0.4 (1.000) | +2.8 (.401) | +2.4 (.461) | +1.6 (.665) | 1.8 ± 1.0 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +6.7 (.012) | +4.7 (.073) | +4.3 (.117) | +6.3 (.014) | 5.5 ± 1.2 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +52.4 (<.001) | +48.4 (<.001) | +48.0 (<.001) | +49.2 (<.001) | 49.5 ± 2.0 |
