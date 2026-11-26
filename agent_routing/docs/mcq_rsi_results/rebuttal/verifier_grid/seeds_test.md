### Training-seed replicates of the manager, pool `test`

Checkpoints: `s1`, `seed1`, `seed2`. `s1` is the paper's checkpoint; `seedK` is the same label file retrained with the recipe printed in the paper-era training logs (LoRA r16 / alpha 32, lr 1e-5, 10 epochs, batch 1 x 8) and training seed K. The original run's seed and maximum sequence length were not logged (command defaults 42 and 4096 are used for the re-trainings), and the re-trainings use the fast linear-attention kernels, so the replicates differ from `s1` in the seed and possibly in those unlogged details. Each cell lists the checkpoints in that order, then mean ± sample SD over the ones present.

**MedQA** (n = 500; seeds present: s1, seed1)

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Candidate only, accuracy (%) | 73.4 | 74.4 | 73.9 ± 0.7 |
| Learned calling (first call), accuracy (%) | 78.8 | 78.0 | 78.4 ± 0.6 |
| Learned calling, calls per question | 0.37 | 0.35 | 0.36 ± 0.01 |
| Random questions, same calls per advisor, accuracy (%) | 76.6 | 77.0 | 76.8 ± 0.3 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 79.6 | 79.0 | 79.3 ± 0.4 |
| Always Verifier with the candidate, accuracy (%) | 83.4 | 82.8 | 83.1 ± 0.4 |
| Always Verifier, candidate removed, accuracy (%) | 84.2 | 84.4 | 84.3 ± 0.1 |
| Empty-output revision turn, accuracy (%) | 72.8 | 73.2 | 73.0 ± 0.3 |
| Share of random draws (same calls per advisor) ≥ learned calling | .008 | .160 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -4.6 (.001) | -4.8 (.002) | -4.7 ± 0.1 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | -0.8 (.672) | -1.0 (.576) | -0.9 ± 0.1 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | -0.8 (.672) | -1.6 (.322) | -1.2 ± 0.6 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +10.6 (<.001) | +9.6 (<.001) | 10.1 ± 0.7 |

**MMLU-Pro** (n = 500; seeds present: s1, seed1)

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Candidate only, accuracy (%) | 50.8 | 51.4 | 51.1 ± 0.4 |
| Learned calling (first call), accuracy (%) | 63.2 | 65.2 | 64.2 ± 1.4 |
| Learned calling, calls per question | 0.33 | 0.35 | 0.34 ± 0.01 |
| Random questions, same calls per advisor, accuracy (%) | 57.2 | 58.4 | 57.8 ± 0.9 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 61.6 | 63.8 | 62.7 ± 1.6 |
| Always Verifier with the candidate, accuracy (%) | 72.4 | 73.6 | 73.0 ± 0.8 |
| Always Verifier, candidate removed, accuracy (%) | 69.2 | 67.4 | 68.3 ± 1.3 |
| Empty-output revision turn, accuracy (%) | 51.4 | 51.6 | 51.5 ± 0.1 |
| Share of random draws (same calls per advisor) ≥ learned calling | <.001 | <.001 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -9.2 (<.001) | -8.4 (<.001) | -8.8 ± 0.6 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +1.6 (.416) | +1.4 (.470) | 1.5 ± 0.1 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +3.2 (.089) | +6.2 (<.001) | 4.7 ± 2.1 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +21.0 (<.001) | +22.0 (<.001) | 21.5 ± 0.7 |

**GPQA** (n = 100; seeds present: s1, seed1)

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Candidate only, accuracy (%) | 41.0 | 40.0 | 40.5 ± 0.7 |
| Learned calling (first call), accuracy (%) | 54.0 | 50.0 | 52.0 ± 2.8 |
| Learned calling, calls per question | 0.52 | 0.55 | 0.54 ± 0.02 |
| Random questions, same calls per advisor, accuracy (%) | 47.7 | 47.6 | 47.6 ± 0.0 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 52.0 | 50.0 | 51.0 ± 1.4 |
| Always Verifier with the candidate, accuracy (%) | 56.0 | 56.0 | 56.0 ± 0.0 |
| Always Verifier, candidate removed, accuracy (%) | 61.0 | 59.0 | 60.0 ± 1.4 |
| Empty-output revision turn, accuracy (%) | 41.0 | 40.0 | 40.5 ± 0.7 |
| Share of random draws (same calls per advisor) ≥ learned calling | .005 | .233 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -2.0 (.774) | -6.0 (.180) | -4.0 ± 2.8 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +2.0 (.754) | +0.0 (1.000) | 1.0 ± 1.4 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | -5.0 (.332) | -3.0 (.607) | -4.0 ± 1.4 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +15.0 (.003) | +16.0 (.002) | 15.5 ± 0.7 |

**AQuA** (n = 254; seeds present: s1, seed1)

| | s1 | seed1 | mean ± SD |
|:--|--:|--:|--:|
| Candidate only, accuracy (%) | 36.2 | 36.2 | 36.2 ± 0.0 |
| Learned calling (first call), accuracy (%) | 77.2 | 76.0 | 76.6 ± 0.8 |
| Learned calling, calls per question | 0.77 | 0.74 | 0.76 ± 0.02 |
| Random questions, same calls per advisor, accuracy (%) | 73.9 | 72.1 | 73.0 ± 1.3 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 76.8 | 73.2 | 75.0 ± 2.5 |
| Always Verifier with the candidate, accuracy (%) | 85.0 | 84.6 | 84.8 ± 0.3 |
| Always Verifier, candidate removed, accuracy (%) | 78.3 | 79.9 | 79.1 ± 1.1 |
| Empty-output revision turn, accuracy (%) | 32.7 | 36.2 | 34.4 ± 2.5 |
| Share of random draws (same calls per advisor) ≥ learned calling | .017 | .007 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -7.9 (<.001) | -8.7 (<.001) | -8.3 ± 0.6 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +0.4 (1.000) | +2.8 (.401) | 1.6 ± 1.7 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +6.7 (.012) | +4.7 (.073) | 5.7 ± 1.4 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +52.4 (<.001) | +48.4 (<.001) | 50.4 ± 2.8 |
