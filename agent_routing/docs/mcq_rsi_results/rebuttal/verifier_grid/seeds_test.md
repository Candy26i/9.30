### Training-seed replicates of the manager, pool `test`

Checkpoints: `s1`, `seed1`, `seed2`. They share the collection, the labels and the SFT recipe and differ only in the training seed. Each cell lists the seeds in that order, then mean ± sample SD over the seeds present.

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

**MMLU-Pro** (n = 500; seeds present: s1)

| | s1 | mean ± SD |
|:--|--:|--:|
| Candidate only, accuracy (%) | 50.8 | 50.8 |
| Learned calling (first call), accuracy (%) | 63.2 | 63.2 |
| Learned calling, calls per question | 0.33 | 0.33 |
| Random questions, same calls per advisor, accuracy (%) | 57.2 | 57.2 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 61.6 | 61.6 |
| Always Verifier with the candidate, accuracy (%) | 72.4 | 72.4 |
| Always Verifier, candidate removed, accuracy (%) | 69.2 | 69.2 |
| Empty-output revision turn, accuracy (%) | 51.4 | 51.4 |
| Share of random draws (same calls per advisor) ≥ learned calling | <.001 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -9.2 (<.001) | -9.2 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +1.6 (.416) | 1.6 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +3.2 (.089) | 3.2 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +21.0 (<.001) | 21.0 |

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

**AQuA** (n = 254; seeds present: s1)

| | s1 | mean ± SD |
|:--|--:|--:|
| Candidate only, accuracy (%) | 36.2 | 36.2 |
| Learned calling (first call), accuracy (%) | 77.2 | 77.2 |
| Learned calling, calls per question | 0.77 | 0.77 |
| Random questions, same calls per advisor, accuracy (%) | 73.9 | 73.9 |
| Confidence-gated Verifier, same number of calls, accuracy (%) | 76.8 | 76.8 |
| Always Verifier with the candidate, accuracy (%) | 85.0 | 85.0 |
| Always Verifier, candidate removed, accuracy (%) | 78.3 | 78.3 |
| Empty-output revision turn, accuracy (%) | 32.7 | 32.7 |
| Share of random draws (same calls per advisor) ≥ learned calling | .017 | |
| Learned − always Verifier (with candidate), pt (exact McNemar p) | -7.9 (<.001) | -7.9 |
| Learned − confidence gate (same number of calls), pt (exact McNemar p) | +0.4 (1.000) | 0.4 |
| Always Verifier: with − without the candidate, pt (exact McNemar p) | +6.7 (.012) | 6.7 |
| Always Verifier − empty-output revision, pt (exact McNemar p) | +52.4 (<.001) | 52.4 |
