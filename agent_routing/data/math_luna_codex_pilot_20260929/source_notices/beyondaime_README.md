---
# YAML metadata block for Hugging Face Hub
license: cc0-1.0
language:
  - en
tags:
  - mathematics
  - math-word-problem
  - problem-solving
  - reasoning
  - competition-math
features:
  - name: problem
    dtype: string
  - name: answer
    dtype: int64
---

# BeyondAIME: Advancing Math Reasoning Evaluation Beyond High School Olympiads

## Table of Contents
- [Dataset Description](#dataset-description)
- [Data Fields](#data-fields)
- [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
- [How to Use](#how-to-use)
- [Citation](#citation)
- [License](#license)

## Dataset Description

**BeyondAIME** is a curated test set designed to benchmark advanced mathematical reasoning. Its creation was guided by the following core principles to ensure a fair and challenging evaluation:

- **High Difficulty**: Problems are sourced from high-school and university mathematics competitions, with a difficulty level greater than or equal to that of AIME Problems #11-15.

- **Contamination-Resistant**: Every problem has been manually revised to be unique, ensuring it will not be found in standard pre-training corpora and providing a true test of a model's reasoning abilities.

- **Focus on Reasoning**, Not Knowledge: The dataset exclusively tests reasoning by ensuring that problems do not require mathematical knowledge beyond the standard university level.

- **Robust Problem Design**: The dataset avoids "pseudo-proof" problems. For problems requiring proof-like steps, they have been reformulated so that guessing the answer is as difficult as formally solving the problem.

- **Automated & Accurate Evaluation**: Each problem's answer is a positive integer, allowing for an unambiguous and 100% accurate automated verification of model performance.

## Data Fields

Each entry in the dataset consists of two fields:

- `problem`: (`string`) - A full statement of the mathematical problem, formatted in Markdown with LaTeX support for mathematical expressions.
- `answer`: (`int`) - The final integer answer to the problem.

Here is an example of a data instance:
```json
{
  "problem": "A sequence of real numbers \\{a_n\\} satisfies that：\\(a_{n + 1}=2^n-7a_n，n = 0,1,2,\\cdots\\). Find the minimal possible value of \\(\\frac{1}{a_0}\\) such that \\(a_{n + 1}>a_n\\) for any positive integer \\(n\\).",
  "answer": 9
}
```

## Data Splits

This dataset consists of a single **`test`** split containing 100 problems, provided in the `test.parquet` file.

## Dataset Creation

Each problem is an original creation at a competition-level difficulty, and the dataset has been balanced by category to ensure coverage across all fields of mathematics competitions.

## How to Use

You can easily load the dataset using the Hugging Face `datasets` library.

```python
from datasets import load_dataset

# Load the dataset from the Hugging Face Hub
ds = load_dataset("ByteDance-Seed/BeyondAIME")

# Access the test split
test_ds = ds['test']

# Print the first example
print(test_ds[0])
```

## Citation

If you use the BeyondAIME dataset in your research or work, please consider citing it:

```bibtex
@misc{bytedance_seed_2025_beyondaime,
  author       = {[ByteDance-Seed]},
  title        = {BeyondAIME: Advancing Math Reasoning Evaluation Beyond High School Olympiads},
  year         = {2025},
  publisher    = {Hugging Face},
  journal      = {Hugging Face repository},
  howpublished = {\url{[https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME](https://huggingface.co/datasets/ByteDance-Seed/BeyondAIME)}},
}
```

## License

BeyondAIME is released under the **CC0 1.0 Universal (CC0 1.0) Public Domain Dedication**.

![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)

This means the work has been dedicated to the public domain by waiving all rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law. You can copy, modify, distribute, and perform the work, even for commercial purposes, all without asking permission. For more details, see the [LICENSE](LICENSE) file or the [full legal text of the CC0 license](https://creativecommons.org/publicdomain/zero/1.0/legalcode).