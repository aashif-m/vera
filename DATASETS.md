# Vera Datasets Summary

This document summarizes the datasets used for training and evaluating the Vera pipeline, including the packaged artifact datasets shipped in this repository.

The bundled files are the paper artifact reference copies. Some upstream regeneration steps depend on live external services, so exact reruns are not guaranteed to be byte-identical; see [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md).

## 1. Seed Datasets for Decomposition

The training data generation for the Decomposer relies on two primary seed sources:

- **FEVER**: Used as the source for fact-checking claims. Stratified sampling was applied to balance label distribution.
- **Wikipedia (via Random API)**: Used as the source for complex, multi-sentence paragraphs to train the decomposer effectively.

Because the Wikipedia seed source is randomized, the packaged `datasets/seeds/` files are the canonical artifact copies used for the published experiments.

### Seed Data Splits

| Source | Train | Validation | Test | Total |
|--------|-------|------------|------|-------|
| **FEVER Claims** | 1,467 | 315 | 318 | 2,100 |
| **Wiki Paragraphs**| 420 | 90 | 90 | 600 |

---

## 2. Distilled Decomposition Datasets

The synthetic datasets for the Decomposer were generated via a Teacher Model (Gemini 3 Flash via OpenRouter) using the FEVER and Wikipedia seed splits above. These datasets were used to fine-tune the 1.2B SLM components.

### 2.1 Decomposition (CoT)
Used to train the `vera-decomposer-cot` model to extract atomic claims with step-by-step reasoning and exact quote alignment.

| Split | Count | Path |
|-------|-------|------|
| **Train** | 1,873 | `datasets/distilled_cot/vera_train.jsonl` |
| **Validation** | 404 | `datasets/distilled_cot/vera_val.jsonl` |
| **Test** | 404 | `datasets/distilled_cot/vera_test.jsonl` |
| **Total** | **2,681** | |

### 2.2 Decomposition (Standard / Non-CoT)
Used to train the standard `vera-decomposer` model for faster inference without the reasoning overhead.

| Split | Count | Path |
|-------|-------|------|
| **Train** | 1,872 | `datasets/distilled_non_cot/vera_train.jsonl` |
| **Validation** | 404 | `datasets/distilled_non_cot/vera_val.jsonl` |
| **Test** | 405 | `datasets/distilled_non_cot/vera_test.jsonl` |
| **Total** | **2,681** | |

`datasets/distilled_non_cot/` is the standard-mode decomposition dataset. It is produced by running `scripts/2_distill_decomposition.py --mode standard`, which restores the original pre-CoT decomposition distillation path used for the standard decomposer artifact.

---

## 3. Verification Dataset

Verification training was based on the **AVeriTeC 2.0** dataset. This dataset was used to train the `vera-verifier-cot` reasoning model to classify claims given relevant evidence. (Note: "Conflicting" labels were scrubbed from these splits post-hoc to align with binary NLI classification `SUPPORTED` and `REFUTED`).

### Obtaining the AVeriTeC Source Data

The AVeriTeC 2.0 dataset is required to reproduce the verification distillation step (Step 3). To obtain it:

1. Visit the [AVeriTeC dataset page](https://fever.ai/dataset/averitec.html)
2. Follow the instructions to download the dataset
3. Place the `train.json` file at `datasets/verification/train.json`
4. Run the distillation script:
   ```bash
   python scripts/3_distill_verification.py
   ```

**Note:** The pre-distilled verification training data is already included in `datasets/distilled_verification/`, so this step is only needed if you wish to rerun the teacher-distillation stage yourself.

### 3.1 Verification (CoT)
| Split | Count | Path |
|-------|-------|------|
| **Train** | 1,708 | `datasets/distilled_verification/vera_train.jsonl` |
| **Validation** | 376 | `datasets/distilled_verification/vera_val.jsonl` |
| **Test** | 373 | `datasets/distilled_verification/vera_test.jsonl` |
| **Total** | **2,457** | |

---

## 4. Evaluation Benchmark

The pipeline is evaluated on the [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) benchmark covering 11 datasets. This is a gated dataset; see `training_notebooks/HUGGINGFACE_SETUP.md` for access instructions. Pre-computed evaluation results are available in `eval_results/`.
