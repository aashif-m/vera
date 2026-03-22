# Vera Evaluation on LLM-AggreFact

These notebooks evaluate Vera's verification models against the [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact) benchmark — a comprehensive suite of 11 fact-verification datasets.

## Notebooks

| Notebook | Model Evaluated | Description |
|----------|----------------|-------------|
| `eval_vera_aggrefact.ipynb` | ModernBERT-large-zeroshot-v2.0 | Standard NLI-based verification (zero-shot) |
| `eval_vera_aggrefact_cot.ipynb` | Vera CoT Verifier (LFM2.5-1.2B fine-tuned) | Chain-of-Thought verification with reasoning |

These notebooks evaluate released models only; training happens in `training_notebooks/`.

## Requirements

- Google Colab with GPU (recommended: A100)
- HuggingFace token — LLM-AggreFact is a gated dataset, see [HUGGINGFACE_SETUP.md](../training_notebooks/HUGGINGFACE_SETUP.md) for access instructions

## Pre-computed Results

If you don't want to re-run the evaluations, pre-computed results are available in `eval_results/evaluations/`:

| Directory | Model | Files |
|-----------|-------|-------|
| `llmaggrefact-modernbert/` | ModernBERT-large-zeroshot-v2.0 | Metrics, predictions, comparison CSV |
| `llmaggrefact-cot-verifier/` | Vera CoT Verifier (fine-tuned) | Metrics, predictions, comparison CSV |

## Running the Evaluation

1. Open the desired notebook in Google Colab
2. Set your HuggingFace token (for gated dataset access)
3. Run all cells in order
4. Results are automatically saved to your Google Drive

### Quick Test Mode

To run on a subset of datasets, modify the target datasets cell:
```python
TARGET_DATASETS = ["Wice", "FactCheck-GPT"]  # Test on 2 datasets
MAX_SAMPLES = 50  # Only evaluate 50 samples per dataset
```

## Output

Results are saved to `drive/MyDrive/vera/evaluations/`:
- `metrics_*.json` — Per-dataset and overall BAcc, Accuracy, Precision, Recall, F1
- `predictions_*.jsonl` — Per-instance predictions
- `results_*.csv` — Results summary table
- `comparison_*.csv` — Comparison with baseline models
