# Vera CFG Ablation Study

Measures the impact of GBNF grammar-constrained decoding on decomposition and verification output quality.

## What it tests

| Metric | Description |
|--------|-------------|
| JSON validity | Can the output be parsed as valid JSON? |
| Schema correctness | Does it have all required fields (`quote`, `atomic_claim`, `type`, etc.)? |
| Quote alignment | Are extracted quotes exact substrings of the input text? |
| Verdict validity | Is the verdict one of `SUPPORTED`/`REFUTED`? |
| Accuracy | Does the verdict match ground truth? |

Two conditions are compared:
- **With grammar** — GBNF constraint passed via `grammar` parameter in llama.cpp API
- **Without grammar** — Unconstrained generation (fine-tuned model alone)

## Quick start (Docker)

```bash
cd eval/ablation
docker compose up --build
```

Results will be in `./outputs/ablation/`.

In this reproducibility repo, the compose file expects:

- GGUF models in `../../models/decomposer-cot/` and `../../models/verifier-cot/`
- Packaged datasets in `../../datasets/distilled_cot/` and `../../datasets/distilled_verification/`

## Quick start (standalone)

If you already have llama.cpp servers running:

```bash
# Run both decomp + verif ablations
python run_ablation.py \
  --task both \
  --mode cot \
  --decomp-url http://localhost:8080 \
  --verif-url http://localhost:8081 \
  --decomp-data ../../datasets/distilled_cot/vera_test.jsonl \
  --verif-data ../../datasets/distilled_verification/vera_test.jsonl \
  --sample-size 100

# Run only decomposition
python run_ablation.py --task decomp --mode cot --sample-size 100

# Run only verification
python run_ablation.py --task verif --sample-size 100
```

## Sample size

The default **100 samples** matches the held-out test set size used in the paper. This is sufficient because:

- The ablation measures **structural** properties (JSON validity, schema) which tend to be binary per-sample
- 100 samples gives tight confidence intervals for binary metrics (e.g. 95% CI of ±~6% at 90% success rate)
- It matches the evaluation methodology already established in the paper

If you want tighter confidence intervals, you can increase to 200–300, but 100 is standard for this type of ablation.

## Output files

Each run produces JSONL files with per-sample results:
- `decomp_cot_with_grammar.jsonl`
- `decomp_cot_without_grammar.jsonl`
- `verif_cot_with_grammar.jsonl`
- `verif_cot_without_grammar.jsonl`

The script prints a summary table at the end of each run.
