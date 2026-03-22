# Reproducibility Notes

This repository is intended to be an artifact-complete reproducibility package for Vera: it includes the released code, packaged datasets, training notebooks, deployment stack, and precomputed evaluation outputs used in the paper.

There are still a few limits to strict byte-for-byte reproducibility that we are keeping explicit:

## 1. Seed regeneration is not fully deterministic

- `scripts/1_fetch_seeds.py` draws from the Wikipedia Random API.
- As a result, rerunning the seed-fetch stage later will not necessarily reproduce the exact packaged `datasets/seeds/` files.
- The bundled seed splits in this repository are therefore the canonical artifact copies for the published results.

## 2. Teacher-model distillation depends on live hosted models

- `scripts/2_distill_decomposition.py` and `scripts/3_distill_verification.py` call OpenRouter teacher models.
- Hosted model behavior can drift over time, and provider-side updates may change outputs even with the same prompts.
- The bundled distilled datasets are the canonical paper artifacts; reruns should be treated as approximate regenerations rather than exact replicas.

## 3. Production compose assumes external TLS provisioning

- `docker-compose.yml` and `caddy/Caddyfile` expect certificate files to already exist at `/opt/certs/fullchain.pem` and `/opt/certs/privkey.pem`.
- For local reproduction, use `docker-compose.dev.yml`, which bypasses Caddy and exposes Streamlit directly on `http://localhost:8501`.

We kept these notes separate so the artifact stays honest about what is bundled, what can be rerun, and what still depends on external infrastructure.
