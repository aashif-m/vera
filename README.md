# Vera — Lightweight Fact-Checking Pipeline

A modular, end-to-end fact-checking framework that verifies long-form LLM-generated text using compact, specialised models. Vera decomposes text into atomic claims, retrieves web evidence, and provides dual-mode verification, all running on CPU-only hardware.

> **Paper:** *Small Models, Big Claims: A Knowledge-Distilled Pipeline for Explainable Fact-Checking of LLM Outputs*
>
> **Models:** [HuggingFace Collection](https://huggingface.co/collections/werstal/vera-models)

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Decomposer  │───>│  Alignment  │───>│  Retrieval   │───>│  Verifier   │
│ (SLM 1.2B)  │    │ (Quote Map) │    │ (Tavily API) │    │ (NLI/CoT)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Four-stage pipeline:**
1. **Decomposition** — Fine-tuned 1.2B SLM extracts atomic claims with verbatim quotes, constrained by GBNF grammar
2. **Alignment** — Multi-strategy cascade maps quotes to exact character offsets in source text
3. **Retrieval** — Web search via Tavily API gathers evidence for factual claims
4. **Verification** — Dual-mode: fast NLI classification (ModernBERT) or Chain-of-Thought reasoning (fine-tuned SLM)

## Quick Start

The repository supports two deployment paths:

- **Local development (recommended first run)** — bypasses Caddy and exposes Streamlit directly on `http://localhost:8501`
- **Production / paper demo deployment** — includes Caddy and expects TLS certificates to already exist on the host

### 1. Clone and configure

```bash
git clone https://github.com/aashif-m/vera.git
cd vera
cp .env.example .env
# Edit .env with your API keys
```

### 2. Download models

```bash
export HF_TOKEN=hf_your_token
bash scripts/download-models.sh
```

### 3. Start locally

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

The frontend will be available at `http://localhost:8501`.

### 4. Production deployment (optional)

The base `docker-compose.yml` is intended for a production-style deployment with Caddy in front of the app.

Before using it, you must:

1. Set `DOMAIN` in `.env`
2. Provision TLS certificate files on the host
3. Mount them at `/opt/certs/fullchain.pem` and `/opt/certs/privkey.pem`

Then run:

```bash
docker compose up -d --build
```

In that mode the frontend is served via `https://your-domain`.

## Services

| Service | Description | Port |
|---------|-------------|------|
| `decomposer` | Standard decomposer (llama.cpp) | 8080 |
| `decomposer-cot` | CoT decomposer (llama.cpp) | 8080 |
| `verifier` | ModernBERT NLI verifier | 8081 |
| `verifier-cot` | CoT verifier (llama.cpp) | 8080 |
| `api` | FastAPI orchestrator | 8000 |
| `frontend` | Streamlit UI | 8501 |
| `caddy` | Reverse proxy with SSL | 80/443 |

## Requirements

- Docker and Docker Compose
- Roughly 10-12 GB disk space for the three GGUF models plus container/image overhead
- CPU-only machine; no GPU is required
- The full default compose stack sets memory limits totaling 28 GB across model services, so plan for a substantially larger machine than 3 GB RAM or tune the compose file for your hardware
- [Tavily API key](https://tavily.com/) for evidence retrieval

## Models

Fine-tuned GGUF models are available on HuggingFace:

| Model | Task | Size | Link |
|-------|------|------|------|
| `vera-decomposer` | Standard claim decomposition | ~2.5 GB | [HuggingFace](https://huggingface.co/werstal/vera-decomposer-lfm2.5-1.2b-gguf) |
| `vera-decomposer-cot` | CoT claim decomposition | ~2.5 GB | [HuggingFace](https://huggingface.co/werstal/vera-decomposer-cot-lfm2.5-1.2b-gguf) |
| `vera-verifier-cot` | CoT verification | ~2.5 GB | [HuggingFace](https://huggingface.co/werstal/vera-verifier-cot-lfm2.5-1.2b-gguf) |

The standard verifier uses [ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) (downloaded automatically).

## Data Pipeline

This repository includes the packaged seed splits, distilled datasets, notebooks, and precomputed evaluation outputs used for the paper artifact.

For dataset provenance and rerun notes, see:

- [DATASETS.md](DATASETS.md)
- [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md)

```
scripts/1_fetch_seeds.py        # Fetch FEVER + Wikipedia seeds
scripts/2_distill_decomposition.py  # Distill decomposition training data (via OpenRouter)
scripts/3_distill_verification.py  # Distill verification training data
```

Install script dependencies: `pip install -r scripts/requirements.txt`

Examples:

```bash
uv run python scripts/2_distill_decomposition.py --mode cot
uv run python scripts/2_distill_decomposition.py --mode standard
```

The `standard` mode restores the original pre-CoT decomposition distillation path used for the released standard decomposer dataset.

## Testing and Ablation

Lightweight validation assets from the main project are included here as well:

- `tests/test_aligner.py` — unit test for quote alignment
- `eval/ablation/` — CFG ablation harness used for the paper tables

For local checks:

```bash
pip install -r scripts/requirements.txt -r requirements-dev.txt
pytest tests/test_aligner.py
```

## License

MIT
