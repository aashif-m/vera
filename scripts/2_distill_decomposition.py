"""
Vera Data Pipeline - Step 2: Distill Decomposition Data

Uses a Teacher Model via OpenRouter to generate decomposition training data
from the packaged seed splits.

Modes:
  - cot:      claim decomposition with reasoning, saved to datasets/distilled_cot/
  - standard: claim decomposition without reasoning, saved to datasets/distilled_non_cot/

Usage:
    uv run python scripts/2_distill_decomposition.py --mode cot
    uv run python scripts/2_distill_decomposition.py --mode standard
"""

import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = SCRIPTS_DIR.parent / "datasets"

try:
    from utils.aligner import align_quote
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from utils.aligner import align_quote

MODE_ALIASES = {
    "cot": "cot",
    "standard": "standard",
    "non-cot": "standard",
    "non_cot": "standard",
}

MODE_CONFIG = {
    "cot": {
        "label": "CoT",
        "prompt_name": "teacher_prompt.txt",
        "output_dir": "distilled_cot",
        "include_reason": True,
        "store_offsets": False,
        "replace_quote_with_source": True,
    },
    "standard": {
        "label": "Standard",
        "prompt_name": "teacher_prompt_standard.txt",
        "output_dir": "distilled_non_cot",
        "include_reason": False,
        "store_offsets": True,
        "replace_quote_with_source": False,
    },
}


def normalize_mode(value: str) -> str:
    mode = MODE_ALIASES.get(value.strip().lower())
    if not mode:
        valid = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Unsupported mode '{value}'. Valid values: {valid}")
    return mode


def build_schema(mode: str) -> dict:
    include_reason = MODE_CONFIG[mode]["include_reason"]
    claim_properties = {
        "quote": {
            "type": "string",
            "description": "Exact verbatim substring from the source text",
        },
        "atomic_claim": {
            "type": "string",
            "description": "The self-contained atomic claim with pronouns resolved",
        },
        "type": {
            "type": "string",
            "enum": ["FACTUAL", "OPINION"],
            "description": "Type of claim",
        },
    }
    required = ["quote", "atomic_claim", "type"]

    if include_reason:
        claim_properties["reason"] = {
            "type": "string",
            "description": "Brief 5-15 word justification for the type classification",
        }
        required.insert(2, "reason")

    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": claim_properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


def validate_and_enrich_output(text: str, result: dict, mode: str) -> tuple[bool, str | None]:
    claims = result.get("claims", [])
    if not claims:
        return False, "No claims found in output"

    config = MODE_CONFIG[mode]
    for claim in claims:
        quote = claim.get("quote")
        if not quote:
            continue

        claim.pop("start", None)
        claim.pop("end", None)

        if config["include_reason"] and not claim.get("reason"):
            return False, "Missing reason field in CoT output"

        indices = align_quote(text, quote)
        if not indices:
            return False, f"Quote alignment failed: '{quote}'"

        start, end = indices
        if config["replace_quote_with_source"]:
            claim["quote"] = text[start:end]
        if config["store_offsets"]:
            claim["start"] = start
            claim["end"] = end

    return True, None


def load_prompt(mode: str) -> str:
    prompt_name = MODE_CONFIG[mode]["prompt_name"]
    path = SCRIPTS_DIR / "prompts" / "decomposition" / prompt_name
    return path.read_text(encoding="utf-8")


def load_seeds(pattern: str) -> list[dict]:
    seeds = []
    for path in (DATA_DIR / "seeds").glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seeds.append(json.loads(line))
    print(f"[LOAD] {pattern}: {len(seeds)} seeds")
    return seeds


def extract_text(seed: dict) -> str:
    return seed.get("text") or seed.get("claim") or ""


async def call_teacher(
    client: httpx.AsyncClient,
    text: str,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    mode: str,
) -> tuple[dict | None, dict | None]:
    schema = build_schema(mode)

    async with semaphore:
        try:
            response = await client.post(
                OPENROUTER_API,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "decomposition",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    "temperature": 0,
                    "max_tokens": 8192,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            result = json.loads(response.json()["choices"][0]["message"]["content"])
            valid, error_msg = validate_and_enrich_output(text, result, mode)
            if not valid:
                return None, {"type": "validation_error", "msg": error_msg, "response": result}
            return result, None
        except httpx.HTTPStatusError as e:
            return None, {"type": "http_error", "status": e.response.status_code, "body": e.response.text[:500]}
        except json.JSONDecodeError as e:
            return None, {"type": "json_error", "msg": str(e)}
        except Exception as e:
            return None, {"type": "unknown_error", "msg": str(e), "exception": type(e).__name__}


async def process_item(
    seed: dict,
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    mode: str,
    max_retries: int = 2,
) -> tuple[str, dict | None, dict | None]:
    text = extract_text(seed)
    if not text or len(text) < 20:
        return text, None, {"type": "skipped", "msg": "Text too short or empty"}

    last_error = None
    for attempt in range(max_retries + 1):
        output, error = await call_teacher(client, text, prompt, model, api_key, semaphore, mode)
        if output:
            return text, output, None
        last_error = error
        if attempt < max_retries:
            await asyncio.sleep(attempt + 1)

    return text, None, last_error


async def distill_dataset_async(
    seeds: list[dict],
    model: str,
    api_key: str,
    output_path: Path,
    mode: str,
    concurrency: int = 5,
) -> tuple[int, int]:
    prompt = load_prompt(mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_inputs = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_inputs.add(json.loads(line)["input"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"[RESUME] {output_path.name}: skipping {len(processed_inputs)} existing items")

    to_process = [seed for seed in seeds if (text := extract_text(seed)) and len(text) >= 20 and text not in processed_inputs]
    if not to_process:
        print("[START] Nothing new to process.")
        return 0, 0

    semaphore = asyncio.Semaphore(concurrency)
    success_count = 0
    fail_count = 0
    error_log_path = output_path.parent / "distillation_errors.jsonl"

    async with httpx.AsyncClient() as client:
        tasks = [
            process_item(seed, client, prompt, model, api_key, semaphore, mode)
            for seed in to_process
        ]

        with open(output_path, "a", encoding="utf-8") as f_out, open(error_log_path, "a", encoding="utf-8") as f_err:
            for future in asyncio.as_completed(tasks):
                text, output, error = await future
                if output:
                    f_out.write(json.dumps({"input": text, "output": output, "model": model}, ensure_ascii=False) + "\n")
                    f_out.flush()
                    success_count += 1
                else:
                    f_err.write(
                        json.dumps(
                            {
                                "input": text,
                                "error": error,
                                "model": model,
                                "timestamp": datetime.datetime.now().isoformat(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f_err.flush()
                    fail_count += 1

    return success_count, fail_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill decomposition data via Teacher Model")
    parser.add_argument("--mode", default="cot", help="cot | standard | non-cot")
    parser.add_argument("--model", default="google/gemini-3-flash-preview", help="Teacher model")
    parser.add_argument("--limit", type=int, default=None, help="Limit items per split")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    args = parser.parse_args()

    mode = normalize_mode(args.mode)
    config = MODE_CONFIG[mode]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return

    print("=" * 60)
    print(f"Vera - Step 2: Distill Decomposition Data ({config['label']} mode)")
    print(f"Model: {args.model}")
    print(f"Output: {DATA_DIR / config['output_dir']}")
    print("=" * 60)

    output_root = DATA_DIR / config["output_dir"]
    split_sizes: dict[str, int] = {}

    for split in ["train", "val", "test"]:
        seeds = load_seeds(f"*_{split}.jsonl")
        if args.limit:
            seeds = seeds[:args.limit]
        split_sizes[split] = len(seeds)
        asyncio.run(
            distill_dataset_async(
                seeds=seeds,
                model=args.model,
                api_key=api_key,
                output_path=output_root / f"vera_{split}.jsonl",
                mode=mode,
                concurrency=args.concurrency,
            )
        )

    with open(output_root / "distill_log.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": args.model,
                "mode": mode,
                "train_seeds": split_sizes["train"],
                "val_seeds": split_sizes["val"],
                "test_seeds": split_sizes["test"],
            },
            f,
            indent=2,
        )

    print("\n✅ Decomposition distillation complete!")


if __name__ == "__main__":
    main()
