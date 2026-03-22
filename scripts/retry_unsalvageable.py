"""
Retry unsalvageable items with temperature=0

Reads from datasets/distilled_cot/unsalvageable.jsonl
Writes successes to the appropriate output files
Writes failures to retry_errors.jsonl
"""
import argparse
import asyncio
import json
import os
import sys
import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from utils.aligner import align_quote

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
DATA_DIR = Path(__file__).parent.parent / "datasets" / "distilled_cot"


def load_prompt() -> str:
    base = Path(__file__).parent
    candidates = [
        base / "prompts" / "decomposition" / "teacher_prompt.txt",
        base / "prompts" / "teacher_prompt.txt",
    ]

    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8")

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find decomposition teacher prompt. Checked:\n{checked}")


def validate_and_enrich(text: str, result: dict) -> tuple[bool, str | None]:
    """Validate and replace quotes with actual source text."""
    claims = result.get("claims", [])
    if not claims:
        return False, "No claims found"
    
    for claim in claims:
        quote = claim.get("quote")
        if not quote:
            continue
        
        indices = align_quote(text, quote, fuzzy_threshold=0.75)
        if indices:
            claim["quote"] = text[indices[0]:indices[1]]
            claim["start"] = indices[0]
            claim["end"] = indices[1]
        else:
            return False, f"Quote alignment failed: '{quote}'"
    
    return True, None


async def call_teacher(
    client: httpx.AsyncClient,
    text: str,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> tuple[dict | None, dict | None]:
    """Call teacher with temperature=0"""
    
    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {"type": "string"},
                        "atomic_claim": {"type": "string"},
                        "reason": {"type": "string"},
                        "type": {"type": "string", "enum": ["FACTUAL", "OPINION"]}
                    },
                    "required": ["quote", "atomic_claim", "reason", "type"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["claims"],
        "additionalProperties": False
    }
    
    async with semaphore:
        try:
            response = await client.post(
                OPENROUTER_API,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "decomposition",
                            "strict": True,
                            "schema": schema
                        }
                    },
                    "temperature": 0,
                    "max_tokens": 8192
                },
                timeout=90.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            
            valid, error_msg = validate_and_enrich(text, result)
            if not valid:
                return None, {"type": "validation_error", "msg": error_msg, "response": result}
            
            return result, None
            
        except Exception as e:
            return None, {"type": "error", "msg": str(e)}


async def retry_items(items: list[dict], model: str, api_key: str, concurrency: int = 10):
    """Retry all items."""
    prompt = load_prompt()
    semaphore = asyncio.Semaphore(concurrency)
    
    successes = []
    failures = []
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for item in items:
            text = item.get("input", "")
            if not text:
                continue
            tasks.append((text, call_teacher(client, text, prompt, model, api_key, semaphore)))
        
        for text, task in tasks:
            result, error = await task
            if result:
                successes.append({
                    "input": text,
                    "output": result,
                    "model": model,
                    "retried": True
                })
                print(f"  ✓ {text[:40]}...")
            else:
                failures.append({
                    "input": text,
                    "error": error,
                    "model": model,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                print(f"  ✗ {text[:40]}...")
    
    return successes, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/mistral-large-2512")
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY")
        return
    
    unsalvageable_path = DATA_DIR / "unsalvageable.jsonl"
    if not unsalvageable_path.exists():
        print("No unsalvageable.jsonl found")
        return
    
    items = []
    with open(unsalvageable_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    print(f"Retrying {len(items)} items with temperature=0...")
    print(f"Model: {args.model}")
    print()
    
    successes, failures = asyncio.run(retry_items(items, args.model, api_key, args.concurrency))
    
    with open(DATA_DIR / "vera_train.jsonl", 'a', encoding='utf-8') as f:
        for item in successes:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(DATA_DIR / "retry_errors.jsonl", 'w', encoding='utf-8') as f:
        for item in failures:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Successes: {len(successes)} (appended to vera_train.jsonl)")
    print(f"✗ Failures: {len(failures)} (written to retry_errors.jsonl)")


if __name__ == "__main__":
    main()
