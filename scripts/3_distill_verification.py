"""
Vera Data Pipeline - Step 3: Distill Verification Data

Uses a Teacher Model to generate CoT reasoning for claim verification.

Input:  datasets/verification/train.json (AVeriTeC 2.0 - see DATASETS.md)
Output: datasets/distilled_verification/vera_*.jsonl

Usage:
    uv run python scripts/3_distill_verification.py
    uv run python scripts/3_distill_verification.py --limit 10 --concurrency 5
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = SCRIPTS_DIR.parent / "datasets"

# Label mapping from source to target
LABEL_MAP = {
    "Supported": "SUPPORTED",
    "Refuted": "REFUTED",
    "Conflicting Evidence/Cherrypicking": "CONFLICTING",
}


def load_prompt(name: str) -> str:
    """Load prompt from prompts/verification directory."""
    path = SCRIPTS_DIR / "prompts" / "verification" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def load_verification_data(path: Path) -> list[dict]:
    """Load verification dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[LOAD] Loaded {len(data)} items from {path.name}")
    return data


def extract_evidence(item: dict) -> list[str]:
    """Extract all evidence snippets from questions/answers."""
    evidence = []
    for q in item.get("questions", []):
        for a in q.get("answers", []):
            # For boolean answers, prefer explanation over raw "Yes"/"No"
            if a.get("answer_type") == "Boolean" and a.get("boolean_explanation"):
                evidence.append(a["boolean_explanation"])
            else:
                answer_text = a.get("answer", "")
                if answer_text and len(answer_text) > 3:  # Skip bare "Yes"/"No"
                    evidence.append(answer_text)
    return evidence


def format_input(claim: str, evidence: list[str]) -> str:
    """Format claim and evidence for the model."""
    lines = [f"CLAIM: {claim}", "", "EVIDENCE:"]
    for i, e in enumerate(evidence, 1):
        lines.append(f"[{i}] {e}")
    return "\n".join(lines)


async def call_teacher(
    client: httpx.AsyncClient,
    input_text: str,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Call Teacher Model for verification."""
    async with semaphore:
        try:
            response = await client.post(
                OPENROUTER_API,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": input_text}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,
                    "max_tokens": 1024
                },
                timeout=120.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            
            if "reasoning" not in result or "verdict" not in result:
                return None
            if result["verdict"] not in ["SUPPORTED", "REFUTED", "CONFLICTING"]:
                return None
                
            return result
            
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] HTTP {e.response.status_code}: {e.response.text[:200]}")
            return None
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            return None


async def process_item(
    item: dict,
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2
) -> dict | None:
    """Process a single verification item."""
    claim = item.get("claim", "")
    evidence = extract_evidence(item)
    ground_truth = item.get("label", "")
    
    if not claim or not evidence:
        return None
    
    input_text = format_input(claim, evidence)
    
    for attempt in range(max_retries + 1):
        output = await call_teacher(client, input_text, prompt, model, api_key, semaphore)
        if output:
            return {
                "input": {
                    "claim": claim,
                    "evidence": evidence
                },
                "output": output,
                "ground_truth": ground_truth,
                "model": model
            }
        if attempt < max_retries:
            await asyncio.sleep(1 * (attempt + 1))
            
    return None


async def distill_dataset_async(
    items: list[dict],
    model: str,
    api_key: str,
    output_path: Path,
    concurrency: int = 5
) -> tuple[int, int, int]:
    """Distill verification data using async I/O."""
    prompt = load_prompt("teacher_prompt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_claims = set()
    if output_path.exists():
        print(f"[RESUME] Found existing: {output_path.name}")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_claims.add(record["input"]["claim"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"[RESUME] Skipping {len(processed_claims)} already processed")
    
    to_process = [i for i in items if i.get("claim") not in processed_claims]
    
    if not to_process:
        print("Nothing new to process.")
        return 0, 0, 0
    
    print(f"[START] Processing {len(to_process)} items (concurrency={concurrency})")
    
    semaphore = asyncio.Semaphore(concurrency)
    success = 0
    failed = 0
    mismatches = 0
    
    async with httpx.AsyncClient() as client:
        tasks = [
            process_item(item, client, prompt, model, api_key, semaphore)
            for item in to_process
        ]
        
        with open(output_path, "a", encoding="utf-8") as f:
            for future in asyncio.as_completed(tasks):
                result = await future
                
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    success += 1
                    
                    expected = LABEL_MAP.get(result["ground_truth"], "")
                    actual = result["output"]["verdict"]
                    if expected and expected != actual:
                        mismatches += 1
                        print(f"  ⚠ Mismatch: {result['input']['claim'][:40]}... (expected={expected}, got={actual})")
                    else:
                        print(f"  ✓ {result['input']['claim'][:50]}... → {actual}")
                else:
                    failed += 1
                    print(f"  ✗ Failed")
    
    return success, failed, mismatches


def split_data(data: list[dict], seed: int = 42) -> tuple[list, list, list]:
    """Split data into train/val/test (70/15/15)."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:]
    )


def compute_metrics(output_dir: Path) -> dict:
    """Compute metrics comparing teacher predictions vs ground truth."""
    from collections import Counter
    
    y_true = []
    y_pred = []
    
    for split in ["train", "val", "test"]:
        path = output_dir / f"vera_{split}.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    gt = LABEL_MAP.get(record.get("ground_truth", ""), "")
                    pred = record.get("output", {}).get("verdict", "")
                    if gt and pred:
                        y_true.append(gt)
                        y_pred.append(pred)
                except (json.JSONDecodeError, KeyError):
                    pass
    
    if not y_true:
        return {"error": "No data found"}
    
    labels = ["SUPPORTED", "REFUTED", "CONFLICTING"]
    
    confusion = {l: {p: 0 for p in labels} for l in labels}
    for gt, pred in zip(y_true, y_pred):
        if gt in confusion and pred in labels:
            confusion[gt][pred] += 1
    
    correct = sum(1 for gt, pred in zip(y_true, y_pred) if gt == pred)
    accuracy = correct / len(y_true)
    
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values())
        }
    
    macro_precision = sum(m["precision"] for m in per_class.values()) / len(labels)
    macro_recall = sum(m["recall"] for m in per_class.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(labels)
    
    return {
        "total_samples": len(y_true),
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "label_distribution": {
            "ground_truth": dict(Counter(y_true)),
            "predictions": dict(Counter(y_pred))
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Distill verification data")
    parser.add_argument("--model", default="google/gemini-3-flash-preview", help="Teacher model")
    parser.add_argument("--limit", type=int, default=None, help="Limit items per split")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        return
    
    print("=" * 50)
    print("Vera - Verification Distillation")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print("=" * 50)
    
    source_path = DATA_DIR / "verification" / "train.json"
    all_data = load_verification_data(source_path)
    
    train, val, test = split_data(all_data, args.seed)
    print(f"[SPLIT] Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    output_dir = DATA_DIR / "distilled_verification"
    
    for name, items in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{'='*20} PROCESSING {name.upper()} {'='*20}")
        
        if args.limit:
            items = items[:args.limit]
            print(f"[LIMIT] Using {len(items)} items")
        
        output_path = output_dir / f"vera_{name}.jsonl"
        asyncio.run(distill_dataset_async(
            items=items,
            model=args.model,
            api_key=api_key,
            output_path=output_path,
            concurrency=args.concurrency
        ))
    
    print("\n" + "=" * 50)
    print("Computing Metrics...")
    print("=" * 50)
    
    metrics = compute_metrics(output_dir)
    metrics["model"] = args.model
    metrics["seed"] = args.seed
    
    metrics_path = output_dir / "distill_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📊 Metrics Summary:")
    print(f"   Total Samples: {metrics.get('total_samples', 0)}")
    print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
    print(f"   Macro F1: {metrics.get('macro_f1', 0):.4f}")
    print(f"\n   Per-Class F1:")
    for label, m in metrics.get("per_class", {}).items():
        print(f"     {label}: {m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f})")
    
    print(f"\n📁 Saved metrics to: {metrics_path}")
    print("\n✅ Verification Distillation Complete!")


if __name__ == "__main__":
    main()
