"""
Fix quotes in distilled dataset by replacing LLM quotes with actual source text.
Uses aligner.py to find the exact match and updates the quote field.
Also removes start/end fields if present.

Usage:
    uv run python scripts/fix_quotes.py
"""

import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))

from utils.aligner import align_quote

DATA_DIR = Path(__file__).parent.parent / "datasets" / "distilled_cot"


def fix_record(record: dict) -> tuple[dict | None, dict]:
    """
    Fix a single record by aligning quotes to source text.
    Returns (fixed_record, stats).
    """
    stats = {"total_claims": 0, "fixed": 0, "failed": 0, "removed_indices": 0}
    
    text = record.get("input", "")
    output = record.get("output", {})
    claims = output.get("claims", [])
    
    if not text or not claims:
        return record, stats
    
    fixed_claims = []
    for claim in claims:
        stats["total_claims"] += 1
        quote = claim.get("quote", "")
        
        if "start" in claim or "end" in claim:
            claim.pop("start", None)
            claim.pop("end", None)
            stats["removed_indices"] += 1
        
        if not quote:
            fixed_claims.append(claim)
            continue
            
        indices = align_quote(text, quote)
        if indices:
            actual_quote = text[indices[0]:indices[1]]
            claim["quote"] = actual_quote
            stats["fixed"] += 1
            fixed_claims.append(claim)
        else:
            # Quote couldn't be aligned - drop the claim
            stats["failed"] += 1
            print(f"  [FAILED] Could not align: '{quote[:50]}...'")
    
    output["claims"] = fixed_claims
    record["output"] = output
    return record, stats


def process_file(input_path: Path, output_path: Path) -> dict:
    """Process a single JSONL file."""
    print(f"\n[PROCESSING] {input_path.name}")
    
    total_stats = {"records": 0, "total_claims": 0, "fixed": 0, "failed": 0, "removed_indices": 0}
    fixed_records = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            fixed_record, stats = fix_record(record)
            
            if fixed_record:
                fixed_records.append(fixed_record)
                total_stats["records"] += 1
                for key in ["total_claims", "fixed", "failed", "removed_indices"]:
                    total_stats[key] += stats[key]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in fixed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"  Records: {total_stats['records']}")
    print(f"  Claims: {total_stats['total_claims']} (Fixed: {total_stats['fixed']}, Failed: {total_stats['failed']})")
    print(f"  Removed start/end indices: {total_stats['removed_indices']}")
    
    return total_stats


def main():
    print("=" * 50)
    print("Vera - Fix Quotes in Distilled Dataset")
    print("=" * 50)
    
    files = ["vera_train.jsonl", "vera_val.jsonl", "vera_test.jsonl"]
    
    for filename in files:
        input_path = DATA_DIR / filename
        if not input_path.exists():
            print(f"\n[SKIP] {filename} not found")
            continue
        
        output_path = input_path
        process_file(input_path, output_path)
    
    print("\n" + "=" * 50)
    print("✅ All files processed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
