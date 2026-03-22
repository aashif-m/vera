"""
Salvage failed items from distillation_errors.jsonl

1. Try to align all quotes at a lower threshold (0.75)
2. Replace quotes with the ACTUAL source text found at aligned positions
3. Write salvaged items back to the appropriate output files
4. Write remaining failures to unsalvageable.jsonl for retry
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, 'scripts')
from utils.aligner import align_quote

DATA_DIR = Path('datasets/distilled_cot')
THRESHOLD = 0.75

def salvage_item(error_record: dict) -> dict | None:
    """
    Attempt to salvage a failed item.
    Returns the fixed record if all claims can be aligned, else None.
    """
    text = error_record.get('input', '')
    response = error_record.get('error', {}).get('response', {})
    model = error_record.get('model', 'unknown')
    claims = response.get('claims', [])
    
    if not claims or not text:
        return None
    
    fixed_claims = []
    
    for claim in claims:
        quote = claim.get('quote', '')
        if not quote:
            continue
        
        result = align_quote(text, quote, fuzzy_threshold=THRESHOLD)
        if not result:
            return None  # Failed to align this claim, can't salvage
        
        start, end = result
        actual_quote = text[start:end]
        
        fixed_claims.append({
            'quote': actual_quote,
            'atomic_claim': claim.get('atomic_claim', ''),
            'type': claim.get('type', 'FACTUAL'),
            'start': start,
            'end': end
        })
    
    if not fixed_claims:
        return None
    
    return {
        'input': text,
        'output': {'claims': fixed_claims},
        'model': model,
        'salvaged': True
    }


def main():
    errors = []
    with open(DATA_DIR / 'distillation_errors.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    errors.append(json.loads(line))
                except:
                    pass
    
    print(f"Loaded {len(errors)} errors")
    
    salvageable_candidates = [
        e for e in errors 
        if e.get('error', {}).get('type') == 'validation_error'
        and e.get('error', {}).get('response')
    ]
    
    print(f"Candidates with responses: {len(salvageable_candidates)}")
    
    salvaged = []
    unsalvageable = []
    
    for err in salvageable_candidates:
        result = salvage_item(err)
        if result:
            salvaged.append(result)
        else:
            unsalvageable.append(err)
    
    other_errors = [e for e in errors if e not in salvageable_candidates]
    unsalvageable.extend(other_errors)
    
    print(f"\nSalvaged: {len(salvaged)}")
    print(f"Unsalvageable: {len(unsalvageable)}")
    
    # We could split by source, but for simplicity append to train
    with open(DATA_DIR / 'vera_train.jsonl', 'a', encoding='utf-8') as f:
        for item in salvaged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nAppended {len(salvaged)} salvaged items to vera_train.jsonl")
    
    with open(DATA_DIR / 'unsalvageable.jsonl', 'w', encoding='utf-8') as f:
        for item in unsalvageable:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Wrote {len(unsalvageable)} items to unsalvageable.jsonl for retry")
    
    # Show sample of salvaged
    if salvaged:
        print("\n--- Sample Salvaged Item ---")
        sample = salvaged[0]
        print(f"Input: {sample['input'][:60]}...")
        print(f"Claims: {len(sample['output']['claims'])}")
        for c in sample['output']['claims'][:2]:
            print(f"  Quote: '{c['quote'][:40]}...' [{c['start']}:{c['end']}]")


if __name__ == '__main__':
    main()
