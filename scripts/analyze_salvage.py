"""
Analyze error log to count salvageable items at different thresholds.
"""
import json
import sys
sys.path.insert(0, 'scripts')
from utils.aligner import align_quote

errors = []
with open('datasets/distilled_cot/distillation_errors.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            try:
                errors.append(json.loads(line))
            except:
                pass

print(f"Total errors to analyze: {len(errors)}")
print()

validation_errors = [e for e in errors if e.get('error', {}).get('type') == 'validation_error' 
                     and e.get('error', {}).get('response')]

print(f"Validation errors with responses: {len(validation_errors)}")

thresholds = [0.85, 0.80, 0.75, 0.70, 0.65]

for threshold in thresholds:
    salvageable = 0
    partial_salvage = 0
    
    for err in validation_errors:
        text = err.get('input', '')
        response = err.get('error', {}).get('response', {})
        claims = response.get('claims', [])
        
        if not claims:
            continue
        
        all_aligned = True
        some_aligned = False
        
        for claim in claims:
            quote = claim.get('quote', '')
            if not quote:
                continue
            result = align_quote(text, quote, fuzzy_threshold=threshold)
            if result:
                some_aligned = True
            else:
                all_aligned = False
        
        if all_aligned and some_aligned:
            salvageable += 1
        elif some_aligned:
            partial_salvage += 1
    
    print(f"Threshold {threshold}: {salvageable} fully salvageable, {partial_salvage} partial")

print()
print("'Fully salvageable' = ALL claims in the response can be aligned")
print("'Partial' = SOME claims align but not all (could keep the good ones)")
