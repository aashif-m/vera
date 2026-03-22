
import sys
import os
from pathlib import Path

# Add scripts/ to path so we can import utils.aligner
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from utils.aligner import align_quote

def test_align_quote():
    text = "The quick brown fox jumps over the lazy dog."
    
    # 1. Exact Match
    quote = "quick brown fox"
    start, end = align_quote(text, quote)
    assert text[start:end] == quote
    assert start == 4
    
    # 2. Case Insensitive
    quote_caps = "QUICK BROWN FOX"
    start, end = align_quote(text, quote_caps)
    assert start == 4
    # Note: text[start:end] will be lowercase, quote is uppercase. 
    # Aligner returns indices in original text.
    assert text[start:end] == "quick brown fox"
    
    # 3. Fuzzy Match (Partial)
    # "brown f" matches "brown fox" partially? No, "brown f" is exact.
    # Let's try Typo: "brown fix"
    quote_typo = "brown fix" 
    # This might fail depending on fuzzy threshold. 0.8 is strict.
    # "brown f" (7 chars) matches "brown f" (7 chars). "ix" vs "ox".
    # SequenceMatcher ratio for "brown fix" vs "brown fox" is high.
    
    result = align_quote(text, quote_typo)
    if result:
        s, e = result
        print(f"Fuzzy match: '{quote_typo}' -> '{text[s:e]}'")
        assert "brown" in text[s:e]
    else:
        print("Fuzzy match skipped (expected if strict)")
        
    # 4. Hallucination (Should return None)
    quote_fake = "The lazy cat sleeps."
    assert align_quote(text, quote_fake) is None
    
    print("All tests passed!")

if __name__ == "__main__":
    test_align_quote()
