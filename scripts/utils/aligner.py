"""
Vera Alignment Utility
Aligns extracted quotes to their exact or approximate character indices in the original text.

Uses multiple strategies:
1. Exact match (fastest)
2. Case-insensitive match
3. Sliding window fuzzy match (handles minor word changes)
"""
import re
from difflib import SequenceMatcher


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single space."""
    return re.sub(r'\s+', ' ', text).strip()


def align_quote(text: str, quote: str, fuzzy_threshold: float = 0.85) -> tuple[int, int] | None:
    """
    Find the start/end indices of a quote in the source text.
    
    Strategies:
    1. Exact match (fastest)
    2. Case-insensitive match
    3. Whitespace-normalized match
    4. Sliding window fuzzy match (finds best approximate location)
    
    Returns:
        (start, end) tuple if found, else None
    """
    if not quote or not text:
        return None
    
    # Normalize inputs for comparison (but keep originals for index mapping)
    quote_clean = normalize_whitespace(quote)
    text_clean = normalize_whitespace(text)
    
    # === Strategy 1: Exact Match ===
    start = text.find(quote)
    if start != -1:
        return start, start + len(quote)
    
    # Try with normalized quote on original text
    start = text.find(quote_clean)
    if start != -1:
        return start, start + len(quote_clean)
    
    # === Strategy 2: Case-Insensitive Match ===
    text_lower = text.lower()
    quote_lower = quote.lower()
    start = text_lower.find(quote_lower)
    if start != -1:
        return start, start + len(quote)
    
    # Normalized + case-insensitive
    quote_clean_lower = quote_clean.lower()
    start = text_lower.find(quote_clean_lower)
    if start != -1:
        return start, start + len(quote_clean)
    
    # === Strategy 3: Sliding Window Fuzzy Match ===
    # Find the window in the text that best matches the quote
    best_match = _find_best_fuzzy_window(text, quote_clean, fuzzy_threshold)
    if best_match:
        return best_match
    
    # === Strategy 4: Subsequence match (for quotes with "..." ellipsis) ===
    if '...' in quote:
        return _align_ellipsis_quote(text, quote, fuzzy_threshold)
    
    return None


def _find_best_fuzzy_window(text: str, quote: str, threshold: float) -> tuple[int, int] | None:
    """
    Slide a window over the text and find the best matching region.
    Uses SequenceMatcher ratio for similarity.
    """
    quote_len = len(quote)
    text_len = len(text)
    
    if quote_len > text_len:
        return None
    
    best_ratio = 0.0
    best_start = -1
    best_end = -1
    
    # Allow some flexibility in window size (±20%)
    min_window = max(1, int(quote_len * 0.8))
    max_window = min(text_len, int(quote_len * 1.2))
    
    # Optimization: use step size for long texts
    step = 1 if text_len < 2000 else max(1, text_len // 1000)
    
    for window_size in range(min_window, max_window + 1, max(1, (max_window - min_window) // 5 + 1)):
        for start in range(0, text_len - window_size + 1, step):
            end = start + window_size
            window = text[start:end]
            
            # Quick pre-filter: check if first/last chars roughly match
            if abs(ord(window[0].lower()) - ord(quote[0].lower())) > 32:
                continue
            
            ratio = SequenceMatcher(None, window.lower(), quote.lower(), autojunk=False).ratio()
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = end
    
    # If we found a good match with step > 1, refine it
    if step > 1 and best_ratio >= threshold * 0.9:
        # Refine around best_start
        refined = _refine_match(text, quote, best_start, threshold)
        if refined:
            return refined
    
    if best_ratio >= threshold:
        return best_start, best_end
    
    return None


def _refine_match(text: str, quote: str, approx_start: int, threshold: float) -> tuple[int, int] | None:
    """Refine a rough match by searching nearby positions."""
    quote_len = len(quote)
    text_len = len(text)
    
    search_radius = min(50, text_len // 10)
    min_window = max(1, int(quote_len * 0.8))
    max_window = min(text_len, int(quote_len * 1.2))
    
    best_ratio = 0.0
    best_start = -1
    best_end = -1
    
    for start in range(max(0, approx_start - search_radius), 
                       min(text_len, approx_start + search_radius)):
        for window_size in range(min_window, max_window + 1):
            if start + window_size > text_len:
                break
            end = start + window_size
            window = text[start:end]
            
            ratio = SequenceMatcher(None, window.lower(), quote.lower(), autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = end
    
    if best_ratio >= threshold:
        return best_start, best_end
    return None


def _align_ellipsis_quote(text: str, quote: str, threshold: float) -> tuple[int, int] | None:
    """
    Handle quotes that contain '...' indicating skipped content.
    E.g., "The film was directed ... by John Smith"
    """
    parts = [p.strip() for p in quote.split('...') if p.strip()]
    if len(parts) < 2:
        return None
    
    # Find first part
    first_match = align_quote(text, parts[0], threshold)
    if not first_match:
        return None
    
    # Find last part (must come after first)
    remaining_text = text[first_match[0]:]
    last_match = align_quote(remaining_text, parts[-1], threshold)
    if not last_match:
        return None
    
    # Return span from start of first to end of last
    start = first_match[0]
    end = first_match[0] + last_match[1]
    
    return start, end
