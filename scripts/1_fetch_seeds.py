"""
Vera Data Pipeline - Step 1: Fetch Seed Data

Fetches seed data from FEVER dataset and Wikipedia for training data generation.

Output: datasets/seeds/
├── fever_claims.jsonl      (1,000 claims)
└── wiki_paragraphs.jsonl   (200 paragraphs)

Usage:
    uv run python scripts/1_fetch_seeds.py
    uv run python scripts/1_fetch_seeds.py --fever-count 500 --wiki-count 100
"""

import argparse
import json
import random
import time
from pathlib import Path

import httpx

FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"
WIKI_RANDOM_API = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "seeds"


def fetch_fever_claims(n: int = 1000) -> list[dict]:
    """Fetch N claims from the FEVER dataset with stratified sampling."""
    print(f"[FEVER] Fetching {n} claims...")
    
    cache_path = OUTPUT_DIR.parent / "cache" / "fever_train.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        print(f"[FEVER] Loading from cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            all_claims = [json.loads(line) for line in f]
    else:
        print(f"[FEVER] Downloading from: {FEVER_TRAIN_URL}")
        all_claims = []
        
        with httpx.stream("GET", FEVER_TRAIN_URL, timeout=300.0, follow_redirects=True) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            data = json.loads(line)
                            all_claims.append({
                                "id": data.get("id"),
                                "claim": data.get("claim"),
                                "label": data.get("label"),
                                "evidence": data.get("evidence", [])
                            })
                        except json.JSONDecodeError:
                            continue
        
        print(f"[FEVER] Caching {len(all_claims)} claims...")
        with open(cache_path, "w", encoding="utf-8") as f:
            for claim in all_claims:
                f.write(json.dumps(claim) + "\n")
    
    labeled = [c for c in all_claims if c.get("label")]
    print(f"[FEVER] Total available: {len(labeled)}")
    
    if len(labeled) <= n:
        return labeled
    
    by_label = {}
    for c in labeled:
        by_label.setdefault(c["label"], []).append(c)
    
    sampled = []
    per_label = n // len(by_label)
    for label, claims in by_label.items():
        sampled.extend(random.sample(claims, min(per_label, len(claims))))
    
    remaining = n - len(sampled)
    if remaining > 0:
        pool = [c for c in labeled if c not in sampled]
        sampled.extend(random.sample(pool, min(remaining, len(pool))))
    
    print(f"[FEVER] Sampled {len(sampled)} claims")
    return sampled


def fetch_wiki_paragraphs(n: int = 200, min_length: int = 100) -> list[dict]:
    """Fetch N random Wikipedia article extracts."""
    print(f"[WIKI] Fetching {n} paragraphs...")
    
    paragraphs = []
    attempts = 0
    max_attempts = n * 3
    
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while len(paragraphs) < n and attempts < max_attempts:
            attempts += 1
            try:
                response = client.get(
                    WIKI_RANDOM_API,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (contact@example.com)"}
                )
                response.raise_for_status()
                data = response.json()
                
                extract = data.get("extract", "")
                title = data.get("title", "")
                
                if len(extract) < min_length:
                    continue
                if "(disambiguation)" in title.lower() or "list of" in title.lower():
                    continue
                
                paragraphs.append({
                    "title": title,
                    "text": extract,
                    "source": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "pageid": data.get("pageid")
                })
                
                print(f"[WIKI] {len(paragraphs)}/{n}: {title[:40]}...")
                time.sleep(1.2)  # Rate limit (increased)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    print(f"[WIKI] 429 Too Many Requests. Sleeping 20s...")
                    time.sleep(20)
                else:
                    print(f"[WIKI] HTTP Error: {e}")
                    time.sleep(1)
            except Exception as e:
                print(f"[WIKI] Error: {e}")
                time.sleep(1)
    
    print(f"[WIKI] Fetched {len(paragraphs)} paragraphs")
    return paragraphs


def save_jsonl(data: list[dict], path: Path) -> None:
    """Save data as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[SAVE] Wrote {len(data)} items to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch seed data for Vera")
    parser.add_argument("--fever-count", type=int, default=1000)
    parser.add_argument("--wiki-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("=" * 50)
    print("Vera - Step 1: Fetch Seeds")
    print("=" * 50)
    
    print(f"[FEVER] Fetching 2100 claims (for 70/15/15 split)...")
    fever = fetch_fever_claims(2100)
    
    # Stratified split by label to maintain class balance
    from collections import defaultdict
    by_label = defaultdict(list)
    for item in fever:
        by_label[item.get("label", "UNKNOWN")].append(item)
    
    train, val, test = [], [], []
    for label, items in by_label.items():
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    save_jsonl(train, OUTPUT_DIR / "fever_train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "fever_val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "fever_test.jsonl")
    print(f"[FEVER] Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    wiki = fetch_wiki_paragraphs(600)
    
    n_wiki = len(wiki)
    n_train = int(n_wiki * 0.70)
    n_val = int(n_wiki * 0.15)
    
    save_jsonl(wiki[:n_train], OUTPUT_DIR / "wiki_train.jsonl")
    save_jsonl(wiki[n_train:n_train + n_val], OUTPUT_DIR / "wiki_val.jsonl")
    save_jsonl(wiki[n_train + n_val:], OUTPUT_DIR / "wiki_test.jsonl")
    print(f"[WIKI] Split: Train={n_train}, Val={n_val}, Test={n_wiki - n_train - n_val}")
    
    import datetime
    log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": args.seed,
        "fever_total": len(fever),
        "fever_train": len(train),
        "fever_val": len(val),
        "fever_test": len(test),
        "wiki_total": n_wiki,
        "wiki_train": n_train,
        "wiki_val": n_val,
        "wiki_test": n_wiki - n_train - n_val,
    }
    with open(OUTPUT_DIR / "split_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"[LOG] Saved split metadata to split_log.json")
    
    print("=" * 50)
    print(f"✅ Complete!")
    print(f"   FEVER: {len(fever)} total → {len(train)}/{len(val)}/{len(test)}")
    print(f"   Wiki:  {n_wiki} total → {n_train}/{n_val}/{n_wiki - n_train - n_val}")
    print("=" * 50)


if __name__ == "__main__":
    main()
