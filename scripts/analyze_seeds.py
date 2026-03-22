"""
Analyze sentence counts in seed data.
"""
import json
import re
from pathlib import Path

def count_sentences(text: str) -> int:
    if not text:
        return 0
    # Simple heuristic: split by sentence terminators followed by space or end of string
    sentences = re.split(r'[.!?]+\s+|[.!?]$', text.strip())
    return len([s for s in sentences if s.strip()])

def analyze_file(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        return

    single = 0
    multi = 0
    total = 0
    
    print(f"\nAnalyzing: {path.name}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            text = data.get("claim") or data.get("text") or ""
            
            count = count_sentences(text)
            if count <= 1:
                single += 1
            else:
                multi += 1
            total += 1
            
    print(f"  Total items: {total}")
    print(f"  Single sentence: {single} ({(single/total)*100:.1f}%)")
    print(f"  Multi sentence:  {multi} ({(multi/total)*100:.1f}%)")

if __name__ == "__main__":
    seeds_dir = Path("datasets/seeds")
    analyze_file(seeds_dir / "fever_train.jsonl")
    analyze_file(seeds_dir / "wiki_train.jsonl")
