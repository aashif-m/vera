"""Remove CONFLICTING samples from verification dataset."""
import json
import shutil
from pathlib import Path

data_dir = Path(__file__).parent.parent / "datasets" / "distilled_verification"

backup_dir = data_dir / "backup_with_conflicting"
backup_dir.mkdir(exist_ok=True)

for split in ["train", "val", "test"]:
    src = data_dir / f"vera_{split}.jsonl"
    if not src.exists():
        continue
    
    backup_file = backup_dir / f"vera_{split}.jsonl"
    if not backup_file.exists():
        shutil.copy(src, backup_file)
    
    kept = 0
    dropped_gt = 0
    dropped_pred = 0
    filtered = []
    
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            
            gt = record.get("ground_truth", "")
            if "Conflicting" in gt:
                dropped_gt += 1
                continue
            
            verdict = record.get("output", {}).get("verdict", "")
            if verdict == "CONFLICTING":
                dropped_pred += 1
                continue
                
            filtered.append(record)
            kept += 1
    
    with open(src, "w", encoding="utf-8") as f:
        for record in filtered:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"{split}: kept={kept}, dropped_gt={dropped_gt}, dropped_pred={dropped_pred}")

if (data_dir / "distill_metrics.json").exists():
    shutil.copy(data_dir / "distill_metrics.json", backup_dir / "distill_metrics.json")

print(f"\n✅ Backup saved to: {backup_dir}")
print("✅ CONFLICTING samples removed (both ground truth and predictions)")
