"""Verify and update common_split.csv so that all 420 slides exist on Hugging Face."""

import csv
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path


def fetch_available_slide_ids() -> set[str]:
    url = "https://huggingface.co/api/datasets/Dearcat/CPathPatchFeature/tree/main/panda/r50/pt_files"
    available = set()
    while url:
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    items = json.loads(r.read().decode("utf-8"))
                    for it in items:
                        sid = it["path"].split("/")[-1].replace(".pt", "")
                        available.add(sid)
                    link = r.headers.get("Link")
                    if link and 'rel="next"' in link:
                        url = link.split(";")[0].strip("<>")
                    else:
                        url = None
                    break
            except Exception as e:
                print(f"Retry {attempt+1}/5 on {url[:60]}: {e}")
                time.sleep(2)
        else:
            raise RuntimeError(f"Failed to fetch {url} after 5 attempts")
    return available


def main():
    root = Path(__file__).resolve().parents[1]
    split_path = root / "data/common_split.csv"
    train_raw_path = root / "data/raw/train.csv"

    print("Fetching 9,999 slide IDs from Hugging Face...")
    available = fetch_available_slide_ids()
    print(f"Total available slides on Hugging Face: {len(available)}")

    # Read current split
    with open(split_path, newline="", encoding="utf-8") as f:
        current_rows = list(csv.DictReader(f))

    # Read raw metadata candidates
    all_candidates = []
    with open(train_raw_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["image_id"] in available:
                all_candidates.append(row)

    current_ids = set(r["image_id"] for r in current_rows)
    kept_rows = []
    replaced_count = 0

    rng = random.Random(42)
    rng.shuffle(all_candidates)

    for r in current_rows:
        sid = r["image_id"]
        if sid in available:
            kept_rows.append(r)
        else:
            target_grade = str(r["isup_grade"])
            target_split = r["split"]
            for cand in all_candidates:
                cid = cand["image_id"]
                if cid not in current_ids and str(cand["isup_grade"]) == target_grade:
                    new_row = dict(cand)
                    new_row["split"] = target_split
                    kept_rows.append(new_row)
                    current_ids.add(cid)
                    replaced_count += 1
                    break

    print(f"Replaced {replaced_count} missing slides with verified available slides.")
    assert len(kept_rows) == 420, f"Expected 420 slides, got {len(kept_rows)}"

    dist = Counter((r["split"], int(r["isup_grade"])) for r in kept_rows)
    for s in ["train", "val", "test"]:
        counts = [dist[(s, g)] for g in range(6)]
        print(f"  {s}: {counts} (total {sum(counts)})")

    # Write updated split
    fieldnames = ["image_id", "data_provider", "isup_grade", "gleason_score", "split"]
    with open(split_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in kept_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"✓ Saved 100% verified common_split.csv ({len(kept_rows)} slides) to {split_path}")


if __name__ == "__main__":
    main()
