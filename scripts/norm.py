"""
scripts/norm.py

Downloads BanglaNMT and normalizes the Bangla side with the csebuetnlp normalizer.
Output: BanglaNMT_normalized/  (English side is left unchanged)

Usage:
    python scripts/norm.py
"""
from pathlib import Path

from datasets import load_dataset
from normalizer import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "BanglaNMT_normalized"


def normalize_bn(batch):
    batch["bn"] = [normalize(text) if text else "" for text in batch["bn"]]
    return batch


if __name__ == "__main__":
    if OUT.exists():
        raise FileExistsError(f"{OUT} already exists")

    ds = load_dataset("csebuetnlp/BanglaNMT")

    for split in ds:
        ds[split] = ds[split].map(
            normalize_bn, batched=True, num_proc=4, desc=f"Normalizing {split}"
        )

    ds.save_to_disk(str(OUT))
    print(ds)
    print("Saved to", OUT)