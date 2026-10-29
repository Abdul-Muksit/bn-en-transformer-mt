"""
scripts/explore.py

Writes 50 random training pairs to a text file for manual inspection.

Usage:
    python scripts/explore.py
"""
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "notebooks" / "reviews" / "sample_50.txt"

ds = load_dataset("csebuetnlp/BanglaNMT")

print({name: len(split) for name, split in ds.items()})

random.seed(42)
indices = random.sample(range(len(ds["train"])), 50)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, encoding="utf-8", mode="w") as f:
    for i in indices:
        ex = ds["train"][i]
        f.write(f"[{i}] 'BN' : {ex['bn']}\n'EN' : {ex['en']}\n\n")

print("Saved to", OUT_PATH)