"""
prepare.py

Builds the final training data:
normalized BanglaNMT -> ratio filter -> LaBSE filter -> clean text -> BanglaNMT_final

Run from the repo root:  python scripts/prepare.py
"""
import re
from pathlib import Path

import numpy as np
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "BanglaNMT_normalized"
OUT = ROOT / "BanglaNMT_final"
LABSE_CACHE = ROOT / "train_labse_cosines.npy"

# Filtering decisions
MIN_WORDS = 5                      # a pair is "long" if both sides have >= 5 words
RATIO_LOW, RATIO_HIGH = -0.4373, 0.5710
LABSE_THRESHOLD = 0.65718675       # 10th percentile of sampled LaBSE cosines
DANDA = "\u0964"                   # Bengali full stop

assert not OUT.exists(), "Output folder already exists"


def word_counts(split):
    en = np.array([len(t.split()) for t in split["en"]])
    bn = np.array([len(t.split()) for t in split["bn"]])
    return en, bn


def compute_labse_scores(split):
    """LaBSE cosine similarity for every long pair (NaN for short pairs)."""
    import torch
    from sentence_transformers import SentenceTransformer

    en_len, bn_len = word_counts(split)
    idx = np.flatnonzero((en_len >= MIN_WORDS) & (bn_len >= MIN_WORDS))
    scores = np.full(len(split), np.nan, dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    for n, start in enumerate(range(0, len(idx), 1024)):
        chunk = idx[start:start + 1024]
        rows = split[chunk.tolist()]
        en_vec = model.encode(rows["en"], batch_size=256, normalize_embeddings=True)
        bn_vec = model.encode(rows["bn"], batch_size=256, normalize_embeddings=True)
        scores[chunk] = (en_vec * bn_vec).sum(axis=1)   # normalized vectors: dot = cosine
        if n % 100 == 0:
            print("LaBSE chunk", n, "of", len(idx) // 1024 + 1)
    return scores


def add_is_long(batch):
    batch["is_long"] = [
        len(e.split()) >= MIN_WORDS and len(b.split()) >= MIN_WORDS
        for e, b in zip(batch["en"], batch["bn"])
    ]
    return batch


def clean_text(batch):
    """Collapse whitespace and remove the space before the danda."""
    batch["en"] = [" ".join(e.split()) for e in batch["en"]]
    batch["bn"] = [re.sub(r"\s+" + DANDA, DANDA, " ".join(b.split())) for b in batch["bn"]]
    return batch


ds = load_from_disk(str(SRC))
train = ds["train"]
print("Raw train:", len(train))

# ---- Step 1: length-ratio filter (long pairs only) ----
en_len, bn_len = word_counts(train)
long_pair = (en_len >= MIN_WORDS) & (bn_len >= MIN_WORDS)
s = (en_len - bn_len) / np.maximum(np.maximum(en_len, bn_len), 1)
bad_ratio = long_pair & ((s <= RATIO_LOW) | (s >= RATIO_HIGH))
train = train.select(np.flatnonzero(~bad_ratio).tolist())
print("After ratio filter:", len(train))
assert len(train) == 2_351_313

# ---- Step 2: LaBSE filter (long pairs only) ----
if LABSE_CACHE.exists():
    scores = np.load(LABSE_CACHE)          # scores were computed once and cached
else:
    scores = compute_labse_scores(train)   # takes hours on a GPU
    np.save(LABSE_CACHE, scores)
assert len(scores) == len(train)

bad_labse = np.isfinite(scores) & (scores < LABSE_THRESHOLD)
train = train.select(np.flatnonzero(~bad_labse).tolist())
print("After LaBSE filter:", len(train))
assert len(train) == 2_177_739

# ---- Step 3: flag long pairs, then clean the text ----
train = train.map(add_is_long, batched=True)   # must run before cleaning
print("Long pairs:", sum(train["is_long"]))
assert sum(train["is_long"]) == 1_320_869

train = train.map(clean_text, batched=True)
ds["train"] = train
ds["validation"] = ds["validation"].map(clean_text, batched=True)  # cleaned, nothing removed
ds["test"] = ds["test"].map(clean_text, batched=True)

# ---- Step 4: save ----
ds.save_to_disk(str(OUT))
print("Saved to", OUT)