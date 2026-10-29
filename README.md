# bn-en-transformer-mt

Encoder-decoder Transformer for English->Bangla machine translation, built from
scratch in PyTorch, together with a data-quality study of the BanglaNMT corpus.

**Research question:** How do quality-based filtering of BanglaNMT (length-ratio
and LaBSE similarity) and the inclusion of very short pairs affect the
translation quality of a from-scratch Transformer trained under a fixed budget?

## Status

- [x] Data analysis, Bangla normalization and filtering
- [x] Reproducible data pipeline (`scripts/`)
- [ ] SentencePiece tokenizer (in progress)
- [ ] Transformer implementation with unit tests
- [ ] Training
- [ ] Evaluation, baselines and data-filtering ablation
- [ ] Error analysis and write-up

## Setup

    python -m venv .venv
    source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
    # Install PyTorch from https://pytorch.org
    pip install -r requirements.txt
    pytest

`requirements.txt` installs `csebuetnlp/normalizer` directly from GitHub, so
`git` must be available.

## Repository layout

    scripts/      data pipeline: explore.py, norm.py, prepare.py
    notebooks/    exploration and the analysis behind the filtering thresholds
    results/      configs and logs produced by the data pipeline
    configs/      experiment configs (training, added later)
    src/          model, tokenizer and training code (added later)
    tests/        unit tests (added later)

## Data pipeline

To rebuild the training data from scratch:

    python scripts/norm.py       # -> BanglaNMT_normalized/
    python scripts/prepare.py    # -> BanglaNMT_final/

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `norm.py` | Downloads `csebuetnlp/BanglaNMT` and normalizes the Bangla side with `csebuetnlp/normalizer`. English is unchanged. |
| 2 | `prepare.py` | Length-ratio filter on long pairs. |
| 3 | `prepare.py` | LaBSE similarity filter on long pairs. |
| 4 | `prepare.py` | Adds an `is_long` flag, collapses whitespace, removes the space before the Bangla danda. Validation and test are cleaned the same way but no pair is removed. |

A pair is **long** if both sides have at least 5 whitespace-separated words.
Pairs with fewer words are **short** and are not filtered (see below).

`prepare.py` asserts the expected pair count after every step, so a changed
result stops the script instead of silently producing different data.

The LaBSE scores are cached in `train_labse_cosines.npy`, which is not tracked
by git. If the file is missing, `prepare.py` recomputes the scores, which takes
hours on a GPU.

### Dataset versions

| Version | Description | Train pairs |
|---------|-------------|-------------|
| v1 | length-ratio filter only | 2,351,313 |
| v2-S | v1 + LaBSE filter, short pairs kept | 2,177,739 |
| v2-N | v2-S without short pairs (`is_long == False` removed) | 1,320,869 |

Validation: 597 pairs. Test: 1,000 pairs. Both are never filtered.
v2-S and v2-N are stored in one dataset (`BanglaNMT_final`); v2-N is obtained by
keeping the rows with `is_long == True`.

## Data findings

### Bangla Unicode normalization
- Dataset: BanglaNMT (train split).
- Checked a random sample of 10,000 Bangla sentences (seed 42) before and after
  `csebuetnlp/normalizer`.
- 5,624 of 10,000 sentences (56%) changed.
- In every changed sentence, the first change was a nukta decomposition:
  - য় -> য + ় : 4,964
  - ড় -> ড + ় : 641
  - ঢ় -> ঢ + ় : 19
- These variants look identical on screen but differ at the code-point level,
  which would otherwise split the same word into two tokens and distort
  BLEU/chrF++.
- Decision: normalize the Bangla side of the whole dataset, and normalize both
  references and model outputs at evaluation time.
- Limitation: only the first change per sentence was counted.

### Length-ratio filter
- Metric: `s = (en_len - bn_len) / max(en_len, bn_len)` over word counts.
- Long pairs with `s <= -0.4373` or `s >= 0.5710` are removed.
- The cut-offs come from the distribution of `s`; samples of the removed pairs
  were inspected manually.
- Short pairs are excluded: with 1-4 words, one extra word changes `s` by a
  large amount, so cut-offs derived from long pairs do not apply to them.
- After this step: 2,351,313 pairs, of which 1,494,443 are long.

### LaBSE filter
- Each long pair is scored by the cosine similarity of its English and Bangla
  LaBSE embeddings.
- Threshold: 0.65718675, the 10th percentile of a sample of long-pair cosines.
- This removed 173,574 pairs (11.6% of the long pairs).
- The threshold is a heuristic. The v1 / v2-S / v2-N comparison below is the
  actual test of whether the filtering helps.

### Short pairs
- 856,870 pairs have fewer than 5 words on at least one side (39% of v2-S).
- They contain useful material (about 25,000 number pairs such as
  `1796732 <-> ১৭৯৬৭৩২`) but also dictionary-style word pairs, some of them wrong.
- They are kept and flagged (`is_long == False`); v2-S vs v2-N measures their
  effect.

### Checks that required no change
- Exact duplicate `(en, bn)` pairs in train: 0.
- Train sources do not appear in validation or test.
- Bangla side: Bengali digits in 21.42% of pairs, Latin digits in 4 pairs.
  The English side contains no Bengali digits.
- Zero-width or control characters: 0 on the Bangla side, 1 on the English side.
- URLs or HTML on the English side: about 0.09% of pairs.
- English side with less than 30% Latin letters: 1.29%. Almost all of these are
  numbers, citations or symbols; at most about 90 pairs contain another script.

### Known issues that are left in the data
- One English sentence can have several different Bangla translations
  (133,542 English sentences) and the reverse (109,939 Bangla sentences). Most
  are valid alternatives; some are noisy. These pairs are kept.
- Partial alignments, where one side contains extra text, and number mismatches
  (for example 2013 vs 2016) pass both filters.
- Sentence-final punctuation is not uniform. In long pairs, 5.71% of Bangla
  sentences end with `.` instead of the danda. The training data is not changed
  for this; evaluation normalizes the final punctuation of references and
  outputs (see below).
- Some pairs look like religious-publication text with scripture citations.
  Domain skew will be examined in the error analysis.

## Experimental plan

**Tokenizer.** One joint SentencePiece BPE model (32k vocabulary), trained on
the v2-S training text only and shared by all runs, so that vocabulary
differences do not enter the comparison. Settings: `byte_fallback`,
`split_digits`, no SentencePiece normalization (the Bangla text is already
normalized), fixed ids for pad/unk/bos/eos.

**Model.** Pre-LN encoder-decoder Transformer (base size as the first target
configuration), label smoothing, AdamW with warmup, token-based batching,
tied embeddings.

**Main experiment.** Train on v1, v2-S and v2-N with identical architecture,
hyperparameters, seeds, tokenizer and training-token budget. Report differences
across seeds where possible.

**Evaluation.**
- sacreBLEU and chrF++ on the test set, with the signature reported.
- chrF++ by length bucket (under 5, 5-15, 15-30, over 30 words).
- An external set (FLORES-200 en->bn) to check generalization beyond BanglaNMT.
- Baseline: the released BanglaT5 en->bn model, evaluated with the same code.
- References and outputs are both passed through `csebuetnlp/normalizer` and a
  final-punctuation normalization before scoring.

## Results

TBD.

## Acknowledgements

- BanglaNMT dataset: https://huggingface.co/datasets/csebuetnlp/BanglaNMT
- Bangla text normalizer: https://github.com/csebuetnlp/normalizer
- LaBSE: https://huggingface.co/sentence-transformers/LaBSE

## License

See `LICENSE`.