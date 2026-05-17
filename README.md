# The Narrow Corridor

> [!WARNING]
> **This repository's central claims have been falsified by the author
> (April 2026). The findings reported below — including the
> `r = −0.990` "frequency–geometry" correlation and the "narrow corridor"
> interpretation — do not survive controlled testing.**
>
> A follow-up paper, **"Destructive Interference in Word Embedding
> Trajectories: A falsification of the frequency hypothesis, and a
> mechanistic replacement"** (Slibar 2026, preprint), documents the
> full falsification ladder and the corrected mechanistic account.
>
> See [`RETRACTION_NOTICE.md`](RETRACTION_NOTICE.md) for the detailed
> summary of what was wrong, what survives, and why.

---

## What this repository is now

This is a **historical artifact** documenting an earlier (and incorrect)
analysis of Skip-Gram word-embedding training dynamics. It is kept
online for:

- transparency about the falsification process,
- reproducibility of the original (artifact-laden) numbers,
- and as the source data for the follow-up paper.

The analytical framing of this repository should **not** be cited as
evidence for its original claims. The new paper supersedes the
interpretation here.

---

## Original (now-falsified) claims

The repository originally analyzed 1.05 M SGD training steps on a
Skip-Gram embedding model with 34-word vocabulary, and reported:

- A Pearson correlation `r(frequency, geodesic_ratio) = −0.990`,
  interpreted as evidence that gradient descent follows "constrained
  pathways rather than exploring freely".
- A "Frequency-Geometry" relationship where high-frequency words
  produced inefficient ("non-geodesic") paths and low-frequency words
  produced near-straight paths.
- A "forbidden transitions" claim that 98 % of possible state
  transitions never occur during training.
- A "narrow corridor" geometric interpretation of these findings.

## Why these claims fail

1. **Sampling artifact.** The original geodesic-ratio measure was
   sensitive to per-word sampling density. Rare words produced
   stationary phases (`v ≈ 0`) followed by sudden jumps that produced
   spurious curvature spikes; frequent words produced smooth motion.
   After arc-length reparametrization (equalizing density across words)
   the correlation collapses to `r ≈ −0.07`. After removing stationary
   segments entirely it collapses to exactly zero.

2. **Diffusion artifact.** The follow-up `efficiency` measure that
   appeared robust under reparametrization is reproduced — and even
   exceeded — by an isotropic random-walk surrogate that keeps the
   step-magnitude distribution of each word but randomizes the
   directions. The correlation `r ≈ −0.81` in the surrogate
   demonstrates that the effect is the expected `1/√N` diffusion
   scaling.

3. **Wrong predictor.** What does survive (a `−0.75` correlation
   between diffusion-normalized directionality and the **context
   entropy** `H(c|w)` of the word) shows that the active variable is
   the breadth of a word's context distribution, not frequency.
   Frequency is a Zipfian proxy with no residual effect after
   controlling for `H` (partial `r = −0.030`).

4. **Underpowered.** The original `n = 34` was too small for the
   rank-based statistics that would have caught the leverage-point
   problem (the Pearson effect at `n = 34` is driven almost entirely
   by a single observation, the word `die`).

## What the follow-up paper establishes

At `n = 392` (replication on a larger template-generated corpus)
the corrected picture is:

```
H(c|w)  →  mean_cos_consec(Δx)  →  structural_ratio ρ(w)
context heterogeneity   vector summation
```

All three links are independently measurable; all three hold at
`p < 10⁻⁴`. Words with broad context distributions receive
SGD updates that **interfere destructively** (consecutive cosine
down to `−0.10`); words with narrow context distributions receive
SGD updates that **interfere constructively** (consecutive cosine
up to `+0.25`). This is the mechanistic basis for the directional
structure of embedding trajectories — and it has nothing to do with
frequency *per se*.

---

## Repository contents (historical)

```
the-narrow-corridor/
├── paper/                    # Original PDF (falsified findings)
├── figures/                  # CFD, FSSA, TPA visualizations
├── src/training/
│   └── experiment.py
├── src/analysis/             # Original analysis scripts
│   ├── curvature_flow.py
│   └── forbidden_state_sequences.py
├── data/                     # 1.05M-step training log
├── RETRACTION_NOTICE.md      # ← READ THIS FIRST
├── README.md                 # this file
└── requirements.txt
```

> The newer, controlled analysis scripts (with surrogate, partial
> correlations, permutation tests, multiprocessing parallelism, and
> the mechanistic update-cosine test) are not in this repo — they are
> in the follow-up paper's accompanying code, which will be released
> alongside the preprint.

---

## How to cite

**Do not cite this repository as evidence for its original claims.**

If you want to cite the corrected position, cite the follow-up:

> Slibar, M. (2026). *Destructive Interference in Word Embedding
> Trajectories: A falsification of the frequency hypothesis, and a
> mechanistic replacement.* Preprint.

If you want to cite this repository as the historical artifact that
the follow-up falsifies, link to the specific commit hash that
contains the original claims, and make clear in your citation that the
findings have been retracted by the author.

---

## License

MIT (original license retained).

---

*Maximilian Slibar, Düsseldorf, Germany — April 2026.*
