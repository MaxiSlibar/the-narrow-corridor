# RETRACTION NOTICE

**Status: This repository's central claims have been falsified by the author.**

**Date of retraction: April 2026.**

---

## What is retracted

The core claims of "The Narrow Corridor" (this repository) are no longer
supported. Specifically:

1. **The reported Pearson correlation
   `r(frequency, geodesic_ratio) = −0.990`** is shown to be approximately
   99 % a sampling and diffusion artifact. Under arc-length reparametrization
   it collapses to `r ≈ −0.07`; under update-time filtering, to exactly
   zero. An isotropic random-walk surrogate with matched step-magnitude
   distribution reproduces and exceeds the effect (`r ≈ −0.81`),
   demonstrating that the correlation is the expected `1/√N` diffusion
   scaling, not a geometric property of the embedding manifold.

2. **The "Frequency-Geometry" framing** — that "word frequency almost
   perfectly predicts trajectory geometry" — does not survive controlled
   tests. After partial-correlation control for context entropy
   `H(c|w)`, the residual effect of raw frequency drops to `r ≈ −0.03`.
   Frequency is merely a Zipfian proxy for the underlying variable,
   which is context entropy, not frequency itself.

3. **The "Narrow Corridor" interpretation** — that gradient descent
   follows constrained pathways "rather than exploring freely" — is not
   supported by the original analysis as published, because the original
   analysis did not include a free-exploration null model (isotropic
   random walk). When that null model is added, much of the apparent
   "constraint" is actually diffusion scaling.

4. **The "die" critical-test claim** (that the article "die" behaves
   like content words because of frequency) was identified by
   leave-one-out diagnostics as a single high-leverage point that drove
   the Pearson correlation. At larger sample sizes (n = 392), the
   behaviour of `die` is one observation among many and the original
   interpretation is not the most parsimonious one.

---

## What survives

After seven rounds of independent statistical and surrogate-based
controls, a weaker but more precise claim does survive at sample size
n = 392:

- The **diffusion-normalized structural ratio**
  `ρ(w) = e_real(w) / e_surrogate(w)` correlates with the **context
  entropy** `H(c|w)` of the word
  (`r_Pearson = −0.753`, `r_Spearman = −0.758`, both `p < 10⁻⁴`).
- A direct **mechanistic test** on consecutive SGD update vectors
  confirms the cause: mean cosine of successive updates correlates
  positively with `ρ` (`+0.375` Pearson, `+0.317` Spearman,
  `p < 10⁻⁴`). Words with broad context distributions receive
  consecutive updates with **destructive interference** (cosine down
  to `−0.10`); words with narrow context distributions receive
  consecutive updates with **constructive interference** (cosine up
  to `+0.25`).
- **Frequency** has **no independent effect** once context entropy is
  controlled for. It is only a proxy.

---

## What the new paper does

A follow-up preprint —
**"Destructive Interference in Word Embedding Trajectories: A
falsification of the frequency hypothesis, and a mechanistic
replacement"** (April 2026) — documents the full falsification ladder
(seven control rounds) and the mechanistic replacement of the original
hypothesis. It is the authoritative reference for the corrected
position. A link will be added here once the preprint is on arXiv.

---

## Why this notice exists

Leaving incorrect findings online without correction harms readers who
rely on them. The original analysis lacked critical null-model controls
(in particular, no isotropic random-walk surrogate, no arc-length
reparametrization control, and no partial-correlation analysis). The
original interpretation was also overconfident given the small sample
(n = 34).

The data-design of the original experiment — granular per-step logging
of every SGD update — was preserved and made the subsequent
falsification and mechanistic refinement possible without retraining.
This is the one methodological aspect of the original work that
remains useful and is highlighted in the new paper.

---

## How to read this repository now

- **Treat all correlations and interpretations as historical.**
  The code reproduces the original numbers but does not include the
  surrogate controls that show those numbers to be artifacts.
- **The PDF in `paper/`** documents the falsified position. It should
  not be cited as evidence for the claims it makes. It may be cited
  as the artifact that the follow-up paper falsifies.
- **The new paper supersedes this repository.** Once it is on arXiv,
  this notice will be updated with the arXiv ID.

---

*Maximilian Slibar, Düsseldorf, April 2026.*
