# UltraTensor — Publishable Paper

This directory holds the formal, publishable research paper for UltraTensor
(Project Raphael), written as a companion to the **HyperTensor volume**
(Stewart 2026, Papers I–XV). It is the peer-review-facing form of
[`../PREPRINT.md`](../PREPRINT.md): every quantitative claim is grounded in a
committed artifact under [`../../outputs/`](../../outputs), produced by a script
under [`../../scripts/`](../../scripts), and validated by the suite under
[`../../tests/`](../../tests).

## Files

| File | Purpose |
|---|---|
| `ultratensor.tex` | The paper source (self-contained; no custom `.sty`). |
| `refs.bib` | Bibliography (biblatex/biber; HyperTensor volume + MoE / quant / spec-decode literature). |
| `ultratensor.pdf` | Built output (20 pp, v0.2). Not committed (repo-wide `*.pdf` ignore) — rebuild with the sequence below. |

## Build

`latexmk` needs Perl, which may not be installed. The manual sequence below
always works with MiKTeX or TeX Live:

```powershell
pdflatex -interaction=nonstopmode ultratensor.tex
biber ultratensor
pdflatex -interaction=nonstopmode ultratensor.tex
pdflatex -interaction=nonstopmode ultratensor.tex
```

If `latexmk` + Perl are available, `latexmk -pdf ultratensor.tex` runs the whole
chain (including `biber`) in one command.

The preamble is standard (`amsmath`, `booktabs`, `biblatex`, `hyperref`, …) and
compiles under MiKTeX with automatic package installation. A clean build reports
zero undefined references, zero font warnings, and one negligible 3.5 pt overfull
box.

## What the paper argues

A trillion-parameter Mixture-of-Experts model (DeepSeek-V4-Pro, 697 GB) is not
one compression target but a **conditional system**. Its *routing* concentrates
sharply by domain (97.66 % of code mass in 64 of 384 experts) even though its
*weights* resist every classical compression lever (low rank, PQ, shared bases,
router refit — all measured negatives). The paper turns that asymmetry into a
pipeline — census → splice → quantise → route → escalation-gate — served from a
32 GB laptop with an 8 GB GPU at 1.2–1.5 tok/s, validated end-to-end against
bit-level numpy oracles of the full architecture.

## Relationship to HyperTensor

The paper builds directly on the HyperTensor framework and cites it throughout:

- **GRC (Part I)** — the geometry-only, measure-honestly discipline, applied here
  to expert operators (where it yields mostly negative results, reported as such).
- **CECI (Part X)** — component interchange, generalised to intra-model expert
  splicing.
- **OTT / GTC (Parts IV, VIII)** — the cached-manifold read model, instantiated
  as the lazy predicted-expert read path.
- **COG + TEH geometric jury (Part XV)** — the confidence-aggregation template
  reused as the reconstruction-risk escalation gate.
