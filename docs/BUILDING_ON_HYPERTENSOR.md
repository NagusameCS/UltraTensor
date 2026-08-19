# Building on HyperTensor — theory and roadmap

What can be built once you can graft models together (ht-graft), split
them apart (Degenerate), route between them (Great Sage), predict
their reliability (Uriel), and recompress them (GRC / UltraTensor)?
This doc is the living theory.  Ranked by evidence-to-effort ratio.

## 1. Escalation inference (build first)

Cascade classifiers meet LLMs.  Raphael-Edge (keep8u/16u IQ2_XS, ~15-26 GB)
handles the confident majority; Uriel scores each request's risk; risky
requests escalate to Raphael-Pro (keep64, 88.5% routing coverage) or the
full model.

Cost model: edge at ~4 t/s, pro at ~0.3 t/s on this hardware.  If edge
handles 70% of requests and the escalation overhead is one Uriel pass
(~ms), effective throughput = 1 / (0.7/4 + 0.3/0.3) ≈ 0.85 t/s vs 0.3
t/s pro-only — a ~2.8x speedup at near-pro quality for the tail.

Evidence we hold: Uriel's 181k controller predicts coverage at
Spearman 0.98 (rho@192); the dispatcher already has the fallback chain.

## 2. Keep-all-compressed (the quality/size dialectic, resolved)

keep-N trades quality for size.  The alternative: keep ALL experts but
at IQ2_XS (~2.36 BPW): full 697 GB -> ~420 GB — still too big for this
machine.  The real resolution is hierarchical: keep64 dense (88.5%
coverage) + GRC-compressed cold experts on disk, streamed on Uriel's
escalation.  Disk is cheap; the hot set is what matters.  This is the
"Stomach + fridge" model: eaten knowledge resident, stored knowledge
streamed.

## 3. The self-improving factory (Raphael's loop)

1. Predator logs (request, domain, expert usage) from live traffic.
2. Nightly: re-census -> per-layer rankings (v4_rebuild_from_census).
3. Degenerate rebuilds specialists; Uriel re-validates (held-out
   coverage, tail veto); rollout only if coverage improved.
4. Gluttony distills new domains from teacher runs (Flash/V100).

The system evolves toward the traffic it actually serves.  Every piece
of this loop exists; only the nightly orchestration needs writing.

## 4. Graft-split recombination (model space search)

ht-graft + Degenerate = bidirectional surgery.  Builds that fall out:

- per-user model assembly: graft the user's domain specialists into a
  single session file (fast context switch, no reload).
- graft Flash's native MTP heads onto the Pro trunk (spec decoding on
  hardware that supports it) — or strip them where they hang.
- expert grafting: replace a weak kept expert with a stronger one from
  a different checkpoint of the same family.

## 5. FACTORED_C serving (fork synergy)

The local llama.cpp fork already carries the Phase-5 FACTORED_C type.
Next: serve factored expert stacks (U @ C) natively -> expert tensors
at ~2 BPW with the fork's kernels.  Combined with #2, the hot set can
be denser (quality) while the cold set is factored (size).

## 6. Provenance chains ("certified models")

Every artifact already embeds a manifest KV (ultratensor.keep_manifest).
Extend: hash each stage (source shards -> census -> split -> quant ->
serve config) and sign.  Publishing on HF then comes with a
reproducibility certificate — a differentiator nobody else has.

## 7. On-device Raphael (HyperTensorARM)

The NEON runtime on Apple Silicon -> Raphael-Edge fully local on
MacBooks; Great Sage + Uriel are tiny (MB) and run anywhere.  The
HyperTensorUI already manages the ecosystem; the end state is a
"model family manager": graft/split/quant as visual operations.

## What NOT to build (measured negatives)

- Linear router refit (ridge): loses to slicing the teacher gate
  (0.801 vs 0.818 coverage, 2026-08-17).  Nonlinear features or direct
  distillation of the gate might reopen this, but evidence says the
  teacher's scores are already optimal among kept experts.
- NPU acceleration of specialists: NPUs are MB-scale INT8 engines;
  only Great Sage (the router) is NPU-appropriate.
