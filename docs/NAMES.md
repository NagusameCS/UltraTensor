# Project Raphael — Naming

The UltraTensor hyper-MoE program is **Project Raphael** — after the
ultimate skill "Raphael, Lord of Wisdom" (Tensura): it observes,
decides, and evolves the system around the extracted specialists.

| Name | Tensura reference | Maps to (code) |
|---|---|---|
| **Raphael** | Lord of Wisdom | the whole program: orchestrator of the hyper-MoE |
| **Degenerate** | two-way skill | `ultratensor/gguf_keep.py` — splice AND split GGUF models (`write_keep_gguf`, `write_uniform_keep_gguf`) |
| **Great Sage** | the advisor | `ultratensor/hypermoe/router.py` + `scripts/hypermoe_serve.py` — routes each request to the right specialist |
| **Uriel** | Lord of Vows | `scripts/v4_rho_predictor.py` + tail gate (`v4_tail_gate.py`) — predicts reliability and vetoes risky routing |
| **Predator** | analysis | the census pipeline (`cluster_dense_trace.py`, `cluster_code_census.py`, `cluster_pl_census.py`) — analyzes traffic to find each domain's active experts |
| **Stomach** | Predator's storage | the keep-N specialists (`v4_coder_keep_uniform.py`, keep16u/12u/8u) — the extracted, stored expertise |
| **Gluttony** | Lord of Gluttony | Phase-4 distillation (`v4_distill_corpus.py` + teacher runs) — ingesting domain knowledge into specialists |

## One-line story

Raphael uses **Predator** to analyze traffic, **Degenerate** to cut the
1.2T model into **Stomach** specialists, **Uriel** to measure their
reliability, **Great Sage** to route requests, and **Gluttony** to feed
them distilled domain data.
