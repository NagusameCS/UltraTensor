# HyperTensor — hyper-MoE architecture

Split the 1.2T V4 into purpose/language specialists; route requests to
the right specialist with a tiny classifier; serve each on commodity
hardware.  "MoE of models" — the same router idea one level up.

## Pipeline (one command per step)

1. Battery: `v4_make_{pl,nl,math,domain}_batteries.py` -> prompt file
   with segment meta.
2. Census (cluster): `cluster_dense_trace.py` + `cluster_code_census.py`
   -> per-segment expert mass -> ranking JSON.
3. Build: `v4_coder_keep_uniform.py --ranking rank_x.json --keep N`
   -> uniform-N specialist GGUF (metadata expert_count=N, so the
   llama.cpp fork accepts it).
4. Requant: IQ2_XS via llama-quantize + imatrix (proven on keep64;
   ~0.61x of Q3_K size).
5. Registry: `v4_hypermoe_registry.py` -> outputs/hypermoe_registry.json
   (specialists, status, model paths, backends).
6. Serve: `hypermoe_serve.py` — DomainRouter classifies, dispatcher
   resolves the best specialist with an existing model, backends:
   llama-cli (native) or numpy (our runtime). HTTP mode: `--serve 8790`
   POST /v1/generate {"prompt", "n"}.

## Specialists (planned)

| id | domain | keep | quant | size est |
|---|---|---|---|---|
| coder-general-64 | coder-general | 64 | Q3_K_M | 156 GB |
| coder-uniform-16 | general | 16 | Q3_K | 42 GB |
| python/rust/sql/javascript-16 | language | 16 | IQ2_XS | ~26 GB |
| backend/frontend/data/devops-16 | purpose | 16 | IQ2_XS | ~26 GB |
| math-16 | math | 16 | IQ2_XS | ~26 GB |

## Router

Heuristic v1 (`ultratensor/hypermoe/router.py`): purpose-first regex
priors, then language. Upgrade path: the 181k score controller
(rho@192 Spearman 0.98) once specialist traces exist.

## Open questions (decided by running censuses)

- exp_pl: are per-language top-32 expert sets disjoint enough to
  justify separate models?
- exp_mid: does the L3 ranking generalize to layers 4-10?
- keep16 quality: naive keep16 routing coverage ~35%; per-specialist
  router refit is the quality lever.
