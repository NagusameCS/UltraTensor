#!/usr/bin/env bash
# node2 batch worker: runs v4_coder_serve once per prompt line, JSONL results.
# Usage: bash node2_queue.sh <prompts.txt> <out.jsonl> <n_tokens>
set -u
cd /home/user/ultratensor-cluster/scripts || exit 1
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
PROMPTS="${1:-node2_prompts.txt}"
OUT="${2:-logs/node2_queue.jsonl}"
N="${3:-32}"
MODEL="/mnt/nas20/models/coder/DeepSeek-V4-Coder-keep8u.gguf"
touch "$OUT"
while IFS= read -r p; do
  [ -z "$p" ] && continue
  echo "== prompt: $p" >> "$OUT"
  python3 v4_coder_serve.py --model "$MODEL" --prompt "$p" --n "$N" --temp 0 \
    >> "$OUT" 2>&1
  echo "== done" >> "$OUT"
done < "$PROMPTS"
echo "ALL_DONE" >> "$OUT"
