#!/usr/bin/env bash
# Lean battery: ONE prompt, persistent MoE layers, 8 greedy tokens.
# Measures end-to-end decode throughput without per-token MoELayer
# recreation (the cost that stalled the first battery).
cd /home/user/ultratensor-cluster/scripts || exit 1
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
export PYTHONPATH=/home/user/ultratensor-cluster
mkdir -p logs
exec > logs/lean_battery.log 2>&1
echo "lean battery $(date)"
python3 -u v4_coder_serve.py \
  --model /mnt/nas20/models/coder/DeepSeek-V4-Coder-keep8u.gguf \
  --prompt "Write a Python function to compute fibonacci numbers." \
  --n 8 --temp 0 --persistent-moe
echo "LEAN_DONE rc=$? $(date)"
