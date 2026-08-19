#!/usr/bin/env bash
# exp_pl chain: dense trace (128 lang tokens, layers 3-10) -> pl census.
set -u
cd /home/user/ultratensor-cluster/scripts || exit 1
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
export PYTHONPATH=/home/user/ultratensor-cluster
mkdir -p logs
exec > logs/exp_pl.log 2>&1
echo "exp_pl trace $(date)"
python3 cluster_dense_trace.py \
  --shards '/mnt/nas20/models/v4pro/*.gguf' \
  --prompts /mnt/nas20/exp_pl/pl_prompts.json \
  --out /mnt/nas20/exp_pl --tokens 128 --max-layer 10
rc=$?
if [ $rc -ne 0 ]; then
  echo "TRACE_FAIL rc=$rc $(date)"
  exit $rc
fi
echo "pl census $(date)"
python3 cluster_pl_census.py \
  --shards '/mnt/nas20/models/v4pro/*.gguf' \
  --in /mnt/nas20/exp_pl \
  --prompts /mnt/nas20/exp_pl/pl_prompts.json
rc=$?
if [ $rc -eq 0 ]; then
  echo "PL_OK $(date)"
  touch /mnt/nas20/exp_pl/DONE.txt
else
  echo "PL_FAIL rc=$rc $(date)"
fi
exit $rc
