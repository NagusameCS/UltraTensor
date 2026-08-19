#!/usr/bin/env bash
# Remaining cluster chain: waits for exp_pl DONE, then runs
# exp_nl -> exp_dom -> exp_math -> exp_rho320 sequentially.
set -u
cd /home/user/ultratensor-cluster/scripts || exit 1
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
export PYTHONPATH=/home/user/ultratensor-cluster
mkdir -p logs
exec > logs/chain_rest.log 2>&1
echo "chain_rest start $(date)"

wait_done() {  # $1 = dir
  while ! ls "$1" 2>/dev/null | grep -q '^DONE.txt$'; do
    sleep 300
  done
}

run_trace() {  # $1 = dir, $2 = prompts json, $3 = tokens
  local d="$1" p="$2" t="$3"
  echo "trace $d $(date)"
  python3 cluster_dense_trace.py \
    --shards '/mnt/nas20/models/v4pro/*.gguf' \
    --prompts "$p" --out "$d" --tokens "$t" --max-layer 10
}

run_analyses() {  # $1 = dir
  echo "census $1 $(date)"
  python3 cluster_code_census.py \
    --shards '/mnt/nas20/models/v4pro/*.gguf' --in "$1" \
    --layers 3 4 5 6 7 8 9 10
  rc=$?
  if [ $rc -eq 0 ]; then
    touch "$1/DONE.txt"
    echo "DONE $1 $(date)"
  else
    echo "FAIL $1 rc=$rc $(date)"
  fi
}

echo "waiting for exp_pl DONE $(date)"
wait_done /mnt/nas20/exp_pl
echo "exp_pl done, starting nl $(date)"

run_trace /mnt/nas20/exp_nl /mnt/nas20/exp_nl/nl_prompts.json 95
run_analyses /mnt/nas20/exp_nl

run_trace /mnt/nas20/exp_dom /mnt/nas20/exp_dom/domain_prompts.json 158
run_analyses /mnt/nas20/exp_dom

run_trace /mnt/nas20/exp_math /mnt/nas20/exp_math/math_prompts.json 317
run_analyses /mnt/nas20/exp_math

mkdir -p /mnt/nas20/exp_rho320
run_trace /mnt/nas20/exp_rho320 /mnt/nas20/exp_code/code_prompts.json 320
run_analyses /mnt/nas20/exp_rho320

echo "CHAIN_REST_COMPLETE $(date)"
