#!/usr/bin/env bash
# Rerun the exp_mid code census (previous run died on a CIFS lock).
set -u
cd /home/user/ultratensor-cluster/scripts || exit 1
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
export PYTHONPATH=/home/user/ultratensor-cluster
mkdir -p logs
exec > logs/census_rerun.log 2>&1
echo "census rerun $(date)"
python3 cluster_code_census.py \
  --shards '/mnt/nas20/models/v4pro/*.gguf' \
  --in /mnt/nas20/exp_mid \
  --general /mnt/nas20/exp96 \
  --layers 3 4 5 6 7 8 9 10
rc=$?
if [ $rc -eq 0 ]; then
  echo "CENSUS_OK $(date)" >> /mnt/nas20/exp_mid/census.log
  touch /mnt/nas20/exp_mid/DONE.txt
else
  echo "CENSUS_FAIL rc=$rc $(date)" >> /mnt/nas20/exp_mid/census.log
fi
exit $rc
