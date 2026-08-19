#!/usr/bin/env bash
# keep64 IQ2_XS quant on node2 (needs ~34 GB f32 scratch -> swap).
# Waits for the trace chain to finish (RAM free), then quants.
set -u
mkdir -p /mnt/nas20/exp_keep64
exec > /home/user/ultratensor-cluster/scripts/logs/keep64_quant.log 2>&1
echo "keep64 quant waiter start $(date)"
while ! ls /mnt/nas20/exp_rho320 2>/dev/null | grep -q '^DONE.txt$'; do
  sleep 600
done
echo "rho320 done; freeing RAM for keep64 quant $(date)"

# swap setup (24G)
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 24G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=24576 status=none
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
fi
swapon --show

export OPENBLAS_NUM_THREADS=20
export OMP_NUM_THREADS=20
echo "launching llama-quantize $(date)"
/home/user/llamacpp-fork/build/bin/llama-quantize \
  --allow-requantize \
  --imatrix /home/user/ultratensor-cluster/imatrix_merged_keep64.dat \
  /mnt/nas20/models/coder/DeepSeek-V4-Coder-keep64-00001-of-00001.gguf \
  /mnt/nas20/models/coder/DeepSeek-V4-Coder-keep64-iq2xxs.gguf \
  IQ2_XS 20
rc=$?
if [ $rc -eq 0 ]; then
  echo "KEEP64_QUANT_OK $(date)"
  touch /mnt/nas20/exp_keep64/DONE.txt
else
  echo "KEEP64_QUANT_FAIL rc=$rc $(date)"
fi
