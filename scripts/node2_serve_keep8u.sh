#!/usr/bin/env bash
# Launch keep8u numpy serve on node2 (second experiment host).
cd /home/user/ultratensor-cluster/scripts || exit 1
export OPENBLAS_NUM_THREADS=12
export OMP_NUM_THREADS=12
export V4_SERVE_LOG=/home/user/ultratensor-cluster/logs/keep8u_serve.log
mkdir -p /home/user/ultratensor-cluster/logs
if [ -f /home/user/ultratensor-cluster/logs/keep8u_serve.pid ]; then
  old=$(cat /home/user/ultratensor-cluster/logs/keep8u_serve.pid 2>/dev/null)
  kill "$old" 2>/dev/null
fi
nohup python3 -u v4_coder_serve.py \
  --model /mnt/nas20/models/coder/DeepSeek-V4-Coder-keep8u.gguf \
  --backend numpy --serve 8791 --host 0.0.0.0 \
  --max-tokens 64 >"$V4_SERVE_LOG" 2>&1 &
echo $! > /home/user/ultratensor-cluster/logs/keep8u_serve.pid
sleep 20
head -20 "$V4_SERVE_LOG"
