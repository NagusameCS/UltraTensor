#!/usr/bin/env bash
# Compile expert_gemv.so and smoke-test against keep8u on the NAS.
set -e
cd /home/user/ultratensor-cluster/ultratensor/kernels || exit 1
gcc -shared -O3 -fPIC -mavx2 -mfma expert_gemv.c -o expert_gemv.so
ls -la expert_gemv.so
cd /home/user/ultratensor-cluster/scripts || exit 1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export PYTHONPATH=/home/user/ultratensor-cluster
python3 - <<'PY'
import numpy as np
from ultratensor.kernels import ExpertGEMV
cg = ExpertGEMV()
cg.open("/mnt/nas20/models/coder/DeepSeek-V4-Coder-keep8u.gguf",
        "blk.3.ffn_gate_exps.weight")
print("shape", cg.shape)
y = cg.gemv(0, np.ones(cg.shape[0], np.float32))
print("gemv ok, sum=", float(y.sum()))
cg.close()
PY
