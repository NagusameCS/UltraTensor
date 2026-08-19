#!/usr/bin/env bash
# One-shot: verify tokenizers on node2, then dry-run the keep8u serve import.
python3 -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
cd /home/user/ultratensor-cluster/scripts || exit 1
python3 -c "import v4_coder_serve; print('v4_coder_serve imports OK')"
