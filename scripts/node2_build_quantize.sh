#!/usr/bin/env bash
# Build llama-quantize (CPU-only) on node2 from the fork source.
set -e
cd /home/user/llamacpp-fork || exit 1
rm -rf build
cmake -B build \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
  > /home/user/cmake_cfg.log 2>&1
echo "configure rc=$?"
tail -5 /home/user/cmake_cfg.log
cmake --build build --target llama-quantize -j 16 \
  > /home/user/cmake_build.log 2>&1
echo "build rc=$?"
tail -3 /home/user/cmake_build.log
ls -la build/bin/llama-quantize
