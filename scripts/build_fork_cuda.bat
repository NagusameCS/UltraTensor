@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin;%PATH%
set PATH=C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor\.venv\Scripts;%PATH%
cd /d C:\Users\legom\OneDrive\Documents\GitHub\llama.cpp
cmake -B build_cuda -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release > C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor\outputs\build_cuda_cfg.log 2>&1
echo CONFIGURE_RC=%ERRORLEVEL% >> C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor\outputs\build_cuda_cfg.log
cmake --build build_cuda --target llama-cli llama-server -j 8 > C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor\outputs\build_cuda_build.log 2>&1
echo BUILD_RC=%ERRORLEVEL% >> C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor\outputs\build_cuda_build.log
echo DONE >> C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor\outputs\build_cuda_build.log
