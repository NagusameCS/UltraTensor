@echo off
cd /d C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor
set CUDA_LAUNCH_BLOCKING=1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\compute-sanitizer.bat" --tool memcheck --launch-timeout 120 --log-file outputs\sanitizer.log "C:\Users\legom\hyperv4flash\engine\llama-cli.exe" -m "D:\hyperv4\models\coder\DeepSeek-V4-Coder-keep16u-iq2xs.gguf" -f outputs\long_prompt.txt -n 1 -t 2 -ngl 12 -c 1024 > outputs\san_run.log 2>&1
