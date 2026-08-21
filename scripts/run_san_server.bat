@echo off
cd /d C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor
set CUDA_LAUNCH_BLOCKING=1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\compute-sanitizer.bat" --tool memcheck --launch-timeout 120 --log-file outputs\sanitizer_server.log "C:\Users\legom\hyperv4flash\engine\llama-server.exe" -m "D:\hyperv4\models\coder\DeepSeek-V4-Coder-keep16u.gguf" --host 127.0.0.1 --port 8781 -ngl 0 -c 512 -t 2 --parallel 1 > outputs\san_server_run.log 2>&1
