# UltraTensor: launch the keep64 IQ2_XS requant once the Pro imatrix is
# done. Scheduled once at 23:00. Polls up to 3h for imatrix completion.
$root = "C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor"
$log = Join-Path $root "outputs\overnight_watch.log"
$imat = Join-Path $root "outputs\imatrix_pro.dat"
$out = "Y:\models\coder\keep64-iq2xxs.gguf"
$quant = "C:\Users\legom\hyperv4flash\engine\llama-quantize.exe"

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') QUANT LAUNCHER start" | Add-Content $log

if ((Test-Path $out) -and ((Get-Item $out).Length -gt 90GB)) {
    "  already quantized (size ok); skipping" | Add-Content $log; exit
}
if (Test-Path $out) {
    "  stale partial output found; removing" | Add-Content $log
    Remove-Item $out -Force
}
if (Get-Process llama-quantize -ErrorAction SilentlyContinue) {
    "  quantize already running; skipping" | Add-Content $log; exit
}

$tries = 0
while (-not (Test-Path $imat) -and $tries -lt 36) {
    Start-Sleep -Seconds 300
    $tries++
}
if (-not (Test-Path $imat)) {
    "  imatrix never appeared after 3h; aborting" | Add-Content $log
    exit 1
}
"  imatrix ready; launching IQ2_XS requants" | Add-Content $log
Start-Process -FilePath $quant -ArgumentList `
    '--allow-requantize', '--imatrix', $imat, `
    'Y:\models\coder\DeepSeek-V4-Coder-keep64-00001-of-00001.gguf', $out, `
    'IQ2_XS', '4' `
    -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $root "outputs\quant_keep64.log") `
    -RedirectStandardError (Join-Path $root "outputs\quant_keep64.err.log")
"  quantize keep64 launched" | Add-Content $log

$out16 = "D:\hyperv4\models\coder\DeepSeek-V4-Coder-keep16u-iq2xs.gguf"
if (Test-Path "D:\hyperv4\models\coder\DeepSeek-V4-Coder-keep16u.gguf") {
    if (-not (Test-Path $out16)) {
        Start-Process -FilePath $quant -ArgumentList `
            '--allow-requantize', '--imatrix', $imat, `
            'D:\hyperv4\models\coder\DeepSeek-V4-Coder-keep16u.gguf', $out16, `
            'IQ2_XS', '4' `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $root "outputs\quant_keep16u.log") `
            -RedirectStandardError (Join-Path $root "outputs\quant_keep16u.err.log")
        "  quantize keep16u launched (D:, ~26GB)" | Add-Content $log
    }
}
