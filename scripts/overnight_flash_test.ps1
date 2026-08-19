# UltraTensor: run the Flash IQ2XXS 1-token load test once the merge
# finishes. Scheduled once at 18:30.
$root = "C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor"
$log = Join-Path $root "outputs\overnight_watch.log"
$done = "Y:\models\flash\MERGE_DONE.txt"
$model = "Y:\models\flash\DeepSeek-V4-Flash-IQ2XXS-merged.gguf"
$out = Join-Path $root "outputs\flash_test.log"

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') FLASH TEST launcher" | Add-Content $log

$tries = 0
while (-not (Test-Path $done) -and $tries -lt 36) {
    Start-Sleep -Seconds 300
    $tries++
}
if (-not (Test-Path $done)) { "  merge never finished; aborting" | Add-Content $log; exit 1 }

"  merge done; running flash 1-token load test" | Add-Content $log
& "C:\Users\legom\hyperv4flash\engine\llama-cli.exe" -m $model -p "def fib" `
    -n 1 -t 4 -ngl 0 -c 128 *> $out
$tail = (Get-Content $out -Tail 3) -join " | "
"  flash test tail: $tail" | Add-Content $log
