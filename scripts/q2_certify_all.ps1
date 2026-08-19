# q2_certify_all.ps1 - overnight Phase-6 certification: stream every
# expert-class tensor of the 17 V4-Pro shards to q2_0, log per-shard results.
$ErrorActionPreference = "Continue"
$py = "C:/Users/legom/AppData/Local/Programs/Python/Python312/python.exe"
$log = "outputs/q2full.log"
"=== q2full start $(Get-Date) ===" | Add-Content $log
foreach ($s in (Get-ChildItem "D:\hyperv4\models\pro\deepseek-ai-DeepSeek-V4-Pro-Q3_K_M-*.gguf" | Sort-Object Name)) {
    "$(Get-Date) START $($s.Name)" | Add-Content $log
    $names = & $py -m ultratensor dry-run $s.FullName 2>$null | ForEach-Object {
        if ($_ -match "^\s*(\S*_exps\.weight)") { $matches[1] }
    }
    if ($names) {
        $only = ($names | Sort-Object -Unique) -join ","
        & $py -m ultratensor compress $s.FullName --out ("outputs/q2full/" + $s.BaseName) --target q2_0 --only $only --manifest-name "manifest.json" *>> $log
        "$(Get-Date) END $($s.Name) rc=$LASTEXITCODE tensors=$($names.Count)" | Add-Content $log
    } else {
        "$(Get-Date) END $($s.Name) no expert tensors" | Add-Content $log
    }
}
"=== q2full done $(Get-Date) ===" | Add-Content $log
