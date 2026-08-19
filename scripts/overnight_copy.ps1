# UltraTensor bulk shard copy: fills the NAS v4pro folder with the
# remaining 14 shards (00004..00017) for full-model cluster work.
# Scheduled once at 04:00 (after exp_mid/exp_pl are done reading).
$src = "D:\hyperv4\models\pro"
$dst = "Y:\models\v4pro"
$log = "C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor\outputs\overnight_watch.log"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') BULK COPY start" | Add-Content $log

$patterns = @("*00004-of-00017.gguf", "*00005-of-00017.gguf",
              "*00006-of-00017.gguf", "*00007-of-00017.gguf",
              "*00008-of-00017.gguf", "*00009-of-00017.gguf",
              "*00010-of-00017.gguf", "*00011-of-00017.gguf",
              "*00012-of-00017.gguf", "*00013-of-00017.gguf",
              "*00014-of-00017.gguf", "*00015-of-00017.gguf",
              "*00016-of-00017.gguf", "*00017-of-00017.gguf")

foreach ($p in $patterns) {
    $target = Join-Path $dst ($p -replace '\*', '')
    if (Test-Path $target) {
        $sz = (Get-Item $target).Length
        $srcFile = Join-Path $src ($p -replace '\*', '')
        if (Test-Path $srcFile -and (Get-Item $srcFile).Length -eq $sz) {
            "$(Get-Date -Format 'HH:mm:ss') $p already complete" | Add-Content $log
            continue
        }
    }
    "$(Get-Date -Format 'HH:mm:ss') copying $p" | Add-Content $log
    robocopy $src $dst $p /NP /NFL /R:2 /W:5 /MT:8
    "$(Get-Date -Format 'HH:mm:ss') $p robocopy exit $LASTEXITCODE" | Add-Content $log
}
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') BULK COPY done" | Add-Content $log
