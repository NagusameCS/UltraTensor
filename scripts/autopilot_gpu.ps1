# UltraTensor GPU autopilot (called by the overnight watchdog, idempotent):
# detects finished IQ2_XS quants, smoke-tests them on the GPU, starts a
# llama-server, and registers the specialist.
$root = "C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor"
$log = Join-Path $root "outputs\autopilot_gpu.log"
$state = Join-Path $root "outputs\autopilot_state.json"
$engine = "C:\Users\legom\hyperv4flash\engine"

function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }

# prefer the rebuilt fork engine (GPU decode fixes: MMQ bypass + repeated-
# expert-id fallback fix, validated 2026-08-19) when present
if (Test-Path "C:\Users\legom\hyperv4flash\engine_rebuilt\llama-server.exe") {
    $engine = "C:\Users\legom\hyperv4flash\engine_rebuilt"
    Log "using rebuilt engine (GPU decode fixes)"
}

# mutex: watchdog ticks and manual runs must not overlap
$lock = Join-Path $root "outputs\autopilot_gpu.lock"
if (Test-Path $lock) {
    $age = (Get-Date) - (Get-Item $lock).LastWriteTime
    if ($age.TotalMinutes -lt 30) {
        Log "another autopilot instance active ($([int]$age.TotalSeconds)s old); skipping"
        exit 0
    }
    Remove-Item $lock -Force
}
Set-Content $lock "$(Get-Date -Format o)"

$defaults = @{ keep8u = @{ smoke = $false; served = $false };
               keep12u = @{ smoke = $false; served = $false } }
$st = $defaults
if (Test-Path $state) {
    try { $st = (Get-Content $state | ConvertFrom-Json) } catch { $st = $defaults }
}

$candidates = @(
    @{ id = "keep8u";  model = "Y:\models\coder\DeepSeek-V4-Coder-keep8u-iq2xs.gguf";
       local = "D:\hyperv4\models\coder\DeepSeek-V4-Coder-keep8u-iq2xs.gguf";
       spec = "coder-gpu-8";  port = 8789 },
    @{ id = "keep12u"; model = "Y:\models\coder\DeepSeek-V4-Coder-keep12u-iq2xs.gguf";
       local = "D:\hyperv4\models\coder\DeepSeek-V4-Coder-keep12u-iq2xs.gguf";
       spec = "coder-gpu-12"; port = 8790 }
)

foreach ($c in $candidates) {
    $k = $c.id
    if (-not (Test-Path $c.model)) { continue }
    $sz = (Get-Item $c.model).Length / 1GB
    if ($sz -lt 1) { continue }                       # partial output
    $quantizing = Get-Process llama-quantize -ErrorAction SilentlyContinue
    if ($quantizing) { continue }                     # another quant still running

    # stage to local SSD: NAS load is too slow for reliable GPU smoke/serve
    if (-not (Test-Path $c.local)) {
        Log "$k staging to local SSD ($([math]::Round($sz,1)) GiB)"
        Copy-Item $c.model $c.local -ErrorAction SilentlyContinue
    }
    if (Test-Path $c.local) { $c.model = $c.local }

    if (-not $st.$k.smoke) {
        Log "$k model present ($([math]::Round($sz,1)) GiB); smoke test"
        $so = Join-Path $root "outputs\smoke_$k.log"
        $se = Join-Path $root "outputs\smoke_$k.err.log"
        $p = Start-Process -FilePath "$engine\llama-cli.exe" -ArgumentList '-m',$c.model,'-p','fibonacci','-n','2','-t','4','-ngl','8','-c','128','--no-op-offload' -WindowStyle Hidden -RedirectStandardOutput $so -RedirectStandardError $se -PassThru
        $p.WaitForExit(900000) | Out-Null
        if (-not $p.HasExited) { $p.Kill() }
        $txt = (Get-Content $so -Raw -ErrorAction SilentlyContinue) +
               (Get-Content $se -Raw -ErrorAction SilentlyContinue)
        if ($txt -match "Generation:\s*([\d\.]+)\s*t/s" -and
            [double]$Matches[1] -gt 0.01) {
            Log "$k smoke OK: generation $($Matches[1]) t/s"
            $st.$k.smoke = $true
        } else {
            Log "$k smoke FAILED"
            $st | ConvertTo-Json | Set-Content $state
            continue
        }
    }

    $listening = Get-NetTCPConnection -LocalPort $c.port -State Listen -ErrorAction SilentlyContinue
    if ($listening) { $st.$k.served = $true }
    if (-not $st.$k.served) {
        Log "$k starting server on port $($c.port)"
        Start-Process -FilePath "$engine\llama-server.exe" -ArgumentList '-m',$c.model,'--host','127.0.0.1','--port',"$($c.port)",'-ngl','8','-c','512','-t','4','--parallel','1','--no-op-offload' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root "outputs\server_$k.log") -RedirectStandardError (Join-Path $root "outputs\server_$k.err.log")
        Start-Sleep -Seconds 60
        if (Get-NetTCPConnection -LocalPort $c.port -State Listen -ErrorAction SilentlyContinue) {
            $st.$k.served = $true
            Log "$k server listening on $($c.port)"
        }
    }

    # register (idempotent)
    $regPath = Join-Path $root "outputs\hypermoe_registry.json"
    $reg = Get-Content $regPath | ConvertFrom-Json
    $found = @($reg.specialists | Where-Object { $_.id -eq $c.spec }).Count
    if ($found -eq 0) {
        $entry = [pscustomobject]@{
            id = $c.spec; domain = "general"; battery = "code_prompts.json";
            keep = [int](($k -replace 'keep', '') -replace 'u$', '');
            model = $c.model;
            backend = "llama-http"; port = $c.port; quant = "IQ2_XS";
            status = "built"
        }
        $reg.specialists += $entry
        $reg | ConvertTo-Json -Depth 8 | Set-Content $regPath -Encoding UTF8
        Log "$k registered as $($c.spec)"
    }
}

$st | ConvertTo-Json | Set-Content $state
Remove-Item $lock -Force -ErrorAction SilentlyContinue
Log "autopilot tick done"
