# UltraTensor overnight watchdog: harvests cluster chains autonomously.
# Idempotent; run every ~10 min by a scheduled task. 12-hour budget.
$root = "C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor"
$log = Join-Path $root "outputs\overnight_watch.log"
$state = Join-Path $root "outputs\overnight_state.json"

function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }

function Notify($m) {
    Log "NOTIFY: $m"
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
        $n = New-Object System.Windows.Forms.NotifyIcon
        $n.Icon = [System.Drawing.SystemIcons]::Information
        $n.Visible = $true
        $n.BalloonTipTitle = "UltraTensor Overnight"
        $n.BalloonTipText = $m
        $n.ShowBalloonTip(8000)
        Start-Sleep -Milliseconds 600
        [System.Windows.Forms.Application]::DoEvents()
        Start-Sleep -Seconds 4
        $n.Dispose()
    } catch { Log "toast unavailable: $($_.Exception.Message)" }
}

function WriteNext {
    $lines = New-Object System.Collections.ArrayList
    [void]$lines.Add("# UltraTensor overnight continuation queue")
    [void]$lines.Add("generated: $(Get-Date -Format o)")
    [void]$lines.Add("")
    [void]$lines.Add("- exp256=$($s.done256) exp_mid=$($s.doneMid) exp_pl=$($s.donePl) exp_nl=$($s.doneNl) exp_dom=$($s.doneDom) exp_math=$($s.doneMath) exp_rho320=$($s.doneRho320)")
    $rfile = Join-Path $root "outputs\rho_192.json"
    if (Test-Path $rfile) {
        try {
            $r = Get-Content $rfile | ConvertFrom-Json
            [void]$lines.Add("- rho@192 2-way ridge rel-L1=$($r.two_way.ridge.mean_abs_error) spearman=$($r.two_way.ridge.spearman); knn rel-L1=$($r.two_way.knn.mean_abs_error)")
            [void]$lines.Add("- rho@192 3-way ridge rel-L1=$($r.three_way.ridge.mean_abs_error) spearman=$($r.three_way.ridge.spearman) (n_eval=$($r.three_way.n_eval))")
        } catch { [void]$lines.Add("- rho@192 present but parse failed") }
    }
    $pfile = Join-Path $root "outputs\pl_census.json"
    if (Test-Path $pfile) {
        try {
            $p = Get-Content $pfile | ConvertFrom-Json
            $langs = ($p.languages.PSObject.Properties | ForEach-Object Name) -join ','
            [void]$lines.Add("- PL census: layer=$($p.layer) languages=$langs")
            $ov = @()
            foreach ($prop in $p.overlap_top32.PSObject.Properties) { $ov += "$($prop.Name)=$($prop.Value)" }
            [void]$lines.Add("- overlap top32: $($ov -join '; ')")
        } catch { [void]$lines.Add("- pl_census present but parse failed") }
    }
    $imat = Join-Path $root "outputs\imatrix_pro.dat"
    if (Test-Path $imat) {
        [void]$lines.Add("- imatrix_pro.dat READY: $([math]::Round((Get-Item $imat).Length/1MB)) MB (quant launcher will pick it up)")
    }
    $smoke = Join-Path $root "outputs\v4coder_tokens.txt"
    if (Test-Path $smoke) {
        $txt = (Get-Content $smoke -First 2) -join ' | '
        [void]$lines.Add("- CODER SERVE SMOKE RESULT: $txt")
    }
    $corp = Join-Path $root "outputs\distill_prompts.json"
    if (Test-Path $corp) {
        [void]$lines.Add("- distill corpus exists (Phase-4 prep)")
    }
    [void]$lines.Add("")
    [void]$lines.Add("NEXT (agent, in order): read rho_192.json + exp256_controller_shrink.json;")
    [void]$lines.Add("mid_census.json L4-10 verify; PL overlap verdict -> per-PL extraction plan;")
    [void]$lines.Add("IQ2_XS requant (~74.5GB) is the next build step.")
    Set-Content (Join-Path $root "outputs\NEXT_FOR_AGENT.md") $lines -Encoding UTF8
    Log "NEXT_FOR_AGENT.md refreshed"
}

Log "tick"
$defaults = @{ start = (Get-Date).ToString("o");
               done256 = $false; doneMid = $false; donePl = $false; doneNl = $false;
               doneDom = $false; doneMath = $false; doneRho320 = $false;
               prevDone256 = $false; prevDoneMid = $false; prevDonePl = $false; prevDoneNl = $false;
               prevDoneDom = $false; prevDoneMath = $false; prevDoneRho320 = $false }
$h = @{}
if (Test-Path $state) {
    $stored = Get-Content $state | ConvertFrom-Json
    foreach ($k in $defaults.Keys) { $h[$k] = $stored.$k }
}
foreach ($k in $defaults.Keys) {
    if ($null -eq $h[$k]) { $h[$k] = $defaults[$k] }
}
$h | ConvertTo-Json | Set-Content $state
$s = $h
$start = [datetime]::Parse($s.start)
if ((Get-Date) -gt $start.AddHours(96)) {
    Log "12h budget expired; writing final queue"
    WriteNext
    Notify "UltraTensor overnight: 12h budget done. See outputs\NEXT_FOR_AGENT.md"
    exit
}

function Harvest256 {
    if ($s.done256) { return }
    if (Test-Path "Y:\exp256\DONE.txt") {
        Log "exp256 DONE detected"
        Copy-Item Y:\exp256\subspace_proj.json (Join-Path $root "outputs\exp256_subspace_proj.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp256\controller_shrink.json (Join-Path $root "outputs\exp256_controller_shrink.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp256\ffn_inputs_dense.npz (Join-Path $root "outputs\exp256_ffn_inputs_dense.npz") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp256\router_trace_dense.json (Join-Path $root "outputs\exp256_router_trace_dense.json") -Force -ErrorAction SilentlyContinue
        Log "exp256 artifacts copied"
        & (Join-Path $root ".venv\Scripts\python.exe") (Join-Path $root "scripts\v4_rho_predictor.py") `
            --rank 8 --train 192 --inputs (Join-Path $root "outputs\exp256_ffn_inputs_dense.npz") `
            --proj3 128 --pred3 64 *> (Join-Path $root "outputs\rho_192_run.log")
        if (Test-Path (Join-Path $root "outputs\rho_predictor_L3.json")) {
            Copy-Item (Join-Path $root "outputs\rho_predictor_L3.json") (Join-Path $root "outputs\rho_192.json") -Force
            Log "rho@192 computed"
        } else { Log "rho@192 FAILED (see rho_192_run.log)" }
        $s.done256 = $true
    }
}

function HarvestMid {
    if ($s.doneMid) { return }
    if (Test-Path "Y:\exp_mid\DONE.txt") {
        Log "exp_mid DONE detected"
        Copy-Item Y:\exp_mid\code_census.json (Join-Path $root "outputs\mid_census.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp_mid\ffn_inputs_dense.npz (Join-Path $root "outputs\exp_mid_ffn_inputs_dense.npz") -Force -ErrorAction SilentlyContinue
        Log "exp_mid census copied"
        $s.doneMid = $true
    }
}

function HarvestPl {
    if ($s.donePl) { return }
    if (Test-Path "Y:\exp_pl\DONE.txt") {
        Log "exp_pl DONE detected"
        Copy-Item Y:\exp_pl\pl_census.json (Join-Path $root "outputs\pl_census.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp_pl\ffn_inputs_dense.npz (Join-Path $root "outputs\exp_pl_ffn_inputs_dense.npz") -Force -ErrorAction SilentlyContinue
        Log "exp_pl census copied"
        $s.donePl = $true
    }
}

function HarvestPl {
    if ($s.donePl) { return }
    if (Test-Path "Y:\exp_pl\DONE.txt") {
        Log "exp_pl DONE detected"
        Copy-Item Y:\exp_pl\pl_census.json (Join-Path $root "outputs\pl_census.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp_pl\ffn_inputs_dense.npz (Join-Path $root "outputs\exp_pl_ffn_inputs_dense.npz") -Force -ErrorAction SilentlyContinue
        Log "exp_pl census copied"
        $s.donePl = $true
    }
}

function HarvestNl {
    if ($s.doneNl) { return }
    if (Test-Path "Y:\exp_nl\DONE.txt") {
        Log "exp_nl DONE detected"
        Copy-Item Y:\exp_nl\nl_census.json (Join-Path $root "outputs\nl_census.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp_nl\ffn_inputs_dense.npz (Join-Path $root "outputs\exp_nl_ffn_inputs_dense.npz") -Force -ErrorAction SilentlyContinue
        Log "exp_nl census copied"
        $s.doneNl = $true
    }
}
function HarvestDom {
    if ($s.doneDom) { return }
    if (Test-Path "Y:\exp_dom\DONE.txt") {
        Log "exp_dom DONE detected"
        Copy-Item Y:\exp_dom\code_census.json (Join-Path $root "outputs\dom_census.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp_dom\ffn_inputs_dense.npz (Join-Path $root "outputs\exp_dom_ffn_inputs_dense.npz") -Force -ErrorAction SilentlyContinue
        Log "exp_dom census copied"
        $s.doneDom = $true
    }
}

function HarvestMath {
    if ($s.doneMath) { return }
    if (Test-Path "Y:\exp_math\DONE.txt") {
        Log "exp_math DONE detected"
        Copy-Item Y:\exp_math\code_census.json (Join-Path $root "outputs\math_census.json") -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp_math\ffn_inputs_dense.npz (Join-Path $root "outputs\exp_math_ffn_inputs_dense.npz") -Force -ErrorAction SilentlyContinue
        Log "exp_math census copied"
        $s.doneMath = $true
    }
}

function HarvestRho320 {
    if ($s.doneRho320) { return }
    if (Test-Path "Y:\exp_rho320\DONE.txt") {
        Log "exp_rho320 DONE detected"
        $npz = Join-Path $root "outputs\exp_rho320_ffn_inputs_dense.npz"
        Copy-Item Y:\exp_rho320\ffn_inputs_dense.npz $npz -Force -ErrorAction SilentlyContinue
        Copy-Item Y:\exp_rho320\code_census.json (Join-Path $root "outputs\rho320_census.json") -Force -ErrorAction SilentlyContinue
        Log "exp_rho320 artifacts copied"
        $py = Join-Path $root ".venv\Scripts\python.exe"
        Start-Process -FilePath $py -ArgumentList (Join-Path $root "scripts\v4_rho_predictor.py"),"--inputs",$npz,"--train","256","--proj3","256","--pred3","64","--rank","8" -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root "outputs\rho_256_run.log") -RedirectStandardError (Join-Path $root "outputs\rho_256_run.err.log")
        Log "rho@256 launched (train 256 / eval 64)"
        $s.doneRho320 = $true
    }
}
Harvest256; HarvestMid; HarvestPl; HarvestNl; HarvestDom; HarvestMath; HarvestRho320

# GPU autopilot: detect finished IQ2_XS quants, smoke-test, serve, register.
& (Join-Path $root "scripts\autopilot_gpu.ps1")
$s | ConvertTo-Json | Set-Content $state
Log "tick done (256=$($s.done256) mid=$($s.doneMid) pl=$($s.donePl) nl=$($s.doneNl))"

# persist any fresh harvest into git so overnight progress is self-documenting
$fresh = (($s.done256 -and -not $s.prevDone256) -or
          ($s.doneMid -and -not $s.prevDoneMid) -or
          ($s.donePl -and -not $s.prevDonePl) -or
          ($s.doneNl -and -not $s.prevDoneNl) -or
          ($s.doneDom -and -not $s.prevDoneDom) -or
          ($s.doneMath -and -not $s.prevDoneMath) -or
          ($s.doneRho320 -and -not $s.prevDoneRho320))
if ($fresh) {
    Log "fresh harvest; committing to git"
    Notify "Fresh harvest committed: exp256=$($s.done256) exp_mid=$($s.doneMid) exp_pl=$($s.donePl) exp_nl=$($s.doneNl)"
    WriteNext
    Set-Location $root
    git add -f outputs/exp256_* outputs/mid_census.json outputs/pl_census.json outputs/nl_census.json outputs/dom_census.json outputs/math_census.json outputs/rho320_census.json outputs/exp_rho320_* outputs/rho_192.json outputs/NEXT_FOR_AGENT.md outputs/v4coder_tokens.txt 2>$null
    git -c user.name="NagusameCS" -c user.email="nagusame@users.noreply.github.com" commit -m "overnight harvest: $(Get-Date -Format 'MM-dd HH:mm')" 2>&1 | Out-Null
    git push origin main 2>&1 | Out-Null
    Log "git commit done"
}
$s.prevDone256 = $s.done256
$s.prevDoneMid = $s.doneMid
$s.prevDonePl = $s.donePl
$s.prevDoneNl = $s.doneNl
$s.prevDoneDom = $s.doneDom
$s.prevDoneMath = $s.doneMath
$s.prevDoneRho320 = $s.doneRho320
$s | ConvertTo-Json | Set-Content $state
