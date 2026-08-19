# UltraTensor morning resume: wakes the operator, reopens the workspace,
# and points at the continuation queue. Scheduled once at 07:00.
$root = "C:\Users\legom\OneDrive\Documents\GitHub\UltraTensor"
$log = Join-Path $root "outputs\overnight_watch.log"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') MORNING RESUME: reopening workspace" | Add-Content $log

try {
    Add-Type -AssemblyName System.Windows.Forms -ErrorAction Stop
    $n = New-Object System.Windows.Forms.NotifyIcon
    $n.Icon = [System.Drawing.SystemIcons]::Information
    $n.Visible = $true
    $n.BalloonTipTitle = "UltraTensor"
    $n.BalloonTipText = "Overnight run finished. Continue the V4-Coder session - see outputs\NEXT_FOR_AGENT.md"
    $n.ShowBalloonTip(10000)
    Start-Sleep -Milliseconds 600
    [System.Windows.Forms.Application]::DoEvents()
    Start-Sleep -Seconds 6
    $n.Dispose()
} catch {
    "toast unavailable: $($_.Exception.Message)" | Add-Content $log
}

& "C:\Users\legom\AppData\Local\Programs\Microsoft VS Code\bin\code.cmd" --reuse-window $root
