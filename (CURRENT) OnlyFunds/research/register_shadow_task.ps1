<#
Register (or refresh) the daily forward shadow update as a Windows Task Scheduler task.
Research only: the task runs `research\shadow_task.py`, which wraps `forward_shadow.py update` (public Binance daily
candles; no orders, no keys) with a single-instance lock, a persistent log and a run journal.

  powershell -NoProfile -ExecutionPolicy Bypass -File research\register_shadow_task.ps1            # register
  powershell -NoProfile -ExecutionPolicy Bypass -File research\register_shadow_task.ps1 -Remove    # remove (records untouched)

The trigger is pinned to 00:30 UTC ("synchronize across time zones"), so it does not drift when local daylight
saving changes. A missed run (PC off or asleep) runs as soon as possible. The task runs only while the user is
logged on, with a 2-hour limit; a second instance is never started.
#>
param([switch]$Remove)
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $PSScriptRoot
$TaskPath = "\OnlyFunds\"
$TaskName = "OF2 Shadow Daily Update"
if ($Remove) {
    Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Confirm:$false
    "removed $TaskPath$TaskName (records untouched)"
    exit 0
}
$py = (Get-Command python.exe).Source
$pyw = Join-Path (Split-Path $py) "pythonw.exe"
$action = New-ScheduledTaskAction -Execute $pyw -Argument "`"$App\research\shadow_task.py`"" -WorkingDirectory $App
$trigger = New-ScheduledTaskTrigger -Daily -At (Get-Date)
$trigger.StartBoundary = [DateTime]::UtcNow.Date.ToString("yyyy-MM-dd") + "T00:30:00Z"   # UTC-pinned
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "OnlyFunds 2 forward SHADOW tracking of frozen T50/RS3 (research only; no orders)." -Force | Out-Null
$t = Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName
"registered $TaskPath$TaskName daily, trigger $($t.Triggers[0].StartBoundary) (00:30 UTC)"
