<#
Register (or refresh) the daily forward shadow update as a Windows Task Scheduler task.
Research only: the task runs `research\forward_shadow.py update` (public Binance daily candles; no orders, no keys).

  powershell -NoProfile -ExecutionPolicy Bypass -File research\register_shadow_task.ps1            # register
  powershell -NoProfile -ExecutionPolicy Bypass -File research\register_shadow_task.ps1 -Remove    # remove (records untouched)

Runs daily at 00:30 UTC (converted to local time at registration), and as soon as possible after a missed run
(PC off or asleep). Runs only while the user is logged on, with a 2-hour time limit and one instance at a time.
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
$utc = [DateTime]::SpecifyKind([DateTime]::UtcNow.Date.AddMinutes(30), [DateTimeKind]::Utc)
$local = $utc.ToLocalTime()
$action = New-ScheduledTaskAction -Execute $pyw -Argument "`"$App\research\forward_shadow.py`" update" -WorkingDirectory $App
$trigger = New-ScheduledTaskTrigger -Daily -At $local
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "OnlyFunds 2 forward SHADOW tracking of frozen T50/RS3 (research only; no orders)." -Force | Out-Null
"registered $TaskPath$TaskName daily at $($local.ToString('HH:mm')) local (= 00:30 UTC)"
