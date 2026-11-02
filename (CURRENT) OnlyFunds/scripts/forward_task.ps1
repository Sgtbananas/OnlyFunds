<#
OnlyFunds 2 - control the forward PAPER deployment through the Windows Task Scheduler.

The supervisor is started by the Task Scheduler service, so it is NOT a child of any terminal, editor or
Claude Code shell: closing those (or a Claude Code turn ending) does not stop it. Paper only: public
Binance market data, no API keys, no orders.

  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\forward_task.ps1 -Action start  [-Deployment binance-1h] [-StateDir <dir>]
  ... -Action status | stop | register | unregister

start      registers (or refreshes) the task, removes a leftover STOP file, starts the task
stop       graceful: creates the STOP file; the session writes its end record, then both exit
status     task state, supervisor and session processes, heartbeat age, last supervisor log lines
register   create or refresh the task only (no start)
unregister remove the task (refused while the supervisor is running); never touches any records

Task: \OnlyFunds\OF2 Forward Paper <deployment>, current user, runs only while the user is logged on,
no time limit, one instance, not stopped on battery, restarted by the Task Scheduler (3x, 1 min apart)
only if the supervisor itself exits with an error. It has no triggers: it runs only when started.
#>
param(
    [ValidateSet("start", "stop", "status", "register", "unregister")] [string]$Action = "status",
    [string]$Deployment = "binance-1h",
    [string]$StateDir = ""
)
$ErrorActionPreference = "Stop"
if ($Deployment -notmatch '^[A-Za-z0-9._-]+$') { throw "invalid deployment name: $Deployment" }

$App = Split-Path -Parent $PSScriptRoot
$TaskPath = "\OnlyFunds\"
$TaskName = "OF2 Forward Paper $Deployment"
# The task runs with its own environment, so a custom OF2_STATE_DIR is passed explicitly as --state-dir.
if (-not $StateDir -and $env:OF2_STATE_DIR) { $StateDir = $env:OF2_STATE_DIR }
if ($StateDir) { $StateRoot = [IO.Path]::GetFullPath($StateDir) }
else { $StateRoot = Join-Path $App "state" }
$Dep = Join-Path $StateRoot "forward_deployments\$Deployment"
$Supervisor = Join-Path $App "scripts\forward_supervisor.py"

function Get-Python {
    $py = (Get-Command python.exe -ErrorAction Stop).Source
    $pyw = Join-Path (Split-Path $py) "pythonw.exe"
    if (-not (Test-Path $pyw)) { throw "pythonw.exe not found next to $py" }
    return @{ py = $py; pyw = $pyw }
}

function Get-SupervisorProcess {
    $pidFile = Join-Path $Dep "supervisor.pid"
    if (-not (Test-Path $pidFile)) { return $null }
    try { $id = [int]((Get-Content $pidFile -Raw | ConvertFrom-Json).pid) } catch { return $null }
    $p = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
    if ($p -and $p.Name -like "python*" -and $p.CommandLine -like "*forward_supervisor.py*") { return $p }
    return $null
}

function Register-Of2Task {
    $p = Get-Python
    $arg = "`"$Supervisor`" --deployment $Deployment"
    if ($StateDir) { $arg += " --state-dir `"$StateRoot`"" }
    $arg += " -- --interval 1h --every 60 --websocket"
    $action = New-ScheduledTaskAction -Execute $p.pyw -Argument $arg -WorkingDirectory $App
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -DontStopOnIdleEnd `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    $user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    $desc = "OnlyFunds 2 forward PAPER trading supervisor ($Deployment). Public market data only; live trading locked. " +
            "Stop with STOP_FORWARD_PAPER.bat (graceful), not with End task."
    Register-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Action $action -Settings $settings `
        -Principal $principal -Description $desc -Force | Out-Null
    "task registered: $TaskPath$TaskName"
    "  runs: $($p.pyw) $arg"
}

function Show-Status {
    $t = Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) {
        $i = $t | Get-ScheduledTaskInfo
        "task: $TaskPath$TaskName  state=$($t.State)  last_run=$($i.LastRunTime)  last_result=$($i.LastTaskResult)"
    } else { "task: not registered" }
    "deployment folder: $Dep"
    $s = Get-SupervisorProcess
    if ($s) {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($s.ParentProcessId)" -ErrorAction SilentlyContinue
        "supervisor: RUNNING pid=$($s.ProcessId) parent=$($s.ParentProcessId) ($($parent.Name))"
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$($s.ProcessId)" | Where-Object { $_.CommandLine -like "*run_forward_session.py*" } |
            ForEach-Object { "session:    RUNNING pid=$($_.ProcessId)" }
    } else { "supervisor: not running" }
    if (Test-Path (Join-Path $Dep "STOP")) { "STOP file: present (a graceful stop was requested)" }
    $last = Get-ChildItem (Join-Path $Dep "sessions") -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($last) {
        $hbPath = Join-Path $last.FullName "heartbeat.json"
        if (Test-Path $hbPath) {
            $hb = Get-Content $hbPath -Raw | ConvertFrom-Json
            $age = [int]((Get-Date).ToUniversalTime() - ([DateTimeOffset]::Parse($hb.time)).UtcDateTime).TotalSeconds
            "latest session: $($last.Name)  heartbeat $age s ago  steps=$($hb.counters.steps) failed=$($hb.counters.failed_steps) ws_alive=$($hb.feed.ws_alive)"
        }
        if (Test-Path (Join-Path $last.FullName "session_end.json")) { "  ended gracefully (session_end.json)" }
        if (Test-Path (Join-Path $last.FullName "session_end_recovered.json")) { "  ended unexpectedly (recovered)" }
    }
    $log = Join-Path $Dep "supervisor.log"
    if (Test-Path $log) { "--- supervisor.log (last 5)"; Get-Content $log -Tail 5 }
}

switch ($Action) {
    "register" { Register-Of2Task }
    "status" { Show-Status }
    "start" {
        $s = Get-SupervisorProcess
        if ($s) { "already running (supervisor pid=$($s.ProcessId)); not starting a second one"; Show-Status; break }
        Register-Of2Task
        $stop = Join-Path $Dep "STOP"
        if (Test-Path $stop) {
            Remove-Item $stop
            New-Item -ItemType Directory -Force $Dep | Out-Null
            Add-Content (Join-Path $Dep "supervisor.log") "$((Get-Date).ToUniversalTime().ToString('s'))+00:00 leftover STOP file removed by forward_task.ps1 start"
        }
        Start-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName
        "task started; waiting for the supervisor..."
        for ($k = 0; $k -lt 30 -and -not (Get-SupervisorProcess); $k++) { Start-Sleep -Seconds 1 }
        Show-Status
    }
    "stop" {
        $p = Get-Python
        $a = @($Supervisor, "--deployment", $Deployment, "--request-stop")
        if ($StateDir) { $a += @("--state-dir", $StateRoot) }
        & $p.py @a
        "the session finishes its current step, writes session_end.json and exits; the supervisor then exits."
        "check with: STATUS_FORWARD_PAPER.bat $Deployment"
    }
    "unregister" {
        if (Get-SupervisorProcess) { throw "supervisor is running; stop it first (STOP_FORWARD_PAPER.bat $Deployment)" }
        Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Confirm:$false
        "task removed: $TaskPath$TaskName (records untouched)"
    }
}
