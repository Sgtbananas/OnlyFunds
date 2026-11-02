"""Local operator notifications for shadow monitoring (Windows). Research only: no network, no credentials.

`notify` shows a native Windows toast using the built-in Windows PowerShell 5.1 and the WinRT notification API (no
extra packages, no services). The toast also stays in the Windows notification centre until dismissed.
- critical: "reminder" toast, which stays on screen until the operator dismisses it;
- warning:  a normal toast.
After showing it, the toast history is read back to confirm delivery. If the toast cannot be shown (notifications
disabled for Windows PowerShell, no interactive session), it falls back to `msg.exe`, a message box in the user's
session. The result is returned so the caller can record it next to the alert; the alert record itself never
depends on the notification succeeding.

Focus assist / Do Not Disturb can hide the banner; the toast is still delivered to the notification centre.

    python research/shadow_notify.py --test critical      # show a clearly labelled test notification
"""
import base64
import os
import subprocess
import sys

APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
CREATE_NO_WINDOW = 0x08000000
TIMEOUT_S = 60

_PS = r"""
$ErrorActionPreference = "Stop"
[void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
[void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
$app = $env:OF2_TOAST_APP
$esc = { param($s) [System.Security.SecurityElement]::Escape($s) }
$title = & $esc $env:OF2_TOAST_TITLE
$line1 = & $esc $env:OF2_TOAST_LINE1
$body = & $esc $env:OF2_TOAST_BODY
if ($env:OF2_TOAST_LEVEL -eq "critical") {
  $head = '<toast scenario="reminder">'
  $actions = '<actions><action content="Dismiss" arguments="dismiss" activationType="system"/></actions>'
} else { $head = '<toast>'; $actions = '' }
$xml = "$head<visual><binding template=""ToastGeneric""><text>$title</text><text>$line1</text><text>$body</text></binding></visual>$actions</toast>"
$doc = New-Object Windows.Data.Xml.Dom.XmlDocument
$doc.LoadXml($xml)
$t = [Windows.UI.Notifications.ToastNotification]::new($doc)
$t.Tag = $env:OF2_TOAST_TAG
$t.Group = "OF2shadow"
$n = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app)
if ($n.Setting -ne "Enabled") { Write-Output ("DISABLED " + $n.Setting); exit 3 }
$n.Show($t)
Start-Sleep -Milliseconds 700
$h = [Windows.UI.Notifications.ToastNotificationManager]::History.GetHistory($app)
if (@($h | Where-Object { $_.Tag -eq $env:OF2_TOAST_TAG }).Count -gt 0) { Write-Output "DELIVERED" } else { Write-Output "SHOWN_NOT_IN_HISTORY" }
"""


def _toast(title, line1, body, level, tag):
    env = dict(os.environ, OF2_TOAST_APP=APP_ID, OF2_TOAST_TITLE=title, OF2_TOAST_LINE1=line1, OF2_TOAST_BODY=body,
               OF2_TOAST_LEVEL=level, OF2_TOAST_TAG=tag[:60])
    enc = base64.b64encode(_PS.encode("utf-16-le")).decode()
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-EncodedCommand", enc], env=env, capture_output=True, text=True, timeout=TIMEOUT_S,
                       creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    out = (r.stdout or "").strip().splitlines()
    status = out[-1] if out else ""
    return {"method": "toast", "delivered": status == "DELIVERED", "detail": status or (r.stderr or "").strip()[-300:],
            "returncode": r.returncode}


def _msgbox(title, line1, body):
    user = os.environ.get("USERNAME") or "*"
    text = f"{title}\n{line1}\n{body}"[:700]
    r = subprocess.run(["msg", user, "/TIME:43200", text], capture_output=True, text=True, timeout=TIMEOUT_S,
                       creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    return {"method": "msg.exe", "delivered": r.returncode == 0, "detail": (r.stderr or r.stdout or "").strip()[-300:],
            "returncode": r.returncode}


def notify(title, line1, body, level="critical", tag="of2"):
    """Show a notification. Returns {"method", "delivered", "detail", ...}; never raises."""
    if os.name != "nt":
        return {"method": None, "delivered": False, "detail": "not Windows: no local notifier"}
    attempts = []
    for fn, args in ((_toast, (title, line1, body, level, tag)), (_msgbox, (title, line1, body))):
        try:
            res = fn(*args)
        except Exception as exc:
            res = {"method": fn.__name__.strip("_"), "delivered": False, "detail": f"{type(exc).__name__}: {exc}"[:300]}
        attempts.append(res)
        if res["delivered"]:
            break
    return {**attempts[-1], "attempts": attempts}


def main(argv=None):
    a = list(argv if argv is not None else sys.argv[1:])
    level = a[1] if len(a) > 1 else "warning"
    if not a or a[0] != "--test" or level not in ("critical", "warning"):
        print("usage: python research/shadow_notify.py --test [critical|warning]")
        return 2
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    res = notify(f"OF2 shadow TEST ({level.upper()})", f"{now:%Y-%m-%d %H:%M} UTC",
                 "Test notification only. Nothing has failed.", level, tag=f"of2-test-{level}")
    print(res)
    return 0 if res["delivered"] else 1


if __name__ == "__main__":
    sys.exit(main())
