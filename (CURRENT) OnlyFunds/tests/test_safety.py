"""Static guards: application code must never be able to place real orders."""
import os
import re

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIRS = ("core", "utils", "scripts", "data")
APP_FILES = ("main.py", "run_paper.py", "run_live.py", "generate_historical.py")


def app_sources():
    for f in APP_FILES:
        yield os.path.join(APP_ROOT, f)
    for d in APP_DIRS:
        for root, _, files in os.walk(os.path.join(APP_ROOT, d)):
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(root, f)


def test_no_app_code_unlocks_trading():
    offenders = []
    for path in app_sources():
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if re.search(r"allow_trading\s*=\s*True", src) and "coinex_client.py" not in path:
            offenders.append(path)
        if re.search(r"^\s*import ccxt|^\s*from ccxt", src, re.M):
            offenders.append(path)
    assert offenders == []


def test_live_lock_constant():
    with open(os.path.join(APP_ROOT, "core", "trade_execution.py"), encoding="utf-8") as fh:
        assert re.search(r"^LIVE_TRADING_LOCKED = True$", fh.read(), re.M)


def test_no_secrets_in_config():
    with open(os.path.join(APP_ROOT, "config", "config.yaml"), encoding="utf-8") as fh:
        src = fh.read()
    assert not re.search(r"(api_key|api_secret|secret_key)\s*:", src)
