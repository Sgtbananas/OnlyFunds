"""Configuration loading.

``config/config.yaml`` holds all non-secret settings. Secrets come only from
environment variables (optionally via a git-ignored ``.env`` file) and are
never written back to disk or shown in the UI.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

import yaml

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_PATH = os.path.join(APP_ROOT, "config", "config.yaml")
PROFILES = ("Conservative", "Normal", "Aggressive")
SECRET_KEYS = ("api_key", "api_secret", "secret_key", "secret")


class Config:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        if not os.path.isabs(config_path):
            candidate = os.path.join(APP_ROOT, config_path)
            config_path = candidate if os.path.exists(candidate) else config_path
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = yaml.safe_load(f) or {}
        if not isinstance(self._data, dict):
            raise ValueError(f"{config_path} must contain a mapping")
        self._strip_secrets()

    def _strip_secrets(self) -> None:
        api = self._data.get("api") or {}
        for k in list(api):
            if k in SECRET_KEYS:
                api.pop(k)

    def get(self, section: str, default=None):
        return self._data.get(section, default)

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self._data.get(name) or {})

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def trading(self) -> Dict[str, Any]:
        return self.section("trading")

    @property
    def trading_pairs(self):
        return self.trading.get("pairs", [])

    @property
    def default_interval(self) -> str:
        return self.trading.get("default_interval", "15m")

    @property
    def profile(self) -> str:
        p = self.trading.get("profile", "Normal")
        return p if p in PROFILES else "Normal"

    def profile_settings(self, profile: Optional[str] = None) -> Dict[str, Any]:
        profiles = self._data.get("profiles") or {}
        return dict(profiles.get(profile or self.profile) or {})

    def save(self, updates: Dict[str, Dict[str, Any]], path: Optional[str] = None) -> None:
        """Merge ``{section: {key: value}}`` into the file atomically. Refuses secrets."""
        data = self.as_dict()
        for section, values in updates.items():
            if section == "api" and any(k in SECRET_KEYS for k in values):
                raise ValueError("refusing to write API secrets to config")
            data.setdefault(section, {})
            data[section].update(values)
        if data.get("trading", {}).get("mode", "paper") != "paper":
            raise ValueError("only paper mode can be saved; live trading is locked")
        target = path or self.config_path
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp, target)
        self._data = data


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Config:
    return Config(path)


def build_settings(cfg: Config, profile: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None):
    """Build (BacktestConfig, PaperSettings) from config + profile + UI overrides.

    ``overrides`` keys: interval, strategy, threshold, initial_capital, use_ml,
    and any RiskLimits / ExitRules field name.
    """
    from core.backtester import BacktestConfig
    from core.paper_trader import PaperSettings
    from core.positions import ExitRules
    from core.risk_manager import RiskLimits

    o = dict(overrides or {})
    t = cfg.trading
    prof = cfg.profile_settings(profile)
    risk_map = {**cfg.section("risk"), **{k: v for k, v in prof.items() if k in RiskLimits.__dataclass_fields__}}
    risk_map.update({k: v for k, v in o.items() if k in RiskLimits.__dataclass_fields__})
    exits_map = cfg.section("exits")
    exits_map.update({k: v for k, v in o.items() if k in ExitRules.__dataclass_fields__})
    exits = ExitRules(**{k: v for k, v in exits_map.items() if k in ExitRules.__dataclass_fields__})
    exits.validate()
    ml = cfg.section("ml")
    common = dict(
        strategy=o.get("strategy", t.get("strategy", "auto")),
        threshold=float(o.get("threshold", prof.get("threshold", 0.35))),
        fee_rate=float(t.get("fee_rate", 0.002)),
        slippage_pct=float(t.get("slippage_pct", 0.0005)),
        ml_threshold=float(ml.get("threshold", 0.55)),
        signal_params=cfg.section("signals"),
        indicator_params=cfg.section("indicators"),
        initial_capital=float(o.get("initial_capital", t.get("initial_capital", 1000))),
    )
    interval = o.get("interval", cfg.default_interval)
    bt = BacktestConfig(interval=interval, exit_rules=exits,
                        risk=RiskLimits.from_mapping(risk_map, kill_switch_file=""), **common)
    paper = PaperSettings(interval=interval, lookback=int(o.get("lookback", 500)),
                          use_ml=bool(o.get("use_ml", ml.get("enabled", False))),
                          exit_rules=copy.deepcopy(exits), risk=RiskLimits.from_mapping(risk_map),
                          trading_window=dict(t.get("trading_windows") or {}), **common)
    return bt, paper
