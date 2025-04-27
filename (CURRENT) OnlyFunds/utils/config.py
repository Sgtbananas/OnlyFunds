import yaml
import os

class Config:
    """
    Loads config from YAML (and environment variables for secrets).
    Supports config versioning, type safety, and helper accessors.
    """
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self._data = self._load_yaml(config_path)
        self._apply_env_fallbacks()

    def _load_yaml(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data

    def _apply_env_fallbacks(self):
        # For keys needing secure ENV (api keys/secrets)
        api = self._data.get("api", {})
        api_key = os.getenv("COINEX_API_KEY", "")
        if api_key:
            api["api_key"] = api_key
        secret_key = os.getenv("COINEX_SECRET_KEY", "")
        if secret_key:
            api["secret_key"] = secret_key
        self._data["api"] = api

        notif = self._data.get("notifications", {})
        for k in ["telegram_bot_token", "telegram_chat_id", "webhook_url"]:
            envval = os.getenv(k.upper(), "")
            if envval:
                notif[k] = envval
        self._data["notifications"] = notif

    def get(self, section, default=None):
        return self._data.get(section, default)

    def __getitem__(self, item):
        return self._data[item]

    def __contains__(self, item):
        return item in self._data

    # --- Direct helpers for frequently used config parts ---

    @property
    def trading_pairs(self):
        return self._data.get("trading", {}).get("pairs", [])

    @property
    def default_interval(self):
        return self._data.get("trading", {}).get("default_interval", "5m")

    @property
    def max_positions(self):
        return self._data.get("trading", {}).get("max_positions", 2)

    @property
    def dry_run(self):
        return self._data.get("trading", {}).get("dry_run", True)

    @property
    def threshold(self):
        return self._data.get("trading", {}).get("threshold", 0.1)

    @property
    def fee(self):
        return self._data.get("trading", {}).get("fee", 0.0004)

    @property
    def default_capital(self):
        return self._data.get("trading", {}).get("default_capital", 10)

    @property
    def min_size(self):
        return self._data.get("trading", {}).get("min_size", 0.0001)

    @property
    def backtest_lookback(self):
        return self._data.get("trading", {}).get("backtest_lookback", 500)

    @property
    def per_trade_risk(self):
        return self._data.get("risk", {}).get("per_trade", 0.01)

    @property
    def stop_loss_pct(self):
        return self._data.get("risk", {}).get("stop_loss_pct", 0.005)

    @property
    def take_profit_pct(self):
        return self._data.get("risk", {}).get("take_profit_pct", 0.01)

    @property
    def trailing_stop_pct(self):
        return self._data.get("risk", {}).get("trailing_stop_pct", 0.008)

    @property
    def max_drawdown_pct(self):
        return self._data.get("risk", {}).get("max_drawdown_pct", 0.15)

    @property
    def max_daily_loss_pct(self):
        return self._data.get("risk", {}).get("max_daily_loss_pct", 0.05)

    @property
    def ml_enabled(self):
        return self._data.get("ml", {}).get("enabled", True)

    @property
    def ml_min_signal_conf(self):
        return self._data.get("ml", {}).get("min_signal_conf", 0.5)

    @property
    def ml_retrain_on_start(self):
        return self._data.get("ml", {}).get("retrain_on_start", False)

    @property
    def strategy_mode(self):
        return self._data.get("strategy", {}).get("mode", "normal")

    @property
    def autotune(self):
        return self._data.get("strategy", {}).get("autotune", True)

    @property
    def debug(self):
        return self._data.get("debug", False)

    @property
    def log_level(self):
        return self._data.get("log_level", "INFO")

    def as_dict(self):
        return self._data

def load_config(path: str = "config/config.yaml") -> Config:
    """
    Convenience function to load and return a Config instance.
    """
    return Config(path)
