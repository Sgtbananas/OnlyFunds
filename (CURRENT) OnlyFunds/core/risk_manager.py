import datetime
import numpy as np

class RiskManager:
    def __init__(self, config):
        self.config = config

    def position_size(self, equity, price, risk_pct=None, volatility=None, v_adj=True):
        """
        Volatility-adjusted sizing. If v_adj is True, scale risk down as volatility rises.
        """
        risk_pct = risk_pct or self.config.per_trade_risk
        usd_to_risk = equity * risk_pct
        base_size = usd_to_risk / price
        min_size = getattr(self.config, "min_size", 0.001)
        if v_adj and volatility is not None and volatility > 0:
            # Use some baseline for "normal" volatility, e.g. median of last N bars
            norm_vol = np.median(volatility[-30:]) if hasattr(volatility, '__getitem__') else volatility
            adj = min(2.0, max(0.5, norm_vol / volatility))
            return max(base_size * adj, min_size)
        return max(base_size, min_size)

    def check_stop_loss(self, entry, current, stop_loss):
        """Return True if stop loss hit."""
        return current <= stop_loss

    def check_take_profit(self, entry, current, take_profit):
        """Return True if take profit hit."""
        return current >= take_profit

    def check_trailing_stop(self, trailing_stop, current):
        """Return True if trailing stop hit (from peak)."""
        return current <= trailing_stop

    def check_max_drawdown(self, equity_curve, max_dd_pct):
        """Return True if global drawdown exceeded."""
        if not equity_curve:
            return False
        peak = max(equity_curve)
        trough = min(equity_curve[equity_curve.index(peak):] or [peak])
        dd = (trough - peak) / peak if peak else 0
        return dd <= -abs(max_dd_pct)

    def check_daily_loss(self, trade_log, equity, max_loss_pct, date=None):
        """Return True if daily loss limit exceeded."""
        date = date or datetime.datetime.utcnow().strftime("%Y-%m-%d")
        daily_trades = [t for t in trade_log if t.get("timestamp", "").startswith(date)]
        start_equity = equity
        for t in daily_trades:
            if "return_pct" in t:
                start_equity *= (1 + t["return_pct"])
        loss = (start_equity - equity) / start_equity if start_equity else 0
        return loss <= -abs(max_loss_pct)