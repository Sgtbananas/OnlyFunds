import datetime

class RiskManager:
    def __init__(self, config):
        self.config = config

    def position_size(self, equity, price, risk_pct=None):
        """Determine position size, capping per-trade risk."""
        risk_pct = risk_pct or self.config["risk"]["per_trade"]
        usd_to_risk = equity * risk_pct
        amount = usd_to_risk / price
        min_size = self.config["risk"].get("min_size", 0.0001)
        return max(amount, min_size)

    def check_stop_loss(self, entry, current, stop_loss_pct):
        """Return True if stop loss hit."""
        return (current - entry) / entry <= -abs(stop_loss_pct)

    def check_take_profit(self, entry, current, take_profit_pct):
        """Return True if take profit hit."""
        return (current - entry) / entry >= abs(take_profit_pct)

    def check_trailing_stop(self, entry, peak, current, trail_pct):
        """Return True if trailing stop hit (from peak)."""
        return (current - peak) / peak <= -abs(trail_pct)

    def check_max_drawdown(self, equity_curve, max_dd_pct):
        """Return True if global drawdown exceeded."""
        peak = max(equity_curve)
        trough = min(equity_curve[equity_curve.index(peak):])
        dd = (trough - peak) / peak
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