# Live trading readiness checklist

Live trading is **locked**: `core/trade_execution.py` has `LIVE_TRADING_LOCKED = True`, there is no
live executor, `CoinExClient.place_order/cancel_order` raise `TradingLockedError` unless the client
is constructed with `allow_trading=True` (nothing in the app does this), and `Config.save` refuses
`trading.mode` other than `paper`.

Do **not** unlock until every item below is done and independently verified.

## Exchange integration
- [ ] Confirm CoinEx API v2 is reachable from the deployment location (from the development
      machine it returned HTTP 503 for every endpoint on 2026-11-01).
- [ ] Run the public-data smoke test against the real API (markets, tickers, klines, depth) and
      compare parsed fields to the official docs.
- [ ] With a read-only key: verify signing via `GET /assets/spot/balance`.
- [ ] With a trade-enabled **sub-account holding a small amount**: place and cancel one far-from-market
      limit order; confirm `client_id`, precision rounding, `min_amount` rejection (error 3127), and
      `GET /spot/order-status` values (`open`, `part_filled`, `filled`, `part_canceled`, `canceled`).
- [ ] Confirm the meaning of `amount` for market **buy** orders (docs imply it is denominated in
      `ccy`; verify before use).
- [ ] Handle `OrderStateUnknown`: on a network error during order placement, reconcile via
      order-status / pending-orders before any retry. Never resend blindly.

## Execution engine (not yet built)
- [ ] Live executor implementing the same interface as `PaperExecutor`, with exchange-side
      reconciliation of balances and open orders at start-up and every step.
- [ ] Partial fills: position quantity from `filled_amount`, fees from `base_fee` / `quote_fee`.
- [ ] Server-side protective orders (stop orders) so positions are protected if the app is offline.
- [ ] Separate live ledger; never mix with the paper ledger.

## Validation
- [ ] At least several weeks of paper trading on the intended pairs, interval and profile, with
      reconciliation clean every step.
- [ ] Walk-forward results that are positive after fees on most folds and beat buy-and-hold.
- [ ] ML filter only enabled if its report shows `useful: true` on recent data.

## Operations
- [ ] API key with trade permission only (no withdrawals), IP whitelist enabled.
- [ ] Alerting on HALT / kill switch / reconciliation failures.
- [ ] Start with the Conservative profile and a small capital cap.
