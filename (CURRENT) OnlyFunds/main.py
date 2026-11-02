"""OnlyFunds 2 — Streamlit app.  Run:  streamlit run main.py

PAPER TRADING ONLY. Live order placement is locked in code
(core.trade_execution.LIVE_TRADING_LOCKED).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

import pandas as pd
import streamlit as st

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(APP_ROOT, ".env"))
except ImportError:
    pass

from core import autotune, dashboard, evaluation
from core.backtester import run_backtest, run_out_of_sample
from core.coinex_client import INTERVAL_MAP, CoinExError, top_usdt_markets
from core.core_data import fetch_klines, get_client, normalize_ohlcv
from core.core_signals import STRATEGY_NAMES, generate_signals
from core.ml_filter import load_bundle, train_filter
from core.paper_trader import LedgerBusy, LedgerCorrupt, PaperTrader, quarantine_ledger
from core.trade_execution import LIVE_TRADING_LOCKED, PaperExecutor
from utils.config import PROFILES, Config, build_settings
from utils.helpers import BLACKLISTED_TOKENS

STATE_DIR = os.environ.get("OF2_STATE_DIR") or os.path.join(APP_ROOT, "state")
PAPER_DIR = os.path.join(STATE_DIR, "paper")
LOG_DIR = os.path.join(APP_ROOT, "logs")
BACKTEST_RESULTS_FILE = os.path.join(STATE_DIR, "backtest_results.json")
ML_MODEL_PATH = os.path.join(STATE_DIR, "ml_filter.pkl")
KILL_SWITCH_FILE = os.path.join(STATE_DIR, "KILL_SWITCH")
INTERVALS = [i for i in ("5m", "15m", "30m", "1h", "4h", "1d") if i in INTERVAL_MAP]

st.set_page_config(page_title="OnlyFunds 2", page_icon="📈", layout="wide")


# ------------------------------------------------------------------ setup
@st.cache_resource
def _setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    root = logging.getLogger()
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        h = RotatingFileHandler(os.path.join(LOG_DIR, "onlyfunds.log"), maxBytes=5_000_000, backupCount=3,
                                encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(h)
        root.setLevel(logging.INFO)
    return logging.getLogger("onlyfunds.ui")


log = _setup_logging()


def load_cfg() -> Config:
    return Config(os.path.join(APP_ROOT, "config", "config.yaml"))


@st.cache_data(ttl=600, show_spinner=False)
def cached_universe(limit: int) -> tuple:
    """(pairs, error). Ranked by 24h quote volume, live from CoinEx."""
    try:
        return top_usdt_markets(get_client(), limit=limit, exclude=BLACKLISTED_TOKENS), ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def cached_klines(pair: str, interval: str, limit: int) -> tuple:
    try:
        df = fetch_klines(pair, interval, limit)
        return df, df.attrs.get("source", "coinex"), ""
    except Exception as exc:
        return pd.DataFrame(), "none", f"{type(exc).__name__}: {exc}"


@st.cache_data(ttl=600, show_spinner=False)
def cached_markets() -> tuple:
    try:
        return get_client().get_markets(), ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def live_price(pair: str) -> float:
    return float(get_client().get_ticker(pair)["last"])


def show_error(context: str, exc: Exception) -> None:
    log.exception(context)
    st.error(f"{context}: {type(exc).__name__}: {exc}")


def fmt_metrics(m: dict) -> pd.DataFrame:
    rows = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, dict)}
    return pd.DataFrame(rows.items(), columns=["metric", "value"]).astype({"value": str})


# ---------------------------------------------------------------- sidebar
cfg = load_cfg()
tcfg = cfg.trading

st.sidebar.title("OnlyFunds 2")
if LIVE_TRADING_LOCKED:
    st.sidebar.success("🧪 Paper trading only. Live trading is locked.")

profile = st.sidebar.selectbox("Trading profile", PROFILES, index=PROFILES.index(cfg.profile),
                               help="Sets threshold, risk per trade and exposure caps (config.yaml → profiles).")
strategy = st.sidebar.selectbox("Strategy", STRATEGY_NAMES,
                                index=STRATEGY_NAMES.index(tcfg.get("strategy", "auto"))
                                if tcfg.get("strategy", "auto") in STRATEGY_NAMES else 0,
                                help="auto = pick by market regime: trend-up → trend following, "
                                     "range → mean reversion, high vol → stacked, down-trend → stand aside.")
interval = st.sidebar.selectbox("Candle interval", INTERVALS,
                                index=INTERVALS.index(cfg.default_interval) if cfg.default_interval in INTERVALS else 1)

universe_size = int(tcfg.get("pair_universe_size", 200))
universe, universe_err = cached_universe(universe_size)
if universe_err:
    st.sidebar.warning(f"CoinEx pair list unavailable — using config pairs. ({universe_err[:120]})")
if st.sidebar.button("🔄 Refresh pair list"):
    cached_universe.clear()
    st.rerun()
cfg_pairs = list(tcfg.get("pairs", ["BTCUSDT"]))
options = list(dict.fromkeys(cfg_pairs + universe))
pairs = st.sidebar.multiselect(f"Trading pairs ({len(universe)} live USDT pairs by volume)", options,
                               default=[p for p in cfg_pairs if p in options] or options[:1])

with st.sidebar.expander("Advanced overrides"):
    _, prof_defaults = build_settings(cfg, profile)
    threshold = st.slider("Entry threshold", 0.05, 0.9, float(prof_defaults.threshold), 0.01)
    stop_mult = st.number_input("Stop-loss (× ATR)", 0.3, 10.0, float(prof_defaults.exit_rules.stop_atr_mult), 0.1)
    tp_mult = st.number_input("Take-profit (× ATR)", 0.3, 20.0, float(prof_defaults.exit_rules.tp_atr_mult), 0.1)
    use_trailing = st.checkbox("Trailing stop", value=prof_defaults.exit_rules.use_trailing)
    trail_mult = st.number_input("Trailing stop (× ATR)", 0.3, 10.0, float(prof_defaults.exit_rules.trail_atr_mult), 0.1)
    max_positions = st.number_input("Max open positions", 1, 20, int(prof_defaults.risk.max_open_positions))

overrides = dict(interval=interval, strategy=strategy, threshold=threshold, stop_atr_mult=stop_mult,
                 tp_atr_mult=tp_mult, use_trailing=use_trailing, trail_atr_mult=trail_mult,
                 max_open_positions=int(max_positions), kill_switch_file=KILL_SWITCH_FILE)
try:
    bt_cfg, paper_settings = build_settings(cfg, profile, overrides)
except ValueError as exc:
    st.error(f"Invalid settings: {exc}")
    st.stop()

if st.sidebar.button("💾 Save as defaults"):
    try:
        cfg.save({"trading": {"profile": profile, "strategy": strategy, "default_interval": interval,
                              "pairs": pairs or cfg_pairs},
                  "exits": {"stop_atr_mult": stop_mult, "tp_atr_mult": tp_mult, "trail_atr_mult": trail_mult,
                            "use_trailing": use_trailing},
                  "risk": {"max_open_positions": int(max_positions)}})
        st.sidebar.success("Saved to config/config.yaml")
    except Exception as exc:
        st.sidebar.error(f"Save failed: {exc}")

st.sidebar.caption(f"Risk/trade {paper_settings.risk.risk_per_trade:.2%} · max position "
                   f"{paper_settings.risk.max_position_pct:.0%} · max exposure {paper_settings.risk.max_exposure_pct:.0%}")

# ------------------------------------------------------------------- tabs
st.title("📈 OnlyFunds 2")
tab_market, tab_bt, tab_paper, tab_ml, tab_status = st.tabs(
    ["Market", "Backtest", "Paper trading", "ML & optimisation", "Risk & status"])


# ================================================================= MARKET
with tab_market:
    if not pairs:
        st.info("Select at least one trading pair in the sidebar.")
    else:
        c1, c2 = st.columns([1, 1])
        mpair = c1.selectbox("Pair", pairs, key="market_pair")
        mlimit = c2.slider("Candles", 200, 1000, 500, 50, key="market_limit")
        if st.button("Refresh market data"):
            cached_klines.clear()
        df, source, err = cached_klines(mpair, interval, mlimit)
        if err:
            st.error(f"No market data for {mpair}: {err}")
        else:
            if source != "coinex":
                st.warning(f"CoinEx unreachable — showing cached candles ({len(df)} bars, last "
                           f"{df['timestamp'].iloc[-1] if 'timestamp' in df else '?'}).")
            try:
                sig = generate_signals(df, strategy=strategy, threshold=threshold,
                                       params=bt_cfg.signal_params, indicator_params=bt_cfg.indicator_params)
                last = sig.iloc[-1]
                k = st.columns(6)
                k[0].metric("Last close", f"{last['Close']:.6g}")
                k[1].metric("Signal", f"{last['signal']:+.2f}")
                k[2].metric("Confidence", f"{last['confidence']:.2f}")
                k[3].metric("Regime", str(last["regime"]))
                k[4].metric("Strategy", str(last["strategy"]))
                k[5].metric("Decision", "ENTRY" if last["entry"] else ("EXIT" if last["exit"] else "—"))
                st.plotly_chart(dashboard.price_chart(sig.tail(mlimit), mpair), use_container_width=True)
                st.plotly_chart(dashboard.signal_chart(sig.tail(mlimit), threshold), use_container_width=True)
                st.caption(f"Data source: {source}. Indicators use completed candles only.")
            except Exception as exc:
                show_error("Signal computation failed", exc)


# =============================================================== BACKTEST
with tab_bt:
    st.subheader("Strategy-aware backtest")
    st.caption("Signals at a bar's close execute at the next bar's open. Includes fees "
               f"({bt_cfg.fee_rate:.2%}) and slippage ({bt_cfg.slippage_pct:.2%}).")
    c1, c2, c3, c4 = st.columns(4)
    bt_pairs = c1.multiselect("Pairs", pairs, default=pairs[:1], key="bt_pairs")
    bt_limit = c2.slider("Candles (max 1000 per request; cache extends history)", 300, 1000, 1000, 50)
    bt_capital = c3.number_input("Starting capital (USDT)", 10.0, 1e7, float(bt_cfg.initial_capital), 10.0)
    oos = c4.checkbox("Out-of-sample split", value=True, help="Report the last 30% separately.")
    upload = st.file_uploader("…or backtest an uploaded OHLCV CSV (columns: timestamp, open, high, low, close)",
                              type=["csv"])
    if st.button("▶ Run backtest", type="primary"):
        bt_cfg.initial_capital = float(bt_capital)
        jobs = []
        if upload is not None:
            try:
                jobs.append((os.path.splitext(upload.name)[0], normalize_ohlcv(pd.read_csv(upload)), "upload"))
            except Exception as exc:
                show_error("Could not read CSV", exc)
        for p in bt_pairs:
            df, source, err = cached_klines(p, interval, bt_limit)
            if err:
                st.error(f"{p}: no data ({err})")
            else:
                jobs.append((p, df, source))
        summaries, saved = [], []
        for name, df, source in jobs:
            try:
                with st.spinner(f"Backtesting {name}…"):
                    if oos:
                        both = run_out_of_sample(df, bt_cfg, name)
                        res = both["out_of_sample"]
                        summaries.append({"pair": name, "segment": "in-sample", **both["in_sample"].metrics})
                        summaries.append({"pair": name, "segment": "out-of-sample", **res.metrics})
                    else:
                        res = run_backtest(df, bt_cfg, name)
                        summaries.append({"pair": name, "segment": "full", **res.metrics})
                st.markdown(f"#### {name} <small>({source}, {len(df)} bars)</small>", unsafe_allow_html=True)
                m = res.metrics
                k = st.columns(6)
                k[0].metric("Return", f"{m['total_return_pct']:.2f}%",
                            f"buy&hold {m['buy_and_hold_return_pct']:.2f}%", delta_color="off")
                k[1].metric("Trades", m["trades"])
                k[2].metric("Win rate", f"{m['win_rate_pct']:.1f}%")
                k[3].metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
                k[4].metric("Profit factor", f"{m['profit_factor']:.2f}")
                k[5].metric("Fees paid", f"{m['fees_paid']:.2f}")
                k2 = st.columns(6)
                k2[0].metric("Expectancy / trade", f"{m['expectancy']:.4f}")
                k2[1].metric("Trades / day", f"{m['trades_per_day']:.2f}")
                k2[2].metric("Sharpe", f"{m['sharpe']:.2f}")
                k2[3].metric("Sortino", f"{m['sortino']:.2f}")
                k2[4].metric("Return / max DD", f"{m['return_over_max_dd']:.2f}")
                k2[5].metric("Exposure", f"{m['exposure_pct']:.1f}%")
                if m["missing_candles"]:
                    st.warning(f"{m['missing_candles']} missing candle(s) in this data; indicators around gaps "
                               "are less reliable.")
                if m["by_regime"]:
                    st.caption("Results by market regime at entry:")
                    st.dataframe(pd.DataFrame(m["by_regime"]).T, use_container_width=True)
                st.plotly_chart(dashboard.equity_chart(res.equity, f"{name} equity"), use_container_width=True)
                if len(res.trades):
                    st.dataframe(res.trades.drop(columns=["id"]), use_container_width=True, height=260)
                if res.open_position:
                    st.info(f"Open at end: {res.open_position['qty']:.6g} @ {res.open_position['entry_price']:.6g}, "
                            f"unrealised PnL {m['unrealised_pnl']:.4f}")
                for s in autotune.suggest_tuning_parameters(m):
                    st.caption("• " + s)
                saved.append({"pair": name, "metrics": m, "trades": res.trades.to_dict("records")})
            except Exception as exc:
                show_error(f"Backtest failed for {name}", exc)
        if summaries:
            st.markdown("### Summary")
            st.dataframe(pd.DataFrame(summaries).drop(columns=["exit_reasons"], errors="ignore"),
                         use_container_width=True)
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(BACKTEST_RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump({"time": time.time(), "config": bt_cfg.as_dict(), "results": saved}, f, default=str)
    elif os.path.exists(BACKTEST_RESULTS_FILE):
        try:
            with open(BACKTEST_RESULTS_FILE, encoding="utf-8") as f:
                prev = json.load(f)
            st.caption(f"Last saved backtest ({time.strftime('%Y-%m-%d %H:%M', time.localtime(prev['time']))}):")
            st.dataframe(pd.DataFrame([{"pair": r["pair"], **{k: v for k, v in r["metrics"].items()
                                                              if not isinstance(v, dict)}}
                                       for r in prev.get("results", [])]), use_container_width=True)
        except Exception as exc:
            st.caption(f"Previous results unreadable: {exc}")


# ================================================================== PAPER
def make_trader() -> PaperTrader:
    markets, _ = cached_markets()
    ex = PaperExecutor(paper_settings.fee_rate, paper_settings.slippage_pct, markets=markets or None,
                       min_notional=paper_settings.risk.min_notional)
    bundle = load_bundle(ML_MODEL_PATH) if paper_settings.use_ml or st.session_state.get("use_ml") else None
    if st.session_state.get("use_ml"):
        paper_settings.use_ml = True
    return PaperTrader(paper_settings, PAPER_DIR, price_fn=live_price, executor=ex, ml_bundle=bundle)


def render_paper(trader: PaperTrader) -> None:
    issues = trader.reconcile()
    if issues:
        st.error("Ledger reconciliation failed; new entries are blocked: " + "; ".join(issues))
    m = trader.metrics()
    k = st.columns(7)
    k[0].metric("Equity", f"{m['equity']:.2f}", f"{m['total_return_pct']:+.2f}%")
    k[1].metric("Cash", f"{m['cash']:.2f}")
    k[2].metric("Open positions", m["open_positions"])
    k[3].metric("Realised PnL", f"{m['realised_pnl']:.4f}")
    k[4].metric("Unrealised PnL", f"{m['unrealised_pnl']:.4f}")
    k[5].metric("Win rate", f"{m['win_rate_pct']:.1f}%", f"{m['trades']} trades", delta_color="off")
    k[6].metric("Max drawdown", f"{m['max_drawdown_pct']:.2f}%")
    if m["kill_switch"]:
        st.error("Kill switch is ACTIVE — no new entries. Clear it in 'Risk & status' when ready.")
    led = trader.ledger
    if led["equity_history"]:
        st.plotly_chart(dashboard.equity_chart(pd.DataFrame(led["equity_history"]), "Paper equity"),
                        use_container_width=True)
    st.markdown("**Open positions**")
    if led["positions"]:
        rows = []
        for pair, p in trader.positions.items():
            px = float(led["last_prices"].get(pair, p.entry_price))
            rows.append({"pair": pair, "qty": p.qty, "entry": p.entry_price, "price": px, "stop": p.effective_stop,
                         "take_profit": p.take_profit, "unrealised_pnl": p.unrealised_pnl(px, trader.executor.fee_rate),
                         "strategy": p.strategy, "opened": p.entry_time})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.caption("No open positions.")
    st.markdown("**Trade ledger**")
    if led["trades"]:
        st.dataframe(pd.DataFrame(led["trades"]).iloc[::-1].drop(columns=["id"], errors="ignore"),
                     use_container_width=True, height=260)
    else:
        st.caption("No closed trades yet.")
    with st.expander(f"Decision log ({len(led['decisions'])})"):
        st.dataframe(pd.DataFrame(led["decisions"][-300:]).iloc[::-1], use_container_width=True, height=300)
    if led["failed_orders"]:
        with st.expander(f"Failed simulated orders ({len(led['failed_orders'])})"):
            st.dataframe(pd.DataFrame(led["failed_orders"]), use_container_width=True)


with tab_paper:
    st.subheader("Paper trading (simulated orders on live CoinEx data)")
    st.caption("Decides once per completed candle. Orders are simulated with fees and slippage; nothing is sent "
               "to the exchange. For unattended operation run `python run_paper.py` instead of keeping this tab open.")
    c1, c2, c3 = st.columns([1, 1, 2])
    run_once = c1.button("▶ Run one step now", type="primary", disabled=not pairs)
    auto = c2.toggle("Auto-run", key="paper_auto", help="Runs a step every refresh interval while this tab is open.")
    refresh_s = int(tcfg.get("paper_refresh_seconds", 60))
    trader = None
    try:
        trader = make_trader()
    except LedgerCorrupt as exc:
        st.error(f"Paper ledger cannot be read: {exc}")
        st.caption("Nothing was overwritten. Inspect state/paper/ first if you need the history.")
        if st.checkbox("I understand the corrupt ledger will be moved aside, not deleted") and \
                st.button("Move corrupt ledger aside and start a new paper account"):
            moved = quarantine_ledger(PAPER_DIR)
            st.success(f"Moved: {moved}. Reload the page.")
    except Exception as exc:
        show_error("Could not open paper ledger", exc)

    def do_step():
        try:
            decisions = trader.step(pairs)
            acted = [d for d in decisions if d["action"] in ("BUY", "SELL", "BUY_FAILED", "SELL_FAILED", "ERROR",
                                                             "DATA_ERROR", "HALT")]
            for d in acted:
                msg = f"{d['pair']}: {d['action']} — {d['reason']}"
                if "FAIL" in d["action"] or "ERROR" in d["action"]:
                    st.error(msg)
                else:
                    st.info(msg)
        except LedgerBusy as exc:
            st.warning(str(exc))
        except Exception as exc:
            show_error("Paper step failed", exc)

    if trader is None:
        pass
    elif run_once:
        do_step()

    if trader is None:
        pass
    elif auto:
        @st.fragment(run_every=refresh_s)
        def auto_panel():
            do_step()
            st.caption(f"Auto-run: last step {time.strftime('%H:%M:%S')}, every {refresh_s}s.")
            render_paper(trader)
        auto_panel()
    else:
        render_paper(trader)


# ===================================================== ML & OPTIMISATION
with tab_ml:
    st.subheader("ML entry filter")
    st.caption("Predicts whether an entry hits take-profit before stop-loss. Trained on a chronological split with "
               "a purge gap. Flagged useful only if, on held-out data, the trades it accepts earn more per trade "
               "after fees and slippage than taking every signal (win rate alone is not enough).")
    bundle = load_bundle(ML_MODEL_PATH)
    if bundle:
        rep = bundle["report"]
        (st.success if rep["useful"] else st.warning)(
            f"Saved model: {'shows an out-of-sample edge' if rep['useful'] else 'NO demonstrated edge'} — "
            f"validation AUC {rep['validation_auc']:.3f}; expectancy after costs "
            f"{rep['validation_expectancy_r_accepted']:+.3f}R on accepted trades vs "
            f"{rep['validation_expectancy_r_all']:+.3f}R taking every signal (trained {rep['trained_at'][:16]}).")
        with st.expander("Model report"):
            st.json(rep)
    else:
        st.info("No ML filter trained yet.")
    use_ml = st.checkbox("Gate paper-trading entries with the ML filter", key="use_ml",
                         disabled=bundle is None,
                         help="Not recommended unless the report shows an out-of-sample edge.")
    c1, c2 = st.columns(2)
    ml_pairs = c1.multiselect("Training pairs", pairs, default=pairs[:3], key="ml_pairs")
    if c2.button("Train ML filter", disabled=not ml_pairs):
        frames = {}
        for p in ml_pairs:
            df, source, err = cached_klines(p, interval, 1000)
            if err:
                st.error(f"{p}: {err}")
            else:
                frames[p] = df
        if frames:
            try:
                with st.spinner("Training…"):
                    b = train_filter(frames, bt_cfg.exit_rules, threshold=bt_cfg.ml_threshold, strategy=strategy,
                                     signal_threshold=threshold, save_path=ML_MODEL_PATH)
                st.json(b["report"])
            except Exception as exc:
                show_error("ML training failed", exc)
    if st.button("Compare rule-only vs ML-gated (out-of-sample)", disabled=not pairs):
        df, source, err = cached_klines(pairs[0], interval, 1000)
        if err:
            st.error(err)
        else:
            try:
                with st.spinner("Training on first 70%, testing on last 30%…"):
                    r = autotune.ml_out_of_sample_backtest(df, bt_cfg, pairs[0])
                st.dataframe(pd.DataFrame({"rule_only": r["rule_only"], "ml_gated": r["ml_gated"]})
                             .drop(index=["exit_reasons"], errors="ignore").astype(str), use_container_width=True)
                st.caption(f"Model trained only on the first 70% of {pairs[0]}; useful={r['report']['useful']}.")
            except Exception as exc:
                show_error("ML comparison failed", exc)

    st.divider()
    st.subheader("Strategy comparison (does anything have an edge?)")
    st.caption("Every strategy on the selected pairs: fixed defaults plus walk-forward out-of-sample. "
               "'credible_edge' requires profit factor > 1.1, beating buy-and-hold, positive walk-forward "
               "results on most folds and at least 30 trades. Slow: a few backtests per pair and strategy.")
    cmp_pairs = st.multiselect("Pairs to compare", pairs, default=pairs[:1], key="cmp_pairs")
    if st.button("Compare strategies", disabled=not cmp_pairs):
        frames = {}
        for p in cmp_pairs:
            df, source, err = cached_klines(p, interval, 1000)
            if err:
                st.error(f"{p}: {err}")
            else:
                frames[p] = df
        if frames:
            try:
                with st.spinner("Evaluating…"):
                    ev = evaluation.evaluate(frames, bt_cfg)
                st.dataframe(ev["verdict"], use_container_width=True)
                st.dataframe(ev["fixed"], use_container_width=True)
                st.dataframe(ev["walk_forward"], use_container_width=True)
            except Exception as exc:
                show_error("Strategy comparison failed", exc)

    st.divider()
    st.subheader("Parameter optimisation (train → out-of-sample)")
    st.caption("Grid search on the first 70% only, then one evaluation on the unseen 30%. Results are never "
               "applied automatically.")
    c1, c2 = st.columns(2)
    opt_pair = c1.selectbox("Pair", pairs or ["BTCUSDT"], key="opt_pair")
    wf = c2.checkbox("Walk-forward (4 folds)")
    if st.button("Run optimisation"):
        df, source, err = cached_klines(opt_pair, interval, 1000)
        if err:
            st.error(err)
        else:
            try:
                with st.spinner("Optimising…"):
                    if wf:
                        r = autotune.walk_forward(df, bt_cfg, pair=opt_pair)
                        st.metric("Compounded out-of-sample return", f"{r['oos_compounded_return_pct']:.2f}%",
                                  f"{r['profitable_folds']}/{r['n_folds']} folds profitable", delta_color="off")
                        st.dataframe(r["folds"].astype(str), use_container_width=True)
                    else:
                        r = autotune.optimise(df, bt_cfg, pair=opt_pair)
                        if r["best_params"] is None:
                            st.warning(r["reason"])
                        else:
                            (st.success if r["confirmed"] else st.warning)(
                                f"Best in-sample params {r['best_params']} — out-of-sample: {r['reason']}.")
                            st.dataframe(pd.DataFrame({"in_sample": r["in_sample"],
                                                       "out_of_sample": r["out_of_sample"]})
                                         .drop(index=["exit_reasons"], errors="ignore").astype(str),
                                         use_container_width=True)
                        st.dataframe(r["table"], use_container_width=True, height=250)
            except Exception as exc:
                show_error("Optimisation failed", exc)


# ========================================================== RISK & STATUS
with tab_status:
    st.subheader("Capital protection")
    c1, c2 = st.columns(2)
    c1.markdown("**Risk limits (active profile + overrides)**")
    c1.json({k: v for k, v in paper_settings.risk.as_dict().items() if k != "kill_switch_file"})
    c2.markdown("**Exit rules**")
    c2.json(paper_settings.exit_rules.__dict__)
    try:
        trader_s = make_trader()
        ks = trader_s.risk.kill_switch_active()
        st.markdown(f"**Kill switch:** {'🛑 ACTIVE' if ks else '✅ off'}")
        b1, b2, b3 = st.columns(3)
        confirm_stop = b1.checkbox("Confirm emergency stop")
        if b1.button("🛑 Emergency stop (close all paper positions)", disabled=not confirm_stop):
            closed = trader_s.emergency_stop("manual emergency stop from UI")
            st.warning(f"Kill switch engaged; {len(closed)} paper position(s) closed.")
        if b2.button("Clear kill switch", disabled=not ks):
            trader_s.risk.clear_kill_switch()
            st.success("Kill switch cleared.")
        if b3.button("Reconcile paper ledger"):
            issues = trader_s.reconcile()
            if issues:
                st.error("; ".join(issues))
            else:
                st.success("Ledger consistent.")
        with st.expander("Reset paper account"):
            new_cap = st.number_input("New starting capital (USDT)", 10.0, 1e7, float(paper_settings.initial_capital))
            if st.checkbox("I understand this erases the paper ledger") and st.button("Reset"):
                trader_s.reset(new_cap)
                st.success("Paper account reset.")
    except LedgerBusy as exc:
        st.warning(str(exc))
    except LedgerCorrupt as exc:
        st.error(f"Paper ledger cannot be read: {exc} (see the Paper trading tab)")
    except Exception as exc:
        show_error("Status panel error", exc)

    st.divider()
    st.subheader("Exchange")
    client = get_client()
    st.markdown(f"- Live trading: **{'LOCKED' if LIVE_TRADING_LOCKED else 'unlocked'}**\n"
                f"- API credentials present: **{'yes' if client.has_credentials else 'no'}** "
                "(read from environment / .env; never shown)")
    if st.button("Test CoinEx connection"):
        try:
            t0 = time.time()
            n = len(client.get_markets(refresh=True))
            st.success(f"CoinEx public API OK: {n} markets ({(time.time() - t0) * 1000:.0f} ms)")
        except Exception as exc:
            st.error(f"CoinEx unreachable: {exc}")
        if client.has_credentials:
            try:
                bal = client.get_spot_balance()
                st.write({k: v for k, v in bal.items() if v["available"] or v["frozen"]})
            except CoinExError as exc:
                st.error(f"Balance check failed: {exc}")
