# backtest.py — Realistic Out-of-Sample Backtest with Fees, Slippage & Position Caps

import argparse, json, logging, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", message="X has feature names, but RandomForestClassifier")

from train_model import (
    fetch_klines, _process_segment, FULL_FEATURES, MODEL_FILE, SYMBOLS,
    ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Cost & Execution Assumptions ──────────────────────────────────────
TAKER_FEE_PCT   = 0.0005   # 0.05% per side — Deribit USDC perpetual taker fee
SLIPPAGE_PCT    = 0.0005   # 0.05% adverse slippage per side
LOOKAHEAD_BARS  = 24       # Matches label lookahead (6h on 15m candles)
MAX_LEVERAGE    = 10.0     # Hard cap on notional as a multiple of equity
META_MODEL_FILE = "meta_pipeline.pkl"


def simulate_symbol(symbol: str, df15: pd.DataFrame, pipeline: dict, threshold: float,
                     meta_pipeline: dict = None, meta_threshold: float = None) -> list:
    """
    Walks every bar of df15, generates ML signals, and resolves outcomes against TP1/SL.
    """
    af        = pipeline["all_features"]
    selector  = pipeline["selector"]
    ensemble  = pipeline["ensemble"]
    label_map = pipeline["label_map"]

    for f in af:
        if f not in df15.columns:
            df15[f] = 0.0

    X = df15[af].replace([np.inf, -np.inf], np.nan).fillna(0)
    Xs = selector.transform(X)
    preds  = ensemble.predict(Xs)
    probas = ensemble.predict_proba(Xs)

    meta_probas = None
    if meta_pipeline is not None:
        mf = meta_pipeline["meta_features"]
        for f in mf:
            if f not in df15.columns:
                df15[f] = 0.0
        Xm = df15[mf].replace([np.inf, -np.inf], np.nan).fillna(0)
        meta_probas = meta_pipeline["meta_ensemble"].predict_proba(Xm)[:, 1]

    trades = []
    n = len(df15)
    highs   = df15["high"].values
    lows    = df15["low"].values
    closes  = df15["close"].values
    opens_t = df15["open_time"].values
    atrs    = df15["atr"].values if "atr" in df15.columns else None

    for i in range(n - LOOKAHEAD_BARS):
        sig  = label_map[int(preds[i])]
        conf = float(max(probas[i]))
        if sig == "NO_TRADE" or conf < threshold:
            continue
        if meta_probas is not None and meta_probas[i] < meta_threshold:
            continue
        if atrs is None or atrs[i] <= 0:
            continue

        entry = float(closes[i])
        atr   = float(atrs[i])

        if sig == "BUY":
            stop = entry - atr * ATR_STOP_MULT
            tp1  = entry + atr * ATR_TARGET1_MULT
        else:
            stop = entry + atr * ATR_STOP_MULT
            tp1  = entry - atr * ATR_TARGET1_MULT

        if stop <= 0 or tp1 <= 0:
            continue

        outcome, exit_price, exit_bar = "TIME_EXIT", float(closes[i + LOOKAHEAD_BARS]), i + LOOKAHEAD_BARS
        for j in range(i + 1, i + LOOKAHEAD_BARS + 1):
            if sig == "BUY":
                hit_sl = lows[j]  <= stop
                hit_tp = highs[j] >= tp1
            else:
                hit_sl = highs[j] >= stop
                hit_tp = lows[j]  <= tp1

            # Conservative tie-breaker: assume SL hit first if both touched on same bar
            if hit_sl:
                outcome, exit_price, exit_bar = "SL", stop, j
                break
            if hit_tp:
                outcome, exit_price, exit_bar = "TP1", tp1, j
                break

        trades.append({
            "symbol": symbol, "signal": sig, "confidence": conf,
            "entry_time": int(opens_t[i]), "exit_time": int(opens_t[exit_bar]),
            "entry": entry, "stop": stop, "tp1": tp1,
            "outcome": outcome, "exit_price": exit_price,
        })

    return trades


def run_portfolio_simulation(all_trades: list, starting_equity: float, risk_per_trade: float, max_open: int) -> dict:
    """
    Applies position caps chronologically, deducts realistic fees & slippage,
    and compounds equity dynamically.
    """
    all_trades.sort(key=lambda t: t["entry_time"])

    taken = []
    open_intervals = []

    for t in all_trades:
        open_intervals = [iv for iv in open_intervals if iv[1] > t["entry_time"]]
        if len(open_intervals) >= max_open:
            continue
        open_intervals.append((t["entry_time"], t["exit_time"]))
        taken.append(t)

    taken.sort(key=lambda t: t["exit_time"])

    equity = starting_equity
    equity_curve = [equity]
    peak = equity
    max_dd = 0.0
    wins, losses = 0, 0
    gross_win, gross_loss = 0.0, 0.0
    r_multiples = []
    capped_count = 0

    for t in taken:
        entry_fill = t["entry"] * (1 + SLIPPAGE_PCT) if t["signal"] == "BUY" else t["entry"] * (1 - SLIPPAGE_PCT)
        exit_fill  = t["exit_price"] * (1 - SLIPPAGE_PCT) if t["signal"] == "BUY" else t["exit_price"] * (1 + SLIPPAGE_PCT)

        stop_dist_pct = abs(t["entry"] - t["stop"]) / t["entry"]
        risk_dollars_target = equity * risk_per_trade
        raw_notional  = risk_dollars_target / stop_dist_pct if stop_dist_pct > 0 else 0

        max_notional = equity * MAX_LEVERAGE
        notional = min(raw_notional, max_notional)
        if raw_notional > max_notional:
            capped_count += 1
        risk_dollars = notional * stop_dist_pct

        raw_pnl_pct = ((exit_fill - entry_fill) / entry_fill) if t["signal"] == "BUY" else ((entry_fill - exit_fill) / entry_fill)
        pnl_dollars = notional * raw_pnl_pct
        fees        = notional * TAKER_FEE_PCT * 2
        pnl_dollars -= fees

        equity += pnl_dollars
        equity_curve.append(equity)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        r_multiple = pnl_dollars / risk_dollars if risk_dollars > 0 else 0
        r_multiples.append(r_multiple)

        if pnl_dollars > 0:
            wins += 1
            gross_win += pnl_dollars
        else:
            losses += 1
            gross_loss += abs(pnl_dollars)

    n = len(taken)
    win_rate = wins / n if n else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0
    avg_r = float(np.mean(r_multiples)) if r_multiples else 0
    std_r = float(np.std(r_multiples)) if r_multiples else 0
    sharpe_per_trade = (avg_r / std_r) if std_r > 0 else 0
    
    total_days = (taken[-1]["exit_time"] - taken[0]["entry_time"]) / (1000 * 60 * 60 * 24) if n > 1 else 1
    trades_per_year = n / total_days * 365 if total_days > 0 else 0
    sharpe_annualized = sharpe_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    return {
        "total_signals_generated": len(all_trades),
        "trades_taken_after_cap": n,
        "skipped_due_to_max_open": len(all_trades) - n,
        "trades_leverage_capped": capped_count,
        "win_rate": round(win_rate * 100, 1),
        "wins": wins, "losses": losses,
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf (no losses)",
        "avg_r_multiple": round(avg_r, 3),
        "sharpe_per_trade": round(sharpe_per_trade, 3),
        "sharpe_annualized_approx": round(sharpe_annualized, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "starting_equity": starting_equity,
        "ending_equity": round(equity, 2),
        "total_return_pct": round((equity - starting_equity) / starting_equity * 100, 2),
        "days_covered": round(total_days, 1),
        "trades_per_year_approx": round(trades_per_year, 0),
        "equity_curve": equity_curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20, help="How many recent days per symbol to backtest")
    parser.add_argument("--threshold", type=float, default=None, help="Primary confidence threshold (default: pipeline recommended)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Subset of symbols (default: all active)")
    parser.add_argument("--use-meta", action="store_true", help="Apply meta_pipeline.pkl as an additional filter")
    parser.add_argument("--meta-threshold", type=float, default=None, help="Meta confidence threshold")
    parser.add_argument("--tag", type=str, default=None, help="Label for this run in backtest_history.json")
    parser.add_argument("--equity", type=float, default=100_000.0, help="Starting simulation equity (default: $100k, use 10.0 for ₹500 test)")
    parser.add_argument("--risk-per-trade", type=float, default=0.01, help="Risk per trade as decimal (default: 0.01, use 0.03 for ₹500 test)")
    parser.add_argument("--max-open", type=int, default=10, help="Max concurrent open trades (default: 10, use 2 for ₹500 test)")
    parser.add_argument("--allow-historical", action="store_true", help="Bypass the post-training cutoff to evaluate against recent historical holdouts")
    args = parser.parse_args()

    pipeline = joblib.load(MODEL_FILE)
    threshold = args.threshold if args.threshold is not None else pipeline.get("recommended_threshold", 0.45)
    symbols = args.symbols if args.symbols else SYMBOLS

    meta_pipeline, meta_threshold = None, None
    if args.use_meta:
        meta_pipeline = joblib.load(META_MODEL_FILE)
        meta_threshold = args.meta_threshold if args.meta_threshold is not None else meta_pipeline.get("recommended_meta_threshold", 0.5)
        log.info(f"META FILTER ACTIVE — meta_threshold={meta_threshold}")

    trained_at_str = pipeline.get("trained_at")
    trained_at_ms = None
    if trained_at_str and trained_at_str != "unknown":
        try:
            trained_at_ms = int(pd.Timestamp(trained_at_str).timestamp() * 1000)
        except Exception as e:
            log.warning(f"Could not parse trained_at: {e}")

    hours_since_training = None
    if trained_at_ms is not None:
        hours_since_training = (pd.Timestamp.now("UTC").timestamp() * 1000 - trained_at_ms) / (1000 * 3600)
        log.info(f"Model trained_at: {trained_at_str} ({hours_since_training:.1f}h ago)")

    if args.allow_historical:
        log.warning("⚠️ --allow-historical flag active — bypassing post-training timestamp filter to test recent historical data.")
        trained_at_ms = None  # Disable cutoff for immediate holdout testing

    log.info(f"Backtesting {len(symbols)} symbols, requested {args.days} days, threshold={threshold}")

    candles = max(args.days, 5) * 96 + 500
    btc_df15 = fetch_klines("BTCUSDT", "15m", candles)
    if trained_at_ms is not None and not btc_df15.empty:
        btc_df15 = btc_df15[btc_df15["open_time"] > trained_at_ms].reset_index(drop=True)
    all_trades = []

    for symbol in symbols:
        try:
            df15 = fetch_klines(symbol, "15m", candles)
            df1h = fetch_klines(symbol, "1h", candles // 4)
            df4h = fetch_klines(symbol, "4h", candles // 16)

            if df15.empty or "open_time" not in df15.columns:
                continue

            if trained_at_ms is not None:
                before = len(df15)
                df15 = df15[df15["open_time"] > trained_at_ms].reset_index(drop=True)
                if before > 0 and len(df15) == 0:
                    continue

            if df15.empty or len(df15) < (100 if trained_at_ms is not None else 30):
                continue

            processed = _process_segment(df15, df1h, df4h, regime="backtest", btc_df15=btc_df15)
            if processed.empty:
                continue

            trades = simulate_symbol(symbol, processed, pipeline, threshold,
                                      meta_pipeline=meta_pipeline, meta_threshold=meta_threshold)
            log.info(f"  [{symbol}] {len(trades)} signals generated")
            all_trades.extend(trades)
        except Exception as e:
            log.warning(f"  [{symbol}] backtest error: {e}")

    if not all_trades:
        log.error("No trades generated — if you recently retrained, use --allow-historical to evaluate recent bars.")
        return

    results = run_portfolio_simulation(all_trades, starting_equity=args.equity, risk_per_trade=args.risk_per_trade, max_open=args.max_open)
    equity_curve = results.pop("equity_curve")

    log.info("\n" + "=" * 60)
    log.info("BACKTEST RESULTS (realistic fees + slippage + position caps)")
    log.info("=" * 60)
    for k, v in results.items():
        log.info(f"  {k}: {v}")
    log.info("=" * 60)

    with open("backtest_results.json", "w") as f:
        json.dump({**results, "equity_curve": equity_curve, "threshold": threshold,
                   "days": args.days, "symbols": symbols}, f, indent=2)
    log.info("Saved: backtest_results.json")

    HISTORY_FILE = "backtest_history.json"
    try:
        with open(HISTORY_FILE) as f: history = json.load(f)
    except Exception:
        history = []

    entry = {
        "run_at": pd.Timestamp.now("UTC").isoformat(),
        "tag": args.tag or ("historical-eval" if args.allow_historical else "oos-live"),
        "model_trained_at": pipeline.get("trained_at", "unknown"),
        "hours_since_training": round(hours_since_training, 1) if hours_since_training is not None else None,
        "primary_threshold": threshold,
        "days": args.days,
        "n_symbols": len(symbols),
        **results,
    }
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Appended to {HISTORY_FILE} ({len(history)} runs tracked total)")

    log.info("\n" + "=" * 78)
    log.info(f"{'LEADERBOARD (all tracked runs, ranked by Sharpe)':^78}")
    log.info("=" * 78)
    log.info(f"{'#':<3}{'run_at':<20}{'tag':<16}{'sharpe':<9}{'maxDD%':<9}{'PF':<8}{'ret%':<8}")
    ranked = sorted(history, key=lambda r: r.get("sharpe_annualized_approx", -999), reverse=True)
    for i, r in enumerate(ranked[:15], 1):
        marker = " <-- THIS RUN" if r is entry else ""
        log.info(f"{i:<3}{r['run_at'][:16]:<20}{str(r.get('tag') or '-'):<16}"
                 f"{r.get('sharpe_annualized_approx','-'):<9}"
                 f"{r.get('max_drawdown_pct','-'):<9}{str(r.get('profit_factor','-'))[:6]:<8}"
                 f"{r.get('total_return_pct','-'):<8}{marker}")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
