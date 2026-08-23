# backtest.py — V4.1: Realistic Out-of-Sample Backtest with Fees, Slippage & Asymmetric Sizing

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", message="X has feature names, but RandomForestClassifier")

from train_model import (
    fetch_klines, _process_segment, FULL_FEATURES, MODEL_FILE, SYMBOLS,
    ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Cost & Execution Assumptions ──────────────────────────────────────
TAKER_FEE_PCT   = 0.0005   # 0.05% per side (Deribit USDC perpetual taker fee)
SLIPPAGE_PCT    = 0.0005   # 0.05% adverse slippage per fill
LOOKAHEAD_BARS  = 24       # 6 hours lookahead (24 × 15m bars)
MAX_LEVERAGE    = 10.0     # Maximum allowable leverage multiplier
META_MODEL_FILE = "meta_pipeline.pkl"
RESULTS_FILE    = "backtest_results.json"
HISTORY_FILE    = "backtest_history.json"


def simulate_symbol(symbol: str, df15: pd.DataFrame, pipeline: dict,
                    buy_thresh: float, sell_thresh: float,
                    meta_pipeline: dict = None, meta_threshold: float = None) -> list:
    """
    Evaluates ML signals across all bars and resolves exits against chronological TP1/SL touches.
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

        if sig == "NO_TRADE":
            continue
        if sig == "BUY" and conf < buy_thresh:
            continue
        if sig == "SELL" and conf < sell_thresh:
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

            # Conservative tie-breaker: prioritize SL if both touched within the same bar
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


def run_portfolio_simulation(all_trades: list, starting_equity: float,
                             risk_per_trade: float, max_open: int) -> dict:
    """
    Applies position caps chronologically, deducts exchange fees & slippage,
    and tracks compounded equity curves.
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
    by_symbol = {}

    for t in taken:
        sym = t["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}

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
        equity_curve.append(round(equity, 2))
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

        r_multiple = pnl_dollars / risk_dollars if risk_dollars > 0 else 0
        r_multiples.append(r_multiple)

        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"] += pnl_dollars

        if pnl_dollars > 0:
            wins += 1
            gross_win += pnl_dollars
            by_symbol[sym]["wins"] += 1
        else:
            losses += 1
            gross_loss += abs(pnl_dollars)
            by_symbol[sym]["losses"] += 1

    n = len(taken)
    win_rate = wins / n if n else 0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (float("inf") if gross_win > 0 else 0)
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
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "avg_r_multiple": round(avg_r, 3),
        "sharpe_per_trade": round(sharpe_per_trade, 3),
        "sharpe_annualized_approx": round(sharpe_annualized, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "starting_equity": starting_equity,
        "ending_equity": round(equity, 2),
        "total_return_pct": round((equity - starting_equity) / starting_equity * 100, 2),
        "days_covered": round(total_days, 1),
        "trades_per_year_approx": round(trades_per_year, 0),
        "by_symbol": by_symbol,
        "equity_curve": equity_curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Days of 15m candles to evaluate")
    parser.add_argument("--threshold", type=float, default=None, help="Global confidence threshold override")
    parser.add_argument("--symbols", nargs="+", default=None, help="Subset of symbols to backtest")
    parser.add_argument("--use-meta", action="store_true", help="Filter trades using meta_pipeline.pkl")
    parser.add_argument("--meta-threshold", type=float, default=None, help="Meta confidence threshold")
    parser.add_argument("--tag", type=str, default=None, help="Label for this run in history")
    parser.add_argument("--equity", type=float, default=10.0, help="Starting account equity in USDT")
    parser.add_argument("--risk-per-trade", type=float, default=0.03, help="Risk per trade fraction (e.g. 0.03 = 3%)")
    parser.add_argument("--max-open", type=int, default=2, help="Max simultaneous open positions")
    parser.add_argument("--allow-historical", action="store_true", help="Bypass training cutoff to evaluate recent holdouts")
    args = parser.parse_args()

    if not Path(MODEL_FILE).exists():
        log.error(f"Primary model {MODEL_FILE} not found. Aborting.")
        return

    pipeline = joblib.load(MODEL_FILE)
    
    # Asymmetric threshold resolution
    if args.threshold is not None:
        buy_thresh = args.threshold
        sell_thresh = args.threshold
    else:
        buy_thresh = float(pipeline.get("recommended_threshold_buy", 0.40))
        sell_thresh = float(pipeline.get("recommended_threshold_sell", 0.45))

    symbols = args.symbols if args.symbols else SYMBOLS

    meta_pipeline, meta_threshold = None, None
    if args.use_meta:
        if Path(META_MODEL_FILE).exists():
            meta_pipeline = joblib.load(META_MODEL_FILE)
            meta_threshold = args.meta_threshold if args.meta_threshold is not None else float(meta_pipeline.get("recommended_meta_threshold", 0.50))
            log.info(f"META FILTER ACTIVE — meta_threshold={meta_threshold}")
        else:
            log.warning(f"{META_MODEL_FILE} not found — proceeding with primary model only.")

    trained_at_str = pipeline.get("trained_at")
    trained_at_ms = None
    if trained_at_str and trained_at_str != "unknown":
        try:
            trained_at_ms = int(pd.Timestamp(trained_at_str).timestamp() * 1000)
            hours_since = (pd.Timestamp.now("UTC").timestamp() * 1000 - trained_at_ms) / (1000 * 3600)
            log.info(f"Model trained_at: {trained_at_str} ({hours_since:.1f}h ago)")
        except Exception as e:
            log.warning(f"Could not parse trained_at: {e}")

    if args.allow_historical:
        log.info("⚠️ --allow-historical flag active — bypassing post-training cutoff.")
        trained_at_ms = None

    log.info(f"Backtesting {len(symbols)} symbols, requested {args.days} days, buy_thresh={buy_thresh:.2f}, sell_thresh={sell_thresh:.2f}")

    candles = max(args.days, 5) * 96 + 500
    btc_df15 = fetch_klines("BTCUSDT", "15m", candles)
    if trained_at_ms is not None and not btc_df15.empty:
        btc_df15 = btc_df15[btc_df15["open_time"] > trained_at_ms].reset_index(drop=True)

    all_trades = []

    for symbol in symbols:
        try:
            df15 = fetch_klines(symbol, "15m", candles)
            df1h = fetch_klines(symbol, "1h", max(candles // 4, 150))
            df4h = fetch_klines(symbol, "4h", max(candles // 16, 250))

            if df15.empty or "open_time" not in df15.columns:
                continue

            if trained_at_ms is not None:
                before = len(df15)
                df15 = df15[df15["open_time"] > trained_at_ms].reset_index(drop=True)
                if before > 0 and len(df15) == 0:
                    continue

            if df15.empty or len(df15) < (100 if trained_at_ms is not None else 30):
                continue

            # Fully aligned 6-parameter call matching train_model.py
            processed = _process_segment(
                symbol=symbol,
                df15=df15,
                df1h=df1h,
                df4h=df4h,
                regime="backtest",
                btc_df15=btc_df15
            )

            if processed.empty:
                continue

            trades = simulate_symbol(
                symbol=symbol,
                df15=processed,
                pipeline=pipeline,
                buy_thresh=buy_thresh,
                sell_thresh=sell_thresh,
                meta_pipeline=meta_pipeline,
                meta_threshold=meta_threshold
            )
            log.info(f"  [{symbol}] {len(trades)} signals generated")
            all_trades.extend(trades)
        except Exception as e:
            log.warning(f"  [{symbol}] backtest error: {e}")

    if not all_trades:
        log.error("No trades generated — if you recently retrained, use --allow-historical to evaluate recent bars.")
        return

    results = run_portfolio_simulation(
        all_trades,
        starting_equity=args.equity,
        risk_per_trade=args.risk_per_trade,
        max_open=args.max_open
    )
    equity_curve = results.pop("equity_curve")
    by_symbol = results.pop("by_symbol")

    log.info("\n" + "=" * 60)
    log.info("BACKTEST RESULTS (realistic fees + slippage + position caps)")
    log.info("=" * 60)
    for k, v in results.items():
        log.info(f"  {k}: {v}")
    log.info("=" * 60)

    # Output per-symbol breakdown
    log.info("\nPER-SYMBOL PERFORMANCE:")
    for sym, stats in by_symbol.items():
        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        log.info(f"  {sym:<10} | Trades: {stats['trades']:<3} | WinRate: {wr:>5.1f}% | PnL: ${stats['pnl']:+6.2f}")

    with open(RESULTS_FILE, "w") as f:
        json.dump({
            **results,
            "by_symbol": by_symbol,
            "equity_curve": equity_curve,
            "buy_threshold": buy_thresh,
            "sell_threshold": sell_thresh,
            "days": args.days,
            "symbols": symbols
        }, f, indent=2)
    log.info(f"Saved: {RESULTS_FILE}")

    history = []
    if Path(HISTORY_FILE).exists():
        try:
            with open(HISTORY_FILE) as f:
                history = json.load(f)
        except Exception:
            history = []

    entry = {
        "run_at": pd.Timestamp.now("UTC").isoformat(),
        "tag": args.tag or ("historical-eval" if args.allow_historical else "oos-live"),
        "model_trained_at": pipeline.get("trained_at", "unknown"),
        "buy_threshold": buy_thresh,
        "sell_threshold": sell_thresh,
        "days": args.days,
        "n_symbols": len(symbols),
        **results,
    }
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-100:], f, indent=2)
    log.info(f"Appended to {HISTORY_FILE} ({len(history)} runs tracked total)")

    log.info("\n" + "=" * 78)
    log.info(f"{'LEADERBOARD (all tracked runs, ranked by Sharpe)':^78}")
    log.info("=" * 78)
    log.info(f"{'#':<3}{'run_at':<20}{'tag':<16}{'sharpe':<9}{'maxDD%':<9}{'PF':<8}{'ret%':<8}")
    ranked = sorted(history, key=lambda r: float(r.get("sharpe_annualized_approx") or -999), reverse=True)
    for i, r in enumerate(ranked[:15], 1):
        marker = " <-- THIS RUN" if r is entry else ""
        log.info(f"{i:<3}{r['run_at'][:16]:<20}{str(r.get('tag') or '-'):<16}"
                 f"{r.get('sharpe_annualized_approx','-'):<9}"
                 f"{r.get('max_drawdown_pct','-'):<9}{str(r.get('profit_factor','-'))[:6]:<8}"
                 f"{r.get('total_return_pct','-'):<8}{marker}")
    log.info("=" * 78)


if __name__ == "__main__":
    main()
