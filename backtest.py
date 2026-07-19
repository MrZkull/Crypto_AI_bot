# backtest.py — P0: realistic out-of-sample backtest with fees, slippage, and
# portfolio-level position caps. This is the tool everything else (meta-labeling,
# vol-targeted sizing, regime features) gets validated against — without this,
# "is the model better now" was only ever answered by classification metrics
# (precision/recall), never by "would this have actually made money."
#
# IMPORTANT — what this does NOT model, on purpose, to keep v1 shippable:
#   - No partial TP1/TP2 split or trailing stop (trade_executor.py does both live).
#     Backtest uses a single target (TP1) as full exit. This UNDERSTATES what a
#     winning trade captures (TP2 upside is ignored) but keeps the simulation
#     honest and simple. Add TP1/TP2 partial modeling in v2 once v1 is trusted.
#   - No correlation filter (check_correlation() in trade_executor.py) — only the
#     flat MAX_OPEN_TRADES cap is modeled. Real live trading will be somewhat more
#     conservative than this backtest suggests.
#   - Funding rate carry cost is not modeled (we don't have reliable historical
#     funding data — see the geo-block saga earlier in this project).
#
# Usage:
#   python backtest.py                          # last ~20 days per symbol, default threshold
#   python backtest.py --days 30 --threshold 0.45
#   python backtest.py --symbols BTCUSDT ETHUSDT

import argparse, json, logging, time
import numpy as np
import pandas as pd
import joblib

from train_model import (
    fetch_klines, _process_segment, FULL_FEATURES, MODEL_FILE, SYMBOLS,
    ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Realistic cost assumptions ──────────────────────────────────────────
TAKER_FEE_PCT   = 0.0005   # 0.05% per side — Deribit USDC perpetual taker fee ballpark
SLIPPAGE_PCT    = 0.0005   # 0.05% adverse slippage per side, beyond fee — conservative
LOOKAHEAD_BARS  = 24       # matches make_targets()'s label lookahead (6h on 15m candles)
MAX_OPEN_TRADES = 10       # matches trade_executor.py's testnet cap
RISK_PER_TRADE  = 0.01     # 1% of current equity risked per trade — matches config.py
MAX_LEVERAGE    = 10.0     # NEW: hard cap on notional as a multiple of equity — prevents
                            # tight-stop trades from implying unrealistic leverage. 10x is
                            # a reasonable ceiling for a testnet perpetual account; tune
                            # down if you want to be more conservative.
STARTING_EQUITY = 100_000.0
META_MODEL_FILE = "meta_pipeline.pkl"  # NEW — see train_meta_model.py


def simulate_symbol(symbol: str, df15: pd.DataFrame, pipeline: dict, threshold: float,
                     meta_pipeline: dict = None, meta_threshold: float = None) -> list:
    """Walk every bar of df15, generate a signal via the trained pipeline exactly
    like generate_signal() does live, and if one fires, resolve the outcome by
    checking forward bars for TP1/SL — first one touched wins. Returns a list of
    trade dicts (not yet filtered by portfolio position caps — that happens after
    merging all symbols chronologically).

    NEW: if meta_pipeline is provided, every primary signal is additionally passed
    through the meta-model — a signal only survives if BOTH the primary confidence
    clears `threshold` AND the meta-model's P(call is correct) clears
    `meta_threshold`. This lets us directly compare "primary alone" vs "primary +
    meta filter" on the exact same realistic-cost, position-capped simulation."""
    af       = pipeline["all_features"]
    selector = pipeline["selector"]
    ensemble = pipeline["ensemble"]
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
    highs  = df15["high"].values
    lows   = df15["low"].values
    closes = df15["close"].values
    opens_t = df15["open_time"].values
    atrs   = df15["atr"].values if "atr" in df15.columns else None

    for i in range(n - LOOKAHEAD_BARS):
        sig  = label_map[int(preds[i])]
        conf = float(max(probas[i]))
        if sig == "NO_TRADE" or conf < threshold:
            continue
        if meta_probas is not None:
            if meta_probas[i] < meta_threshold:
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
            # Conservative: if both could have been hit on the same bar, assume the
            # worse outcome (SL) happened first — matches "don't flatter yourself"
            # backtesting convention.
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


def run_portfolio_simulation(all_trades: list) -> dict:
    """Apply MAX_OPEN_TRADES cap chronologically, apply fees/slippage, compound
    equity risking RISK_PER_TRADE of CURRENT equity per trade (not fixed $), and
    compute the metrics that actually matter: Sharpe, max drawdown, profit factor —
    not just win rate, which is misleading on its own with an asymmetric R:R."""
    all_trades.sort(key=lambda t: t["entry_time"])

    taken = []
    open_intervals = []  # list of (entry_time, exit_time) for currently-tracked opens

    for t in all_trades:
        open_intervals = [iv for iv in open_intervals if iv[1] > t["entry_time"]]
        if len(open_intervals) >= MAX_OPEN_TRADES:
            continue  # matches live bot's "MAX TRADES — skip"
        open_intervals.append((t["entry_time"], t["exit_time"]))
        taken.append(t)

    taken.sort(key=lambda t: t["exit_time"])

    equity = STARTING_EQUITY
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
        risk_dollars_target = equity * RISK_PER_TRADE
        raw_notional  = risk_dollars_target / stop_dist_pct if stop_dist_pct > 0 else 0

        # FIX: cap implied leverage. Without this, a tight-stop trade (small
        # stop_dist_pct) can imply absurd notional — e.g. a 0.1% stop with 1%
        # target risk implies 10x leverage; a 0.02% stop implies 50x. Nothing
        # was capping this, so a handful of tight-stop trades produced huge PnL
        # swings that then compounded exponentially (this is exactly how a
        # 19.5-day run turned $100k into $4.66M — not a real edge, a sizing bug).
        # MAX_LEVERAGE mirrors the reality that real exchanges enforce hard
        # position/margin ceilings (see: Deribit's non_pme_max_future_position_size
        # limit we hit live in trade_executor.py).
        max_notional = equity * MAX_LEVERAGE
        notional = min(raw_notional, max_notional)
        if raw_notional > max_notional:
            capped_count += 1
        # Actual dollar risk taken is whatever the (possibly capped) notional
        # implies — NOT the target risk_dollars, since capping means the account
        # is risking LESS than 1% on these trades, same as a real leveraged
        # account would if it hit its own margin ceiling.
        risk_dollars = notional * stop_dist_pct

        raw_pnl_pct = ((exit_fill - entry_fill) / entry_fill) if t["signal"] == "BUY" else ((entry_fill - exit_fill) / entry_fill)
        pnl_dollars = notional * raw_pnl_pct
        fees        = notional * TAKER_FEE_PCT * 2  # entry + exit
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
    # Rough annualization: assumes ~trades_per_year based on observed frequency
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
        "starting_equity": STARTING_EQUITY,
        "ending_equity": round(equity, 2),
        "total_return_pct": round((equity - STARTING_EQUITY) / STARTING_EQUITY * 100, 2),
        "days_covered": round(total_days, 1),
        "trades_per_year_approx": round(trades_per_year, 0),
        "equity_curve": equity_curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20, help="How many recent days per symbol to backtest")
    parser.add_argument("--threshold", type=float, default=None, help="Primary confidence threshold (default: pipeline's recommended_threshold)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Subset of symbols (default: all)")
    parser.add_argument("--use-meta", action="store_true", help="Apply meta_pipeline.pkl as an additional filter — compare against a run WITHOUT this flag on the same days/symbols")
    parser.add_argument("--meta-threshold", type=float, default=None, help="Meta confidence threshold (default: meta_pipeline's recommended_meta_threshold)")
    parser.add_argument("--tag", type=str, default=None, help="Label for this run in backtest_history.json (e.g. 'post-orphan-fix', 'meta-v1') — makes the leaderboard readable")
    args = parser.parse_args()

    pipeline = joblib.load(MODEL_FILE)
    threshold = args.threshold if args.threshold is not None else pipeline.get("recommended_threshold", 0.45)
    symbols = args.symbols if args.symbols else SYMBOLS

    meta_pipeline, meta_threshold = None, None
    if args.use_meta:
        meta_pipeline = joblib.load(META_MODEL_FILE)
        meta_threshold = args.meta_threshold if args.meta_threshold is not None else meta_pipeline.get("recommended_meta_threshold", 0.5)
        log.info(f"META FILTER ACTIVE — meta_threshold={meta_threshold} "
                 f"(meta model trained_at={meta_pipeline.get('trained_at', 'unknown')})")

    # FIX: previously this only LOGGED a reminder to make sure the backtest window
    # was after training — never enforced it. train_model.py's "recent" segment
    # pulls the latest ~52 days of candles AS OF WHENEVER TRAINING RAN, so a
    # same-day (or even same-week) backtest using "last N days as of now" mostly
    # overlaps data the model was already fit on. That's exactly how a 20-day
    # backtest showed a fake 65.3% win rate and 2,546% return — not a sizing bug,
    # the model was graded on data it had already partially seen. Now this is a
    # hard cutoff: only candles with open_time strictly AFTER trained_at are used.
    trained_at_str = pipeline.get("trained_at")
    trained_at_ms = None
    if trained_at_str and trained_at_str != "unknown":
        try:
            trained_at_ms = int(pd.Timestamp(trained_at_str).timestamp() * 1000)
        except Exception as e:
            log.warning(f"Could not parse trained_at ({trained_at_str}): {e} — cannot enforce leak-free cutoff!")

    hours_since_training = None
    if trained_at_ms is not None:
        hours_since_training = (pd.Timestamp.now("UTC").timestamp() * 1000 - trained_at_ms) / (1000 * 3600)
        log.info(f"Model trained_at: {trained_at_str} ({hours_since_training:.1f}h ago) — "
                 f"ONLY candles after this timestamp will be used, regardless of --days requested")
        if hours_since_training < 24:
            log.warning(f"⚠️ Only {hours_since_training:.1f}h have passed since training — there is "
                        f"very little genuinely fresh data to test on yet. Results below will be "
                        f"based on a small, statistically weak sample. Consider waiting longer after "
                        f"training before trusting a backtest.")
    else:
        log.error("🚨 No trained_at timestamp on this model — CANNOT enforce leak-free cutoff. "
                  "Results may be contaminated by training-window overlap. Proceed with caution.")

    log.info(f"Backtesting {len(symbols)} symbols, requested {args.days} days, threshold={threshold}")

    # Fetch generously (requested days + enough padding to survive the post-cutoff
    # filter) since we don't know in advance how much gets filtered out.
    candles = max(args.days, 5) * 96 + 500
    btc_df15 = fetch_klines("BTCUSDT", "15m", candles)
    if trained_at_ms is not None:
        btc_df15 = btc_df15[btc_df15["open_time"] > trained_at_ms].reset_index(drop=True)
    all_trades = []

    for symbol in symbols:
        try:
            df15 = fetch_klines(symbol, "15m", candles)
            df1h = fetch_klines(symbol, "1h", candles // 4)
            df4h = fetch_klines(symbol, "4h", candles // 16)

            if trained_at_ms is not None:
                before = len(df15)
                df15 = df15[df15["open_time"] > trained_at_ms].reset_index(drop=True)
                if before > 0 and len(df15) == 0:
                    log.warning(f"  [{symbol}] ALL fetched candles are pre-training — "
                                f"no genuinely fresh data available yet, skipping")
                    continue

            if df15.empty or len(df15) < 100:
                log.warning(f"  [{symbol}] insufficient POST-TRAINING data ({len(df15)} candles) — skipping "
                            f"(this is expected if training just ran recently — wait longer, don't lower "
                            f"the minimum to force a result)")
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
        log.error("No trades generated at all — check threshold/data window")
        return

    results = run_portfolio_simulation(all_trades)
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

    # ── Persistent history — every run gets appended, never overwritten, so you
    # can track which training/config actually held up best over time instead of
    # only ever seeing the most recent run's numbers. ──
    HISTORY_FILE = "backtest_history.json"
    try:
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    except Exception:
        history = []

    entry = {
        "run_at": pd.Timestamp.now("UTC").isoformat(),
        "tag": args.tag,
        "model_trained_at": pipeline.get("trained_at", "unknown"),
        "hours_since_training": round(hours_since_training, 1) if hours_since_training is not None else None,
        "meta_used": bool(args.use_meta),
        "meta_trained_at": meta_pipeline.get("trained_at", "unknown") if meta_pipeline else None,
        "primary_threshold": threshold,
        "meta_threshold": meta_threshold,
        "days": args.days,
        "n_symbols": len(symbols),
        **results,
    }
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Appended to {HISTORY_FILE} ({len(history)} runs tracked total)")

    # ── Leaderboard — rank every tracked run by risk-adjusted return (Sharpe),
    # not raw total return, since a higher-return run with a much worse drawdown
    # isn't actually "better" for a live account. ──
    log.info("\n" + "=" * 78)
    log.info(f"{'LEADERBOARD (all tracked runs, ranked by Sharpe)':^78}")
    log.info("=" * 78)
    log.info(f"{'#':<3}{'run_at':<20}{'tag':<16}{'meta':<6}{'sharpe':<9}{'maxDD%':<9}{'PF':<8}{'ret%':<8}")
    ranked = sorted(history, key=lambda r: r.get("sharpe_annualized_approx", -999), reverse=True)
    for i, r in enumerate(ranked[:15], 1):
        marker = " <-- THIS RUN" if r is entry else ""
        log.info(f"{i:<3}{r['run_at'][:16]:<20}{str(r.get('tag') or '-'):<16}"
                 f"{'Y' if r['meta_used'] else 'N':<6}{r.get('sharpe_annualized_approx','-'):<9}"
                 f"{r.get('max_drawdown_pct','-'):<9}{str(r.get('profit_factor','-'))[:6]:<8}"
                 f"{r.get('total_return_pct','-'):<8}{marker}")
    log.info("=" * 78)



if __name__ == "__main__":
    main()
        
