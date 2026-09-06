# trade_executor.py — V4.1: Native EV Authority, Post-Fill Guard,
# Prediction Ledger, Ensemble Disagreement, and Adaptation Overrides

import os, json, time, logging, requests, joblib, base64, math
import pandas as pd, numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import config
from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, MODEL_FILE, LOG_FILE, get_tier,
    TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM, TIMEFRAME_TREND, LIVE_LIMIT
)
from deribit_client import DeribitClient, TRADEABLE_SYMBOLS
from feature_engineering import add_indicators
from smart_scheduler import (
    should_scan, get_mode_thresholds, get_effective_risk, check_correlation,
    check_btc_momentum, check_fear_and_greed
)
from whale_tracker import get_exchange_netflow 

TRADES_FILE            = "trades.json"
HISTORY_FILE           = "trade_history.json"
SIGNALS_FILE           = "signals.json"
BALANCE_FILE           = "balance.json"
LOCK_FILE              = "scan_lock.json"
SCAN_STATUS_FILE       = "scan_status.json"
STALE_LOCK_MINUTES     = 20
EXECUTION_SANITY_FLOOR = 35.0         
ORPHAN_TRACKER_FILE    = "orphan_candidates.json"
ORPHAN_CONFIRM_SECONDS = 600          

COOLDOWN_FILE        = "cooldown.json"
RELIABILITY_FILE     = "reliability.json"
COOLDOWN_HOURS       = 2
GHOST_STRIKE_LIMIT   = 3
MAX_DAILY_TRADES     = 20
FUNDING_WARN_PCT     = 0.05
FUNDING_SKIP_PCT     = 0.10
ENTRY_MAX_SPREAD_PCT = 0.005

CONSECUTIVE_LOSS_BENCH_THRESHOLD = 3      
PROBATION_WIN_GOAL               = 3      
PROBATION_MIN_WINS_FOR_TIME_EXIT = 2      
PROBATION_CONSECUTIVE_LOSS_RESET = 3      
PROBATION_CONFIDENCE_OFFSET      = 10.0   
BENCH_MAX_COOLDOWN_DAYS          = 7      

ROLLING_WINDOW_TRADES            = 6
ROLLING_MIN_WIN_RATE             = 0.40   
ROLLING_MAX_NET_LOSS             = 0.0    

# ── NEW: Prediction Ledger + Ensemble Disagreement ─────────────────────
PREDICTIONS_FILE       = "predictions.json"
OVERRIDES_FILE         = "adaptation_overrides.json"
MAX_PREDICTIONS_KEPT   = 10000

# PLACEHOLDER — not yet validated. Once prediction_auditor.py has a few
# hundred audited records, replace this with whatever threshold actually
# separates correct/incorrect calls in the bucketed dossier analysis.
HIGH_DISAGREEMENT_THRESHOLD = 0.15

GH_TOKEN  = os.getenv("GH_PAT_TOKEN", "")
GH_REPO   = os.getenv("GITHUB_REPO", "MrZkull/Crypto_AI_bot")
GH_BRANCH = os.getenv("GITHUB_BRANCH", "main")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
log = logging.getLogger(__name__)


def load_json(path, default):
    try:
        for p in [Path(path), Path("data") / Path(path).name]:
            if p.exists():
                with open(p) as f: return json.load(f)
    except Exception: pass
    return default

def save_json(path, data):
    for dest in [Path(path), Path("data") / Path(path).name]:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(dest) + ".tmp"
            with open(tmp, "w") as f: json.dump(data, f, indent=2, default=str)
            os.replace(tmp, str(dest))
        except Exception as e: log.error(f"save_json {dest}: {e}")

load_trades  = lambda: load_json(TRADES_FILE,  {})
save_trades  = lambda d: save_json(TRADES_FILE, d)
load_history = lambda: load_history_json()

def load_history_json():
    return load_json(HISTORY_FILE, [])

def append_history(rec):
    h = load_history(); h.append(rec); save_json(HISTORY_FILE, h)

def save_signal(sig):
    s = load_json(SIGNALS_FILE, [])
    s.append({**sig, "generated_at": datetime.now(timezone.utc).isoformat()})
    save_json(SIGNALS_FILE, s[-500:])

def load_cooldown() -> dict: return load_json(COOLDOWN_FILE, {})
def save_cooldown(d: dict): save_json(COOLDOWN_FILE, d)


# ── NEW: Adaptation overrides (approved via dashboard, never auto-applied) ──
def get_symbol_overrides(symbol: str) -> dict:
    overrides = load_json(OVERRIDES_FILE, {})
    return overrides.get(symbol, {}) if isinstance(overrides, dict) else {}


# ── NEW: Ensemble disagreement — variance across base estimators' probas ──
def compute_ensemble_disagreement(pipeline, x_selected_row: np.ndarray) -> dict:
    """Measures decision-boundary ambiguity: do XGB/RF/GB actually agree,
    independent of whether the input itself is novel. Cheap, no retrain needed."""
    calibrated = pipeline["ensemble"]
    try:
        base_voting = calibrated.calibrated_classifiers_[0].estimator.estimator
        named = base_voting.named_estimators_
        if not named:
            raise ValueError("named_estimators_ is empty — unexpected VotingClassifier state")
        probas = np.array([est.predict_proba(x_selected_row.reshape(1, -1))[0]
                            for _, est in named.items()])
        disagreement = float(np.mean(np.std(probas, axis=0)))
        return {"ensemble_disagreement": round(disagreement, 4),
                "high_disagreement": disagreement > HIGH_DISAGREEMENT_THRESHOLD}
    except Exception as e:
        if not getattr(compute_ensemble_disagreement, "_warned", False):
            log.warning(f"⚠️ ensemble disagreement unwrap failed — will be null for ALL predictions this run: {e}")
            compute_ensemble_disagreement._warned = True
        return {"ensemble_disagreement": None, "high_disagreement": False}


# ── NEW: Prediction ledger — logs every model opinion, executed or not ──
def _pred_record(symbol, sig, conf, pipeline, row, disagreement, reject_reason=None, pred_id=None):
    entry = float(row.get("close", 0) or 0)
    atr   = float(row.get("atr", 0) or 0)
    stop  = (entry - atr*ATR_STOP_MULT) if sig == "BUY" else (entry + atr*ATR_STOP_MULT) if sig == "SELL" else None
    tp1   = (entry + atr*ATR_TARGET1_MULT) if sig == "BUY" else (entry - atr*ATR_TARGET1_MULT) if sig == "SELL" else None
    return {
        "pred_id":               pred_id or f"{symbol}_{int(time.time()*1000)}",
        "symbol":                symbol,
        "predicted_signal":      sig,
        "confidence":            conf,
        "entry_ref":             entry,
        "atr_ref":               atr,
        "predicted_stop":        round(stop, 6) if stop else None,
        "predicted_tp1":         round(tp1, 6) if tp1 else None,
        "rsi_15m":               float(row.get("rsi", 50)),
        "adx_15m":               float(row.get("adx", 0)),
        "ensemble_disagreement": disagreement.get("ensemble_disagreement"),
        "high_disagreement":     disagreement.get("high_disagreement"),
        "ood_distance":          None,       # populated once Mahalanobis/LedoitWolf gate ships post-retrain
        "ood_tier":              "PENDING",
        "model_version":         pipeline.get("trained_at"),
        "was_executed":          False,
        "reject_reason":         reject_reason,
        "lookahead_bars":        24,          # must match make_targets() exactly
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "audited":               False,
    }


def save_prediction(rec: dict):
    preds = load_json(PREDICTIONS_FILE, [])
    preds.append(rec)
    if len(preds) > MAX_PREDICTIONS_KEPT:
        unaudited = [p for p in preds if not p.get("audited")]
        audited   = [p for p in preds if p.get("audited")]
        room = max(0, MAX_PREDICTIONS_KEPT - len(unaudited))
        preds = audited[-room:] + unaudited if room else unaudited
    save_json(PREDICTIONS_FILE, preds)


def mark_prediction_executed(pred_id: str):
    preds = load_json(PREDICTIONS_FILE, [])
    for p in preds:
        if p.get("pred_id") == pred_id:
            p["was_executed"] = True
            break
    save_json(PREDICTIONS_FILE, preds)


# ── NEW: Trade snapshot — market physics at signal-generation time ──────
def build_trade_snapshot(row, r4h, e20_4h, e50_4h, rsi_4h, fng_data, whale_flow, btc_momentum):
    return {
        "entry_atr":      float(row.get("atr", 0)),
        "rsi_15m":        float(row.get("rsi", 50)),
        "adx_15m":        float(row.get("adx", 0)),
        "rsi_4h":         rsi_4h,
        "ema_4h_bullish": bool(e20_4h > e50_4h),
        "fng_value":      fng_data.get("value") if fng_data else None,
        "whale_bias":     whale_flow.get("bias") if whale_flow else None,
        "btc_bias":       btc_momentum.get("bias") if btc_momentum else None,
        "btc_corr_20":    float(row.get("btc_corr_20", 0)),
        "btc_beta_20":    float(row.get("btc_beta_20", 1.0)),
    }


def gh_fetch(filename: str):
    if not GH_TOKEN or not GH_REPO:
        return None
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    for path in [filename, f"data/{filename}"]:
        try:
            url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                raw_content = base64.b64decode(r.json()["content"]).decode("utf-8")
                return json.loads(raw_content) if filename.endswith(".json") else raw_content
        except Exception as e:
            log.debug(f"gh_fetch failed for {path}: {e}")
    return None

def load_reliability() -> dict:
    data = load_json(RELIABILITY_FILE, None)
    if data is not None and isinstance(data, dict):
        return data
    remote_data = gh_fetch(RELIABILITY_FILE)
    if remote_data and isinstance(remote_data, dict):
        save_reliability(remote_data)
        return remote_data
    return {}

def save_reliability(d: dict): save_json(RELIABILITY_FILE, d)


def _acquire_scan_lock() -> bool:
    lock = load_json(LOCK_FILE, {})
    if lock.get("locked_at"):
        try:
            age_min = (datetime.now(timezone.utc) - datetime.fromisoformat(
                lock["locked_at"].replace("Z", ""))).total_seconds() / 60
        except Exception:
            age_min = 0
        if age_min < STALE_LOCK_MINUTES:
            log.warning(f"  🔒 Scan already in progress (started {age_min:.1f}m ago) — skipping this run")
            return False
        log.warning(f"  🔓 Stale lock from {age_min:.1f}m ago — clearing and proceeding")

    save_json(LOCK_FILE, {"locked_at": datetime.now(timezone.utc).isoformat()})
    return True

def _release_scan_lock():
    save_json(LOCK_FILE, {"locked_at": None})

def _add_cooldown(symbol: str, reason: str):
    cd = load_cooldown()
    cd[symbol] = {
        "blocked_until": (datetime.now(timezone.utc).timestamp() + COOLDOWN_HOURS * 3600),
        "reason": reason,
        "blocked_at": datetime.now(timezone.utc).isoformat(),
    }
    save_cooldown(cd)
    log.info(f"  🔒 {symbol}: cooldown {COOLDOWN_HOURS}h — {reason}")

def _is_on_cooldown(symbol: str) -> bool:
    cd = load_cooldown()
    if symbol not in cd: return False
    if datetime.now(timezone.utc).timestamp() < cd[symbol]["blocked_until"]:
        remaining = (cd[symbol]["blocked_until"] - datetime.now(timezone.utc).timestamp()) / 3600
        log.info(f"  ⏳ {symbol}: cooldown {remaining:.1f}h remaining — skip")
        return True
    cd.pop(symbol); save_cooldown(cd)
    return False


def _record_ghost(symbol: str):
    rel = load_reliability()
    if symbol not in rel:
        rel[symbol] = {
            "normal_consecutive_losses": 0, "probation_wins": 0,
            "probation_consecutive_losses": 0, "is_benched": False,
            "benched_at": 0, "wins": 0, "losses": 0, "ghosts": 0
        }
    rel[symbol]["ghosts"] = rel[symbol].get("ghosts", 0) + 1
    log.warning(f"  👻 {symbol}: Ghost trade recorded (Total ghosts: {rel[symbol]['ghosts']})")
    save_reliability(rel)

def _check_rolling_performance(symbol: str) -> bool:
    hist = load_history()
    recent = [
        h for h in hist
        if h.get("symbol") == symbol
        and h.get("close_reason") not in ("Ghost — PnL unrecoverable", "Ghost — broken record")
    ]
    recent = sorted(recent, key=lambda h: h.get("closed_at", ""))[-ROLLING_WINDOW_TRADES:]

    if len(recent) < ROLLING_WINDOW_TRADES:
        return False

    wins = sum(1 for h in recent if float(h.get("pnl", 0)) > 0)
    net_pnl = sum(float(h.get("pnl", 0)) for h in recent)
    win_rate = wins / len(recent)

    if win_rate < ROLLING_MIN_WIN_RATE and net_pnl < ROLLING_MAX_NET_LOSS:
        log.warning(
            f"  🚫 {symbol}: Rolling {len(recent)}-trade win rate={win_rate:.0%} | "
            f"Net PnL=${net_pnl:+.2f} — BENCHING ON PROBATION (Rolling Filter)"
        )
        rel = load_reliability()
        if symbol not in rel:
            rel[symbol] = {
                "normal_consecutive_losses": 0, "probation_wins": 0,
                "probation_consecutive_losses": 0, "is_benched": False,
                "benched_at": 0, "wins": 0, "losses": 0, "ghosts": 0
            }
        rel[symbol]["is_benched"] = True
        rel[symbol]["benched_at"] = time.time()
        save_reliability(rel)
        return True
    return False


# ── MODIFIED: now applies approved adaptation override on every return path ──
def get_required_confidence(symbol: str, current_baseline_conf: float = 35.0) -> float:
    sym_overrides = get_symbol_overrides(symbol)
    extra = float(sym_overrides.get("probation_offset_override", {}).get("extra_conf_pct", 0.0))

    rel = load_reliability()
    if symbol not in rel:
        return current_baseline_conf + extra

    data = rel[symbol]
    now = time.time()

    if data.get("is_benched", False):
        benched_at = data.get("benched_at", 0)
        p_wins = data.get("probation_wins", 0)
        p_losses = data.get("probation_consecutive_losses", 0)
        days_elapsed = (now - benched_at) / 86400 if benched_at > 0 else 0

        if days_elapsed >= BENCH_MAX_COOLDOWN_DAYS:
            if p_wins >= PROBATION_MIN_WINS_FOR_TIME_EXIT:
                data["is_benched"] = False
                data["benched_at"] = 0
                data["probation_wins"] = 0
                data["probation_consecutive_losses"] = 0
                save_reliability(rel)
                log.info(f"  ⏱️ {symbol}: 7 days elapsed with {p_wins} wins. Reinstated to baseline ({current_baseline_conf:.1f}%).")
                return current_baseline_conf + extra
            else:
                data["benched_at"] = now
                save_reliability(rel)
                log.warning(f"  🔒 {symbol}: 7 days elapsed but only {p_wins} win(s) logged. Probation timer extended.")

        probation_required_conf = current_baseline_conf + PROBATION_CONFIDENCE_OFFSET + extra
        log.info(
            f"  🔒 {symbol} ON PROBATION: Requiring {probation_required_conf:.1f}% Conf "
            f"(Base: {current_baseline_conf:.1f}% + {PROBATION_CONFIDENCE_OFFSET}%"
            f"{f' + override {extra}%' if extra else ''} | "
            f"Wins: {p_wins}/{PROBATION_WIN_GOAL} | Probation Losses: {p_losses}/{PROBATION_CONSECUTIVE_LOSS_RESET})"
        )
        return probation_required_conf

    return current_baseline_conf + extra


def _is_unreliable(symbol: str, signal_conf: float = 0.0, current_baseline_conf: float = 35.0) -> bool:
    rel = load_reliability()
    if symbol not in rel:
        return False

    data = rel[symbol]
    ghosts = data.get("ghosts", 0)
    wins = data.get("wins", 0)

    if ghosts >= GHOST_STRIKE_LIMIT and ghosts > wins * 2:
        log.warning(f"  🚫 {symbol}: unreliable ({ghosts} ghosts, {wins} wins) — skip")
        return True

    required_conf = get_required_confidence(symbol, current_baseline_conf)
    if signal_conf > 0 and signal_conf < required_conf:
        log.warning(f"  🚫 {symbol}: Signal confidence ({signal_conf:.1f}%) < required probation threshold ({required_conf:.1f}%) — skip")
        return True

    return False

def _record_outcome(symbol: str, won: bool):
    rel = load_reliability()
    if symbol not in rel:
        rel[symbol] = {
            "normal_consecutive_losses": 0, "probation_wins": 0,
            "probation_consecutive_losses": 0, "is_benched": False,
            "benched_at": 0, "wins": 0, "losses": 0, "ghosts": 0
        }

    data = rel[symbol]
    now = time.time()

    if won:
        data["wins"] = data.get("wins", 0) + 1
    else:
        data["losses"] = data.get("losses", 0) + 1

    if data.get("is_benched", False):
        if won:
            data["probation_wins"] = data.get("probation_wins", 0) + 1
            data["probation_consecutive_losses"] = 0
            current_wins = data["probation_wins"]
            log.info(f"  🎯 {symbol} PROBATION WIN! Progress: {current_wins}/{PROBATION_WIN_GOAL} wins.")

            if current_wins >= PROBATION_WIN_GOAL:
                data["is_benched"] = False
                data["benched_at"] = 0
                data["probation_wins"] = 0
                data["probation_consecutive_losses"] = 0
                data["normal_consecutive_losses"] = 0
                log.info(f"  🎉 {symbol}: Achieved {PROBATION_WIN_GOAL} probation wins! Fully reinstated to Normal Mode.")
        else:
            data["probation_consecutive_losses"] = data.get("probation_consecutive_losses", 0) + 1
            p_losses = data["probation_consecutive_losses"]
            current_wins = data.get("probation_wins", 0)

            if p_losses >= PROBATION_CONSECUTIVE_LOSS_RESET:
                data["probation_wins"] = 0
                data["probation_consecutive_losses"] = 0
                data["benched_at"] = now
                log.warning(f"  🚨 {symbol}: 3 consecutive losses in probation! Win progress wiped to 0 and timer restarted.")
            else:
                log.warning(f"  ⚠️ {symbol} Probation Loss ({p_losses}/{PROBATION_CONSECUTIVE_LOSS_RESET} consecutive). Wins retained: {current_wins}/{PROBATION_WIN_GOAL}.")
    else:
        if won:
            data["normal_consecutive_losses"] = 0
            log.info(f"  ✅ {symbol} Normal Trade Win. Loss streak cleared.")
        else:
            data["normal_consecutive_losses"] = data.get("normal_consecutive_losses", 0) + 1
            n_losses = data["normal_consecutive_losses"]
            log.warning(f"  ⚠️ {symbol} Normal Trade Loss ({n_losses}/{CONSECUTIVE_LOSS_BENCH_THRESHOLD} consecutive).")

            if n_losses >= CONSECUTIVE_LOSS_BENCH_THRESHOLD:
                data["is_benched"] = True
                data["benched_at"] = now
                data["normal_consecutive_losses"] = 0
                data["probation_wins"] = 0
                data["probation_consecutive_losses"] = 0
                log.warning(f"  🚫 {symbol}: {CONSECUTIVE_LOSS_BENCH_THRESHOLD} consecutive losses! Benched on Probation (+{PROBATION_CONFIDENCE_OFFSET}% Conf Required).")

    save_reliability(rel)
    _check_rolling_performance(symbol)

record_trade_outcome = _record_outcome
record_ghost         = _record_ghost


def _get_daily_trade_count() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hist  = load_history()
    return sum(1 for h in hist if h.get("opened_at", "")[:10] == today and h.get("close_reason") != "Ghost — PnL unrecoverable")

def _check_funding_rate(deribit, symbol: str, signal: str) -> float:
    try:
        rate = deribit.get_funding_rate(symbol)
        rate_pct = abs(rate) * 100
        if signal == "BUY" and rate > 0:
            if rate_pct >= FUNDING_SKIP_PCT:
                log.warning(f"  💸 {symbol}: funding {rate_pct:.3f}% (8h) — skip LONG (>{FUNDING_SKIP_PCT}%)")
                return 0.0
            elif rate_pct >= FUNDING_WARN_PCT:
                log.warning(f"  ⚠️ {symbol}: funding {rate_pct:.3f}% (8h) — high for LONG (50% size penalty)")
                return 0.5
        if signal == "SELL" and rate < 0:
            if rate_pct >= FUNDING_SKIP_PCT:
                log.warning(f"  💸 {symbol}: funding -{rate_pct:.3f}% (8h) — skip SHORT (>{FUNDING_SKIP_PCT}%)")
                return 0.0
            elif rate_pct >= FUNDING_WARN_PCT:
                log.warning(f"  ⚠️ {symbol}: funding -{rate_pct:.3f}% (8h) — high for SHORT (50% size penalty)")
                return 0.5
    except Exception as e: log.debug(f"  funding check {symbol}: {e}")
    return 1.0

def _wait_for_position(deribit, symbol: str, side: str, entry_oid: str, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    step     = 0.5
    while time.time() < deadline:
        time.sleep(step)
        try:
            size = deribit.get_position_size(symbol)
            if abs(size) > 0:
                log.info(f"  ✅ Position confirmed: {symbol} size={size}")
                return True
        except Exception: pass
        if entry_oid:
            try:
                chk = deribit.get_order(entry_oid)
                if chk.get("order_state", "") == "filled": return True
            except Exception: pass
        step = min(step * 1.5, 2.0)
    log.warning(f"  ⚠️ {symbol}: position not confirmed after {timeout}s")
    return False

def _place_tp_with_fallback(deribit, symbol: str, side: str, qty, price: float, label: str, trade: dict, key: str, dec: int) -> str:
    try:
        actual_pos = abs(deribit.get_position_size(symbol))
        if actual_pos > 0:
            qty = min(qty, actual_pos)
            qty = deribit.round_amount(symbol, qty)
    except Exception: pass

    if qty <= 0:
        log.warning(f"  {label} {symbol}: qty=0 after position check — skip")
        return ""

    try:
        res = deribit.place_limit_order(symbol, side, qty, price, use_reduce_only=False)
        o   = res.get("order", res)
        oid = str(o.get("order_id", ""))
        if oid:
            log.info(f"  🛠 Re-placed {label} {symbol} @ {price:.{dec}f}  id:{oid}")
            return oid
    except Exception as e:
        log.warning(f"  {label} re-place {symbol}: {e}")
    return ""

def _cancel_all_open_orders_for_symbol(deribit: DeribitClient, symbol: str):
    try:
        orders = deribit.get_open_orders(symbol)
        if orders:
            log.info(f"  🧹 Found {len(orders)} active open order(s) on Deribit for {symbol} — cancelling all...")
            for o in orders:
                oid = str(o.get("order_id", ""))
                if oid:
                    try:
                        deribit.cancel_order(oid)
                        log.info(f"    ✓ Cancelled exchange order {oid}")
                    except Exception as ce:
                        log.debug(f"    Failed to cancel {oid}: {ce}")
    except Exception as e:
        log.warning(f"  _cancel_all_open_orders_for_symbol {symbol}: {e}")

def _get_safe_close_info(deribit: DeribitClient, symbol: str, trade: dict) -> tuple:
    for attempt in range(2):
        try:
            actual_pos = deribit.get_position_size(symbol)
            if abs(actual_pos) > 0:
                close_side = "SELL" if actual_pos > 0 else "BUY"
                close_qty = deribit.round_amount(symbol, abs(actual_pos))
                return close_side, close_qty
        except Exception as e:
            log.debug(f"  _get_safe_close_info {symbol} attempt {attempt+1} failed: {e}")
            if attempt == 0: time.sleep(0.3)

    signal = trade.get("signal", "BUY")
    close_side = "SELL" if signal == "BUY" else "BUY"
    recorded_qty = float(trade.get("qty_tp2", 0)) if trade.get("tp1_hit") else float(trade.get("qty", 0))
    return close_side, recorded_qty

def _verify_actually_closed(deribit: DeribitClient, symbol: str, tolerance: float = 0.01) -> bool:
    try:
        remaining = abs(deribit.get_position_size(symbol))
        return remaining <= tolerance
    except Exception as e:
        log.warning(f"  _verify_actually_closed {symbol}: could not check ({e}) — assuming NOT closed")
        return False

def save_balance(deribit: DeribitClient) -> float:
    try:
        bals  = deribit.get_all_balances()
        total = deribit.get_total_equity_usd()
        pos   = deribit.get_positions()
        upnl  = sum(float(p.get("floating_profit_loss_usd") or p.get("floating_profit_loss") or 0) for p in pos)
        assets = [{"asset":c,"free":str(round(i["available"],6)), "total":str(round(i["equity_usd"],2))} for c,i in bals.items()]
        save_json(BALANCE_FILE, {
            "usdt":round(total,2),"equity":round(total+upnl,2),
            "unrealised":round(upnl,4),"assets":assets,
            "updated_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "mode":"deribit_testnet","exchange":"Deribit(by Coinbase) Testnet",
            "open_positions":len(pos),
        })
        log.info(f"  Balance: ${total:.2f} | upnl:{upnl:+.2f} | positions:{len(pos)}")
        return total
    except Exception as e:
        log.error(f"  save_balance: {e}"); return 0.0


def _fetch_deribit_klines_live(symbol: str, interval: str, limit: int = LIVE_LIMIT) -> pd.DataFrame:
    try:
        base = symbol.replace("USDT", "").upper()
        inst = f"{base}_USDC-PERPETUAL"
        res_map = {"15m": "15", "1h": "60", "4h": "240"}
        res = res_map.get(interval, "15")
        mins = int(res) * limit
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (mins * 60 * 1000)
        r = requests.get(
            "https://www.deribit.com/api/v2/public/get_tradingview_chart_data",
            params={"instrument_name": inst, "resolution": res, "start_timestamp": start_ms, "end_timestamp": now_ms},
            timeout=8
        )
        if r.status_code == 200:
            res_data = r.json().get("result", {})
            ticks = res_data.get("ticks", [])
            if ticks:
                df = pd.DataFrame({
                    "open_time": ticks,
                    "open": [float(x) for x in res_data.get("open", [])],
                    "high": [float(x) for x in res_data.get("high", [])],
                    "low": [float(x) for x in res_data.get("low", [])],
                    "close": [float(x) for x in res_data.get("close", [])],
                    "volume": [float(x) for x in res_data.get("volume", [])],
                    "taker_buy_base_vol": [float(x) * 0.5 for x in res_data.get("volume", [])]
                })
                return df.sort_values("open_time").reset_index(drop=True)
    except Exception: pass
    return pd.DataFrame()


def get_data(symbol: str, interval: str) -> pd.DataFrame:
    for url in ["https://data-api.binance.vision/api/v3/klines", "https://binance.com/api/v3/klines", "https://fapi.binance.com/fapi/v1/klines"]:
        try:
            r = requests.get(url, params={"symbol":symbol,"interval":interval,"limit":LIVE_LIMIT}, timeout=8)
            if r.status_code == 200:
                raw = r.json()
                if raw and isinstance(raw, list):
                    df = pd.DataFrame(raw)
                    cols = ["open_time","open","high","low","close","volume",
                            "close_time","quote_vol","trades",
                            "taker_buy_base_vol","taker_buy_quote_vol","ignore"]
                    df.columns = cols[:df.shape[1]]
                    for c in ["open","high","low","close","volume","taker_buy_base_vol"]:
                        if c in df.columns:
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                    keep = [c for c in ["open_time","open","high","low","close","volume","taker_buy_base_vol"]
                            if c in df.columns]
                    return df[keep]
        except Exception: continue

    df_deribit = _fetch_deribit_klines_live(symbol, interval, limit=LIVE_LIMIT)
    if not df_deribit.empty:
        return df_deribit

    return pd.DataFrame()


def _merge_extra_features_live(df15: pd.DataFrame, btc_df15: pd.DataFrame) -> pd.DataFrame:
    if df15.empty or "open_time" not in df15.columns:
        return df15

    df = df15.copy()
    if btc_df15 is not None and not btc_df15.empty and "close" in btc_df15.columns and "open_time" in btc_df15.columns:
        btc_slim = (btc_df15[["open_time","close"]].rename(columns={"close":"btc_close"})
                    .assign(open_time=lambda d: d["open_time"].astype("int64")).sort_values("open_time"))
        df_work = df.assign(open_time=lambda d: d["open_time"].astype("int64")).sort_values("open_time")
        df = pd.merge_asof(df_work, btc_slim, on="open_time", direction="backward")

    if "btc_close" in df.columns and df["btc_close"].notna().sum() > 30:
        btc_ret  = df["btc_close"].pct_change()
        coin_ret = df["close"].pct_change()
        roll_cov = coin_ret.rolling(20, min_periods=10).cov(btc_ret)
        roll_var = btc_ret.rolling(20, min_periods=10).var()
        df["btc_corr_20"]      = coin_ret.rolling(20, min_periods=10).corr(btc_ret)
        df["btc_beta_20"]      = roll_cov / roll_var.replace(0, np.nan)
        df["btc_rel_strength"] = (df["close"].pct_change(6) - df["btc_close"].pct_change(6)) * 100
    else:
        df["btc_corr_20"] = 0.0
        df["btc_beta_20"] = 1.0
        df["btc_rel_strength"] = 0.0

    df["btc_corr_20"]      = df["btc_corr_20"].fillna(0.0).clip(-1, 1)
    df["btc_beta_20"]      = df["btc_beta_20"].fillna(1.0).clip(-5, 5)
    df["btc_rel_strength"] = df["btc_rel_strength"].fillna(0.0).clip(-50, 50)

    return df.sort_values("open_time").reset_index(drop=True)


# ── MODIFIED: generate_signal now computes ensemble disagreement and logs
#              every prediction (executed or rejected) to predictions.json ──
def generate_signal(symbol, pipeline, thresholds, btc_momentum=None, whale_flow=None, fng_data=None, btc_df15_live=None):
    try:
        raw15 = get_data(symbol, TIMEFRAME_ENTRY)
        if raw15 is None or raw15.empty or len(raw15) < 30:
            log.warning(f"    [{symbol}] Insufficient/missing 15m candle data — skip")
            return None

        if "open_time" not in raw15.columns:
            if "ticks" in raw15.columns: raw15["open_time"] = raw15["ticks"]
            elif "timestamp" in raw15.columns: raw15["open_time"] = raw15["timestamp"]
            else:
                log.warning(f"    [{symbol}] Missing open_time column — skip")
                return None

        df15 = add_indicators(raw15)
        if df15.empty or "open_time" not in df15.columns: return None

        df15 = _merge_extra_features_live(df15, btc_df15_live).fillna(0)

        df1h_raw = get_data(symbol, TIMEFRAME_CONFIRM)
        df1h = add_indicators(df1h_raw).fillna(0) if (df1h_raw is not None and not df1h_raw.empty) else pd.DataFrame()
        df4h_raw = get_data(symbol, TIMEFRAME_TREND)
        df4h = add_indicators(df4h_raw).fillna(0) if (df4h_raw is not None and not df4h_raw.empty) else pd.DataFrame()

        row = df15.iloc[-1].copy()
        r1h = df1h.iloc[-1] if not df1h.empty else pd.Series(0, index=df15.columns)
        r4h = df4h.iloc[-1] if not df4h.empty else pd.Series(0, index=df15.columns)

        row["rsi_1h"]   = float(r1h.get("rsi",  50))
        row["adx_1h"]   = float(r1h.get("adx",   0))
        row["trend_1h"] = float(r1h.get("trend", 0))
        row["rsi_4h"]   = float(r4h.get("rsi",   50))
        row["trend_4h"] = float(r4h.get("trend",  0))

        af   = pipeline["all_features"]
        for col in af:
            if col not in row: row[col] = 0.0

        X    = pd.DataFrame([row[af].values], columns=af).replace([np.inf,-np.inf],0).fillna(0)
        Xs   = pipeline["selector"].transform(X)

        # ── NEW: ensemble disagreement, computed once per signal ──
        disagreement = compute_ensemble_disagreement(pipeline, Xs[0])

        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        
        sig  = pipeline["label_map"][int(pred)]
        conf = round(float(max(prob))*100, 1)

        rec_buy  = float(pipeline.get("recommended_threshold_buy", 0.40)) * 100.0
        rec_sell = float(pipeline.get("recommended_threshold_sell", 0.45)) * 100.0
        raw_target = rec_buy if sig == "BUY" else rec_sell

        effective_base_conf = max(raw_target, EXECUTION_SANITY_FLOOR)
        min_required_conf = get_required_confidence(symbol, effective_base_conf)

        log.info(f"    ML: {sig} {conf:.1f}% (need ≥{min_required_conf:.1f}%) | disagreement={disagreement.get('ensemble_disagreement')}")
        if sig == "NO_TRADE" or conf < min_required_conf:
            save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                             reject_reason="NO_TRADE" if sig == "NO_TRADE" else f"conf {conf}<{min_required_conf}"))
            return None

        fg_override_active = False
        if fng_data:
            is_blocked = (sig == "SELL" and fng_data.get("fg_blocks_sell", False)) or \
                         (sig == "BUY" and fng_data.get("fg_blocks_buy", False))
            
            if is_blocked:
                if conf >= 70.0:
                    fg_override_active = True
                    log.info(f"    ⚡ [FG_OVERRIDE] F&G={fng_data.get('value')} extreme bypassed (High Conviction: {conf:.1f}% ≥ 70%)")
                else:
                    direction_str = "SELL at panic bottom" if sig == "SELL" else "BUY at euphoric top"
                    log.info(f"    [FILTER:FG_HARD] F&G={fng_data.get('value')} — blocking new {direction_str} (Conf: {conf:.1f}% < 70%)")
                    save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                                     reject_reason=f"FG_HARD_BLOCK fg={fng_data.get('value')}"))
                    return None

        adx = float(row.get("adx", 0))
        min_adx_req = thresholds.get("min_adx", getattr(config, "MIN_ADX", 15.0))
        log.info(f"    [{symbol}] ML={sig} {conf:.1f}% ADX={adx:.1f} — evaluating filters")
        log.info(f"    ADX: {adx:.1f} (need ≥{min_adx_req})")
        if adx < min_adx_req:
            save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                             reject_reason=f"ADX {adx:.1f}<{min_adx_req}"))
            return None

        score = 0
        reasons = []

        e20_4h = float(r4h.get("ema20", 0)) if not df4h.empty else 0
        e50_4h = float(r4h.get("ema50", 0)) if not df4h.empty else 0
        rsi_4h = float(r4h.get("rsi", 50))  if not df4h.empty else 50
        trend_bars = 0

        if not df4h.empty:
            if sig == "BUY" and e20_4h < e50_4h:
                log.info(f"    [FILTER:4H_BIAS] 4h bearish (EMA20 < EMA50) — hard block BUY {symbol}")
                save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                                 reject_reason="4H_BIAS_bearish_block_buy"))
                return None
            
            if sig == "SELL" and e20_4h > e50_4h:
                if rsi_4h < 75:
                    log.info(f"    [FILTER:4H_BIAS] 4h bullish (EMA20 > EMA50 | RSI {rsi_4h:.0f} < 75) — hard block SELL {symbol}")
                    save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                                     reject_reason="4H_BIAS_bullish_block_sell"))
                    return None
                else:
                    score -= 1
                    reasons.append("4h extreme overbought counter-trend (-1)")

            for i in range(1, min(6, len(df4h))):
                r_prev = df4h.iloc[-(i+1)]
                if sig == "BUY" and float(r_prev.get("ema20", 0)) > float(r_prev.get("ema50", 0)):
                    trend_bars += 1
                elif sig == "SELL" and float(r_prev.get("ema20", 0)) < float(r_prev.get("ema50", 0)):
                    trend_bars += 1
                else: break

            if trend_bars < 1:
                log.info(f"    [FILTER:4H_FRESH] 4h trend too fresh ({trend_bars} bars) — skip {symbol}")
                save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                                 reject_reason=f"4H_FRESH trend_bars={trend_bars}"))
                return None

        if conf >= (min_required_conf + 15): score+=2; reasons.append(f"Strong conf ({conf:.0f}%)")
        elif conf >= min_required_conf:     score+=1; reasons.append(f"Valid conf ({conf:.0f}%)")
        
        adx_val = adx
        if adx_val > 25:   score+=1; reasons.append(f"Strong ADX {adx_val:.0f}")
        elif adx_val > 18: score+=1; reasons.append(f"ADX {adx_val:.0f}")
        
        rsi = float(row.get("rsi", 50))
        if sig=="BUY"  and rsi < 65: score+=1; reasons.append(f"RSI not overbought ({rsi:.0f})")
        elif sig=="SELL" and rsi > 35: score+=1; reasons.append(f"RSI not oversold ({rsi:.0f})")
        
        e20=float(row.get("ema20",0)); e50=float(row.get("ema50",0))
        if sig=="BUY"  and e20>e50: score+=1; reasons.append("EMA bullish")
        elif sig=="SELL" and e20<e50: score+=1; reasons.append("EMA bearish")
        
        c20=float(r1h.get("ema20",0)); c50=float(r1h.get("ema50",0))
        if sig=="BUY"  and c20>c50: score+=1; reasons.append("1h confirms")
        elif sig=="SELL" and c20<c50: score+=1; reasons.append("1h confirms")

        if sig == "BUY"  and e20_4h > e50_4h and rsi_4h > 45:
            score += 1; reasons.append(f"4h trend aligns (RSI {rsi_4h:.0f})")
        elif sig == "SELL" and e20_4h < e50_4h and rsi_4h < 55:
            score += 1; reasons.append(f"4h trend aligns (RSI {rsi_4h:.0f})")

        if trend_bars >= 4:
            score += 1; reasons.append(f"4h trend established ({trend_bars*4}h)")

        if btc_momentum and btc_momentum.get("bias"):
            if sig == btc_momentum["bias"]:
                score += btc_momentum["score_mod"]
                reasons.append(f"BTC {btc_momentum['bias']} momentum ({btc_momentum['strength']})")
            elif btc_momentum["score_mod"] > 0:
                score -= btc_momentum["score_mod"]
                reasons.append(f"BTC counter-momentum (-{btc_momentum['score_mod']})")

        if whale_flow and whale_flow.get("bias"):
            if sig == whale_flow["bias"]:
                score += whale_flow.get("score_mod", 1)
                reasons.append(f"Whale flow aligned (+{whale_flow.get('score_mod', 1)})")
            else:
                score -= whale_flow.get("score_mod", 1)
                reasons.append(f"Whale counter-flow (-{whale_flow.get('score_mod', 1)})")

        if fng_data and fng_data.get("bias"):
            f_bias = fng_data["bias"]
            f_mod  = fng_data.get("score_mod", 1)
            if sig == f_bias:
                score += f_mod
                reasons.append(f"Contrarian F&G aligned (+{f_mod})")
                log.info(f"    F&G ALIGNED (+{f_mod}) — {fng_data['message']}")
            elif f_mod > 0:
                score -= f_mod
                reasons.append(f"F&G extreme countered (-{f_mod})")
                log.info(f"    F&G OPPOSED (-{f_mod}) — {fng_data['message']}")

        vol_prev = float(df15["volume"].iloc[-2])
        vol_ma20 = float(df15["volume"].rolling(20).mean().iloc[-2])
        if vol_ma20 <= 0 or vol_prev <= 0:
            log.info("    Volume data missing — skipping volume gate")
        else:
            if vol_prev < vol_ma20 * 0.5:
                log.info(f"    [FILTER:VOL] Low volume ({vol_prev:.0f} < {vol_ma20:.0f}) — skip {symbol}")
                save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                                 reject_reason=f"LOW_VOLUME {vol_prev:.0f}<{vol_ma20:.0f}"))
                return None
            if vol_prev > vol_ma20 * 1.5:
                score += 1; reasons.append(f"Volume surge {vol_prev/vol_ma20:.1f}×")

        effective_min = thresholds.get("min_score", getattr(config, "MIN_SCORE", 3))
        log.info(f"    Score: {score} (need ≥{effective_min})")
        if score < effective_min:
            log.info(f"    [FILTER:SCORE] Too low ({score} < {effective_min}) — skip {symbol}")
            save_signal({"symbol":symbol,"signal":sig,"confidence":conf,"score":score,
                "reasons":reasons,"rejected":True,"reject_reason":f"score {score}<{effective_min}"})
            save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement,
                             reject_reason=f"SCORE {score}<{effective_min}"))
            return None

        if not reasons: reasons.append(f"ML {conf:.0f}%")

        entry = float(row["close"]); atr = float(row["atr"])
        if not math.isfinite(entry) or entry <= 0 or not math.isfinite(atr) or atr <= 0:
            log.error(f"    🚨 {symbol}: bad entry price/ATR — aborting signal")
            return None

        # ── NEW: log the successful prediction + build the trade snapshot ──
        snapshot = build_trade_snapshot(row, r4h, e20_4h, e50_4h, rsi_4h, fng_data, whale_flow, btc_momentum)
        pred_id  = f"{symbol}_{int(time.time()*1000)}"
        save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement, pred_id=pred_id))

        return {
            "symbol": symbol, "signal": sig, "confidence": conf, "score": score,
            "entry": entry, "atr": atr, "stop": 0, "tp1": 0, "tp2": 0,
            "reasons": reasons, "conf_tier": "high" if conf >= 60.0 else "normal",
            "fg_override": fg_override_active,
            "pred_id": pred_id,
            "snapshot": snapshot,
        }
    except Exception as e:
        log.error(f"    Signal {symbol}: {e}")
        return None


# ── MODIFIED: execute_trade now applies approved stop-mult override and
#              carries pred_id/snapshot through to the trade record ──
def execute_trade(deribit: DeribitClient, sig: dict, risk_mult: float, balance: float, vol_state: str = "NORMAL", base_min_conf: float = None) -> bool:
    if base_min_conf is None:
        raise ValueError("execute_trade() requires base_min_conf — no implicit default allowed.")

    symbol = sig["symbol"]
    signal = sig["signal"]
    entry  = sig["entry"]
    atr    = sig["atr"]

    # ── NEW: approved adaptation override for this symbol, if any ──
    sym_overrides = get_symbol_overrides(symbol)
    effective_stop_mult = ATR_STOP_MULT
    if "atr_stop_mult_override" in sym_overrides:
        effective_stop_mult = ATR_STOP_MULT * sym_overrides["atr_stop_mult_override"].get("multiplier", 1.0)
        log.info(f"  🧪 {symbol}: applying approved stop-mult override — {ATR_STOP_MULT} → {effective_stop_mult:.2f}")
    
    trades     = load_trades()
    open_count = len([t for t in trades.values() if not t.get("closed", False)])
    
    max_open_trades = int(getattr(config, "MAX_OPEN_TRADES", 8))
    max_same_dir    = int(getattr(config, "MAX_SAME_DIRECTION", 4))

    if open_count >= max_open_trades:
        log.info(f"  🛑 MAX TRADES ({open_count}/{max_open_trades}) — skip {symbol}")
        return False

    same_dir_count = sum(1 for t in trades.values() if not t.get("closed", False) and t.get("signal") == signal)
    if same_dir_count >= max_same_dir:
        log.info(f"  🛑 MAX SAME DIRECTION ({same_dir_count}/{max_same_dir} {signal}) — skip {symbol}")
        return False

    if symbol in trades and not trades[symbol].get("closed", False):
        log.info(f"  {symbol}: already open — skip")
        return False

    try:
        real_pos = abs(deribit.get_position_size(symbol))
    except Exception as e:
        real_pos = 0.0
        log.debug(f"  {symbol}: real position check failed ({e}) — proceeding on trades.json only")
    if real_pos > 0:
        log.warning(f"  🚫 {symbol}: real exchange position ({real_pos}) already exists — SKIPPING entry.")
        return False

    if not deribit.is_supported(symbol):
        log.info(f"  {symbol}: not on Deribit — skip")
        return False

    daily_count = _get_daily_trade_count()
    if daily_count >= MAX_DAILY_TRADES:
        log.info(f"  📊 Daily trade limit reached ({daily_count}/{MAX_DAILY_TRADES}) — skip {symbol}")
        return False
    if _is_on_cooldown(symbol):
        return False
    if _is_unreliable(symbol, signal_conf=sig.get("confidence", 0.0), current_baseline_conf=base_min_conf):
        return False
    
    funding_mult = _check_funding_rate(deribit, symbol, signal)
    if funding_mult <= 0:
        return False
    risk_mult *= funding_mult

    spread_info = deribit.get_order_book_spread(symbol)
    if spread_info.get("spread_pct", 999) > ENTRY_MAX_SPREAD_PCT:
        log.warning(f"  🚫 {symbol}: entry spread {spread_info.get('spread_pct', 0)*100:.2f}% > {ENTRY_MAX_SPREAD_PCT*100:.2f}% ceiling — book too thin, skip entry")
        return False
    
    if risk_mult <= 0:
        log.info(f"  🛑 Risk multiplier is {risk_mult} — skip {symbol}")
        return False

    live_price = deribit.get_live_price(symbol)
    if live_price > 0:
        drift = abs(live_price - entry) / entry * 100
        if drift > 0.5:
            log.warning(f"  [EXPIRED] {symbol} price drifted {drift:.2f}% from signal — skip")
            return False
        entry = live_price  
        sig["entry"] = entry

    dyn_tp1 = ATR_TARGET1_MULT
    dyn_tp2 = ATR_TARGET2_MULT
    if vol_state == "VERY_HIGH":
        dyn_tp1 *= 1.5
        dyn_tp2 *= 1.5
    elif vol_state in ("DEAD", "UNKNOWN"):
        dyn_tp1 *= 0.8
        dyn_tp2 *= 0.8

    dec = 4 if entry < 10 else 2
    side    = "BUY"  if signal == "BUY" else "SELL"
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"

    if signal == "BUY":
        stop = round(entry - atr * effective_stop_mult, dec)
        tp1  = round(entry + atr * dyn_tp1, dec)
        tp2  = round(entry + atr * dyn_tp2, dec)
    else:
        stop = round(entry + atr * effective_stop_mult, dec)
        tp1  = round(entry - atr * dyn_tp1, dec)
        tp2  = round(entry - atr * dyn_tp2, dec)

    for name, val in [("stop", stop), ("tp1", tp1), ("tp2", tp2)]:
        if not math.isfinite(val) or val <= 0:
            log.error(f"  🚨 {symbol}: computed {name}={val!r} is invalid — aborting trade")
            return False

    sig["stop"] = stop
    sig["tp1"]  = tp1
    sig["tp2"]  = tp2

    vol_scalar = 1.0
    try:
        atr_pct_this    = (atr / entry) if entry > 0 else 0
        btc_ref_atr_pct = 0.003
        if atr_pct_this > 0:
            vol_scalar = btc_ref_atr_pct / atr_pct_this
            vol_scalar = max(0.4, min(2.0, vol_scalar))
    except Exception as e:
        log.debug(f"  vol_scalar calc failed for {symbol}: {e}")

    if sig.get("fg_override", False):
        final_risk_mult = (risk_mult * vol_scalar) * 0.5
        log.info(f"  🛡️ [FG_OVERRIDE_SIZING] 50% discount applied, high-conf boost suppressed (final_risk_mult={final_risk_mult:.3f})")
    else:
        risk_boost = 1.5 if sig.get("conf_tier") == "high" else 1.0
        final_risk_mult = risk_mult * risk_boost * vol_scalar

    total_q = deribit.calc_contracts(symbol, balance, entry, stop, final_risk_mult)
    min_lot = deribit.get_min_trade_amount(symbol)

    if total_q < min_lot and sig.get("fg_override", False):
        min_lot_risk = abs(entry - stop) * min_lot
        max_allowed_normal_risk = balance * RISK_PER_TRADE * 1.0
        if min_lot_risk <= max_allowed_normal_risk:
            total_q = min_lot
            log.info(f"  ℹ️ [FG_OVERRIDE_SIZING] Clamped to min lot ({min_lot} {symbol}) | Risk: ${min_lot_risk:.2f} <= ${max_allowed_normal_risk:.2f}")
        else:
            log.info(f"  [SKIP] FG override {symbol}: min lot risk (${min_lot_risk:.2f}) > normal risk budget (${max_allowed_normal_risk:.2f}) — skip rather than oversize")
            return False

    if total_q < min_lot or total_q <= 0:
        log.warning(f"  [SKIP] Sizing below exchange min lot size. symbol={symbol} target_qty={total_q} min_lot={min_lot}")
        return False

    qty_tp1, qty_tp2 = deribit.split_amount(symbol, total_q)
    exit_mode = "DUAL_TP" if qty_tp2 > 0 else "SINGLE_TP"
    
    budget_risk_usd = round(balance * RISK_PER_TRADE * final_risk_mult, 2)
    actual_risk_usd = round(abs(entry - stop) * total_q, 2)

    log.info(f"  {signal} {symbol} Total={total_q} (TP1={qty_tp1}, TP2={qty_tp2} | Mode={exit_mode}) | Real Risk=${actual_risk_usd:.2f} (Budget: ${budget_risk_usd:.2f})")
    log.info(f"  SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids = {}
    actual_entry = entry
    try:
        from deribit_client import DEFAULT_LEVERAGE
        existing_pos = abs(deribit.get_position_size(symbol))
        if existing_pos == 0:
            deribit.set_leverage(symbol, DEFAULT_LEVERAGE)

        er = deribit.place_market_order(symbol, side, total_q)
        if not er:
            log.warning(f"  {symbol}: entry order aborted (thin book / slippage) — skip")
            return False
        eo = er.get("order", er)
        order_ids["entry"] = str(eo.get("order_id", ""))
        actual_entry = deribit.get_fill_price(er, entry) or entry
        
        o_state = eo.get("order_state", "").lower()
        filled  = float(eo.get("filled_amount", 0) or 0)
        
        if o_state == "cancelled" and filled == 0:
            log.warning("  Market cancelled (thin book) — skip")
            return False
            
        if o_state == "open" and filled == 0:
            log.warning("  Market order stuck as 'open' — cancelling & skipping")
            try: deribit.cancel_order(order_ids["entry"])
            except Exception: pass
            return False

        log.info(f"  ✅ Entry @ {actual_entry:.{dec}f}")
        
        position_confirmed = _wait_for_position(deribit, symbol, side, order_ids.get("entry", ""))
        if not position_confirmed:
            log.warning(f"  ⚠️ {symbol}: position unconfirmed — SL/TP will retry next scan")

        # ── POST-FILL RECOMPUTE WITH STRICT EMERGENCY CLOSE VALIDATION ──
        if signal == "BUY":
            recomputed_stop = deribit.round_price(symbol, actual_entry - atr * effective_stop_mult)
            recomputed_tp1  = deribit.round_price(symbol, actual_entry + atr * dyn_tp1)
            recomputed_tp2  = deribit.round_price(symbol, actual_entry + atr * dyn_tp2)
        else:
            recomputed_stop = deribit.round_price(symbol, actual_entry + atr * effective_stop_mult)
            recomputed_tp1  = deribit.round_price(symbol, actual_entry - atr * dyn_tp1)
            recomputed_tp2  = deribit.round_price(symbol, actual_entry - atr * dyn_tp2)

        post_fill_valid = True
        for name, val in [("stop", recomputed_stop), ("tp1", recomputed_tp1), ("tp2", recomputed_tp2)]:
            if not math.isfinite(val) or val <= 0:
                post_fill_valid = False
                log.critical(f"  🚨 {symbol}: POST-FILL {name}={val!r} is invalid (entry={actual_entry}, atr={atr})")
                break

        if not post_fill_valid:
            log.critical(f"  🚨🚨 {symbol}: Position LIVE ({total_q}) but post-fill SL/TP invalid — EMERGENCY FLATTENING NOW")
            try:
                deribit.place_market_order(symbol, sl_side, total_q, reduce_only=True)
                _send(f"🚨 *EMERGENCY CLOSE — {symbol}*\nPost-fill bracket prices invalid. Position flattened immediately.")
            except Exception as ce:
                log.critical(f"  🚨🚨 {symbol}: Emergency close failed: {ce}")
                _send(f"🚨🚨 *MANUAL ACTION NEEDED — {symbol}*\nPosition open with invalid SL/TP and emergency close failed!")
            return False

        stop = recomputed_stop
        tp1  = recomputed_tp1
        tp2  = recomputed_tp2
        actual_risk_usd = round(abs(actual_entry - stop) * total_q, 2)

        tick = deribit.get_tick_size(symbol)
        sl_limit = deribit.round_price(symbol, stop - (tick * 3) if signal == "BUY" else stop + (tick * 3))

        try:
            actual_pos_size = abs(deribit.get_position_size(symbol))
            if actual_pos_size > 0 and actual_pos_size < total_q:
                qty_tp1 = min(qty_tp1, actual_pos_size)
                qty_tp2 = min(qty_tp2, max(0, actual_pos_size - qty_tp1))
                qty_tp1 = deribit.round_amount(symbol, qty_tp1)
                qty_tp2 = deribit.round_amount(symbol, qty_tp2)
        except Exception as pos_e:
            log.debug(f"  Position size check: {pos_e}")

        order_plan = [("SL", total_q, sl_limit, stop, "stop_loss"), ("TP1", qty_tp1, tp1, None, "tp1")]
        if qty_tp2 > 0:
            order_plan.append(("TP2", qty_tp2, tp2, None, "tp2"))

        for label, qty, price, sl_p, key in order_plan:
            if qty <= 0: continue
            try:
                res = deribit.place_limit_order(
                    symbol, sl_side if label == "SL" else tp_side,
                    qty, price, stop_price=sl_p, use_reduce_only=(label == "SL")
                )
                o   = res.get("order", res)
                oid = str(o.get("order_id", ""))
                if oid: order_ids[key] = oid
                log.info(f"  ✅ {label} @ {price:.{dec}f} × {qty} id:{oid or 'MISSING'}")
            except Exception as e:
                log.warning(f"  {label} placement error: {e}")

        # ── VERIFY STOP LOSS WAS ACCEPTED BY DERIBIT ──
        if not order_ids.get("stop_loss"):
            log.critical(f"  🚨 {symbol}: Stop-Loss order failed on exchange — attempting emergency immediate retry")
            try:
                res = deribit.place_limit_order(symbol, sl_side, total_q, sl_limit, stop_price=stop, use_reduce_only=True)
                o   = res.get("order", res)
                oid = str(o.get("order_id", ""))
                if oid:
                    order_ids["stop_loss"] = oid
                    log.info(f"  ✅ Fallback SL placed successfully id:{oid}")
            except Exception as fe:
                log.critical(f"  🚨 {symbol}: Emergency SL retry failed: {fe}")

        if not order_ids.get("stop_loss"):
            log.critical(f"  🚨🚨 {symbol}: SL completely failed on exchange — FLATTENING POSITION NOW to avoid naked exposure")
            try:
                deribit.place_market_order(symbol, sl_side, total_q, reduce_only=True)
                _send(f"🚨 *EMERGENCY CLOSE — {symbol}*\nExchange rejected SL order. Position flattened immediately to prevent naked risk.")
            except Exception as ce:
                log.critical(f"  🚨🚨 {symbol}: Emergency flatten also failed: {ce}")
                _send(f"🚨🚨 *MANUAL ACTION NEEDED — {symbol}*\nNaked position open on Deribit and auto-flatten failed!")
            return False

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        _send(f"⚠️ {symbol}: {e}")
        return False

    # ── MODIFIED: record now carries pred_id + snapshot for later audit/taxonomy ──
    record = {
        "symbol": symbol, "signal": signal, "entry": actual_entry,
        "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": total_q, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        "risk_usd": actual_risk_usd, "budget_risk_usd": budget_risk_usd, "balance_at_open": balance,
        "risk_mult": final_risk_mult, "exit_mode": exit_mode,
        "order_ids": order_ids,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": sig["confidence"], "score": sig["score"],
        "reasons": sig.get("reasons", []), "tier": get_tier(symbol),
        "fg_override": sig.get("fg_override", False),
        "exchange": "deribit_testnet",
        "pred_id": sig.get("pred_id"),
        "snapshot": sig.get("snapshot", {}),
        "stop_mult_used": effective_stop_mult,
    }
    trades[symbol] = record
    save_trades(trades)
    save_signal({**record, "type": "executed"})
    if sig.get("pred_id"):
        mark_prediction_executed(sig["pred_id"])
    _send_open_alert(symbol, signal, sig["confidence"], sig["score"],
                     actual_entry, stop, tp1, tp2, total_q, qty_tp1, qty_tp2, actual_risk_usd, balance)
    log.info(f"  ✅✅ TRADE OPENED: {symbol} {signal} ({exit_mode})")
    return True


def _safe_get_order(deribit, oid_str):
    if not oid_str or oid_str in ("", "None"): return {}
    try: return deribit.get_order(oid_str)
    except Exception: return {}

def fp(o, fb):
    p = float(o.get("average_price") or o.get("last_price") or o.get("price") or 0)
    return p if p > 0 else fb

def _pnl(t, cp, ct):
    qty  = float(t["qty_tp1"] if ct == "tp1" else t["qty_tp2"] if ct == "tp2" else t["qty"])
    diff = (cp - float(t["entry"])) if t["signal"] == "BUY" else (float(t["entry"]) - cp)
    return round(diff * qty, 4)

def _close_record(t, cp, pnl, reason):
    append_history({**t, "close_price": cp, "pnl": pnl,
        "closed_at": datetime.now(timezone.utc).isoformat(), "close_reason": reason})


def _replace_missing_orders(deribit: DeribitClient, symbol: str, trade: dict) -> bool:
    oids   = trade.get("order_ids", {})
    signal = trade["signal"]
    entry  = float(trade["entry"])
    stop   = float(trade.get("stop", 0))
    tp1    = float(trade.get("tp1", 0))
    tp2    = float(trade.get("tp2", 0))
    qty    = float(trade.get("qty", 0))
    qty_t1 = float(trade.get("qty_tp1", 0))
    qty_t2 = float(trade.get("qty_tp2", 0))
    dec    = 4 if entry < 10 else 2
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"
    changed = False

    sl_oid  = str(oids.get("stop_loss", ""))
    has_sl  = bool(sl_oid and sl_oid not in ("", "None"))

    # ── HEAL CORRUPTED OR ZERO STOP-LOSS DYNAMICALLY ──
    if not has_sl and not trade.get("closed") and qty > 0:
        if stop <= 0:
            log.warning(f"  ⚠️ {symbol}: Saved stop is {stop!r} but position is live — recomputing dynamically")
            try:
                live_p = deribit.get_live_price(symbol)
                ref_p = entry if entry > 0 else live_p
                raw15 = get_data(symbol, TIMEFRAME_ENTRY)
                fresh_atr = (float(raw15["close"].iloc[-1]) * 0.015) if raw15.empty else float(add_indicators(raw15)["atr"].iloc[-1])
                stop = deribit.round_price(symbol, ref_p - fresh_atr * ATR_STOP_MULT if signal == "BUY" else ref_p + fresh_atr * ATR_STOP_MULT)
                trade["stop"] = stop
                changed = True
                log.info(f"  ✓ {symbol} stop recomputed: {stop:.{dec}f}")
            except Exception as heale:
                log.error(f"  Failed to heal stop for {symbol}: {heale}")

        if stop > 0:
            try:
                tick     = deribit.get_tick_size(symbol)
                sl_limit = deribit.round_price(
                    symbol, stop - (tick * 3) if sl_side == "SELL" else stop + (tick * 3)
                )
                res = deribit.place_limit_order(
                    symbol, sl_side, qty, sl_limit,
                    stop_price=stop, use_reduce_only=True
                )
                o   = res.get("order", res)
                oid = str(o.get("order_id", ""))
                if oid:
                    trade["order_ids"]["stop_loss"] = oid
                    log.info(f"  🛠 Re-placed missing SL {symbol} @ {sl_limit:.{dec}f} trigger={stop:.{dec}f} id:{oid}")
                    changed = True
            except Exception as e:
                err = str(e).lower()
                log.warning(f"  SL re-place {symbol}: {e}")
                if any(x in err for x in ("trigger_price_too_low", "trigger_price_too_high", "10035", "10036")):
                    log.warning(f"  🚨 {symbol}: SL trigger passed — MARKET CLOSE")
                    try:
                        _cancel_all_open_orders_for_symbol(deribit, symbol)
                        close_side, close_qty = _get_safe_close_info(deribit, symbol, trade)
                        if close_qty > 0:
                            deribit.place_market_order(symbol, close_side, close_qty, reduce_only=True)
                        live = deribit.get_live_price(symbol)
                        pnl  = _pnl(trade, live if live > 0 else stop, "sl")
                        if _verify_actually_closed(deribit, symbol):
                            _close_record(trade, live if live > 0 else stop, pnl, "SL missed — market close")
                            _send(f"🚨 *SL MISSED — {symbol}*\nTrigger passed. Closed @ `{live:.{dec}f}` | PnL≈`{pnl:+.4f}`")
                            if pnl < 0:
                                _add_cooldown(symbol, f"SL missed market close pnl={pnl:+.4f}")
                                _record_outcome(symbol, won=False)
                            trade["closed"] = True
                            changed = True
                    except Exception as me:
                        log.error(f"  Emergency SL close {symbol}: {me}")

    tp1_oid  = str(oids.get("tp1", ""))
    need_tp1 = (not tp1_oid or tp1_oid in ("", "None")) and tp1 > 0 and qty_t1 > 0
    if need_tp1 and not trade.get("tp1_hit") and not trade.get("closed"):
        oid = _place_tp_with_fallback(deribit, symbol, tp_side, qty_t1, tp1, "TP1", trade, "tp1", dec)
        if oid:
            trade["order_ids"]["tp1"] = oid
            changed = True

    tp2_oid  = str(oids.get("tp2", ""))
    need_tp2 = (not tp2_oid or tp2_oid in ("", "None")) and tp2 > 0 and qty_t2 > 0
    if need_tp2 and not trade.get("tp2_hit") and not trade.get("closed"):
        oid = _place_tp_with_fallback(deribit, symbol, tp_side, qty_t2, tp2, "TP2", trade, "tp2", dec)
        if oid:
            trade["order_ids"]["tp2"] = oid
            changed = True

    return changed


def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    live_positions = set()
    try:
        for p in deribit.get_positions():
            if float(p.get("size", 0)) != 0:
                inst = p.get("instrument_name", "")
                base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                live_positions.add(f"{base}USDT")
    except Exception as e:
        log.warning(f"  Could not fetch live positions: {e}")

    for symbol, trade in list(trades.items()):

        if trade.get("closed"):
            to_remove.append(symbol)
            continue

        entry  = float(trade["entry"])
        stop   = float(trade.get("stop", 0))
        tp1_p  = float(trade.get("tp1",  0))
        tp2_p  = float(trade.get("tp2",  0))
        dec    = 4 if entry < 10 else 2
        signal = trade["signal"]
        oids   = trade.get("order_ids", {})

        if _replace_missing_orders(deribit, symbol, trade): save_trades(trades)

        if trade.get("closed"):
            to_remove.append(symbol)
            continue

        try:
            live = deribit.get_live_price(symbol)
        except Exception as e: log.warning(f"  {symbol}: live price error — {e}"); continue
        if live <= 0: log.warning(f"  {symbol}: live price = 0 — skip"); continue

        if live_positions and symbol not in live_positions and not trade.get("tp1_hit"):
            log.warning(f"  {symbol}: no live position found — ghost cleaner will handle")
            continue

        risk_usd = float(trade.get("risk_usd", 0))
        if risk_usd > 0:
            close_side, mae_qty = _get_safe_close_info(deribit, symbol, trade)
            mae_diff = (live - entry) if signal == "BUY" else (entry - live)
            mae_pnl  = round(mae_diff * mae_qty, 4)

            if mae_pnl < -(risk_usd * 3):
                log.warning(f"  🚨 {symbol}: MAE ${mae_pnl:.2f} > 3× risk ${risk_usd:.2f} — FORCE CLOSE")
                _cancel_all_open_orders_for_symbol(deribit, symbol)

                recorded_qty = float(trade.get("qty", 0))
                size_mismatch = recorded_qty > 0 and abs(mae_qty - recorded_qty) > max(recorded_qty * 0.2, 0.0)
                if size_mismatch:
                    log.error(f"  🚨🚨 {symbol}: SIZE MISMATCH — recorded qty={recorded_qty}, real exchange position={mae_qty}.")

                try:
                    if mae_qty > 0:
                        deribit.place_market_order(symbol, close_side, mae_qty, reduce_only=True)
                    if _verify_actually_closed(deribit, symbol):
                        lbl = "Max adverse excursion ❌" + (" [SIZE MISMATCH]" if size_mismatch else "")
                        _close_record(trade, live, mae_pnl, lbl)
                        _send(f"🚨 *FORCE CLOSE — {symbol}*\nLoss `{mae_pnl:+.4f}` exceeded 3× risk\nLive @ `{live:.{dec}f}`")
                        if mae_pnl < 0:
                            _add_cooldown(symbol, f"MAE close pnl={mae_pnl:+.4f}")
                            _record_outcome(symbol, won=False)
                        trade["closed"] = True
                        to_remove.append(symbol)
                    continue
                except Exception as e:
                    log.error(f"  MAE force-close {symbol}: {e}")

        # TP1 Monitoring
        try:
            if not trade.get("tp1_hit"):
                o      = _safe_get_order(deribit, str(oids.get("tp1", "")))
                state  = o.get("order_state", "").lower()
                tp1_order_filled = deribit.is_order_filled(o)
                filled_amt = float(o.get("filled_amount", 0) or 0)
                total_amt  = float(o.get("amount", 0) or 0)
                tp1_partial = (total_amt > 0 and filled_amt / total_amt >= 0.8 and filled_amt > 0)
                tp1_price_hit = tp1_p > 0 and ((signal == "BUY" and live >= tp1_p) or (signal == "SELL" and live <= tp1_p))
                tp1_o_gone = state in ("filled", "cancelled", "closed", "rejected", "")

                if tp1_order_filled or tp1_partial or (tp1_price_hit and tp1_o_gone):
                    trade["tp1_hit"] = True
                    method = ("order" if tp1_order_filled else "partial" if tp1_partial else "price-fallback")
                    fill   = fp(o, tp1_p) if (tp1_order_filled or tp1_partial) else live
                    pnl    = _pnl(trade, fill, "tp1")
                    log.info(f"  🎯 TP1 {symbol} @ {fill:.{dec}f}  pnl≈{pnl:+.4f}  [{method}]")
                    _send(f"🎯 *TP1 HIT — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}` | [{method}]")
                    
                    append_history({
                        **trade,
                        "close_price": fill,
                        "pnl":          pnl,
                        "qty":          float(trade.get("qty_tp1", 0)),
                        "closed_at":    datetime.now(timezone.utc).isoformat(),
                        "close_reason": f"TP1 hit [{method}]",
                        "partial":      (float(trade.get("qty_tp2", 0)) > 0),
                    })

                    if float(trade.get("qty_tp2", 0)) <= 0:
                        trade["closed"] = True
                        _record_outcome(symbol, won=True)
                        if oids.get("stop_loss"):
                            try: deribit.cancel_order(oids["stop_loss"])
                            except Exception: pass
                        to_remove.append(symbol)
                        continue

                    if oids.get("stop_loss") and float(trade.get("qty_tp2", 0)) > 0:
                        try: deribit.cancel_order(oids["stop_loss"])
                        except Exception: pass

                        try:
                            sl_s = "SELL" if signal == "BUY" else "BUY"
                            tick = deribit.get_tick_size(symbol)
                            be_limit = entry + tick if sl_s == "BUY" else entry - tick

                            be   = deribit.place_limit_order(
                                symbol, sl_s, float(trade["qty_tp2"]),
                                be_limit, stop_price=entry
                            )
                            be_o = be.get("order", be)
                            nid  = str(be_o.get("order_id", ""))
                            if nid:
                                trade["order_ids"]["stop_loss"] = nid
                                trade["stop"] = entry
                            _send(f"🛡️ *{symbol} RISK-FREE* SL→entry `{entry:.{dec}f}`")
                        except Exception as be_e:
                            log.warning(f"  BE SL {symbol}: {be_e}")

            if trade.get("tp1_hit") and not trade.get("tp2_hit") and oids.get("stop_loss"):
                halfway = (entry + tp2_p) / 2
                at_half = ((signal == "BUY"  and live >= halfway) or (signal == "SELL" and live <= halfway))
                sl_at_be = abs(float(trade.get("stop", 0)) - entry) < entry * 0.001
                if at_half and sl_at_be and float(trade.get("qty_tp2", 0)) > 0:
                    try:
                        try: deribit.cancel_order(oids["stop_loss"])
                        except Exception: pass

                        sl_s = "SELL" if signal == "BUY" else "BUY"
                        tick = deribit.get_tick_size(symbol)
                        sl_lim = tp1_p + tick if sl_s == "BUY" else tp1_p - tick

                        sl_r = deribit.place_limit_order(
                            symbol, sl_s, float(trade["qty_tp2"]),
                            sl_lim, stop_price=tp1_p
                        )
                        sl_o = sl_r.get("order", sl_r)
                        nid  = str(sl_o.get("order_id", ""))
                        if nid:
                            trade["order_ids"]["stop_loss"] = nid
                            trade["stop"] = tp1_p
                        _send(f"🚀 *{symbol}* Trail SL→TP1 `{tp1_p:.{dec}f}` locked!")
                    except Exception as e:
                        log.warning(f"  Trail SL {symbol}: {e}")

            if not trade.get("tp2_hit") and float(trade.get("qty_tp2", 0)) > 0:
                o      = _safe_get_order(deribit, str(oids.get("tp2", "")))
                state2 = o.get("order_state", "").lower()
                tp2_order_filled = deribit.is_order_filled(o)
                filled_amt2 = float(o.get("filled_amount", 0) or 0)
                total_amt2  = float(o.get("amount", 0) or 0)
                tp2_partial = (total_amt2 > 0 and filled_amt2 / total_amt2 >= 0.8 and filled_amt2 > 0)
                tp2_price_hit = tp2_p > 0 and ((signal == "BUY" and live >= tp2_p) or (signal == "SELL" and live <= tp2_p))
                tp2_o_gone = state2 in ("filled", "cancelled", "closed", "rejected", "")

                if tp2_order_filled or tp2_partial or (tp2_price_hit and tp2_o_gone):
                    if not trade.get("tp1_hit"):
                        trade["tp1_hit"] = True
                    method2 = ("order" if tp2_order_filled else "partial" if tp2_partial else "price-fallback")
                    fill2   = fp(o, tp2_p) if (tp2_order_filled or tp2_partial) else live
                    pnl2    = _pnl(trade, fill2, "tp2")

                    trade["tp2_hit"] = True
                    trade["closed"]  = True
                    log.info(f"  ✅ TP2 {symbol} @ {fill2:.{dec}f}  pnl≈{pnl2:+.4f}  [{method2}]")
                    _send(f"✅ *FULL WIN — {symbol}*\nTP2 @ `{fill2:.{dec}f}` | PnL ≈ `{pnl2:+.4f}` | [{method2}]")
                    _close_record(trade, fill2, pnl2, "TP2 hit")
                    _record_outcome(symbol, won=True)

                    cd = load_cooldown(); cd.pop(symbol, None); save_cooldown(cd)
                    if oids.get("stop_loss"):
                        try: deribit.cancel_order(oids["stop_loss"])
                        except Exception: pass

                    to_remove.append(symbol)
                    continue

            if not trade.get("closed") and oids.get("stop_loss"):
                sl_o     = _safe_get_order(deribit, str(oids["stop_loss"]))
                sl_state = sl_o.get("order_state", "").lower()
                sl_hit   = deribit.is_sl_triggered(sl_o)

                if not sl_hit and sl_state == "not_found":
                    try:
                        real_pos_size = abs(deribit.get_position_size(symbol))
                        already_flat  = real_pos_size <= 0.01
                        if not already_flat:
                            recorded_qty = float(trade.get("qty", 0))
                            if abs(recorded_qty - real_pos_size) > 0.0001:
                                trade["qty"] = real_pos_size
                                save_trades(trades)
                    except Exception: already_flat = False
                    if already_flat: sl_hit = True

                sl_breached = stop > 0 and ((signal == "BUY" and live <= stop * 0.999) or (signal == "SELL" and live >= stop * 1.001))

                mark_price = deribit.get_mark_price(symbol)
                mark_breached = stop > 0 and (
                    (signal == "BUY"  and mark_price <= stop * 0.998) or
                    (signal == "SELL" and mark_price >= stop * 1.002)
                )

                # ── HEARTBEAT: alert on unresolved breach BEFORE attempting
                # the emergency close, independent of whether that close
                # succeeds — so a stuck exchange trigger is never silent. ──
                heartbeat_attr = f"_breach_streak_{symbol}"
                if (mark_breached or sl_breached) and not sl_hit:
                    streak = getattr(check_open_trades, heartbeat_attr, 0) + 1
                    setattr(check_open_trades, heartbeat_attr, streak)
                    if streak >= 2:
                        _send(f"🚨 *UNRESOLVED BREACH — {symbol}*\n"
                              f"Price past SL for {streak} consecutive scans; exchange trigger "
                              f"state=`{sl_state}`. Attempting emergency close now — verify manually if this repeats.")
                else:
                    setattr(check_open_trades, heartbeat_attr, 0)

                # FIX: the old `sl_not_waiting` gate required the SL order's
                # own reported state to be OUT of "untriggered"/"open" before
                # allowing an emergency close — but that's precisely the
                # normal, expected state of a healthy, unfired stop order.
                # That inverted the safety net: it refused to intervene
                # exactly when the exchange's trigger mechanism was stuck.
                # deribit.is_sl_triggered(sl_o) above already correctly
                # handles the "fired normally" case — that's the only gate
                # this needs. Gate removed from both branches below.
                if mark_breached and not sl_hit:
                    try:
                        _cancel_all_open_orders_for_symbol(deribit, symbol)
                        close_side, close_qty = _get_safe_close_info(deribit, symbol, trade)
                        if close_qty > 0:
                            deribit.place_market_order(symbol, close_side, close_qty, reduce_only=True)
                        sl_hit = True
                    except Exception as e: log.error(f"  Mark-breach close {symbol}: {e}")

                if not sl_hit and sl_breached:
                    try:
                        _cancel_all_open_orders_for_symbol(deribit, symbol)
                        close_side, close_qty = _get_safe_close_info(deribit, symbol, trade)
                        if close_qty > 0:
                            deribit.place_market_order(symbol, close_side, close_qty, reduce_only=True)
                        sl_hit = True
                    except Exception as e: log.error(f"  Scenario-B close {symbol}: {e}")

                if sl_hit:
                    if _verify_actually_closed(deribit, symbol):
                        trade["closed"] = True
                        fill = fp(sl_o, stop)
                        if fill == 0 or fill == stop or fill == float(trade.get("stop", 0)):
                            fill = live if live > 0 else stop

                        pnl = _pnl(trade, fill, "sl")
                        lbl = ("BREAK-EVEN ⚖️" if abs(fill - entry) < entry * 0.002 else "STOPPED OUT ❌")
                        log.info(f"  ❌ SL {symbol} @ {fill:.{dec}f}  pnl≈{pnl:+.4f}  [state={sl_state}]")
                        _send(f"{'⚖️' if 'BREAK' in lbl else '❌'} *{lbl} — {symbol}*\n@ `{fill:.{dec}f}` | PnL ≈ `{pnl:+.4f}`")
                        _close_record(trade, fill, pnl, lbl)
                        
                        if pnl < 0:
                            _add_cooldown(symbol, f"SL hit pnl={pnl:+.4f}")
                            _record_outcome(symbol, won=False)
                        else:
                            _record_outcome(symbol, won=True)

                        for k in ("tp1", "tp2"):
                            if oids.get(k) and not trade.get(f"{k}_hit"):
                                try: deribit.cancel_order(oids[k])
                                except Exception: pass
                        to_remove.append(symbol)

        except Exception as e:
            log.error(f"  Monitor {symbol}: {e}")

    save_trades(trades)
    for sym in set(to_remove): trades.pop(sym, None)
    save_trades(trades)


def check_stale_trades(deribit: DeribitClient):
    trades = load_trades()
    now    = datetime.now(timezone.utc)
    to_remove = []

    pre_tp1_max  = getattr(config, "MAX_TRADE_AGE_HOURS_PRE_TP1", 12)
    post_tp1_max = getattr(config, "MAX_TRADE_AGE_HOURS_POST_TP1", 48)

    for symbol, trade in trades.items():
        if trade.get("closed") or trade.get("tp2_hit"):
            continue
        try:
            opened_str = str(trade.get("opened_at", "")).replace("Z", "+00:00")
            age_h = (now - datetime.fromisoformat(opened_str)).total_seconds() / 3600.0
        except Exception:
            continue

        tp1_secured = trade.get("tp1_hit", False)
        max_allowed_age = post_tp1_max if tp1_secured else pre_tp1_max

        if age_h <= max_allowed_age:
            continue

        tier_label = "POST-TP1 RUNNER" if tp1_secured else "PRE-TP1 STALL"
        log.warning(f"  ⏰ {symbol}: {age_h:.1f}h exceeds {max_allowed_age}h ({tier_label}) — time exit")

        try:
            _cancel_all_open_orders_for_symbol(deribit, symbol)
            live = deribit.get_live_price(symbol)
            close_side, close_qty = _get_safe_close_info(deribit, symbol, trade)
            if close_qty > 0:
                try:
                    deribit.place_market_order(symbol, close_side, close_qty, reduce_only=True)
                except Exception as me:
                    log.warning(f"  Time-exit market order {symbol}: {me}")

            if _verify_actually_closed(deribit, symbol):
                close_price = live if live > 0 else float(trade["entry"])
                pnl         = _pnl(trade, close_price, "sl")
                reason      = f"Time exit ({age_h:.0f}h [{tier_label}])"
                _close_record(trade, close_price, pnl, reason)
                _send(f"⏰ *TIME EXIT ({tier_label}) — {symbol}*\n{age_h:.1f}h | PnL≈`{pnl:+.4f}` | @ `{close_price:.4f}`")
                to_remove.append(symbol)
        except Exception as e:
            log.error(f"  Time exit {symbol}: {e}")

    if to_remove:
        for sym in to_remove:
            trades.pop(sym, None)
        save_trades(trades)


def clean_ghost_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: return
    live_pos = {}
    for p in deribit.get_positions():
        if float(p.get("size", 0)) != 0:
            inst = p.get("instrument_name", "")
            base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            live_pos[f"{base}USDT"] = True

    to_remove = []
    for symbol, trade in trades.items():
        if float(trade.get("stop", 0)) == 0 or float(trade.get("tp1", 0)) == 0:
            log.warning(f"  ⚠️ {symbol}: Incomplete bracket state — attempting dynamic recovery instead of deleting")
            try:
                actual = abs(deribit.get_position_size(symbol))
                if actual > 0:
                    _replace_missing_orders(deribit, symbol, trade)
                    if float(trade.get("stop", 0)) == 0:
                        log.critical(f"  🚨 {symbol}: Bracket unrecoverable — FLATTENING ON EXCHANGE before removal")
                        _cancel_all_open_orders_for_symbol(deribit, symbol)
                        close_side = "SELL" if trade.get("signal") == "BUY" else "BUY"
                        deribit.place_market_order(symbol, close_side, actual, reduce_only=True)
                        _close_record(trade, float(trade.get("entry", 0)), 0.0, "Corrupted state auto-flattened")
                        to_remove.append(symbol)
                else:
                    to_remove.append(symbol)
            except Exception as ge:
                log.error(f"  Ghost healing failed for {symbol}: {ge}")
            continue
            
        if float(trade.get("score", 1)) == 0 and float(trade.get("confidence", 1)) == 0:
            try: actual = deribit.get_position_size(symbol)
            except Exception: actual = 0
            if abs(actual) > 0:
                try:
                    _cancel_all_open_orders_for_symbol(deribit, symbol)
                    close_side = "SELL" if actual > 0 else "BUY"
                    deribit.place_market_order(symbol, close_side, abs(actual), reduce_only=True)
                except Exception: pass
            _close_record(trade, float(trade.get("entry", 0)), 0.0, "Ghost — broken record")
            to_remove.append(symbol)
            continue
            
        if symbol not in live_pos:
            log.warning(f"  🕵️ {symbol}: no position — recovering PnL...")
            real_pnl = None; real_close = None; reason = "Closed on exchange"
            try:
                fills = deribit.get_trade_history_for_instrument(symbol, count=20)
                entry = float(trade["entry"]); entry_dir = trade["signal"]
                close_fills = [f for f in fills
                    if (entry_dir == "BUY" and f.get("direction") == "sell") or
                       (entry_dir == "SELL" and f.get("direction") == "buy")]
                if close_fills:
                    latest = close_fills[0]; real_close = float(latest.get("price", 0) or 0)
                    qty = float(trade.get("qty", 0))
                    if real_close > 0 and qty > 0:
                        diff = (real_close - entry) if entry_dir == "BUY" else (entry - real_close)
                        real_pnl = round(diff * qty, 4)
                        tp1_p = float(trade.get("tp1", 0)); tp2_p = float(trade.get("tp2", 0))
                        
                        if entry_dir == "BUY":
                            if real_close >= tp2_p * 0.998: reason = "TP2 hit"
                            elif real_close >= tp1_p * 0.998: reason = "TP1 hit"
                            elif real_pnl > 0: reason = "Manual close (Profit)"
                            else: reason = "SL hit"
                        else:
                            if real_close <= tp2_p * 1.002: reason = "TP2 hit"
                            elif real_close <= tp1_p * 1.002: reason = "TP1 hit"
                            elif real_pnl > 0: reason = "Manual close (Profit)"
                            else: reason = "SL hit"
                        log.info(f"  ✅ Recovered {symbol}: close={real_close:.4f} pnl={real_pnl:+.4f} ({reason})")
            except Exception as e:
                log.warning(f"  PnL recovery {symbol}: {e}")

            if real_pnl is not None and real_close is not None:
                _close_record(trade, real_close, real_pnl, reason)
                _send(f"{'✅' if real_pnl > 0 else '❌'} *{reason} — {symbol}*\nPnL:`{real_pnl:+.4f}`")
            else:
                _close_record(trade, float(trade.get("entry", 0)), 0.0, "Ghost — PnL unrecoverable")
                _record_ghost(symbol)
                _add_cooldown(symbol, "Ghost trade — position unrecoverable")
            to_remove.append(symbol)

    if to_remove:
        for sym in to_remove: trades.pop(sym, None)
        save_trades(trades)


def check_funding_rates(deribit) -> None:
    trades = load_trades()
    if not trades: return
    for symbol, trade in trades.items():
        if trade.get("closed"): continue
        try:
            rate     = deribit.get_funding_rate(symbol)
            rate_pct = rate * 100
            signal   = trade["signal"]
            is_paying = (signal == "BUY" and rate > 0) or (signal == "SELL" and rate < 0)
            if is_paying and abs(rate_pct) >= FUNDING_WARN_PCT:
                age_h = 0
                try:
                    age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(
                        trade.get("opened_at","").replace("Z",""))).total_seconds() / 3600
                except Exception: pass
                total_drag = abs(rate_pct) * (age_h / 8)
                log.warning(f"  💸 {symbol} {signal}: funding {rate_pct:+.3f}%/8h | age={age_h:.0f}h | drag≈{total_drag:.3f}%")
                if abs(rate_pct) >= FUNDING_SKIP_PCT:
                    _send(f"💸 *FUNDING ALERT — {symbol}*\n{signal} paying `{abs(rate_pct):.3f}%` per 8h\n"
                          f"Position age: `{age_h:.0f}h` | Total drag ≈ `{total_drag:.3f}%`")
        except Exception as e: log.debug(f"  funding monitor {symbol}: {e}")


def _send(text):
    tok = os.getenv("TELEGRAM_TOKEN",""); cid = os.getenv("TELEGRAM_CHAT_ID","")
    if not tok or not cid: return
    try: requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                       data={"chat_id":cid,"text":text,"parse_mode":"Markdown"}, timeout=8)
    except Exception: pass

def _send_open_alert(sym,sig,conf,score,entry,stop,tp1,tp2,qty,q1,q2,risk,bal):
    e="🟢" if sig=="BUY" else "🔴"; d=4 if entry<10 else 2
    sp=abs((stop-entry)/entry*100); t1=abs((tp1-entry)/entry*100); t2=abs((tp2-entry)/entry*100)
    _send(f"🤖 *DERIBIT TRADE*\n━━━━━━━━━━━━━━━━━━━━\n"
          f"{e} *{sig} — {sym}* ⭐×{score}\n🎯 {conf:.1f}% conf\n\n"
          f"⚡ Entry: `{entry:.{d}f}`\n"
          f"🛑 SL:    `{stop:.{d}f}` (-{sp:.1f}%)\n"
          f"🎯 TP1:   `{tp1:.{d}f}` (+{t1:.1f}%) × {q1}\n"
          f"🎯 TP2:   `{tp2:.{d}f}` (+{t2:.1f}%) × {q2}\n"
          f"📦 {qty} contracts · Real Risk: ${risk:.2f} · Bal: ${bal:.2f}\n"
          f"━━━━━━━━━━━━━━━━━━━━")


def run_execution_scan():
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    if not _acquire_scan_lock():
        return
    try:
        _run_execution_scan_locked()
    finally:
        _release_scan_lock()


def _run_execution_scan_locked():
    scan_started_at = datetime.now(timezone.utc).isoformat()
    save_json(SCAN_STATUS_FILE, {"phase": "started", "started_at": scan_started_at, "completed_at": None})

    run, mode, vol, reason = should_scan()
    if not run:
        log.info(f"  Scan skipped: {reason}")
        save_json(SCAN_STATUS_FILE, {"phase": "completed", "started_at": scan_started_at, "completed_at": datetime.now(timezone.utc).isoformat()})
        return

    deribit    = DeribitClient(os.getenv("DERIBIT_CLIENT_ID",""), os.getenv("DERIBIT_CLIENT_SECRET",""))
    deribit.test_connection()
    pipeline   = joblib.load(MODEL_FILE)
    thresholds = get_mode_thresholds(mode)
    risk_mult  = get_effective_risk(mode, vol)

    max_open_trades = int(getattr(config, "MAX_OPEN_TRADES", 8))
    max_same_dir    = int(getattr(config, "MAX_SAME_DIRECTION", 4))

    rec_buy  = float(pipeline.get("recommended_threshold_buy", pipeline.get("recommended_threshold", 0.40))) * 100.0
    rec_sell = float(pipeline.get("recommended_threshold_sell", pipeline.get("recommended_threshold", 0.45))) * 100.0

    log.info(f"  {mode['label']} | Active Targets: BUY≥{rec_buy:.1f}% SELL≥{rec_sell:.1f}% "
             f"| score≥{thresholds['min_score']} | ADX≥{thresholds['min_adx']} | risk:{risk_mult:.2f}")

    log.info("\n[0] Balance..."); balance = save_balance(deribit)
    log.info("\n[1] Monitor trades..."); check_open_trades(deribit)
    log.info("\n[2] Stale trade check..."); check_stale_trades(deribit)
    log.info("\n[3] Ghost trade recovery..."); clean_ghost_trades(deribit)
    log.info("\n[3b] Funding rate check..."); check_funding_rates(deribit)
    
    log.info("\n[3c] Reconciler: Auditing exchange positions vs local ledger...")
    trades = load_trades()
    tracked_symbols = set(trades.keys())

    orphan_tracker = load_json(ORPHAN_TRACKER_FILE, {})
    now_ts = time.time()
    active_orphans_this_scan = set()

    try:
        raw_positions = deribit.get_positions()
        fetch_succeeded = True
    except Exception as e:
        raw_positions = []
        fetch_succeeded = False
        log.warning(f"  [3c] could not fetch live positions: {e}")

    for p in raw_positions:
        inst = p.get("instrument_name", "")
        if not inst:
            continue
        base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
        sym  = f"{base}USDT"
        size = deribit.get_position_size(sym)

        if abs(size) <= 0.0001:
            continue

        if sym not in tracked_symbols:
            active_orphans_this_scan.add(sym)
            first_seen = orphan_tracker.get(sym)

            if first_seen is None:
                orphan_tracker[sym] = now_ts
                log.warning(f"  ⚠️ [RECONCILER] Candidate orphan detected: {sym} (size={size}) — awaiting confirmation on next scan cycle.")
                continue

            elapsed = now_ts - float(first_seen)
            if elapsed < ORPHAN_CONFIRM_SECONDS:
                log.info(f"  ⏳ [RECONCILER] Candidate {sym} pending confirmation ({elapsed:.0f}s / {ORPHAN_CONFIRM_SECONDS}s).")
                continue

            log.warning(f"  🚨 [RECONCILER] Confirmed orphan across multiple cycles: {sym} (size={size}, active {elapsed/60:.1f}m) — Flattening...")
            _cancel_all_open_orders_for_symbol(deribit, sym)
            
            flatten_side = "SELL" if size > 0 else "BUY"
            flatten_qty  = deribit.round_amount(sym, abs(size))
            
            try:
                if flatten_qty > 0:
                    deribit.place_market_order(symbol=sym, side=flatten_side, amount=flatten_qty, reduce_only=True)
                
                if _verify_actually_closed(deribit, sym):
                    orphan_tracker.pop(sym, None)
                    log.info(f"  ✓ [RECONCILER] Successfully flattened orphan position for {sym}")
                    _send(f"🛡️ *[RECONCILER] Orphan Flattened — {sym}*\nConfirmed across scans. Closed `{flatten_qty}` contracts via reduce_only.")
                else:
                    log.error(f"  🚨 [RECONCILER] Flatten failed to verify for {sym} — Check exchange!")
            except Exception as re_err:
                log.error(f"  [RECONCILER] Flatten failed: {re_err}")

    if fetch_succeeded:
        for tracked_cand in list(orphan_tracker.keys()):
            if tracked_cand not in active_orphans_this_scan:
                orphan_tracker.pop(tracked_cand, None)
        save_json(ORPHAN_TRACKER_FILE, orphan_tracker)
    else:
        log.warning("  [3c] Exchange fetch failed — preserving orphan_tracker state for next cycle.")
        
    save_balance(deribit)

    open_count = len([t for t in load_trades().values() if not t.get("closed",False)])
    log.info(f"\n[4] Scanning {len(SYMBOLS)} coins | Open:{open_count}/{max_open_trades} (Max Direction: {max_same_dir})")

    found = 0
    btc_momentum = check_btc_momentum()
    log.info(f"\n  BTC momentum: {btc_momentum['message']}")
    
    whale_flow = get_exchange_netflow("BTC")
    log.info(f"  Whale flow:   {whale_flow['message']}")

    fng_data = check_fear_and_greed()
    log.info(f"  Fear & Greed: {fng_data['message']}")
    
    vol_state = vol.get("status", "NORMAL")
    btc_df15_live = get_data("BTCUSDT", TIMEFRAME_ENTRY)

    for symbol in SYMBOLS:
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds, btc_momentum, whale_flow, fng_data, btc_df15_live=btc_df15_live)
        if sig is None: time.sleep(0.2); continue
        
        found += 1
        base_hurdle = rec_buy if sig["signal"] == "BUY" else rec_sell
        
        if execute_trade(deribit, sig, risk_mult, balance, vol_state, base_min_conf=base_hurdle):
            time.sleep(1.5)

    save_balance(deribit)
    log.info(f"\n{'═'*56}\nDONE — {found} signal(s) | ${balance:.2f}\n{'═'*56}")

    save_json(SCAN_STATUS_FILE, {
        "phase": "completed", 
        "started_at": scan_started_at, 
        "completed_at": datetime.now(timezone.utc).isoformat()
    })

if __name__ == "__main__":
    run_execution_scan()
