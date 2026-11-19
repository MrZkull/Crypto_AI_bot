# trade_executor.py — V4.2: Enforced EV Clamp, Code 11030 Prevention,
# Symmetrical State Guard, Prediction Ledger, and Adaptation Overrides

import os, json, time, logging, requests, joblib, base64, math, hashlib, uuid, hmac
import pandas as pd, numpy as np
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import config
from config import (
    SYMBOLS, ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    RISK_PER_TRADE, MODEL_FILE, LOG_FILE, get_tier,
    TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM, TIMEFRAME_TREND, LIVE_LIMIT
)
from deribit_client import (
    DeribitClient, TRADEABLE_SYMBOLS, OrderOutcomeUnknown, DeribitRPCError,
    ProtectiveOrderVerificationError,
)
from exchange_reliability import (
    InstrumentSpec, RiskCapital, RiskLimits, calculate_position_size,
    round_price_to_tick,
)
from daily_risk_control import evaluate_daily_risk_guard
from execution_state_persistence import (
    StatePersistenceError,
    canonical_json_bytes,
    read_json_mirrors,
    write_json_mirrors,
)
from exchange_environment_guard import (
    EnvironmentGuardError,
    validate_deribit_testnet_mutation_environment,
)
from feature_engineering import add_indicators
from legacy_production_features import add_legacy_production_indicators
from market_data_integrity import (
    sanitize_closed_candles,
    build_observation_contract,
    validate_observation_contract,
    observation_identity,
)
from smart_scheduler import (
    should_scan, get_mode_thresholds, get_effective_risk, check_correlation,
    check_btc_momentum, check_fear_and_greed
)
from whale_tracker import get_exchange_netflow

try:
    from candidate_identity import TRADE_IDENTITY_FIELDS
except ImportError:
    TRADE_IDENTITY_FIELDS = (
        "candidate_id", "model_sha256", "feature_schema_hash",
        "feature_code_hash", "execution_policy_hash", "config_hash"
    )

TRADES_FILE            = "trades.json"
HISTORY_FILE           = "trade_history.json"
SIGNALS_FILE           = "signals.json"
BALANCE_FILE           = "balance.json"
LOCK_FILE              = "scan_lock.json"
SCAN_STATUS_FILE       = "scan_status.json"

# Stage2E scanner liveness telemetry.
# This records whether the scanner actually ran, how many symbols
# reached ML scoring, and how many prediction records were persisted.
_STAGE2E_LIVENESS_V2 = True
_SCAN_LIVENESS = {
    "scan_ran": False,
    "symbols_attempted": 0,
    "symbols_scored": 0,
    "predictions_saved": 0,
    "features_zero_filled": 0,
    "feature_validation_rejects": 0,
    "skip_reason": None,
}

# The Testnet gateway lease is explicitly released at the end of every scan.
_ACTIVE_DERIBIT_CLIENT = None
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
try:
    from config import PROBATION_CONFIDENCE_OFFSET
except ImportError:
    PROBATION_CONFIDENCE_OFFSET = 10.0   
BENCH_MAX_COOLDOWN_DAYS          = 7      

ROLLING_WINDOW_TRADES            = 6
ROLLING_MIN_WIN_RATE             = 0.40   
ROLLING_MAX_NET_LOSS             = 0.0    

PREDICTIONS_FILE       = "predictions.json"
OVERRIDES_FILE         = "adaptation_overrides.json"
MAX_PREDICTIONS_KEPT   = 10000
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

def get_symbol_overrides(symbol: str) -> dict:
    overrides = load_json(OVERRIDES_FILE, {})
    return overrides.get(symbol, {}) if isinstance(overrides, dict) else {}

def compute_ensemble_disagreement(pipeline, x_selected_row: np.ndarray) -> dict:
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
            log.warning(f"⚠️ ensemble disagreement unwrap failed: {e}")
            compute_ensemble_disagreement._warned = True
        return {"ensemble_disagreement": None, "high_disagreement": False}

def _candidate_identity_fields() -> dict:
    """Return the active immutable candidate identity, if one is present.

    Missing manifest is represented by an empty dict so production/legacy
    prediction flows remain operational; when a manifest exists, its exact
    hashes are carried into prediction and trade records.
    """
    path = Path("candidate_manifest.json")
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        return {k: m[k] for k in TRADE_IDENTITY_FIELDS if m.get(k) not in (None, "")}
    except Exception as e:
        log.error(f"Candidate manifest unreadable; identity omitted: {e}")
        return {}


def _latest_contract_source(
    df: pd.DataFrame,
    *,
    interval: str,
    logical_symbol: str,
    observation_market: str,
    observation_source: dict,
    primary_close_time: int,
):
    # Return the latest completed source candle usable at the primary observation.
    if df is None or df.empty:
        return None
    if not {"open_time", "close_time"}.issubset(df.columns):
        return None

    eligible = df.copy()
    eligible["open_time"] = pd.to_numeric(
        eligible["open_time"], errors="coerce"
    )
    eligible["close_time"] = pd.to_numeric(
        eligible["close_time"], errors="coerce"
    )
    eligible = eligible.dropna(subset=["open_time", "close_time"])
    eligible = eligible[
        eligible["close_time"] <= int(primary_close_time)
    ].sort_values("close_time")

    if eligible.empty:
        return None

    source_row = eligible.iloc[-1]
    return {
        "logical_symbol": str(logical_symbol).upper(),
        "observation_market": str(observation_market).upper(),
        "observation_source": dict(observation_source),
        "interval": interval,
        "open_time": int(source_row["open_time"]),
        "close_time": int(source_row["close_time"]),
    }


def _build_signal_observation_contract(
    symbol: str,
    row: pd.Series,
    *,
    df1h_raw: pd.DataFrame,
    df4h_raw: pd.DataFrame,
    btc_df15_live: pd.DataFrame,
    retrieved_at_ms: int,
):
    # Build and validate the canonical contract for one production observation.
    primary_close_time = int(row["close_time"])
    logical_symbol = str(symbol).upper()
    observation_market = "BINANCE_SPOT"
    observation_source = {
        "exchange": "binance",
        "market_type": "spot",
    }

    htf_sources = {}

    source_1h = _latest_contract_source(
        df1h_raw,
        interval="1h",
        logical_symbol=logical_symbol,
        observation_market=observation_market,
        observation_source=observation_source,
        primary_close_time=primary_close_time,
    )
    if source_1h is not None:
        htf_sources["1h"] = source_1h

    source_4h = _latest_contract_source(
        df4h_raw,
        interval="4h",
        logical_symbol=logical_symbol,
        observation_market=observation_market,
        observation_source=observation_source,
        primary_close_time=primary_close_time,
    )
    if source_4h is not None:
        htf_sources["4h"] = source_4h

    btc_source = _latest_contract_source(
        btc_df15_live,
        interval="15m",
        logical_symbol="BTCUSDT",
        observation_market=observation_market,
        observation_source=observation_source,
        primary_close_time=primary_close_time,
    )

    contract = build_observation_contract(
        logical_symbol=logical_symbol,
        observation_market=observation_market,
        observation_source=observation_source,
        execution_instrument=None,
        interval=str(TIMEFRAME_ENTRY),
        open_time=int(row["open_time"]),
        close_time=primary_close_time,
        retrieved_at_ms=int(retrieved_at_ms),
        htf_sources=htf_sources,
        btc_source=btc_source,
    )
    validate_observation_contract(contract)
    return contract


def _pred_record(symbol, sig, conf, pipeline, row, disagreement, reject_reason=None, pred_id=None):
    observation_contract = row.get("_observation_contract")
    if observation_contract is None:
        if "close_time" not in row:
            raise ValueError("Prediction record requires observation close_time")
        observation_contract = build_observation_contract(
            logical_symbol=str(symbol).upper(),
            observation_market="BINANCE_SPOT",
            observation_source={
                "exchange": "binance",
                "market_type": "spot",
            },
            execution_instrument=None,
            interval=str(TIMEFRAME_ENTRY),
            open_time=int(row["open_time"]),
            close_time=int(row["close_time"]),
            retrieved_at_ms=int(time.time() * 1000),
        )
    validate_observation_contract(observation_contract)
    identity = list(observation_identity(observation_contract))

    entry = float(row.get("close", 0) or 0)
    atr   = float(row.get("atr", 0) or 0)
    stop  = (entry - atr*ATR_STOP_MULT) if sig == "BUY" else (entry + atr*ATR_STOP_MULT) if sig == "SELL" else None
    tp1   = (entry + atr*ATR_TARGET1_MULT) if sig == "BUY" else (entry - atr*ATR_TARGET1_MULT) if sig == "SELL" else None
    return {
        "pred_id":               pred_id or f"{symbol}_{int(time.time()*1000)}",
        **_candidate_identity_fields(),
        "symbol":                symbol,
        "logical_symbol":        str(symbol).upper(),
        "open_time":             int(row.get("open_time", 0)),
        "interval":              str(TIMEFRAME_ENTRY),
        "observation_source":    {
            "exchange": "binance",
            "market_type": "spot",
        },
        "observation_market":    "BINANCE_SPOT",
        "observation_contract":  observation_contract,
        "observation_identity":  identity,
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
        "ood_distance":          None,
        "ood_tier":              "PENDING",
        "model_version":         pipeline.get("trained_at"),
        "was_executed":          False,
        "reject_reason":         reject_reason,
        "lookahead_bars":        24,
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "audited":               False,
    }

def save_prediction(rec: dict):
    preds = load_json(PREDICTIONS_FILE, [])
    preds.append(rec)
    if _SCAN_LIVENESS.get("scan_ran"):
        _SCAN_LIVENESS["predictions_saved"] += 1
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

def get_required_confidence(symbol: str, current_baseline_conf: float = 52.0) -> float:
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

def _is_unreliable(symbol: str, signal_conf: float = 0.0, current_baseline_conf: float = 52.0) -> bool:
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
        log.warning(f"  🚫 {symbol}: Signal confidence ({signal_conf:.1f}%) < required threshold ({required_conf:.1f}%) — skip")
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
                log.info(f"  🎉 {symbol}: Achieved {PROBATION_WIN_GOAL} probation wins! Reinstated to Normal Mode.")
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
                log.warning(f"  🚫 {symbol}: {CONSECUTIVE_LOSS_BENCH_THRESHOLD} consecutive losses! Benched on Probation.")

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
        if rate is None or not math.isfinite(float(rate)):
            log.warning(f"  🚫 {symbol}: funding rate unavailable/invalid — FAIL-CLOSED, skip entry")
            return 0.0
        rate = float(rate)
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
    except Exception as e:
        log.warning(f"  🚫 {symbol}: funding check failed — FAIL-CLOSED, skip entry: {e}")
        return 0.0

def _wait_for_position(deribit, symbol: str, side: str, entry_oid: str, timeout: float = 8.0) -> bool:
    """Confirm a complete, side-correct position snapshot after the fill."""
    expected_sign = 1 if str(side).upper() == "BUY" else -1
    deadline = time.time() + timeout
    step = 0.5
    while time.time() < deadline:
        time.sleep(step)
        try:
            size = float(deribit.get_position_size(symbol))
            if not math.isfinite(size):
                raise ValueError("position quantity is non-finite")
        except Exception as exc:
            log.critical(
                "  %s: position verification failed (%s); refusing to infer exposure from order status",
                symbol,
                type(exc).__name__,
            )
            return False
        if abs(size) > 0 and (1 if size > 0 else -1) == expected_sign:
            log.info("  ✅ Side-correct position confirmed: %s size=%s", symbol, size)
            return True
        if abs(size) > 0 and (1 if size > 0 else -1) != expected_sign:
            log.critical("  %s: position direction conflicts with submitted entry side", symbol)
            return False
        step = min(step * 1.5, 2.0)
    log.critical("  %s: position not confirmed after %ss — FAIL-CLOSED", symbol, timeout)
    return False


def _trade_state_mirror_paths():
    """Return the legacy root/data mirror paths in their established order."""
    return (Path(TRADES_FILE), Path("data") / Path(TRADES_FILE).name)


def _persist_trade_snapshot(symbol: str, trade: dict) -> bool:
    """Strictly persist and verify a complete lifecycle snapshot before mutation.

    Unlike the generic UI/reporting JSON helper, this critical path refuses
    corrupt or divergent mirrors, flushes each temporary file before atomic
    replacement, and compares the complete read-back snapshot. A failed or
    partial mirror write returns False so the caller must not submit/cancel.
    """
    try:
        paths = _trade_state_mirror_paths()
        current = read_json_mirrors(paths, default={})
        if not isinstance(current, dict):
            raise StatePersistenceError("Trade state root must be a JSON object")
        current[str(symbol)] = dict(trade)
        write_json_mirrors(paths, current)
        persisted = read_json_mirrors(paths, default=None)
        expected_trade = json.loads(canonical_json_bytes(dict(trade)).decode("utf-8"))
        return isinstance(persisted, dict) and persisted.get(str(symbol)) == expected_trade
    except Exception as exc:
        log.critical("  %s: lifecycle marker persistence failed (%s)", symbol, type(exc).__name__)
        return False


def _order_label_matches(deribit, symbol: str, label: str) -> list:
    """Query Deribit label endpoints without submitting/cancelling anything."""
    instrument = deribit.get_instrument_name(symbol)
    info = deribit.get_instrument_info(symbol)
    if not isinstance(info, dict) or not info:
        raise ValueError("instrument metadata unavailable for label reconciliation")
    currency = str(info.get("base_currency", "")).strip().upper()
    if not currency:
        raise ValueError("instrument base currency missing for label reconciliation")
    states = deribit.get_order_state_by_label(currency, label)
    opens = deribit.get_open_orders_by_label(currency, label)
    rows = list(states) + list(opens)
    matches = [
        row for row in rows
        if str(row.get("label", "")) == label
        and str(row.get("instrument_name", "")) == instrument
    ]
    unique = {}
    for row in matches:
        oid = str(row.get("order_id", "")).strip()
        if oid:
            unique[oid] = row
    combined = list(unique.values())
    if getattr(deribit, "require_execution_gateway", False):
        deribit.reconcile_execution_order_label(label, combined, queries_complete=True)
    return combined


def _verify_reduce_only_limit_ack(deribit, symbol: str, side: str, qty, price: float, result: dict, client_label: str = None) -> dict:
    """Independently verify a reduce-only TP acknowledgement against exchange state."""
    immediate = result.get("order", result) if isinstance(result, dict) else None
    order_id = str(immediate.get("order_id", "")).strip() if isinstance(immediate, dict) else ""
    if not order_id:
        raise OrderOutcomeUnknown("TP acknowledgement omitted exchange order ID")
    try:
        observed = deribit.get_order(order_id)
    except Exception as exc:
        raise OrderOutcomeUnknown(
            f"TP order {order_id} state could not be independently read ({type(exc).__name__})"
        ) from None
    if not isinstance(observed, dict) or str(observed.get("order_id", "")) != order_id:
        raise OrderOutcomeUnknown("TP order-state identity could not be verified")
    instrument = str(deribit.get_instrument_name(symbol))
    try:
        actual_amount = float(observed.get("amount"))
        actual_price = float(observed.get("price"))
        expected_amount = float(deribit.round_amount(symbol, qty))
        expected_price = float(deribit.round_price(symbol, price))
        tick = float(deribit.get_tick_size(symbol))
    except Exception:
        raise ProtectiveOrderVerificationError("TP amount/price precision could not be verified") from None
    state = str(observed.get("order_state", "")).lower()
    valid = (
        str(observed.get("instrument_name", "")) == instrument
        and str(observed.get("order_type", "")).lower() == "limit"
        and str(observed.get("direction", "")).lower() == str(side).lower()
        and observed.get("reduce_only") is True
        and math.isfinite(actual_amount)
        and math.isclose(actual_amount, expected_amount, rel_tol=1e-8, abs_tol=max(1e-12, expected_amount * 1e-8))
        and math.isfinite(actual_price)
        and math.isfinite(tick) and tick > 0
        and abs(actual_price - expected_price) <= tick * 0.5
        and state in {"open", "untriggered", "filled"}
        and (client_label is None or str(observed.get("label", client_label)) == client_label)
    )
    if not valid:
        raise ProtectiveOrderVerificationError(
            f"TP order {order_id} failed exchange contract verification (state={state or 'missing'})"
        )
    return observed


def _place_tp_with_fallback(deribit, symbol: str, side: str, qty, price: float, label: str, trade: dict, key: str, dec: int) -> str:
    states = trade.setdefault("order_submission_states", {})
    labels = trade.setdefault("order_client_labels", {})
    prior_submission = states.get(key)
    if trade.get("emergency_recovery_required") and prior_submission not in {"SUBMITTING", "UNKNOWN", "REJECTED"}:
        return ""
    if trade.get("manual_reconciliation_required") and prior_submission not in {"SUBMITTING", "UNKNOWN", "REJECTED"}:
        return ""
    client_label = labels.get(key) or _deterministic_order_label(trade, symbol, key.upper())
    labels[key] = client_label
    if states.get(key) in {"SUBMITTING", "UNKNOWN", "REJECTED"}:
        try:
            matches = _order_label_matches(deribit, symbol, client_label)
        except Exception as exc:
            matches = []
            trade["manual_reconciliation_required"] = True
            trade["protection_reconciliation_required"] = True
            trade["protection_reconciliation_reason"] = f"{key} label query unavailable ({type(exc).__name__})"
            states[key] = "UNKNOWN"
            _persist_trade_snapshot(symbol, trade)
            return ""
        if len(matches) == 1:
            match = matches[0]
            oid = str(match.get("order_id", "")).strip()
            order_state = str(match.get("order_state", "")).lower()
            try:
                matched_amount = float(match.get("amount"))
                matched_price = float(match.get("price"))
                expected_price = float(deribit.round_price(symbol, price))
                expected_amount = float(deribit.round_amount(symbol, qty))
                amount_ok = math.isfinite(matched_amount) and math.isclose(
                    matched_amount, expected_amount, rel_tol=1e-8,
                    abs_tol=max(1e-12, expected_amount * 1e-8),
                )
                tick = float(deribit.get_tick_size(symbol))
                price_ok = math.isfinite(matched_price) and tick > 0 and abs(matched_price - expected_price) <= tick * 0.5
            except Exception:
                amount_ok = price_ok = False
            contract_ok = bool(
                oid
                and str(match.get("order_type", "")).lower() == "limit"
                and match.get("reduce_only") is True
                and str(match.get("direction", "")).lower() == str(side).lower()
                and amount_ok
                and price_ok
                and str(match.get("label", "")) == client_label
            )
            if contract_ok and order_state in {"open", "untriggered", "triggered", "filled"}:
                states[key] = "RECONCILED"
                trade.setdefault("order_ids", {})[key] = oid
                other_pending = any(
                    name != key and status in {"SUBMITTING", "UNKNOWN", "REJECTED"}
                    for name, status in states.items()
                )
                if not other_pending and trade.get("stop_loss_submission_state") not in {"SUBMITTING", "UNKNOWN", "REJECTED"}:
                    trade.pop("manual_reconciliation_required", None)
                    trade.pop("protection_reconciliation_required", None)
                    trade.pop("protection_reconciliation_reason", None)
                _persist_trade_snapshot(symbol, trade)
                return oid
            if contract_ok and order_state in {"cancelled", "canceled", "rejected", "closed", "expired"}:
                # A terminal order is evidence of its identity, not permission
                # to create another order with the same intent automatically.
                states[key] = "REJECTED"
                trade.setdefault("order_ids", {})[key] = oid
                trade["manual_reconciliation_required"] = True
                trade["protection_reconciliation_required"] = True
                trade["protection_reconciliation_reason"] = f"{key} matched a terminal order ({order_state}); inspect fills and remaining exposure"
                _persist_trade_snapshot(symbol, trade)
                return ""
        trade["manual_reconciliation_required"] = True
        trade["protection_reconciliation_required"] = True
        trade["protection_reconciliation_reason"] = f"{key} prior submission remains unmatched or ambiguous"
        states[key] = "UNKNOWN"
        _persist_trade_snapshot(symbol, trade)
        return ""
    try:
        actual_pos = abs(deribit.get_position_size(symbol))
        min_lot = deribit.get_min_trade_amount(symbol)
        if actual_pos < min_lot:
            log.warning(f"  {label} {symbol}: actual position {actual_pos} < min lot {min_lot} — skip")
            return ""

        # ── 1. ADOPT EXISTING RESTING ORDER (Prevent Duplicate 11030) ──
        open_orders = deribit.get_open_orders(symbol)
        target_side = side.lower()
        tick = deribit.get_tick_size(symbol)

        existing_reduce_qty = 0.0
        expected_instrument = deribit.get_instrument_name(symbol)
        for o in open_orders:
            if not isinstance(o, dict):
                continue
            o_side = str(o.get("direction", "")).lower()
            o_type = str(o.get("order_type", "")).lower()
            o_instrument = str(o.get("instrument_name", ""))
            if (
                o_instrument != expected_instrument
                or o.get("reduce_only") is not True
                or o_type != "limit"
                or o_side != target_side
            ):
                continue
            try:
                o_price = float(o.get("price", 0) or 0)
                o_trig = float(o.get("trigger_price", 0) or o.get("stop_price", 0) or 0)
                o_qty = float(o.get("amount", 0) or 0) - float(o.get("filled_amount", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) for value in (o_price, o_trig, o_qty)) or o_qty <= 0:
                continue
            # Adopt only an exact-identity, reduce-only limit order.
            if o_trig <= 0:
                if abs(o_price - price) <= (tick * 1.5):
                    oid = str(o.get("order_id", "")).strip()
                    if oid:
                        log.info(f"  ✓ Adopted existing {label} {symbol} reduce-only order @ {o_price:.{dec}f} id:{oid}")
                        return oid
                existing_reduce_qty += o_qty

        # ── 2. CLAMP TO REMAINING UNRESERVED CAPACITY ──
        unreserved_pos = max(0.0, actual_pos - existing_reduce_qty)
        if unreserved_pos < min_lot:
            log.warning(f"  {label} {symbol}: unreserved capacity ({unreserved_pos:.4f}) < min lot ({min_lot}) (reserved: {existing_reduce_qty:.4f}/{actual_pos:.4f}) — skip")
            return ""

        qty = min(qty, unreserved_pos)
        qty = deribit.round_amount(symbol, qty)
        if qty < min_lot:
            log.warning(f"  {label} {symbol}: qty {qty} < min lot {min_lot} — skip")
            return ""

    except Exception as e:
        log.warning(
            "  %s pre-check %s failed (%s); refusing replacement order",
            label,
            symbol,
            type(e).__name__,
        )
        return ""

    # ── 3. PERSIST ONE ATTEMPT, THEN SUBMIT EXACTLY ONCE ──
    states[key] = "SUBMITTING"
    trade.setdefault("order_ids", {})[f"{key}_label"] = client_label
    if not _persist_trade_snapshot(symbol, trade):
        states[key] = "UNKNOWN"
        trade["manual_reconciliation_required"] = True
        trade["protection_reconciliation_required"] = True
        trade["protection_reconciliation_reason"] = f"{key} pre-submit intent persistence failed"
        _persist_trade_snapshot(symbol, trade)
        return ""
    try:
        res = deribit.place_limit_order(
            symbol, side, qty, price, use_reduce_only=True, client_label=client_label
        )
        verified_order = _verify_reduce_only_limit_ack(
            deribit, symbol, side, qty, price, res, client_label
        )
        oid = str(verified_order.get("order_id", "")).strip()
        trade.setdefault("order_ids", {})[key] = oid
        states[key] = "ACKNOWLEDGED"
        if not _persist_trade_snapshot(symbol, trade):
            trade["manual_reconciliation_required"] = True
            trade["protection_reconciliation_required"] = True
            trade["protection_reconciliation_reason"] = f"{key} acknowledgement persistence failed"
            _persist_trade_snapshot(symbol, trade)
            return ""
        log.info(f"  🛠 Re-placed {label} {symbol} @ {price:.{dec}f} × {qty} id:{oid}")
        return oid
    except OrderOutcomeUnknown as exc:
        states[key] = "UNKNOWN"
        trade["manual_reconciliation_required"] = True
        trade["protection_reconciliation_required"] = True
        trade["protection_reconciliation_reason"] = f"{key} outcome unknown ({type(exc).__name__})"
        _persist_trade_snapshot(symbol, trade)
        log.critical("  %s %s outcome UNKNOWN; no retry; manual reconciliation required", label, symbol)
    except Exception as exc:
        states[key] = "REJECTED"
        trade["manual_reconciliation_required"] = True
        trade["protection_reconciliation_required"] = True
        trade["protection_reconciliation_reason"] = f"{key} submission rejected/failed ({type(exc).__name__})"
        _persist_trade_snapshot(symbol, trade)
        log.warning("  %s re-place %s failed; same parameters will not be retried (%s)", label, symbol, type(exc).__name__)
    return ""
    
def _cancel_all_open_orders_for_symbol(deribit: DeribitClient, symbol: str) -> bool:
    """Cancel and then independently verify that no open orders remain.

    A read/cancellation error is not equivalent to an empty order book.
    """
    try:
        orders = deribit.get_open_orders(symbol)
        if not isinstance(orders, list):
            log.error("  Open-order snapshot for %s is malformed; cleanup unverified", symbol)
            return False
    except Exception as exc:
        log.error(
            "  Open-order snapshot for %s unavailable (%s); cleanup unverified",
            symbol,
            type(exc).__name__,
        )
        return False

    if orders:
        log.info(
            "  🧹 Found %s active open order(s) on Deribit for %s — cancelling all...",
            len(orders), symbol,
        )
    for order in orders:
        if not isinstance(order, dict):
            log.error("  Malformed open-order row for %s; cleanup unverified", symbol)
            continue
        order_id = str(order.get("order_id", "")).strip()
        if not order_id:
            log.error("  Open order without an ID for %s; cleanup unverified", symbol)
            continue
        try:
            deribit.cancel_order(order_id)
            log.info("    ✓ Cancellation requested for exchange order %s", order_id)
        except Exception as exc:
            log.warning(
                "    Cancellation for %s failed (%s); checking remaining orders",
                order_id,
                type(exc).__name__,
            )

    try:
        remaining = deribit.get_open_orders(symbol)
    except Exception as exc:
        log.error(
            "  Cannot verify remaining open orders for %s (%s)",
            symbol,
            type(exc).__name__,
        )
        return False

    if not isinstance(remaining, list):
        log.error("  Remaining open-order snapshot for %s is malformed", symbol)
        return False
    if remaining:
        log.critical(
            "  %s: %s open order(s) remain after cleanup; retaining trade tracking",
            symbol,
            len(remaining),
        )
        return False
    return True

def _get_safe_close_info(deribit: DeribitClient, symbol: str, trade: dict) -> tuple:
    for attempt in range(2):
        try:
            actual_pos = deribit.get_position_size(symbol)
            close_side = "SELL" if actual_pos > 0 else "BUY"
            if abs(actual_pos) <= 1e-8:
                # A verified flat result is different from a failed position read.
                return close_side, 0.0
            close_qty = deribit.round_amount(symbol, abs(actual_pos))
            if close_qty <= 0:
                return close_side, 0.0
            return close_side, close_qty
        except Exception as exc:
            log.warning(
                "  _get_safe_close_info %s attempt %s failed (%s)",
                symbol,
                attempt + 1,
                type(exc).__name__,
            )
            if attempt == 0:
                time.sleep(0.3)

    raise RuntimeError(
        f"Cannot verify Deribit position for {symbol}; refusing tracked-quantity close fallback"
    ) from None

def _verify_actually_closed(deribit: DeribitClient, symbol: str, tolerance: float = 1e-8) -> bool:
    try:
        remaining = abs(deribit.get_position_size(symbol))
        return remaining <= tolerance
    except Exception as e:
        log.warning(f"  _verify_actually_closed {symbol}: could not check ({e}) — assuming NOT closed")
        return False

def _record_durable_account_risk_snapshot(deribit, snapshot: dict, observed_at: float) -> None:
    """Persist an immutable USDC equity checkpoint through the fenced execution ledger."""
    if not getattr(deribit, "require_execution_gateway", False):
        return
    gateway = getattr(deribit, "execution_gateway", None)
    coordinator = getattr(gateway, "coordinator", None) if gateway is not None else None
    ledger = getattr(gateway, "ledger", None) if gateway is not None else None
    recorder = getattr(ledger, "record_account_risk_snapshot", None)
    window_reader = getattr(ledger, "first_account_risk_snapshot_between", None)
    if coordinator is None or not callable(recorder) or not callable(window_reader):
        raise RuntimeError("Durable fenced account-risk storage is unavailable")
    stamp = float(observed_at)
    observed_utc = datetime.fromtimestamp(stamp, timezone.utc)
    day_start = datetime(
        observed_utc.year, observed_utc.month, observed_utc.day, tzinfo=timezone.utc
    ).timestamp()
    baseline_deadline = day_start + (20 * 60)
    # Persist only the first checkpoint in the UTC opening window. The risk
    # ledger remains a compact daily baseline store, not an unbounded copy of
    # every per-symbol account refresh.
    if stamp < day_start or stamp > baseline_deadline:
        return
    existing = window_reader(
        exchange=coordinator.exchange, environment=coordinator.environment,
        account_id=coordinator.account_id, currency="USDC",
        start=day_start, end=baseline_deadline,
    )
    if existing is not None:
        return
    renew = getattr(gateway, "_renew", None)
    if callable(renew):
        renew()
    fence_factory = getattr(coordinator, "_lease_kwargs", None)
    fence = fence_factory() if callable(fence_factory) else {}
    recorder(
        exchange=coordinator.exchange,
        environment=coordinator.environment,
        account_id=coordinator.account_id,
        currency="USDC",
        equity=snapshot.get("risk_equity"),
        observed_at=stamp,
        source=str(snapshot.get("source", "private/get_account_summaries")),
        **fence,
    )


def save_balance(deribit: DeribitClient) -> float:
    """Persist a validated account snapshot and a fenced USDC risk checkpoint."""
    from risk_accounting import build_balance_payload, unavailable_balance_payload
    try:
        snapshot = deribit.get_risk_capital_snapshot()
        positions = deribit.get_positions()
        local_observed_at = time.time()
        observed_at = local_observed_at
        if getattr(deribit, "require_execution_gateway", False):
            server_time_ms = deribit.get_server_time_ms()
            local_time_ms = int(local_observed_at * 1000)
            if abs(int(server_time_ms) - local_time_ms) > int(
                getattr(config, "MAX_DERIBIT_CLOCK_SKEW_MS", 5000)
            ):
                raise RuntimeError("Deribit server clock skew exceeds the configured safety limit")
            observed_at = int(server_time_ms) / 1000.0
        payload, risk_equity = build_balance_payload(
            snapshot, positions,
            updated_at=datetime.fromtimestamp(observed_at, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        if snapshot.get("entry_enabled") is True:
            _record_durable_account_risk_snapshot(deribit, snapshot, observed_at)
        save_json(BALANCE_FILE, payload)
        snapshot_copy = dict(snapshot)
        snapshot_copy["_observed_at_epoch"] = observed_at
        # Freshness is measured with a monotonic clock so a small positive
        # Deribit server-time offset does not create a negative snapshot age.
        snapshot_copy["_observed_at_monotonic"] = time.monotonic()
        deribit._last_risk_capital_snapshot = snapshot_copy
        deribit._last_risk_positions_snapshot = [dict(item) for item in positions]
        if snapshot.get("entry_enabled") is not True:
            log.warning("New entries disabled by account risk policy: %s", snapshot.get("risk_block_reason"))
            return 0.0
        log.info(
            "Risk equity: %.2f USDC | available: %.2f USDC | positions: %d",
            risk_equity, payload["available_funds"], len(positions),
        )
        return risk_equity
    except Exception as exc:
        try:
            deribit._last_risk_capital_snapshot = {
                "entry_enabled": False,
                "risk_equity": None,
                "risk_currency": "USDC",
                "available_funds": None,
                "margin_mode": "unknown",
                "risk_block_reason": "ACCOUNT_SNAPSHOT_UNAVAILABLE",
            }
            deribit._last_risk_positions_snapshot = None
            save_json(BALANCE_FILE, unavailable_balance_payload())
        except Exception:
            pass
        log.error("save_balance failed; new-entry risk capital unavailable (%s)", type(exc).__name__)
        return 0.0


def _verified_open_portfolio_risk_usdc(deribit, trades: dict, positions: list[dict]) -> float:
    """Sum conservative open stop-risk only when exchange/local state agrees.

    Exchange positions are authoritative for live quantity. Every tradable
    position must map to a local record with a verified stop acknowledgement.
    The stored initial risk is intentionally retained after partial exits until
    management persists a fresh risk amount, preventing accidental risk release.
    """
    from deribit_client import SYMBOL_MAP

    if not isinstance(trades, dict) or not isinstance(positions, list):
        raise ValueError("Portfolio state snapshots are malformed")
    symbol_by_instrument = {
        str(value.get("instrument", "")): symbol
        for symbol, value in SYMBOL_MAP.items()
        if isinstance(value, dict) and value.get("instrument")
    }
    active_trades = {
        str(symbol).upper(): item
        for symbol, item in trades.items()
        if isinstance(item, dict) and not item.get("closed", False)
    }
    dust_records = {
        str(symbol).upper(): item
        for symbol, item in trades.items()
        if isinstance(item, dict) and isinstance(item.get("dust_residual"), dict)
    }
    from capital_feasibility import resolve_active_capital_tier_policy
    tier_policy = resolve_active_capital_tier_policy(config, environment="testnet")
    dust_reserve_total = 0.0
    live_positions: dict[str, dict] = {}
    for position in positions:
        if not isinstance(position, dict):
            raise ValueError("Exchange position row is malformed")
        instrument = str(position.get("instrument_name", ""))
        symbol = symbol_by_instrument.get(instrument)
        try:
            size = float(position.get("size_currency"))
        except (TypeError, ValueError, OverflowError):
            if position.get("direction") == "zero" and float(position.get("size", 0) or 0) == 0:
                size = 0.0
            else:
                raise ValueError("Exchange position quantity is invalid") from None
        if not math.isfinite(size) or size < 0:
            raise ValueError("Exchange position quantity is invalid")
        if size == 0:
            continue
        if symbol is None:
            raise ValueError("Unmapped non-flat exchange position blocks new entries")
        if symbol in live_positions:
            raise ValueError(f"Duplicate live exchange position for {symbol}")
        live_positions[symbol] = position

    total_risk = 0.0
    for symbol, position in live_positions.items():
        trade = active_trades.get(symbol)
        if trade is None:
            dust_trade = dust_records.get(symbol)
            if dust_trade is None:
                raise ValueError(f"Untracked non-zero exchange position blocks new entries: {symbol}")
            minimum = float(deribit.get_min_trade_amount(symbol))
            live_signed = float(deribit.get_position_size(symbol))
            mark_price = float(deribit.get_mark_price(symbol))
            open_orders = deribit.get_open_orders(symbol)
            from dust_residual_policy import validate_verified_dust_residual
            dust_validation = validate_verified_dust_residual(
                dust_trade,
                signed_position=live_signed,
                min_trade_amount=minimum,
                mark_price=mark_price,
                open_orders=open_orders,
                max_position_notional_usdc=tier_policy["max_dust_position_notional_usdc"],
                reserve_per_position_usdc=tier_policy["dust_reserve_per_position_usdc"],
            )
            dust_reserve_total += float(dust_validation["reserve_usdc"])
            if dust_reserve_total > float(tier_policy["max_total_dust_reserve_usdc"]):
                raise ValueError("Combined verified dust reserve exceeds the active capital-tier ceiling")
            total_risk += float(dust_validation["reserve_usdc"])
            continue
        if str(trade.get("risk_currency", "")).upper() != "USDC":
            raise ValueError(f"Open trade has unknown risk currency: {symbol}")
        if str(trade.get("total_risk_currency", "")).upper() != "USDC":
            raise ValueError(f"Open trade has unknown total-risk currency: {symbol}")
        signal = str(trade.get("signal", "")).upper()
        direction = str(position.get("direction", "")).lower()
        if signal not in {"BUY", "SELL"} or direction != ("buy" if signal == "BUY" else "sell"):
            raise ValueError(f"Local/exchange position direction mismatch: {symbol}")
        order_ids = trade.get("order_ids")
        stop_id = order_ids.get("stop_loss") if isinstance(order_ids, dict) else None
        if not stop_id or trade.get("stop_loss_submission_state") != "ACKNOWLEDGED":
            raise ValueError(f"Protective stop is not verified in local state: {symbol}")
        # Re-read the protective order at admission time. A historical local
        # ACK is not proof that the stop is still resting on the exchange.
        stop_order = deribit.get_order(str(stop_id))
        if not isinstance(stop_order, dict):
            raise ValueError(f"Protective stop read is malformed: {symbol}")
        expected_instrument = str(position.get("instrument_name") or "")
        expected_stop_side = "SELL" if signal == "BUY" else "BUY"
        try:
            stop_amount = float(stop_order.get("amount"))
            stop_trigger = float(stop_order.get("trigger_price"))
            live_quantity = float(position.get("size_currency"))
            local_stop = float(trade.get("stop"))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Protective stop fields are missing: {symbol}") from None
        stop_state = str(stop_order.get("order_state") or "").strip().lower()
        if (
            str(stop_order.get("order_id") or "") != str(stop_id)
            or str(stop_order.get("instrument_name") or "") != expected_instrument
            or str(stop_order.get("order_type") or "").lower() != "stop_market"
            or str(stop_order.get("direction") or "").upper() != expected_stop_side
            or str(stop_order.get("trigger") or "").lower() != "mark_price"
            or stop_order.get("reduce_only") is not True
            or stop_state not in {"open", "untriggered"}
            or not all(math.isfinite(value) and value > 0 for value in (stop_amount, stop_trigger, live_quantity, local_stop))
            # A reduce-only stop may be larger than the live quantity, but it
            # must never be smaller: even a sub-lot uncovered remainder is risk.
            or stop_amount + 1e-10 < live_quantity
            or not math.isclose(stop_trigger, local_stop, rel_tol=1e-8, abs_tol=1e-10)
        ):
            raise ValueError(f"Protective stop does not match current live position: {symbol}")
        try:
            risk = float(trade.get("total_risk_with_fees"))
            local_qty = float(trade.get("qty"))
            live_qty = float(position.get("size_currency"))
            step = float(deribit.get_min_trade_amount(symbol))
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"Open trade risk or quantity is missing: {symbol}") from None
        if (
            not all(math.isfinite(value) for value in (risk, local_qty, live_qty, step))
            or risk <= 0 or local_qty <= 0 or live_qty <= 0 or step <= 0
            or abs(local_qty - live_qty) > max(step * 0.51, 1e-10)
        ):
            raise ValueError(f"Local/exchange quantity or risk mismatch: {symbol}")
        total_risk += risk

    for symbol in active_trades:
        if symbol not in live_positions:
            raise ValueError(f"Local open trade has no verified exchange position: {symbol}")
    if not math.isfinite(total_risk):
        raise ValueError("Aggregate portfolio risk is non-finite")
    return total_risk


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
                    "close_time": [int(x) + int(res) * 60 * 1000 - 1 for x in ticks],
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

def get_data(
    symbol: str,
    interval: str,
    canonical_observation: bool = False,
) -> pd.DataFrame:
    """Fetch market data with explicit observation-source semantics.

    canonical_observation=True permits Binance Spot only and fails closed
    rather than silently switching the observation source to Deribit.

    Existing non-canonical callers retain the historical fallback behavior.
    """
    spot_urls = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://binance.com/api/v3/klines",
    ]

    urls = list(spot_urls)

    if not canonical_observation:
        urls.append("https://fapi.binance.com/fapi/v1/klines")

    for url in urls:
        try:
            r = requests.get(
                url,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": LIVE_LIMIT,
                },
                timeout=8,
            )

            if r.status_code != 200:
                continue

            raw = r.json()

            if not raw or not isinstance(raw, list):
                continue

            df = pd.DataFrame(raw)

            cols = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_vol",
                "trades",
                "taker_buy_base_vol",
                "taker_buy_quote_vol",
                "ignore",
            ]

            df.columns = cols[:df.shape[1]]

            for c in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "taker_buy_base_vol",
            ]:
                if c in df.columns:
                    df[c] = pd.to_numeric(
                        df[c],
                        errors="coerce",
                    )

            keep = [
                c
                for c in [
                    "open_time",
                    "close_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "taker_buy_base_vol",
                ]
                if c in df.columns
            ]

            out = df[keep].copy()

            # Provenance is metadata only at this stage.
            out.attrs["logical_symbol"] = str(symbol).upper()
            out.attrs["observation_market"] = "BINANCE_SPOT"
            out.attrs["observation_source"] = {
                "exchange": "binance",
                "market_type": "spot",
            }

            return out

        except Exception:
            continue

    # Canonical signal observations must fail closed.
    if canonical_observation:
        return pd.DataFrame()

    # Preserve the existing Deribit fallback for non-canonical callers.
    df_deribit = _fetch_deribit_klines_live(
        symbol,
        interval,
        limit=LIVE_LIMIT,
    )

    if not df_deribit.empty:
        df_deribit = df_deribit.copy()

        df_deribit.attrs["logical_symbol"] = str(symbol).upper()
        df_deribit.attrs["observation_market"] = (
            "DERIBIT_LINEAR_USDC_PERPETUAL"
        )
        df_deribit.attrs["observation_source"] = {
            "exchange": "deribit",
            "market_type": "linear_usdc_perpetual",
        }

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

def _aggregate_completed_1h_to_4h(df1h, observation_time_ms=None):
    """Derive completed UTC 4H candles from completed Binance Spot 1H candles."""
    required = {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    if df1h is None or df1h.empty:
        return pd.DataFrame()

    if not required.issubset(df1h.columns):
        return pd.DataFrame()

    hour_ms = 60 * 60 * 1000
    four_hour_ms = 4 * hour_ms

    work = df1h.copy()

    try:
        work["open_time"] = pd.to_numeric(
            work["open_time"], errors="raise"
        ).astype("int64")
        work["close_time"] = pd.to_numeric(
            work["close_time"], errors="raise"
        ).astype("int64")

        for col in ["open", "high", "low", "close", "volume"]:
            work[col] = pd.to_numeric(work[col], errors="raise")
    except Exception:
        return pd.DataFrame()

    if observation_time_ms is not None:
        try:
            observation_time_ms = int(observation_time_ms)
        except Exception:
            return pd.DataFrame()

        work = work[
            work["close_time"] <= observation_time_ms
        ].copy()

    if work.empty:
        return pd.DataFrame()

    expected_close = work["open_time"] + hour_ms - 1

    if not (work["close_time"] == expected_close).all():
        return pd.DataFrame()

    if work["open_time"].duplicated().any():
        return pd.DataFrame()

    if not np.isfinite(
        work[["open", "high", "low", "close", "volume"]]
        .to_numpy(dtype=float)
    ).all():
        return pd.DataFrame()

    work = work.sort_values("open_time").reset_index(drop=True)
    work["_bucket"] = work["open_time"] // four_hour_ms

    rows = []

    for _, group in work.groupby("_bucket", sort=True):
        group = group.sort_values("open_time").reset_index(drop=True)

        if len(group) != 4:
            continue

        first_open = int(group.loc[0, "open_time"])

        expected_times = [
            first_open + i * hour_ms
            for i in range(4)
        ]

        if group["open_time"].astype("int64").tolist() != expected_times:
            continue

        # Only accept UTC-aligned 4H buckets.
        if first_open % four_hour_ms != 0:
            continue

        row = {
            "open_time": first_open,
            "close_time": first_open + four_hour_ms - 1,
            "open": float(group.loc[0, "open"]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group.loc[3, "close"]),
            "volume": float(group["volume"].sum()),
        }

        if "taker_buy_base_vol" in group.columns:
            row["taker_buy_base_vol"] = float(
                pd.to_numeric(
                    group["taker_buy_base_vol"],
                    errors="coerce",
                ).sum()
            )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    out.attrs["logical_symbol"] = work.attrs.get(
        "logical_symbol", ""
    )
    out.attrs["observation_market"] = work.attrs.get(
        "observation_market",
        "BINANCE_SPOT",
    )
    out.attrs["observation_source"] = work.attrs.get(
        "observation_source",
        {
            "exchange": "binance",
            "market_type": "spot",
        },
    )
    out.attrs["derived_from_interval"] = "1h"
    out.attrs["derivation"] = (
        "completed Binance Spot 1h -> UTC-aligned 4h"
    )

    return out.sort_values("open_time").reset_index(drop=True)

def _required_production_features(pipeline):
    """Resolve the model's explicit feature contract without inference."""
    selector = pipeline.get("selector")

    selected = getattr(selector, "selected_features", None)
    if selected:
        features = list(selected)
        source = "selector.selected_features"
    else:
        legacy = getattr(selector, "feature_names", None)
        if legacy:
            features = list(legacy)
            source = "selector.feature_names"
        else:
            artifact = pipeline.get("best_features")
            if artifact:
                features = list(artifact)
                source = "pipeline.best_features"
            else:
                features = list(pipeline.get("all_features") or [])
                source = "pipeline.all_features"

    if not features:
        raise RuntimeError(
            "Production model exposes no usable feature schema"
        )

    if len(features) != len(set(features)):
        raise RuntimeError(
            f"Production model feature schema contains duplicates: {features}"
        )

    artifact = pipeline.get("best_features")
    if selected and artifact and list(artifact) != features:
        raise RuntimeError(
            "Production model selected_features disagree with best_features"
        )

    if legacy and artifact and list(artifact) != features:
        raise RuntimeError(
            "Production model feature_names disagree with best_features"
        )

    strict = bool(
        selected
        or getattr(selector, "feature_names", None)
        or artifact
    )

    return features, source, strict


def _validate_production_feature_row(row, required_features):
    """Return missing/invalid explicit model features; never repair them."""
    missing = [
        feature
        for feature in required_features
        if feature not in row.index
    ]

    invalid = []

    for feature in required_features:
        if feature in missing:
            continue

        try:
            value = float(row[feature])
        except (TypeError, ValueError, OverflowError):
            invalid.append(feature)
            continue

        if not math.isfinite(value):
            invalid.append(feature)

    return missing, invalid


def generate_signal(symbol, pipeline, thresholds, btc_momentum=None, whale_flow=None, fng_data=None, btc_df15_live=None):
    try:
        raw15 = get_data(
            symbol, TIMEFRAME_ENTRY, canonical_observation=True
        )
        if raw15 is None or raw15.empty or len(raw15) < 30:
            log.warning(f"    [{symbol}] Insufficient/missing 15m candle data — skip")
            return None

        if "open_time" not in raw15.columns:
            if "ticks" in raw15.columns: raw15["open_time"] = raw15["ticks"]
            elif "timestamp" in raw15.columns: raw15["open_time"] = raw15["timestamp"]
            else:
                log.warning(f"    [{symbol}] Missing open_time column — skip")
                return None

        now_ms = int(time.time() * 1000)

        if "close_time" not in raw15.columns:
            log.warning(f"    [{symbol}] Missing close_time on 15m production data — skip")
            return None

        clean_15m = sanitize_closed_candles(
            raw15,
            candle_duration_ms=15 * 60 * 1000,
            observation_time_ms=now_ms,
        )
        raw15 = pd.DataFrame(clean_15m)
        if raw15.empty or len(raw15) < 30:
            log.warning(f"    [{symbol}] Insufficient completed 15m candles — skip")
            return None

        # Production v2.0 compatibility frame:
        # use the exact locked legacy formulas on the closed Binance
        # observation stream. Canonical feature_engineering.py remains
        # unchanged and continues to define the candidate/canonical schema.
        df15 = add_legacy_production_indicators(raw15)
        if df15.empty or "open_time" not in df15.columns:
            return None

        if btc_df15_live is not None and not btc_df15_live.empty:
            if "close_time" not in btc_df15_live.columns:
                btc_df15_live = pd.DataFrame()
            else:
                btc_df15_live = pd.DataFrame(sanitize_closed_candles(
                    btc_df15_live,
                    candle_duration_ms=15 * 60 * 1000,
                    observation_time_ms=now_ms,
                ))

        df15 = _merge_extra_features_live(
            df15,
            btc_df15_live,
        )

        df1h_raw = get_data(
            symbol, TIMEFRAME_CONFIRM, canonical_observation=True
        )
        if (
            df1h_raw is not None
            and not df1h_raw.empty
            and "close_time" in df1h_raw.columns
        ):
            df1h_clean = sanitize_closed_candles(
                df1h_raw,
                candle_duration_ms=60 * 60 * 1000,
                observation_time_ms=now_ms,
            )
            df1h_raw = pd.DataFrame(df1h_clean)
        else:
            df1h_raw = pd.DataFrame()

        df1h = (
            add_legacy_production_indicators(df1h_raw)
            if not df1h_raw.empty
            else pd.DataFrame()
        )

        df4h_raw = _aggregate_completed_1h_to_4h(
            df1h_raw,
            observation_time_ms=now_ms,
        )
        df4h = (
            add_legacy_production_indicators(df4h_raw)
            if not df4h_raw.empty
            else pd.DataFrame()
        )

        row = df15.iloc[-1].copy()
        observation_time_ms = int(row["open_time"])

        r1h = (
            df1h.iloc[-1]
            if not df1h.empty
            else pd.Series(dtype=float)
        )
        r4h = (
            df4h.iloc[-1]
            if not df4h.empty
            else pd.Series(dtype=float)
        )

        # Populate HTF fields only from actual completed frames.
        # Production-model validation below will fail closed when any
        # selected HTF feature is unavailable.
        if not df1h.empty:
            row["rsi_1h"] = float(r1h.get("rsi", 50))
            row["adx_1h"] = float(r1h.get("adx", 0))
            row["trend_1h"] = float(r1h.get("trend", 0))

        if not df4h.empty:
            row["rsi_4h"] = float(r4h.get("rsi", 50))
            row["trend_4h"] = float(r4h.get("trend", 0))

        observation_contract = _build_signal_observation_contract(
            symbol,
            row,
            df1h_raw=df1h_raw,
            df4h_raw=df4h_raw,
            btc_df15_live=btc_df15_live,
            retrieved_at_ms=now_ms,
        )
        row["_observation_contract"] = observation_contract

        required_features, feature_source, strict_schema = (
            _required_production_features(pipeline)
        )

        missing_features, invalid_features = (
            _validate_production_feature_row(
                row,
                required_features,
            )
        )

        if strict_schema:
            if missing_features or invalid_features:
                if missing_features:
                    _SCAN_LIVENESS["features_zero_filled"] += len(
                        missing_features
                    )

                _SCAN_LIVENESS["feature_validation_rejects"] += 1

                details = []

                if missing_features:
                    details.append(
                        "missing=" + ",".join(missing_features)
                    )

                if invalid_features:
                    details.append(
                        "invalid=" + ",".join(invalid_features)
                    )

                reject_reason = (
                    "PRODUCTION_FEATURE_INVALID "
                    + "; ".join(details)
                )

                log.warning(
                    f"    [{symbol}] Production feature contract failed "
                    f"({feature_source}): {reject_reason}"
                )

                # Fail closed before model inference. At this point there is
                # deliberately no valid sig/conf to persist as a prediction.
                return None

        else:
            # Non-production/test selectors may expose only all_features.
            # Preserve the lightweight test contract without weakening the
            # real production artifact contract above.
            for feature in required_features:
                if feature not in row.index:
                    row[feature] = 0.0

        X = pd.DataFrame(
            [[row[feature] for feature in required_features]],
            columns=required_features,
        )

        if not np.isfinite(
            X.to_numpy(dtype=float)
        ).all():
            raise RuntimeError(
                "Production feature frame contains non-finite selected values "
                "after validation"
            )

        # Production-model selector compatibility:
        # 1) Current ImportanceSelector state: selected_features
        # 2) Legacy serialized ImportanceSelector state: feature_names
        # 3) Generic sklearn/test selector exposing transform()
        # 4) Final frozen-artifact fallback: pipeline["best_features"]
        selector = pipeline.get("selector")
        selected_features = getattr(
            selector,
            "selected_features",
            None,
        )
        legacy_feature_names = getattr(
            selector,
            "feature_names",
            None,
        )
        artifact_features = pipeline.get("best_features")

        if selected_features:
            Xs = selector.transform(X)

        elif legacy_feature_names:
            legacy_feature_names = list(
                legacy_feature_names
            )

            Xs = X[legacy_feature_names].to_numpy()

        elif selector is not None and hasattr(
            selector,
            "transform",
        ):
            Xs = selector.transform(X)

        elif artifact_features:
            Xs = X[
                list(artifact_features)
            ].to_numpy()

        else:
            raise RuntimeError(
                "Production model selector is incompatible and exposes "
                "no explicit feature schema"
            )

        disagreement = compute_ensemble_disagreement(pipeline, Xs[0])

        pred = pipeline["ensemble"].predict(Xs)[0]
        prob = pipeline["ensemble"].predict_proba(Xs)[0]
        
        sig  = pipeline["label_map"][int(pred)]
        conf = round(float(max(prob))*100, 1)

        # A symbol is "scored" only after the ensemble has actually
        # produced a prediction/probability result.
        if _SCAN_LIVENESS.get("scan_ran"):
            _SCAN_LIVENESS["symbols_scored"] += 1

        # ── CLAMP: Enforce config.MIN_CONFIDENCE as absolute minimum floor ──
        config_floor = float(getattr(config, "MIN_CONFIDENCE", 52.0))
        rec_buy  = max(config_floor, float(pipeline.get("recommended_threshold_buy", 0.40)) * 100.0)
        rec_sell = max(config_floor, float(pipeline.get("recommended_threshold_sell", 0.45)) * 100.0)
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
                    log.info(f"    ⚡ [FG_OVERRIDE] F&G={fng_data.get('value')} extreme bypassed ({conf:.1f}% ≥ 70%)")
                else:
                    direction_str = "SELL at panic bottom" if sig == "SELL" else "BUY at euphoric top"
                    log.info(f"    [FILTER:FG_HARD] F&G={fng_data.get('value')} — blocking {direction_str} (Conf: {conf:.1f}% < 70%)")
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
            elif f_mod > 0:
                score -= f_mod
                reasons.append(f"F&G extreme countered (-{f_mod})")

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

        snapshot = build_trade_snapshot(row, r4h, e20_4h, e50_4h, rsi_4h, fng_data, whale_flow, btc_momentum)
        pred_id  = f"{symbol}_{int(time.time()*1000)}"
        save_prediction(_pred_record(symbol, sig, conf, pipeline, row, disagreement, pred_id=pred_id))

        return {
            "symbol": symbol, "signal": sig, "confidence": conf, "score": score,
            "entry": entry, "atr": atr, "stop": 0, "tp1": 0, "tp2": 0,
            "reasons": reasons, "conf_tier": "high" if conf >= 60.0 else "normal",
            "fg_override": fg_override_active,
            "pred_id": pred_id,
            "open_time": observation_time_ms,
            "logical_symbol": str(symbol).upper(),
            "interval": str(TIMEFRAME_ENTRY),
            "observation_source": {
                "exchange": "binance",
                "market_type": "spot",
            },
            "observation_market": "BINANCE_SPOT",
            "observation_contract": observation_contract,
            "observation_identity": list(observation_identity(observation_contract)),
            "snapshot": snapshot,
        }
    except Exception as e:
        log.error(f"    Signal {symbol}: {e}")
        return None

def _deribit_instrument_spec_for_decimal_sizing(deribit, symbol: str) -> tuple:
    """Build a sizing spec only from the verified live instrument response.

    Deribit documents min_trade_amount as both minimum order quantity and the
    quantity increment. For linear USDC perpetuals, order amounts are base-coin
    quantities, so one unit of order quantity represents one base coin for
    price-move/notional calculations; contract_size is validated separately
    and is not multiplied a second time into a base-quantity order amount.
    """
    from decimal import Decimal

    info = deribit.get_instrument_info(symbol)
    if not isinstance(info, dict) or not info:
        raise ValueError("verified Deribit instrument metadata unavailable")
    expected_name = deribit.get_instrument_name(symbol)
    if str(info.get("instrument_name", "")) != expected_name:
        raise ValueError("Deribit instrument identity mismatch")
    if info.get("is_active") is not True or str(info.get("state", "")).lower() != "open":
        raise ValueError("Deribit instrument is not currently open for new orders")
    if str(info.get("instrument_type", "")).lower() != "linear":
        raise ValueError("only verified linear Deribit instruments may use this sizing model")
    if str(info.get("kind", "")).lower() != "future":
        raise ValueError("Deribit instrument kind is not future")
    if str(info.get("settlement_period", "")).lower() != "perpetual":
        raise ValueError("Deribit instrument is not perpetual")
    if str(info.get("settlement_currency", "")).upper() != "USDC":
        raise ValueError("Deribit instrument settlement currency is not USDC")
    if not str(info.get("base_currency", "")).strip():
        raise ValueError("Deribit instrument base currency is missing")

    try:
        min_step = Decimal(str(info["min_trade_amount"]))
        contract_size = Decimal(str(info["contract_size"]))
        tick_size = Decimal(str(deribit.get_tick_size(symbol)))
        taker_fee = Decimal(str(info["taker_commission"]))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        raise ValueError("Deribit quantity, contract, tick or fee specification is incomplete") from None
    if not all(value.is_finite() for value in (min_step, contract_size, tick_size, taker_fee)):
        raise ValueError("Deribit instrument specification contains non-finite values")
    if min_step <= 0 or contract_size <= 0 or tick_size <= 0 or taker_fee < 0:
        raise ValueError("Deribit instrument specification contains invalid values")

    max_qty = None
    raw_max = info.get("max_trade_amount")
    if raw_max is not None:
        max_qty = Decimal(str(raw_max))
        if not max_qty.is_finite() or max_qty < min_step:
            raise ValueError("Deribit maximum quantity is invalid")

    spec = InstrumentSpec(
        exchange="DERIBIT",
        symbol=str(symbol).upper(),
        instrument_id=expected_name,
        settlement_currency="USDC",
        quantity_unit="BASE",
        contract_value_base=Decimal("1"),
        qty_step=min_step,
        min_qty=min_step,
        max_qty=max_qty,
        tick_size=tick_size,
        min_notional=Decimal("0"),
        linear=True,
    )
    return spec, taker_fee


def _calculate_deribit_decimal_size(
    deribit, symbol: str, risk_snapshot: dict, *, entry_price, stop_price, risk_multiplier, side: str,
    tier_policy: dict | None = None,
):
    """Use the shared Decimal sizing engine in the actual Deribit entry path."""
    from decimal import Decimal

    if not isinstance(risk_snapshot, dict) or risk_snapshot.get("entry_enabled") is not True:
        raise ValueError("validated risk-capital snapshot is unavailable")
    if str(risk_snapshot.get("risk_currency", "")).upper() != "USDC":
        raise ValueError("risk-capital currency must be USDC")
    if str(risk_snapshot.get("margin_mode", "")).lower() != "segregated":
        raise ValueError("new entries require segregated USDC margin")
    equity = Decimal(str(risk_snapshot.get("risk_equity")))
    available = Decimal(str(risk_snapshot.get("available_funds")))
    multiplier = Decimal(str(risk_multiplier))
    base_risk_fraction = (
        tier_policy["risk_fraction"] if isinstance(tier_policy, dict)
        else Decimal(str(getattr(config, "CAPITAL_TIER_RISK_FRACTION", RISK_PER_TRADE)))
    )
    effective_risk_fraction = base_risk_fraction * multiplier
    # Keep effective risk, including any strategy multiplier, within the
    # hard 5% per-trade maximum in the shared sizing policy.
    if not effective_risk_fraction.is_finite() or effective_risk_fraction <= 0 or effective_risk_fraction > Decimal("0.05"):
        raise ValueError("effective per-trade risk exceeds the 5% hard ceiling")

    capital = RiskCapital(
        equity=equity,
        currency="USDC",
        available_funds=available,
        margin_mode="segregated",
        source=str(risk_snapshot.get("source", "deribit_account_snapshot")),
    )
    spec, taker_fee = _deribit_instrument_spec_for_decimal_sizing(deribit, symbol)
    if isinstance(tier_policy, dict):
        reserve_fraction = tier_policy["available_funds_reserve_fraction"]
        tier_cap = tier_policy.get("capital_cap_usdc")
        max_notional_fraction = tier_policy["max_notional_fraction"]
    else:
        reserve_fraction = Decimal(str(getattr(config, "AVAILABLE_FUNDS_RESERVE_FRACTION", "0.20")))
        tier_cap_value = getattr(config, "CAPITAL_TIER_CAP_USDC", None)
        tier_cap = Decimal(str(tier_cap_value)) if tier_cap_value is not None else None
        max_notional_fraction = Decimal(str(getattr(config, "CAPITAL_TIER_MAX_NOTIONAL_FRACTION", "0.05")))
    limits = RiskLimits(
        risk_fraction=effective_risk_fraction,
        max_notional_fraction=max_notional_fraction,
        tier_cap=tier_cap,
        taker_fee_rate=taker_fee,
        available_funds_reserve_fraction=reserve_fraction,
    )
    result = calculate_position_size(
        capital,
        spec,
        entry_price=entry_price,
        stop_price=stop_price,
        limits=limits,
        risk_multiplier=Decimal("1"),
        side=str(side or "").upper(),
    )
    sizing_equity = min(equity, tier_cap) if tier_cap is not None else equity
    if not sizing_equity.is_finite() or sizing_equity <= 0:
        raise ValueError("capital tier equity cap is invalid")
    return result, spec, taker_fee, effective_risk_fraction, sizing_equity


def _deterministic_order_label(sig: dict, symbol: str, slot: str) -> str:
    """Deribit label derived from prediction identity and lifecycle slot."""
    identity = str(sig.get("pred_id") or sig.get("open_time") or sig.get("opened_at") or "")
    if not identity:
        raise ValueError("prediction identity is required for deterministic order labels")
    raw = f"DERIBIT|TESTNET|{str(symbol).upper()}|{identity}|{slot}"
    return "cb_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _confirm_execution_gateway_stop(
    deribit, trade: dict, *, stop_client_label: str, stop_order_id: str,
    protected_quantity: float, evidence: dict = None,
) -> None:
    """Record the stop only after DeribitClient independently verified it."""
    if not getattr(deribit, "require_execution_gateway", False):
        return
    order_ids = trade.get("order_ids", {}) if isinstance(trade, dict) else {}
    attempt = trade.get("entry_attempt", {}) if isinstance(trade, dict) else {}
    entry_label = ""
    if isinstance(order_ids, dict):
        entry_label = str(order_ids.get("entry_label") or "").strip()
    if not entry_label and isinstance(attempt, dict):
        entry_label = str(attempt.get("client_label") or "").strip()
    if not entry_label:
        raise RuntimeError("Cannot link stop protection without the durable entry client label")
    deribit.confirm_execution_protection(
        entry_client_order_id=entry_label,
        stop_client_order_id=str(stop_client_label),
        protective_order_id=str(stop_order_id),
        protected_quantity=protected_quantity,
        evidence=evidence or {"verification_source": "DeribitClient.place_stop_loss"},
    )


def _logical_entry_intent_id(sig: dict, symbol: str, signal: str) -> str:
    """Stable identity for one prediction's entry intent, independent of retries."""
    identity = str(sig.get("pred_id") or sig.get("open_time") or sig.get("opened_at") or "").strip()
    if not identity:
        raise ValueError("prediction identity is required for entry intent identity")
    raw = f"DERIBIT|TESTNET|ENTRY|{str(symbol).upper()}|{str(signal).upper()}|{identity}"
    return "li_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _entry_attempt_client_label(logical_intent_id: str, attempt_id: str) -> str:
    """Create a Deribit label unique to one persisted order attempt (not an idempotency key)."""
    if not logical_intent_id or not attempt_id:
        raise ValueError("logical intent and attempt identities are required")
    raw = f"DERIBIT|TESTNET|ENTRY_ATTEMPT|{logical_intent_id}|{attempt_id}"
    return "cb_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class EntryIntentReplayBlocked(RuntimeError):
    """A previously attempted logical entry must not be submitted again implicitly."""


def _copy_entry_attempt_history(previous_trade: dict) -> list:
    history = previous_trade.get("entry_attempt_history", []) if isinstance(previous_trade, dict) else []
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise EntryIntentReplayBlocked("persisted entry attempt history is malformed; manual review required")
    copied = [dict(item) for item in history]
    seen_ids = set()
    for item in copied:
        attempt_id = str(item.get("attempt_id", "")).strip()
        logical_id = str(item.get("logical_intent_id", "")).strip()
        try:
            attempt_number = int(item.get("attempt_number"))
        except (TypeError, ValueError, OverflowError):
            raise EntryIntentReplayBlocked("persisted attempt number is malformed; manual review required") from None
        if (
            not attempt_id or not logical_id or attempt_id in seen_ids
            or isinstance(item.get("attempt_number"), bool) or attempt_number < 1
        ):
            raise EntryIntentReplayBlocked("persisted attempt identity is missing or duplicated; manual review required")
        seen_ids.add(attempt_id)
    return copied


def _has_unresolved_entry_attempt(trade: dict) -> bool:
    """Fail closed when any historical attempt lacks a terminal disposition."""
    if not isinstance(trade, dict):
        return True
    history = trade.get("entry_attempt_history", [])
    if not isinstance(history, list):
        return True
    unresolved_states = {"SUBMITTING", "ACKNOWLEDGED", "UNKNOWN", "SUBMISSION_UNKNOWN"}
    for item in history:
        if not isinstance(item, dict):
            return True
        if str(item.get("submission_state", "")).strip().upper() in unresolved_states:
            return True
    return False


def _legacy_entry_intent_matches(previous_trade: dict, sig: dict, symbol: str, signal: str, logical_intent_id: str) -> bool:
    """Recognize a same-prediction record written before attempt identities existed."""
    if not isinstance(previous_trade, dict):
        return False
    if str(previous_trade.get("logical_intent_id", "")) == logical_intent_id:
        return True
    previous_pred_id = previous_trade.get("pred_id")
    current_pred_id = sig.get("pred_id")
    if previous_pred_id not in (None, "") and current_pred_id not in (None, ""):
        # A prediction ID is itself unique; a changed direction must not turn
        # an already-attempted prediction into a fresh entry intent.
        return str(previous_pred_id) == str(current_pred_id)
    if previous_pred_id not in (None, ""):
        previous_signal = str(previous_trade.get("signal") or signal)
        try:
            return _logical_entry_intent_id({"pred_id": previous_pred_id}, symbol, previous_signal) == logical_intent_id
        except ValueError:
            return True
    prior_orders = previous_trade.get("order_ids", {})
    prior_labels = previous_trade.get("order_client_labels", {})
    prior_label = ""
    if isinstance(prior_orders, dict):
        prior_label = str(prior_orders.get("entry_label", ""))
    if not prior_label and isinstance(prior_labels, dict):
        prior_label = str(prior_labels.get("entry", ""))
    try:
        legacy_label = _deterministic_order_label(sig, symbol, "ENTRY")
    except ValueError:
        return False
    return bool(prior_label and prior_label == legacy_label)


def _prepare_entry_attempt(
    sig: dict,
    symbol: str,
    signal: str,
    requested_qty: float,
    previous_trade: dict = None,
) -> dict:
    """Build next attempt identity; replay is blocked unless a verified zero-fill retry is explicitly authorized."""
    logical_id = _logical_entry_intent_id(sig, symbol, signal)
    history = _copy_entry_attempt_history(previous_trade or {})
    same_intent = [
        item for item in history
        if str(item.get("logical_intent_id", "")) == logical_id
    ]
    retry_authorization = None

    if same_intent:
        prior = max(same_intent, key=lambda item: int(item.get("attempt_number", 0) or 0))
        prior_id = str(prior.get("attempt_id", "")).strip()
        retryable_states = {
            "ZERO_FILL_VERIFIED", "EXCHANGE_REJECTED_ZERO_FILL", "PREFLIGHT_BLOCKED"
        }
        if (
            not prior_id
            or str(prior.get("submission_state", "")) not in retryable_states
            or prior.get("zero_fill_position_unchanged") is not True
        ):
            raise EntryIntentReplayBlocked(
                "previous logical entry is unresolved, filled, or lacks verified zero-fill disposition"
            )
        expected = f"{logical_id}:{prior_id}"
        supplied = os.getenv("CRYPTOBOT_ENTRY_RETRY_AUTHORIZATION", "").strip()
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise EntryIntentReplayBlocked(
                "same logical entry already has a verified zero-fill attempt; explicit attempt-scoped operator authorization is required"
            )
        retry_authorization = {
            "method": "CRYPTOBOT_ENTRY_RETRY_AUTHORIZATION",
            "previous_attempt_id": prior_id,
            "authorized_at": datetime.now(timezone.utc).isoformat(),
        }
        attempt_number = max(int(item.get("attempt_number", 0) or 0) for item in same_intent) + 1
    else:
        if _legacy_entry_intent_matches(previous_trade or {}, sig, symbol, signal, logical_id):
            raise EntryIntentReplayBlocked(
                "matching legacy trade exists without verified attempt disposition; refusing replay"
            )
        attempt_number = 1

    attempt_id = uuid.uuid4().hex
    label = _entry_attempt_client_label(logical_id, attempt_id)
    now = datetime.now(timezone.utc).isoformat()
    attempt = {
        "logical_intent_id": logical_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "client_label": label,
        "requested_qty": float(requested_qty),
        "created_at": now,
        "submission_state": "SUBMITTING",
        "zero_fill_position_unchanged": False,
    }
    if retry_authorization is not None:
        attempt["retry_authorization"] = retry_authorization
    history.append(dict(attempt))
    return {
        "logical_intent_id": logical_id,
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "client_label": label,
        "attempt": attempt,
        "history": history,
    }


def _update_entry_attempt_history(record: dict, attempt_id: str, **updates) -> None:
    history = record.get("entry_attempt_history")
    if not isinstance(history, list):
        raise StatePersistenceError("entry attempt history is absent or malformed")
    matches = [item for item in history if isinstance(item, dict) and item.get("attempt_id") == attempt_id]
    if len(matches) != 1:
        raise StatePersistenceError("entry attempt identity is missing or non-unique in persisted history")
    matches[0].update(updates)
    record["entry_attempt"] = dict(matches[0])
    record["entry_attempt_id"] = attempt_id


def _verify_entry_position_unchanged(deribit, symbol: str, before_signed_position: float) -> dict:
    """Require a fresh, finite signed position identical to the pre-submit snapshot."""
    try:
        before = float(before_signed_position)
        after = float(deribit.get_position_size(symbol))
    except Exception as exc:
        raise OrderOutcomeUnknown(
            f"entry zero-fill position comparison unavailable ({type(exc).__name__})"
        ) from None
    if not math.isfinite(before) or not math.isfinite(after):
        raise OrderOutcomeUnknown("entry zero-fill position comparison is non-finite")
    tolerance = max(1e-12, abs(before) * 1e-9)
    if abs(after - before) > tolerance:
        raise OrderOutcomeUnknown("entry reports no fill but signed position changed")
    return {
        "verified": True,
        "position_before": before,
        "position_after": after,
        "absolute_tolerance": tolerance,
    }


def _assert_entry_order_contract(immediate: dict, observed: dict, expected: dict) -> None:
    """Require acknowledgement and independent readback to match the complete IOC entry contract."""
    if not all(isinstance(item, dict) for item in (immediate, observed, expected)):
        raise OrderOutcomeUnknown("entry order contract evidence is malformed")
    aliases = {"order_type": ("order_type", "type")}
    fields = ("instrument_name", "direction", "amount", "order_type", "time_in_force", "reduce_only", "label")

    def read_field(row, field):
        for key in aliases.get(field, (field,)):
            if key in row:
                return row.get(key)
        return None

    for field in fields:
        acknowledged = read_field(immediate, field)
        independently_observed = read_field(observed, field)
        if acknowledged is None or independently_observed is None:
            raise OrderOutcomeUnknown(f"entry {field} is missing from acknowledgement or exchange readback")
        target = expected.get(field)
        if field == "amount":
            try:
                ack_value = float(acknowledged)
                observed_value = float(independently_observed)
                target_value = float(target)
            except (TypeError, ValueError, OverflowError):
                raise OrderOutcomeUnknown("entry amount contract is invalid") from None
            if not all(math.isfinite(v) for v in (ack_value, observed_value, target_value)):
                raise OrderOutcomeUnknown("entry amount contract is non-finite")
            tolerance = max(1e-12, abs(target_value) * 1e-9)
            if abs(ack_value - target_value) > tolerance or abs(observed_value - target_value) > tolerance:
                raise OrderOutcomeUnknown("entry amount differs from the requested contract")
        elif field == "reduce_only":
            if acknowledged is not False or independently_observed is not False or target is not False:
                raise OrderOutcomeUnknown("entry reduce_only contract mismatch")
        else:
            ack_value = str(acknowledged).strip().lower()
            observed_value = str(independently_observed).strip().lower()
            target_value = str(target).strip().lower()
            if ack_value != target_value or observed_value != target_value:
                raise OrderOutcomeUnknown(f"entry {field} contract mismatch")

    try:
        ack_price = float(immediate.get("price"))
        observed_price = float(observed.get("price"))
    except (TypeError, ValueError, OverflowError):
        raise OrderOutcomeUnknown("entry limit price is missing or invalid") from None
    try:
        expected_price = float(expected.get("price"))
    except (TypeError, ValueError, OverflowError):
        raise OrderOutcomeUnknown("entry expected limit price is missing or invalid") from None
    if (
        not all(math.isfinite(v) for v in (ack_price, observed_price, expected_price))
        or ack_price <= 0 or observed_price <= 0 or expected_price <= 0
        or not math.isclose(ack_price, observed_price, rel_tol=1e-12, abs_tol=1e-12)
        or not math.isclose(ack_price, expected_price, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise OrderOutcomeUnknown("entry limit price differs from the persisted acknowledgement/readback contract")


def _build_entry_submission_intent(
    sig: dict,
    symbol: str,
    signal: str,
    model_entry: float,
    execution_entry_ref: float,
    requested_qty: float,
    client_label: str,
    *,
    attempt: dict = None,
    attempt_history: list = None,
) -> dict:
    """Build a durable local intent before the first entry mutation.

    Requested quantity is kept separate from ``qty`` so an in-flight order is
    never represented as an exchange-confirmed position.
    """
    now = datetime.now(timezone.utc).isoformat()
    record = {
        **_candidate_identity_fields(),
        "symbol": str(symbol),
        "signal": str(signal),
        "entry": float(execution_entry_ref),
        "model_entry": float(model_entry),
        "execution_entry_ref": float(execution_entry_ref),
        "actual_entry": 0.0,
        "qty": 0.0,
        "requested_qty": float(requested_qty),
        "filled_qty": 0.0,
        "quantity_source": "entry_order_not_yet_confirmed",
        "pred_id": sig.get("pred_id"),
        "opened_at": now,
        "closed": False,
        "order_ids": {"entry_label": str(client_label)},
        "order_submission_states": {"entry": "SUBMITTING"},
        "order_client_labels": {"entry": str(client_label)},
        "entry_submission_state": "SUBMITTING",
        "stop_loss_submission_state": "NOT_SUBMITTED",
        "manual_reconciliation_required": True,
        "protection_reconciliation_required": True,
        "protection_reconciliation_reason": "entry submission in progress",
        "exchange": "deribit_testnet",
        "pnl_currency": "USDC",
        "snapshot": sig.get("snapshot", {}) if isinstance(sig.get("snapshot", {}), dict) else {},
    }
    if attempt is not None:
        record.update({
            "logical_intent_id": str(attempt["logical_intent_id"]),
            "entry_attempt_id": str(attempt["attempt_id"]),
            "entry_attempt_number": int(attempt["attempt_number"]),
            "entry_attempt": dict(attempt),
            "entry_attempt_history": [dict(item) for item in (attempt_history or [attempt])],
        })
    elif attempt_history is not None:
        record["entry_attempt_history"] = [dict(item) for item in attempt_history]
    return record


def _entry_terminal_has_confirmed_fill(order: dict) -> bool:
    """Return whether a terminal entry order has exposure that needs protection."""
    if not isinstance(order, dict):
        raise OrderOutcomeUnknown("terminal entry order is not a structured object")
    state = str(order.get("order_state", "")).strip().lower()
    if state not in {"filled", "cancelled", "canceled", "rejected", "closed", "expired"}:
        raise OrderOutcomeUnknown("entry order is not in a verified terminal state")
    raw_filled = order.get("filled_amount")
    if raw_filled is None or isinstance(raw_filled, bool):
        raise OrderOutcomeUnknown("terminal entry order has no confirmed filled_amount")
    try:
        filled = float(raw_filled)
    except (TypeError, ValueError, OverflowError):
        raise OrderOutcomeUnknown("terminal entry filled_amount is invalid") from None
    if not math.isfinite(filled) or filled < 0:
        raise OrderOutcomeUnknown("terminal entry filled_amount is non-finite or negative")
    if state == "rejected" and filled > 0:
        raise OrderOutcomeUnknown("rejected entry unexpectedly reports a non-zero fill")
    return filled > 0


def _reconcile_unknown_entry_by_label_read_only(deribit, symbol: str, client_label: str) -> dict:
    """Collect read-only Deribit evidence after an ambiguous entry submission.

    This routine never retries, cancels, or otherwise mutates an order. A missing
    label match is not interpreted as proof of rejection because Deribit's
    closed-order label search has a limited recency window.
    """
    evidence = {
        "client_label": str(client_label),
        "symbol": str(symbol).upper(),
        "label_lookup_window": "recent closed orders only (Deribit documents 30 minutes)",
        "recent_trade_lookup_window": "recent trade endpoint (Deribit documents 24 hours; maximum 1,000 rows per request)",
        "read_only": True,
        "order_state_matches": [],
        "open_order_matches": [],
        "position_snapshot": {"verified": False},
        "recent_trade_snapshot": {"verified": False},
        "read_errors": [],
    }
    instrument = None
    currency = None
    try:
        instrument = deribit.get_instrument_name(symbol)
        info = deribit.get_instrument_info(symbol)
        if not isinstance(info, dict) or not info:
            raise ValueError("instrument metadata missing")
        currency = str(info.get("base_currency", "")).strip().upper()
        if not currency:
            raise ValueError("instrument base currency missing")
        evidence["instrument_name"] = str(instrument)
        evidence["label_currency"] = currency
    except Exception as exc:
        evidence["read_errors"].append({"source": "instrument_identity", "error": type(exc).__name__})

    state_matches = []
    open_matches = []
    if currency:
        try:
            rows = deribit.get_order_state_by_label(currency, client_label)
            state_matches = [
                row for row in rows
                if str(row.get("label", "")) == client_label
                and str(row.get("instrument_name", "")) == str(instrument)
            ]
            evidence["order_state_matches"] = [
                {
                    "order_id": str(row.get("order_id", "")),
                    "order_state": str(row.get("order_state", "")),
                    "instrument_name": str(row.get("instrument_name", "")),
                    "filled_amount": row.get("filled_amount"),
                    "average_price": row.get("average_price"),
                    "direction": row.get("direction"),
                    "label": str(row.get("label", "")),
                }
                for row in state_matches
            ]
            evidence["order_state_query_succeeded"] = True
        except Exception as exc:
            evidence["read_errors"].append({"source": "order_state_by_label", "error": type(exc).__name__})
        try:
            rows = deribit.get_open_orders_by_label(currency, client_label)
            open_matches = [
                row for row in rows
                if str(row.get("label", "")) == client_label
                and str(row.get("instrument_name", "")) == str(instrument)
            ]
            evidence["open_order_matches"] = [
                {
                    "order_id": str(row.get("order_id", "")),
                    "order_state": str(row.get("order_state", "")),
                    "instrument_name": str(row.get("instrument_name", "")),
                    "amount": row.get("amount"),
                    "filled_amount": row.get("filled_amount"),
                    "direction": row.get("direction"),
                    "label": str(row.get("label", "")),
                }
                for row in open_matches
            ]
            evidence["open_orders_query_succeeded"] = True
        except Exception as exc:
            evidence["read_errors"].append({"source": "open_orders_by_label", "error": type(exc).__name__})

    if getattr(deribit, "require_execution_gateway", False):
        deribit.reconcile_execution_order_label(
            client_label, list(state_matches) + list(open_matches),
            queries_complete=bool(evidence.get("order_state_query_succeeded") and evidence.get("open_orders_query_succeeded")),
        )

    try:
        position = float(deribit.get_position_size(symbol))
        if not math.isfinite(position):
            raise ValueError("position snapshot is non-finite")
        evidence["position_snapshot"] = {"verified": True, "signed_quantity": position}
    except Exception as exc:
        evidence["read_errors"].append({"source": "position_snapshot", "error": type(exc).__name__})

    trade_matches = []
    trade_order_ids = set()
    historical_order_matches = []
    historical_trade_matches = []
    historical_order_ids = set()
    historical_trade_order_ids = set()
    historical_order_query_succeeded = False
    historical_history_complete = False
    historical_trade_query_succeeded_ids = set()
    historical_trade_query_failed_ids = set()
    if instrument:
        try:
            # Deribit returns at most 1,000 recent trades per request. Trade
            # rows can carry the original user-defined order label even after
            # the closed-order-by-label 30-minute window has elapsed.
            # Recent instrument trades are capped at 1,000 rows per request.
            # Continue backward by sequence number when a full page is returned.
            # Bound the scan at 10,000 rows; a cap or missing sequence makes the
            # result incomplete, never a reason to infer rejection.
            rows = []
            recent_trade_page_size = 1000
            max_recent_trade_pages = 10
            recent_trade_pages = 0
            recent_trade_history_complete = False
            next_end_seq = None
            for _page in range(max_recent_trade_pages):
                kwargs = {"count": recent_trade_page_size, "strict": True}
                if next_end_seq is not None:
                    kwargs["end_seq"] = next_end_seq
                page_rows = deribit.get_trade_history_by_instrument_name(
                    instrument, **kwargs
                )
                if not isinstance(page_rows, list) or any(not isinstance(row, dict) for row in page_rows):
                    raise ValueError("Malformed recent trade-history page")
                recent_trade_pages += 1
                rows.extend(page_rows)
                if len(page_rows) < recent_trade_page_size:
                    recent_trade_history_complete = True
                    break
                oldest_seq = page_rows[-1].get("trade_seq")
                if isinstance(oldest_seq, bool) or not isinstance(oldest_seq, int) or oldest_seq <= 0:
                    break
                next_end_seq = oldest_seq - 1

            known_ids = {
                str(row.get("order_id", ""))
                for row in state_matches + open_matches
                if str(row.get("order_id", "")).strip()
            }
            trade_matches = [
                row for row in rows
                if isinstance(row, dict)
                and str(row.get("instrument_name", "")) == str(instrument)
                and (
                    str(row.get("order_id", "")) in known_ids
                    or str(row.get("label", "")) == client_label
                )
            ]
            trade_order_ids = {
                str(row.get("order_id", "")).strip()
                for row in trade_matches
                if str(row.get("order_id", "")).strip()
            }
            evidence["recent_trade_snapshot"] = {
                "verified": True,
                "rows_returned": len(rows),
                "pages_returned": recent_trade_pages,
                "matching_order_ids": sorted(trade_order_ids | known_ids),
                "matching_trade_count": len(trade_matches),
                "matching_trade_ids": [
                    str(row.get("trade_id", ""))
                    for row in trade_matches if row.get("trade_id")
                ],
                "matched_by_exact_label_count": sum(
                    1 for row in trade_matches
                    if str(row.get("label", "")) == client_label
                ),
                "history_complete": recent_trade_history_complete,
                "pagination_capped": not recent_trade_history_complete,
                "note": (
                    "Recent trades are queried in descending sequence pages up to "
                    "10,000 rows. Empty or incomplete history is not proof of no fill. "
                    "Historical order/trade queries and position reconciliation are also required."
                ),
            }
        except Exception as exc:
            evidence["read_errors"].append({"source": "recent_trade_history", "error": type(exc).__name__})

        # The label-state endpoint is limited to recently closed orders. Query
        # Deribit's indexed historical order store as a second, read-only source.
        # Historical=true does not include the recent window, so both result
        # sets are retained separately and absence from either remains UNKNOWN.
        try:
            # The endpoint caps each response at 1,000 rows and supports offset
            # pagination. Search a bounded window (up to 10,000 historical orders)
            # so a busy instrument cannot silently hide an older matching label.
            # If the cap is reached, the snapshot is explicitly incomplete and an
            # absent label remains UNKNOWN.
            historical_rows = []
            historical_page_count = 0
            historical_history_complete = False
            historical_page_size = 1000
            max_historical_pages = 10
            for page_number in range(max_historical_pages):
                page_offset = page_number * historical_page_size
                page_rows = deribit.get_order_history_by_instrument_name(
                    instrument,
                    count=historical_page_size,
                    historical=True,
                    include_old=True,
                    include_unfilled=True,
                    offset=page_offset,
                    strict=True,
                )
                if not isinstance(page_rows, list):
                    raise ValueError("Malformed historical order page")
                historical_page_count += 1
                historical_rows.extend(page_rows)
                if len(page_rows) < historical_page_size:
                    historical_history_complete = True
                    break
            historical_order_query_succeeded = True
            historical_order_matches = [
                row for row in historical_rows
                if isinstance(row, dict)
                and str(row.get("instrument_name", "")) == str(instrument)
                and str(row.get("label", "")) == client_label
            ]
            evidence["historical_order_snapshot"] = {
                "query_succeeded": True,
                "verified": True,
                "rows_returned": len(historical_rows),
                "pages_returned": historical_page_count,
                "matching_orders": [
                    {
                        "order_id": str(row.get("order_id", "")),
                        "order_state": str(row.get("order_state", "")),
                        "filled_amount": row.get("filled_amount"),
                        "average_price": row.get("average_price"),
                        "direction": row.get("direction"),
                        "label": str(row.get("label", "")),
                        "instrument_name": str(row.get("instrument_name", "")),
                    }
                    for row in historical_order_matches
                ],
                "query_mode": "historical=true; indexed historical records only",
                "page_size": historical_page_size,
                "max_pages": max_historical_pages,
                "history_complete": historical_history_complete,
                "pagination_capped": not historical_history_complete,
            }
        except Exception as exc:
            evidence["historical_order_snapshot"] = {
                "query_succeeded": False,
                "verified": False,
                "history_complete": False,
            }
            evidence["read_errors"].append({"source": "historical_order_history", "error": type(exc).__name__})

        for order_row in historical_order_matches[:10]:
            order_id = str(order_row.get("order_id", "")).strip()
            if not order_id:
                continue
            try:
                order_trades = deribit.get_user_trades_by_order(
                    order_id, historical=True, strict=True
                )
                if not isinstance(order_trades, list) or any(not isinstance(row, dict) for row in order_trades):
                    raise ValueError("Malformed historical trades-by-order response")
                historical_trade_query_succeeded_ids.add(order_id)
                historical_trade_matches.extend(
                    row for row in order_trades
                    if str(row.get("instrument_name", "")) == str(instrument)
                    and str(row.get("order_id", "")) == order_id
                    and str(row.get("label", client_label)) in ("", client_label)
                )
            except Exception as exc:
                historical_trade_query_failed_ids.add(order_id)
                evidence["read_errors"].append({
                    "source": "historical_trades_by_order",
                    "order_id": order_id,
                    "error": type(exc).__name__,
                })

        trade_lookup_coverage_complete = (
            len(historical_order_matches) <= 10
            and len(historical_trade_query_succeeded_ids) == len(historical_order_matches)
            and not historical_trade_query_failed_ids
        )
        historical_trade_queries_complete = (
            historical_order_query_succeeded
            and historical_history_complete
            and trade_lookup_coverage_complete
        )
        evidence["historical_trade_snapshot"] = {
            "query_succeeded": historical_order_query_succeeded and not historical_trade_query_failed_ids,
            "verified": historical_trade_queries_complete,
            "history_complete": historical_history_complete,
            "order_trade_lookups_complete": trade_lookup_coverage_complete,
            "historical_order_count": len(historical_order_matches),
            "historical_order_lookup_limit": 10,
            "order_ids_queried": sorted(historical_trade_query_succeeded_ids),
            "order_ids_failed": sorted(historical_trade_query_failed_ids),
            "matching_trade_count": len(historical_trade_matches),
            "matching_trade_ids": [
                str(row.get("trade_id", "")) for row in historical_trade_matches if row.get("trade_id")
            ],
            "matching_order_ids": sorted({
                str(row.get("order_id", "")) for row in historical_trade_matches if row.get("order_id")
            }),
            "query_mode": "historical=true by exact historical order ID",
        }
        historical_order_ids = {
            str(row.get("order_id", "")).strip()
            for row in historical_order_matches
            if str(row.get("order_id", "")).strip()
        }
        historical_trade_order_ids = {
            str(row.get("order_id", "")).strip()
            for row in historical_trade_matches
            if str(row.get("order_id", "")).strip()
        }
        evidence["historical_order_matches_count"] = len(historical_order_matches)
        evidence["historical_trade_matches_count"] = len(historical_trade_matches)
        evidence["historical_reconciliation_note"] = (
            "Historical APIs are indexed separately and may not immediately contain recent activity. "
            "Empty historical results do not prove rejection or no fill."
        )

    evidence["matched_order_ids"] = sorted({
        str(row.get("order_id", ""))
        for row in state_matches + open_matches
        if str(row.get("order_id", "")).strip()
    } | trade_order_ids | historical_order_ids | historical_trade_order_ids)
    return evidence



def _resume_pending_entry_attempt_read_only(deribit, sig: dict) -> bool:
    """Resume reconciliation of the same persisted attempt without any mutation.

    A durable SUBMITTING marker may survive a process crash either before or
    after the network request. The next scan must not infer which happened and
    must never create a replacement order. It reuses the persisted attempt ID
    and client label for read-only reconciliation, then leaves the attempt
    UNKNOWN for operator disposition regardless of whether a match is found.

    Returns True when this signal belongs to an unresolved persisted attempt
    (including malformed identity/state that must fail closed), meaning the
    caller must stop before all entry-preparation and exchange-mutation paths.
    """
    if not isinstance(sig, dict):
        return False
    symbol = str(sig.get("symbol", "")).strip().upper()
    signal = str(sig.get("signal", "")).strip().upper()
    if not symbol or signal not in {"BUY", "SELL"}:
        return False
    try:
        logical_id = _logical_entry_intent_id(sig, symbol, signal)
    except (TypeError, ValueError):
        return False

    try:
        trades = load_trades()
    except Exception as exc:
        log.critical(
            "  %s: cannot load persisted lifecycle state before entry; fail-closed (%s)",
            symbol,
            type(exc).__name__,
        )
        return True
    if not isinstance(trades, dict):
        log.critical("  %s: persisted lifecycle root is not an object; fail-closed", symbol)
        return True

    previous = trades.get(symbol)
    if not isinstance(previous, dict):
        return False

    try:
        history = _copy_entry_attempt_history(previous)
    except EntryIntentReplayBlocked as exc:
        same_top_level = str(previous.get("logical_intent_id", "")) == logical_id
        same_prediction = (
            previous.get("pred_id") not in (None, "")
            and sig.get("pred_id") not in (None, "")
            and str(previous.get("pred_id")) == str(sig.get("pred_id"))
        )
        if same_top_level or same_prediction:
            log.critical(
                "  %s: matching persisted entry history is malformed; no exchange read or submit (%s)",
                symbol,
                exc,
            )
            return True
        return False

    matching = [
        item for item in history
        if str(item.get("logical_intent_id", "")) == logical_id
    ]
    if not matching:
        if str(previous.get("logical_intent_id", "")) == logical_id:
            log.critical(
                "  %s: persisted logical intent has no corresponding attempt history; fail-closed",
                symbol,
            )
            return True
        return False

    unresolved_states = {"SUBMITTING", "ACKNOWLEDGED", "UNKNOWN", "SUBMISSION_UNKNOWN"}
    unresolved = [
        item for item in matching
        if str(item.get("submission_state", "")).strip().upper() in unresolved_states
    ]
    if not unresolved:
        return False

    def attempt_number(item):
        value = item.get("attempt_number")
        if isinstance(value, bool):
            raise ValueError("boolean attempt number")
        return int(value)

    try:
        latest = max(matching, key=attempt_number)
        if len(unresolved) != 1 or unresolved[0].get("attempt_id") != latest.get("attempt_id"):
            raise ValueError("multiple or non-latest unresolved attempts")
        attempt_id = str(latest.get("attempt_id", "")).strip()
        expected_label = _entry_attempt_client_label(logical_id, attempt_id)
        if (
            not attempt_id
            or str(latest.get("client_label", "")) != expected_label
            or str(previous.get("logical_intent_id", "")) != logical_id
            or str(previous.get("entry_attempt_id", "")) != attempt_id
            or not isinstance(previous.get("entry_attempt"), dict)
            or str(previous["entry_attempt"].get("attempt_id", "")) != attempt_id
            or str(previous["entry_attempt"].get("client_label", "")) != expected_label
        ):
            raise ValueError("persisted attempt ID or client-label identity mismatch")
        stored_order_ids = previous.get("order_ids")
        stored_labels = previous.get("order_client_labels")
        if (
            not isinstance(stored_order_ids, dict)
            or str(stored_order_ids.get("entry_label", "")) != expected_label
            or not isinstance(stored_labels, dict)
            or str(stored_labels.get("entry", "")) != expected_label
        ):
            raise ValueError("persisted order-label identity mismatch")
        contract = previous.get("entry_order_contract")
        if not isinstance(contract, dict) or str(contract.get("label", "")) != expected_label:
            raise ValueError("persisted entry order contract does not match attempt label")
    except Exception as exc:
        log.critical(
            "  %s: unresolved entry attempt identity is inconsistent; no exchange read or submit (%s)",
            symbol,
            type(exc).__name__,
        )
        return True

    # The helper is explicitly read-only. Any exception or malformed result is
    # recorded as an incomplete reconciliation and remains UNKNOWN.
    try:
        evidence = _reconcile_unknown_entry_by_label_read_only(
            deribit, symbol, expected_label
        )
        if not isinstance(evidence, dict) or evidence.get("read_only") is not True:
            raise ValueError("read-only reconciliation returned malformed evidence")
    except Exception as exc:
        evidence = {
            "read_only": True,
            "client_label": expected_label,
            "symbol": symbol,
            "read_errors": [{"source": "resume_reconciliation", "error": type(exc).__name__}],
            "matched_order_ids": [],
            "complete": False,
        }

    checked_at = datetime.now(timezone.utc).isoformat()
    matched_ids = evidence.get("matched_order_ids", [])
    if not isinstance(matched_ids, list):
        matched_ids = []
    matched_ids = sorted({str(value).strip() for value in matched_ids if str(value).strip()})

    record = dict(previous)
    record["entry_attempt_history"] = history
    record["order_ids"] = dict(stored_order_ids)
    if len(matched_ids) == 1:
        record["order_ids"]["entry"] = matched_ids[0]
        record["order_ids"].pop("entry_label_matches", None)
    elif matched_ids:
        record["order_ids"]["entry_label_matches"] = matched_ids
    record["order_ids"]["entry_label"] = expected_label

    _update_entry_attempt_history(
        record,
        attempt_id,
        submission_state="UNKNOWN",
        reconciliation_read_only=True,
        reconciliation_checked_at=checked_at,
        reconciliation_matched_order_ids=matched_ids,
        reconciliation_evidence=evidence,
    )
    record["entry_submission_state"] = "UNKNOWN"
    submission_states = record.get("order_submission_states")
    if not isinstance(submission_states, dict):
        submission_states = {}
    else:
        submission_states = dict(submission_states)
    submission_states["entry"] = "UNKNOWN"
    record["order_submission_states"] = submission_states
    record["manual_reconciliation_required"] = True
    record["protection_reconciliation_required"] = True
    record["protection_reconciliation_reason"] = (
        "persisted entry attempt resumed after restart; read-only evidence recorded; operator disposition required"
    )
    record["entry_reconciliation_evidence"] = evidence
    record["entry_reconciliation_read_only"] = True
    record["entry_reconciliation_checked_at"] = checked_at
    record["entry_reconciliation_matched_order_ids"] = matched_ids

    if not _persist_trade_snapshot(symbol, record):
        log.critical(
            "  %s: resumed read-only reconciliation could not be persisted; attempt remains blocked",
            symbol,
        )
    else:
        log.critical(
            "  %s: resumed persisted attempt %s using existing label; matches=%d; remains UNKNOWN; no submit/cancel",
            symbol,
            attempt_id,
            len(matched_ids),
        )
    return True

def _confirm_terminal_entry_order(deribit, order_result: dict, requested_qty: float, expected_contract: dict = None) -> dict:
    """Resolve one IOC entry to a verified terminal exchange state.

    An open/partial IOC is cancelled once and then fetched once more. A missing
    order ID, unreadable state, or non-terminal post-cancel state is UNKNOWN;
    no order is submitted again.
    """
    if not isinstance(order_result, dict):
        raise OrderOutcomeUnknown("entry submission returned no structured response")
    immediate = order_result.get("order", order_result)
    if not isinstance(immediate, dict):
        raise OrderOutcomeUnknown("entry submission returned no order object")
    order_id = str(immediate.get("order_id", "")).strip()
    if not order_id:
        raise OrderOutcomeUnknown("entry acknowledgement has no exchange order ID")

    try:
        observed = deribit.get_order(order_id)
    except Exception:
        raise OrderOutcomeUnknown("entry order state could not be read after submission") from None
    if not isinstance(observed, dict) or str(observed.get("order_id", "")) != order_id:
        raise OrderOutcomeUnknown("entry order state identity mismatch")
    if expected_contract is not None:
        if not isinstance(expected_contract, dict):
            raise OrderOutcomeUnknown("entry expected contract is malformed")
        _assert_entry_order_contract(immediate, observed, expected_contract)

    state = str(observed.get("order_state", "")).strip().lower()
    terminal_states = {"filled", "cancelled", "canceled", "rejected", "closed", "expired"}
    if state not in terminal_states:
        # A live or partially-filled IOC must be cancelled before protection
        # sizing. Never treat a requested quantity as an executed quantity.
        try:
            deribit.cancel_order(order_id)
        except Exception:
            # The cancel response may itself be ambiguous; perform only a read.
            pass
        try:
            observed = deribit.get_order(order_id)
        except Exception:
            raise OrderOutcomeUnknown("entry cancel outcome could not be reconciled") from None
        if not isinstance(observed, dict) or str(observed.get("order_id", "")) != order_id:
            raise OrderOutcomeUnknown("entry post-cancel order state identity mismatch")
        if expected_contract is not None:
            _assert_entry_order_contract(immediate, observed, expected_contract)
        state = str(observed.get("order_state", "")).strip().lower()
        if state not in terminal_states:
            raise OrderOutcomeUnknown("entry order remains non-terminal after one cancel/reconcile cycle")

    raw_filled = observed.get("filled_amount")
    if raw_filled is None or isinstance(raw_filled, bool):
        raise OrderOutcomeUnknown("terminal entry order has no confirmed filled_amount")
    try:
        filled = float(raw_filled)
        requested = float(requested_qty)
    except (TypeError, ValueError, OverflowError):
        raise OrderOutcomeUnknown("terminal entry order quantity is invalid") from None
    if not math.isfinite(filled) or not math.isfinite(requested) or filled < 0 or requested <= 0:
        raise OrderOutcomeUnknown("terminal entry order quantity is non-finite or negative")
    if filled > requested + max(1e-12, requested * 1e-8):
        raise OrderOutcomeUnknown("terminal entry filled quantity exceeds requested quantity")
    if state == "rejected" and filled > 0:
        raise OrderOutcomeUnknown("rejected entry order unexpectedly reports a non-zero fill")
    return observed


def execute_trade(deribit: DeribitClient, sig: dict, risk_mult: float, balance: float, vol_state: str = "NORMAL", base_min_conf: float = None) -> bool:
    # Read-only recovery is intentionally ahead of all new-entry gates. It must
    # remain available after a restart even in predict/manage-only modes, and it
    # must stop this signal before any new order can be considered.
    if _resume_pending_entry_attempt_read_only(deribit, sig):
        return False
    if not _deribit_testnet_mutations_enabled() or not _execution_allows_new_entries():
        log.warning(
            f"  [EXECUTION_MODE:{get_execution_mode()}] "
            f"New entry blocked for {sig.get('symbol', 'UNKNOWN')}"
        )
        return False
    if base_min_conf is None:
        raise ValueError("execute_trade() requires base_min_conf — no implicit default allowed.")
    from deribit_client import SYMBOL_MAP
    symbol = str(sig.get("symbol", "")).upper()
    signal = str(sig.get("signal", "")).upper()
    settlement_currency = str(SYMBOL_MAP.get(symbol, {}).get("currency", "")).upper()
    if signal not in {"BUY", "SELL"} or settlement_currency != "USDC":
        log.warning("  [SKIP] Entry blocked: invalid signal or non-USDC settlement symbol.")
        return False
    try:
        from capital_feasibility import resolve_active_capital_tier_policy
        tier_policy = resolve_active_capital_tier_policy(config, environment="testnet")
    except Exception as tier_error:
        log.critical("  [HALT] Active capital-tier policy is missing or invalid (%s)", type(tier_error).__name__)
        return False
    if symbol not in set(tier_policy["allowed_symbols"]):
        log.info("  [SKIP] %s is outside active capital tier %s", symbol, tier_policy["name"])
        return False
    # Account equity and available funds are refreshed immediately before
    # sizing. The scanner-level balance argument is intentionally not trusted.
    balance = 0.0
    risk_snapshot = None

    try:
        model_entry = float(sig["entry"])
        atr = float(sig["atr"])
    except (TypeError, ValueError, OverflowError, KeyError):
        log.warning("  [SKIP] Signal price/ATR is missing or invalid for %s", symbol)
        return False
    if not all(math.isfinite(value) and value > 0 for value in (model_entry, atr)):
        log.warning("  [SKIP] Signal price/ATR is non-positive or non-finite for %s", symbol)
        return False
    entry = model_entry

    sym_overrides = get_symbol_overrides(symbol)
    effective_stop_mult = ATR_STOP_MULT
    if "atr_stop_mult_override" in sym_overrides:
        effective_stop_mult = ATR_STOP_MULT * sym_overrides["atr_stop_mult_override"].get("multiplier", 1.0)
        log.info(f"  🧪 {symbol}: applying approved stop-mult override — {ATR_STOP_MULT} → {effective_stop_mult:.2f}")

    trades = load_trades()
    unresolved = [
        name for name, item in trades.items()
        if isinstance(item, dict) and (
            (
                not item.get("closed", False)
                and (
                    item.get("emergency_recovery_required")
                    or item.get("manual_reconciliation_required")
                    or item.get("protection_reconciliation_required")
                )
            )
            or _has_unresolved_entry_attempt(item)
        )
    ]
    if unresolved:
        log.critical(
            "  [HALT] New entries blocked by unresolved execution reconciliation: %s",
            ", ".join(sorted(map(str, unresolved))),
        )
        return False
    open_count = len([t for t in trades.values() if not t.get("closed", False)])

    max_open_trades = int(tier_policy["max_positions"])
    max_same_dir    = int(getattr(config, "MAX_SAME_DIRECTION", 4))

    if open_count >= max_open_trades:
        log.info(f"  🛑 MAX TRADES ({open_count}/{max_open_trades}) — skip {symbol}")
        return False

    same_dir_count = sum(1 for t in trades.values() if not t.get("closed", False) and t.get("signal") == signal)
    if same_dir_count >= max_same_dir:
        log.info(f"  🛑 MAX SAME DIRECTION ({same_dir_count}/{max_same_dir} {signal}) — skip {symbol}")
        return False

    if not check_correlation(trades, signal, symbol):
        log.info(f"  🛑 PORTFOLIO CORRELATION/SECTOR ADMISSION — skip {symbol}")
        return False

    if symbol in trades and not trades[symbol].get("closed", False):
        log.info(f"  {symbol}: already open in trades.json — skip")
        return False

    # Reject a replay of the same prediction even when its previous trade is
    # closed. Any retry must pass the attempt-scoped authorization gate below.
    try:
        logical_entry_id = _logical_entry_intent_id(sig, symbol, signal)
    except ValueError as identity_error:
        log.critical("  %s: entry identity unavailable (%s); no submission", symbol, identity_error)
        return False
    prior_symbol_trade = trades.get(symbol, {})
    if _legacy_entry_intent_matches(
        prior_symbol_trade, sig, symbol, signal, logical_entry_id
    ):
        prior_history = _copy_entry_attempt_history(prior_symbol_trade)
        if not any(item.get("logical_intent_id") == logical_entry_id for item in prior_history):
            log.critical(
                "  %s: matching legacy prediction record has no verified attempt disposition; refusing replay",
                symbol,
            )
            return False

    min_lot = float(deribit.get_min_trade_amount(symbol))
    if not math.isfinite(min_lot) or min_lot <= 0:
        log.critical(f"  🚨 {symbol}: invalid exchange minimum lot {min_lot!r} — FAIL-CLOSED")
        return False

    # ── DUST GUARD: Sub-lot remnants (< min_lot) will NOT block valid setups ──
    try:
        pre_entry_signed_position = float(deribit.get_position_size(symbol))
        if not math.isfinite(pre_entry_signed_position):
            raise ValueError("pre-entry position is non-finite")
        real_pos = abs(pre_entry_signed_position)
    except Exception as e:
        log.critical(
            "  %s: real position check failed (%s) — FAIL-CLOSED, skip entry",
            symbol,
            type(e).__name__,
        )
        return False

    if real_pos >= min_lot:
        log.warning(f"  🚫 {symbol}: real exchange position ({real_pos}) >= min lot ({min_lot}) — SKIPPING entry.")
        return False
    elif real_pos > 0:
        # A verified dust disposition may allow unrelated symbols through the
        # portfolio risk gate, but never pyramid into the same residual symbol.
        log.info(
            "  %s: non-zero sub-minimum residual remains; skipping this symbol only. "
            "Other symbols must still pass the portfolio dust-reserve check.",
            symbol,
        )
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
        log.warning(f"  🚫 {symbol}: entry spread {spread_info.get('spread_pct', 0)*100:.2f}% > ceiling — skip entry")
        return False

    if risk_mult <= 0:
        log.info(f"  🛑 Risk multiplier is {risk_mult} — skip {symbol}")
        return False

    live_price = deribit.get_live_price(symbol)
    if not math.isfinite(float(live_price)) or float(live_price) <= 0:
        log.warning(f"  🚫 {symbol}: live execution price unavailable/invalid — FAIL-CLOSED, no entry")
        return False
    drift = abs(float(live_price) - entry) / entry * 100
    if drift > 0.5:
        log.warning(f"  [EXPIRED] {symbol} price drifted {drift:.2f}% from signal — skip")
        return False
    entry = float(live_price)

    dyn_tp1 = ATR_TARGET1_MULT
    dyn_tp2 = ATR_TARGET2_MULT
    if vol_state == "VERY_HIGH":
        dyn_tp1 *= 1.5; dyn_tp2 *= 1.5
    elif vol_state in ("DEAD", "UNKNOWN"):
        dyn_tp1 *= 0.8; dyn_tp2 *= 0.8

    side    = "BUY" if signal == "BUY" else "SELL"
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"

    def _directional_tick_price(raw_price, rounding):
        raw_decimal = Decimal(str(raw_price))
        for _ in range(3):
            tick = Decimal(str(deribit.get_tick_size(symbol, price=float(raw_decimal))))
            rounded = round_price_to_tick(raw_decimal, tick, rounding=rounding)
            next_tick = Decimal(str(deribit.get_tick_size(symbol, price=float(rounded))))
            if next_tick == tick:
                return float(rounded), tick
            raw_decimal = rounded
        raise ValueError("Dynamic exchange tick size did not converge")

    try:
        if signal == "BUY":
            stop_raw = Decimal(str(entry)) - Decimal(str(atr)) * Decimal(str(effective_stop_mult))
            tp1_raw = Decimal(str(entry)) + Decimal(str(atr)) * Decimal(str(dyn_tp1))
            tp2_raw = Decimal(str(entry)) + Decimal(str(atr)) * Decimal(str(dyn_tp2))
            stop, stop_tick = _directional_tick_price(stop_raw, "DOWN")
            tp1, tp1_tick = _directional_tick_price(tp1_raw, "DOWN")
            tp2, tp2_tick = _directional_tick_price(tp2_raw, "DOWN")
        else:
            stop_raw = Decimal(str(entry)) + Decimal(str(atr)) * Decimal(str(effective_stop_mult))
            tp1_raw = Decimal(str(entry)) - Decimal(str(atr)) * Decimal(str(dyn_tp1))
            tp2_raw = Decimal(str(entry)) - Decimal(str(atr)) * Decimal(str(dyn_tp2))
            stop, stop_tick = _directional_tick_price(stop_raw, "UP")
            tp1, tp1_tick = _directional_tick_price(tp1_raw, "UP")
            tp2, tp2_tick = _directional_tick_price(tp2_raw, "UP")
        dec = max(max(0, -tick.normalize().as_tuple().exponent) for tick in (stop_tick, tp1_tick, tp2_tick))
    except Exception as tick_error:
        log.critical("  [HALT] Exchange tick rounding failed for %s (%s)", symbol, type(tick_error).__name__)
        return False

    for name, val in [("stop", stop), ("tp1", tp1), ("tp2", tp2)]:
        if not math.isfinite(val) or val <= 0:
            log.error(f"  🚨 {symbol}: computed {name}={val!r} is invalid — aborting trade")
            return False

    if signal == "BUY" and not (stop < entry < tp1 <= tp2):
        log.critical("  [HALT] BUY stop/target tick geometry invalid for %s", symbol)
        return False
    if signal == "SELL" and not (stop > entry > tp1 >= tp2):
        log.critical("  [HALT] SELL stop/target tick geometry invalid for %s", symbol)
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
    else:
        risk_boost = 1.5 if sig.get("conf_tier") == "high" else 1.0
        final_risk_mult = risk_mult * risk_boost * vol_scalar

    # Refresh account capital and the complete positions snapshot immediately
    # before each eligible entry. A prior symbol's collateral may no longer be
    # available after its order has filled.
    balance = save_balance(deribit)
    risk_snapshot = getattr(deribit, "_last_risk_capital_snapshot", None)
    observed_at = risk_snapshot.get("_observed_at_epoch") if isinstance(risk_snapshot, dict) else None
    observed_monotonic = risk_snapshot.get("_observed_at_monotonic") if isinstance(risk_snapshot, dict) else None
    try:
        snapshot_equity = float(risk_snapshot.get("risk_equity"))
        available_funds = float(risk_snapshot.get("available_funds"))
        snapshot_age = time.monotonic() - float(observed_monotonic)
        balance_value = float(balance)
        risk_asof = datetime.fromtimestamp(float(observed_at), timezone.utc)
    except (TypeError, ValueError, OverflowError, AttributeError):
        log.critical("  [HALT] Fresh USDC risk snapshot missing or malformed for %s", symbol)
        return False
    if (
        risk_snapshot.get("entry_enabled") is not True
        or str(risk_snapshot.get("risk_currency", "")).upper() != "USDC"
        or str(risk_snapshot.get("margin_mode", "")).lower() != "segregated"
        or not all(math.isfinite(value) for value in (snapshot_equity, available_funds, snapshot_age, balance_value))
        or snapshot_equity <= 0 or available_funds <= 0
        or snapshot_age < 0 or snapshot_age > 15
        or not math.isclose(snapshot_equity, balance_value, rel_tol=1e-9, abs_tol=1e-7)
    ):
        log.critical("  [HALT] Fresh per-entry account snapshot failed validation for %s", symbol)
        return False

    daily_loss_limit = tier_policy["max_daily_loss_fraction"]
    daily_loss_amount_value = tier_policy.get("max_daily_loss_usdc")
    daily_guard = evaluate_daily_risk_guard(
        deribit, current_equity=Decimal(str(snapshot_equity)),
        max_daily_loss_fraction=daily_loss_limit,
        max_daily_loss_amount=(
            Decimal(str(daily_loss_amount_value)) if daily_loss_amount_value is not None else None
        ),
        now=risk_asof,
    )
    if not daily_guard.allowed:
        log.critical(
            "  [HALT] Daily USDC risk gate blocks %s: %s (loss_fraction=%s)",
            symbol, daily_guard.reason, daily_guard.loss_fraction,
        )
        return False

    current_positions = getattr(deribit, "_last_risk_positions_snapshot", None)
    fresh_trades = load_trades()
    fresh_unresolved = [
        name for name, item in fresh_trades.items()
        if isinstance(item, dict) and (
            item.get("emergency_recovery_required")
            or item.get("manual_reconciliation_required")
            or item.get("protection_reconciliation_required")
            or _has_unresolved_entry_attempt(item)
        )
    ]
    if fresh_unresolved:
        log.critical("  [HALT] Fresh trade ledger has unresolved records: %s", ", ".join(sorted(map(str, fresh_unresolved))))
        return False
    try:
        open_portfolio_risk = _verified_open_portfolio_risk_usdc(
            deribit, fresh_trades, current_positions,
        )
    except Exception as portfolio_error:
        log.critical(
            "  [HALT] Portfolio exposure cannot be verified for %s (%s)",
            symbol, type(portfolio_error).__name__,
        )
        return False

    # A previously verified dust residual never blocks unrelated symbols, but
    # it must block a fresh entry on the same instrument until the exchange is flat.
    prior_dust = fresh_trades.get(symbol)
    if isinstance(prior_dust, dict) and isinstance(prior_dust.get("dust_residual"), dict):
        matching_position = next((
            item for item in current_positions
            if isinstance(item, dict)
            and str(item.get("instrument_name") or "") == str(SYMBOL_MAP.get(symbol, {}).get("instrument") or "")
            and float(item.get("size_currency") or 0) > 0
        ), None)
        if matching_position is not None:
            log.info("  [SKIP] %s still has a verified dust residual; same-symbol entries remain disabled", symbol)
            return False

    current_open_count = sum(
        1 for item in fresh_trades.values()
        if isinstance(item, dict) and not item.get("closed", False)
    )
    tier_position_limit = int(tier_policy["max_positions"])
    if tier_position_limit < 1 or current_open_count >= tier_position_limit:
        log.info("  [SKIP] Capital-tier position cap reached (%d/%d)", current_open_count, tier_position_limit)
        return False

    portfolio_risk_cap_fraction = tier_policy["max_portfolio_risk_fraction"]
    if (
        not portfolio_risk_cap_fraction.is_finite()
        or portfolio_risk_cap_fraction <= 0
        or portfolio_risk_cap_fraction > Decimal("0.05")
    ):
        log.critical("  [HALT] MAX_PORTFOLIO_RISK_FRACTION is invalid")
        return False
    portfolio_risk_cap = Decimal(str(snapshot_equity)) * portfolio_risk_cap_fraction
    portfolio_risk_amount_value = tier_policy.get("max_portfolio_risk_usdc")
    if portfolio_risk_amount_value is not None:
        portfolio_risk_amount = Decimal(str(portfolio_risk_amount_value))
        if not portfolio_risk_amount.is_finite() or portfolio_risk_amount <= 0:
            log.critical("  [HALT] CAPITAL_TIER_MAX_PORTFOLIO_RISK_USDC is invalid")
            return False
        portfolio_risk_cap = min(portfolio_risk_cap, portfolio_risk_amount)

    try:
        size_result, instrument_spec, taker_fee_rate, effective_risk_fraction, sizing_equity = _calculate_deribit_decimal_size(
            deribit,
            symbol,
            risk_snapshot,
            entry_price=entry,
            stop_price=stop,
            risk_multiplier=final_risk_mult,
            side=side,
            tier_policy=tier_policy,
        )
    except Exception as sizing_error:
        log.warning(
            "  [SKIP] Decimal risk sizing rejected %s (%s): %s",
            symbol,
            type(sizing_error).__name__,
            sizing_error,
        )
        return False

    if not size_result.accepted:
        log.warning(
            "  [SKIP] Decimal sizing says no trade: symbol=%s reason=%s",
            symbol,
            size_result.reason,
        )
        return False

    combined_portfolio_risk = Decimal(str(open_portfolio_risk)) + size_result.total_risk
    if combined_portfolio_risk > portfolio_risk_cap:
        log.warning(
            "  [SKIP] Aggregate portfolio stop-risk %.8f exceeds cap %.8f USDC",
            float(combined_portfolio_risk), float(portfolio_risk_cap),
        )
        return False

    total_q = float(size_result.quantity)
    budget_risk_usd = float(sizing_equity * effective_risk_fraction)
    pretrade_risk_usd = float(size_result.stop_risk)
    pretrade_total_risk_usd = float(size_result.total_risk)
    if (
        not all(math.isfinite(value) for value in (total_q, budget_risk_usd, pretrade_risk_usd, pretrade_total_risk_usd))
        or total_q < min_lot
        or pretrade_risk_usd <= 0
        or pretrade_total_risk_usd <= 0
        or pretrade_total_risk_usd > budget_risk_usd * 1.000001
    ):
        log.critical(
            "  %s: Decimal pre-trade risk invariant failed | stop_risk=%.8f "
            "fees=%.8f total=%.8f budget=%.8f USDC qty=%.12g",
            symbol,
            pretrade_risk_usd,
            float(size_result.estimated_round_trip_fees),
            pretrade_total_risk_usd,
            budget_risk_usd,
            total_q,
        )
        return False

    qty_tp1, qty_tp2 = deribit.split_amount(symbol, total_q)
    exit_mode = "DUAL_TP" if qty_tp2 > 0 else "SINGLE_TP"

    actual_risk_usd = round(pretrade_risk_usd, 8)
    estimated_round_trip_fees = float(size_result.estimated_round_trip_fees)
    total_risk_with_fees = float(size_result.total_risk)

    log.info(
        f"  {signal} {symbol} Total={total_q} (TP1={qty_tp1}, TP2={qty_tp2} | Mode={exit_mode}) "
        f"| Stop Risk={actual_risk_usd:.4f} USDC | Estimated Fees={estimated_round_trip_fees:.4f} USDC"
    )
    log.info(f"  SL={stop:.{dec}f} TP1={tp1:.{dec}f} TP2={tp2:.{dec}f}")

    order_ids = {}
    actual_entry = entry
    try:
        # Daily transaction-log reconciliation and portfolio reads can take
        # time. Re-check freshness at the final admission boundary so the
        # account snapshot cannot age out between sizing and durable intent.
        final_snapshot_age = time.monotonic() - float(observed_monotonic)
        if not math.isfinite(final_snapshot_age) or final_snapshot_age < 0 or final_snapshot_age > 15:
            log.critical("  [HALT] Account snapshot aged out before entry intent for %s", symbol)
            return False
        # Do not mutate account leverage on a normal entry. Sizing is based
        # on verified USDC collateral and never assumes exchange leverage.

        try:
            attempt_identity = _prepare_entry_attempt(
                sig, symbol, signal, total_q, previous_trade=trades.get(symbol)
            )
        except EntryIntentReplayBlocked as replay_error:
            log.critical("  %s: entry replay blocked (%s); no new order attempt", symbol, replay_error)
            return False
        client_label = attempt_identity["client_label"]
        attempt_id = attempt_identity["attempt_id"]
        order_ids["entry_label"] = client_label
        entry_contract = {
            "instrument_name": str(deribit.get_instrument_name(symbol)),
            "direction": str(side).lower(),
            "amount": float(total_q),
            "order_type": "limit",
            "time_in_force": "immediate_or_cancel",
            "reduce_only": False,
            "label": client_label,
        }
        # Every attempt retains its own immutable pre-submit order contract;
        # later attempts never overwrite the evidence for an earlier one.
        attempt_identity["attempt"]["order_contract"] = dict(entry_contract)
        attempt_identity["history"][-1]["order_contract"] = dict(entry_contract)
        entry_intent_record = _build_entry_submission_intent(
            sig, symbol, signal, model_entry, entry, total_q, client_label,
            attempt=attempt_identity["attempt"],
            attempt_history=attempt_identity["history"],
        )
        entry_intent_record["entry_order_contract"] = dict(entry_contract)
        trades[symbol] = entry_intent_record
        if not _persist_trade_snapshot(symbol, entry_intent_record):
            log.critical(
                "  %s: entry intent could not be persisted; no entry order submitted",
                symbol,
            )
            return False

        try:
            er = deribit.place_market_order(
                symbol, side, total_q, client_label=client_label
            )
            if not isinstance(er, dict):
                raise OrderOutcomeUnknown("entry submission returned no structured response")
        except DeribitRPCError as rpc_error:
            try:
                position_evidence = _verify_entry_position_unchanged(
                    deribit, symbol, pre_entry_signed_position
                )
            except OrderOutcomeUnknown as position_error:
                _update_entry_attempt_history(
                    entry_intent_record, attempt_id,
                    submission_state="UNKNOWN",
                    exchange_rejection={"method": rpc_error.method, "code": rpc_error.code},
                    position_check_error=str(position_error),
                )
                entry_intent_record["order_submission_states"]["entry"] = "UNKNOWN"
                entry_intent_record["entry_submission_state"] = "UNKNOWN"
                entry_intent_record["manual_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_reason"] = "RPC rejection followed by an unverified position snapshot"
                _persist_trade_snapshot(symbol, entry_intent_record)
                _retain_emergency_recovery_record(
                    trades, sig, symbol, signal, entry, 0.0, order_ids,
                    "RPC rejection but post-rejection position could not be verified",
                    exchange_quantity_verified=False,
                    reconciliation_evidence={"position_check_error": str(position_error)},
                )
                return False
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                submission_state="EXCHANGE_REJECTED_ZERO_FILL",
                exchange_rejection={"method": rpc_error.method, "code": rpc_error.code},
                zero_fill_position_unchanged=True,
                zero_fill_position_evidence=position_evidence,
                resolved_at=datetime.now(timezone.utc).isoformat(),
            )
            entry_intent_record["order_submission_states"]["entry"] = "REJECTED"
            entry_intent_record["entry_submission_state"] = "REJECTED"
            entry_intent_record["entry_rejection"] = {"method": rpc_error.method, "code": rpc_error.code}
            entry_intent_record["entry_zero_fill_position_check"] = position_evidence
            entry_intent_record["closed"] = True
            entry_intent_record["manual_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_reason"] = None
            if not _persist_trade_snapshot(symbol, entry_intent_record):
                log.critical("  %s: confirmed RPC rejection disposition could not be persisted", symbol)
            else:
                log.warning("  %s: Deribit explicitly rejected entry (%s/%s); no retry", symbol, rpc_error.method, rpc_error.code)
            return False
        except PermissionError as preflight_error:
            try:
                position_evidence = _verify_entry_position_unchanged(
                    deribit, symbol, pre_entry_signed_position
                )
            except OrderOutcomeUnknown as position_error:
                _update_entry_attempt_history(
                    entry_intent_record, attempt_id,
                    submission_state="UNKNOWN",
                    preflight_error=type(preflight_error).__name__,
                    position_check_error=str(position_error),
                )
                entry_intent_record["order_submission_states"]["entry"] = "UNKNOWN"
                entry_intent_record["entry_submission_state"] = "UNKNOWN"
                entry_intent_record["manual_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_reason"] = "preflight block followed by an unverified position snapshot"
                _persist_trade_snapshot(symbol, entry_intent_record)
                _retain_emergency_recovery_record(
                    trades, sig, symbol, signal, entry, 0.0, order_ids,
                    "preflight block but post-check position could not be verified",
                    exchange_quantity_verified=False,
                    reconciliation_evidence={"position_check_error": str(position_error)},
                )
                return False
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                submission_state="PREFLIGHT_BLOCKED",
                preflight_error=type(preflight_error).__name__,
                zero_fill_position_unchanged=True,
                zero_fill_position_evidence=position_evidence,
                resolved_at=datetime.now(timezone.utc).isoformat(),
            )
            entry_intent_record["order_submission_states"]["entry"] = "PREFLIGHT_BLOCKED"
            entry_intent_record["entry_submission_state"] = "PREFLIGHT_BLOCKED"
            entry_intent_record["entry_zero_fill_position_check"] = position_evidence
            entry_intent_record["closed"] = True
            entry_intent_record["manual_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_reason"] = None
            if not _persist_trade_snapshot(symbol, entry_intent_record):
                log.critical("  %s: preflight block disposition could not be persisted", symbol)
            else:
                log.warning("  %s: entry blocked before submission (%s); no order was sent", symbol, type(preflight_error).__name__)
            return False
        except OrderOutcomeUnknown as unknown_error:
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                submission_state="UNKNOWN",
                unknown_error=type(unknown_error).__name__,
            )
            entry_intent_record["order_submission_states"]["entry"] = "UNKNOWN"
            entry_intent_record["entry_submission_state"] = "UNKNOWN"
            entry_intent_record["manual_reconciliation_required"] = True
            entry_intent_record["protection_reconciliation_required"] = True
            entry_intent_record["protection_reconciliation_reason"] = "entry submission outcome unknown"
            _persist_trade_snapshot(symbol, entry_intent_record)
            reconciliation = _reconcile_unknown_entry_by_label_read_only(
                deribit, symbol, client_label
            )
            order_ids["entry_label"] = client_label
            matched_ids = reconciliation.get("matched_order_ids", [])
            if len(matched_ids) == 1:
                order_ids["entry"] = matched_ids[0]
            elif matched_ids:
                order_ids["entry_label_matches"] = matched_ids
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                reconciliation_matched_order_ids=list(matched_ids),
                reconciliation_read_only=True,
            )
            _persist_trade_snapshot(symbol, entry_intent_record)
            log.critical(
                "  %s: entry outcome UNKNOWN (%s); read-only label/position/trade reconciliation completed; no resubmit",
                symbol,
                type(unknown_error).__name__,
            )
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, entry, 0.0, order_ids,
                "entry outcome unknown; read-only reconciliation recorded; operator acknowledgement required",
                exchange_quantity_verified=False,
                reconciliation_evidence=reconciliation,
            )
            return False
        if not er:
            try:
                position_evidence = _verify_entry_position_unchanged(
                    deribit, symbol, pre_entry_signed_position
                )
            except OrderOutcomeUnknown as position_error:
                _update_entry_attempt_history(
                    entry_intent_record, attempt_id,
                    submission_state="UNKNOWN",
                    preflight_result="empty_response",
                    position_check_error=str(position_error),
                )
                entry_intent_record["order_submission_states"]["entry"] = "UNKNOWN"
                entry_intent_record["entry_submission_state"] = "UNKNOWN"
                entry_intent_record["manual_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_reason"] = "empty preflight response followed by an unverified position snapshot"
                _persist_trade_snapshot(symbol, entry_intent_record)
                _retain_emergency_recovery_record(
                    trades, sig, symbol, signal, entry, 0.0, order_ids,
                    "empty entry preflight response but position change/read failure detected",
                    exchange_quantity_verified=False,
                    reconciliation_evidence={"position_check_error": str(position_error)},
                )
                return False
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                submission_state="PREFLIGHT_BLOCKED",
                preflight_result="empty_response",
                zero_fill_position_unchanged=True,
                zero_fill_position_evidence=position_evidence,
                resolved_at=datetime.now(timezone.utc).isoformat(),
            )
            log.warning(f"  {symbol}: local entry preflight blocked the request; no exchange order was sent")
            entry_intent_record["order_submission_states"]["entry"] = "PREFLIGHT_BLOCKED"
            entry_intent_record["entry_submission_state"] = "PREFLIGHT_BLOCKED"
            entry_intent_record["entry_zero_fill_position_check"] = position_evidence
            entry_intent_record["closed"] = True
            entry_intent_record["manual_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_reason"] = None
            _persist_trade_snapshot(symbol, entry_intent_record)
            return False

        immediate_order = er.get("order", er) if isinstance(er, dict) else {}
        immediate_order_id = str(immediate_order.get("order_id", "")).strip() if isinstance(immediate_order, dict) else ""
        if immediate_order_id:
            order_ids["entry"] = immediate_order_id
        entry_contract["price"] = immediate_order.get("price") if isinstance(immediate_order, dict) else None
        entry_intent_record["entry_order_contract"] = dict(entry_contract)
        entry_intent_record["order_ids"] = dict(order_ids)
        _update_entry_attempt_history(
            entry_intent_record, attempt_id,
            submission_state="ACKNOWLEDGED",
            exchange_order_id=immediate_order_id or None,
            order_contract=dict(entry_contract),
            acknowledged_at=datetime.now(timezone.utc).isoformat(),
        )
        entry_intent_record["order_submission_states"]["entry"] = "ACKNOWLEDGED"
        entry_intent_record["entry_submission_state"] = "ACKNOWLEDGED"
        if not _persist_trade_snapshot(symbol, entry_intent_record):
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                submission_state="UNKNOWN",
                unknown_error="acknowledgement_persistence_failed",
            )
            entry_intent_record["order_submission_states"]["entry"] = "UNKNOWN"
            entry_intent_record["entry_submission_state"] = "UNKNOWN"
            entry_intent_record["manual_reconciliation_required"] = True
            entry_intent_record["protection_reconciliation_required"] = True
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, entry, 0.0, order_ids,
                "entry acknowledgement could not be persisted; read-only reconciliation required",
                exchange_quantity_verified=False,
            )
            return False

        try:
            eo = _confirm_terminal_entry_order(
                deribit, er, total_q, expected_contract=entry_contract
            )
        except OrderOutcomeUnknown as unknown_error:
            immediate = er.get("order", er) if isinstance(er, dict) else {}
            if isinstance(immediate, dict) and immediate.get("order_id"):
                order_ids["entry"] = str(immediate.get("order_id"))
            entry_intent_record["order_ids"] = dict(order_ids)
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                submission_state="UNKNOWN",
                unknown_error=type(unknown_error).__name__,
            )
            entry_intent_record["order_submission_states"]["entry"] = "UNKNOWN"
            entry_intent_record["entry_submission_state"] = "UNKNOWN"
            entry_intent_record["manual_reconciliation_required"] = True
            entry_intent_record["protection_reconciliation_required"] = True
            entry_intent_record["protection_reconciliation_reason"] = "entry terminal state could not be verified"
            _persist_trade_snapshot(symbol, entry_intent_record)
            log.critical(
                "  %s: entry order remains UNKNOWN after read/cancel/reconcile (%s); no resubmit",
                symbol,
                type(unknown_error).__name__,
            )
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, entry, 0.0, order_ids,
                "entry state remains unknown; manual read-only reconciliation required",
                exchange_quantity_verified=False,
            )
            return False

        order_ids["entry"] = str(eo.get("order_id", ""))
        filled = float(eo.get("filled_amount", 0.0))
        o_state = str(eo.get("order_state", "")).lower()
        has_confirmed_fill = _entry_terminal_has_confirmed_fill(eo)
        if not has_confirmed_fill:
            try:
                position_evidence = _verify_entry_position_unchanged(
                    deribit, symbol, pre_entry_signed_position
                )
            except OrderOutcomeUnknown as position_error:
                _update_entry_attempt_history(
                    entry_intent_record, attempt_id,
                    submission_state="UNKNOWN",
                    terminal_state=o_state,
                    filled_qty=filled,
                    position_check_error=str(position_error),
                )
                entry_intent_record["order_submission_states"]["entry"] = "UNKNOWN"
                entry_intent_record["entry_submission_state"] = "UNKNOWN"
                entry_intent_record["manual_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_required"] = True
                entry_intent_record["protection_reconciliation_reason"] = "terminal zero-fill order but unchanged position could not be verified"
                _persist_trade_snapshot(symbol, entry_intent_record)
                _retain_emergency_recovery_record(
                    trades, sig, symbol, signal, entry, 0.0, order_ids,
                    "terminal zero-fill entry but signed position was changed or unreadable",
                    exchange_quantity_verified=False,
                    reconciliation_evidence={"position_check_error": str(position_error)},
                )
                return False
            _update_entry_attempt_history(
                entry_intent_record, attempt_id,
                submission_state="ZERO_FILL_VERIFIED",
                terminal_state=o_state,
                filled_qty=0.0,
                zero_fill_position_unchanged=True,
                zero_fill_position_evidence=position_evidence,
                resolved_at=datetime.now(timezone.utc).isoformat(),
            )
            entry_intent_record["entry_zero_fill_position_check"] = position_evidence
            log.warning("  %s: terminal entry state=%s filled=%.12g; unchanged position verified", symbol, o_state, filled)
            entry_intent_record["order_submission_states"]["entry"] = o_state.upper() or "CANCELLED"
            entry_intent_record["entry_submission_state"] = o_state.upper() or "CANCELLED"
            entry_intent_record["closed"] = True
            entry_intent_record["manual_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_required"] = False
            entry_intent_record["protection_reconciliation_reason"] = None
            _persist_trade_snapshot(symbol, entry_intent_record)
            return False

        entry_intent_record["order_submission_states"]["entry"] = (
            "FILLED" if o_state == "filled" else "PARTIALLY_FILLED_TERMINAL"
        )
        entry_intent_record["entry_submission_state"] = entry_intent_record["order_submission_states"]["entry"]
        entry_intent_record["filled_qty"] = filled
        entry_intent_record["order_ids"] = dict(order_ids)
        _update_entry_attempt_history(
            entry_intent_record, attempt_id,
            submission_state="FILLED" if o_state == "filled" else "PARTIALLY_FILLED_TERMINAL",
            terminal_state=o_state,
            filled_qty=filled,
            exchange_order_id=order_ids.get("entry"),
            resolved_at=datetime.now(timezone.utc).isoformat(),
        )
        if not _persist_trade_snapshot(symbol, entry_intent_record):
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, entry, 0.0, order_ids,
                "terminal entry fill could not be persisted; exchange position reconciliation required",
                exchange_quantity_verified=False,
            )
            return False

        actual_entry = float(deribit.get_order_fill_price(eo, 0.0) or 0.0)
        if actual_entry <= 0:
            actual_entry = float(deribit.get_fill_price(er, 0.0) or 0.0)
        if actual_entry > 0:
            entry_intent_record["entry"] = actual_entry
            entry_intent_record["actual_entry"] = actual_entry
            entry_intent_record["entry_fill_price"] = actual_entry
            if not _persist_trade_snapshot(symbol, entry_intent_record):
                _retain_emergency_recovery_record(
                    trades, sig, symbol, signal, entry, 0.0, order_ids,
                    "verified entry fill price could not be persisted; exchange position reconciliation required",
                    exchange_quantity_verified=False,
                )
                return False

        if actual_entry <= 0:
            log.critical(f"  🚨 {symbol}: NO AUTHORITATIVE ENTRY FILL — preserving unknown exposure without guessed size")
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, entry, 0.0, order_ids,
                "terminal entry fill has no authoritative average price",
                exchange_quantity_verified=False,
            )
            return False

        log.info(f"  ✅ Terminal entry @ {actual_entry:.{dec}f}; confirmed filled={filled}")
        
        position_confirmed = _wait_for_position(deribit, symbol, side, order_ids.get("entry", ""))
        if not position_confirmed:
            log.critical(
                "  %s: filled entry lacks a verified side-correct position snapshot; "
                "retaining recovery state",
                symbol,
            )
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, actual_entry, 0.0, order_ids,
                "post-entry position could not be verified",
                exchange_quantity_verified=False,
            )
            return False

        position_below_minimum = False
        try:
            signed_position_after_entry = float(deribit.get_position_size(symbol))
            if not math.isfinite(signed_position_after_entry):
                raise ValueError("post-entry position is non-finite")
            expected_delta_sign = 1 if signal == "BUY" else -1
            signed_delta = signed_position_after_entry - pre_entry_signed_position
            if signed_delta * expected_delta_sign <= 0:
                raise ValueError("post-entry position delta does not match signal direction")
            if abs(abs(signed_delta) - filled) > max(min_lot, filled * 0.02):
                raise ValueError("post-entry position delta does not reconcile to confirmed order fill")
            actual_pos_size = abs(signed_position_after_entry)
            position_below_minimum = actual_pos_size < min_lot
        except Exception as position_error:
            log.critical(
                "  %s: post-entry fill/position reconciliation failed (%s); preserving recovery state",
                symbol,
                type(position_error).__name__,
            )
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, actual_entry, 0.0, order_ids,
                "post-entry position quantity did not reconcile to terminal fill",
                exchange_quantity_verified=False,
            )
            return False

        if position_below_minimum:
            # The order fill and the side-correct exchange position were both
            # confirmed, but the net exposure is below the instrument's minimum
            # lot. Do not continue into bracket sizing. Attempt the common
            # verified recovery path; it re-reads the position and order book and
            # refuses a guessed or unroundable close quantity.
            log.critical(
                "  %s: post-fill position below minimum lot (%.12g < %.12g); "
                "attempting verified recovery",
                symbol, actual_pos_size, min_lot,
            )
            recovery_verified = _emergency_close_and_verify(
                deribit, trades, sig, symbol, signal, actual_entry,
                actual_pos_size, order_ids, "post-fill position below minimum lot",
            )
            if recovery_verified:
                # Do not invent PnL for a dust recovery. Record only the verified
                # flat/clear outcome so this rejected entry cannot block later
                # scans as an open trade.
                entry_intent_record["closed"] = True
                entry_intent_record["emergency_recovery_required"] = False
                entry_intent_record["manual_reconciliation_required"] = False
                entry_intent_record["protection_reconciliation_required"] = False
                entry_intent_record["protection_reconciliation_reason"] = None
                entry_intent_record["emergency_recovery_status"] = "VERIFIED_FLAT_AND_ORDERS_CLEARED"
                _persist_trade_snapshot(symbol, entry_intent_record)
            return False

        if signal == "BUY":
            recomputed_stop = _directional_tick_price(
                Decimal(str(actual_entry)) - Decimal(str(atr)) * Decimal(str(effective_stop_mult)), "DOWN"
            )[0]
            recomputed_tp1 = _directional_tick_price(
                Decimal(str(actual_entry)) + Decimal(str(atr)) * Decimal(str(dyn_tp1)), "DOWN"
            )[0]
            recomputed_tp2 = _directional_tick_price(
                Decimal(str(actual_entry)) + Decimal(str(atr)) * Decimal(str(dyn_tp2)), "DOWN"
            )[0]
        else:
            recomputed_stop = _directional_tick_price(
                Decimal(str(actual_entry)) + Decimal(str(atr)) * Decimal(str(effective_stop_mult)), "UP"
            )[0]
            recomputed_tp1 = _directional_tick_price(
                Decimal(str(actual_entry)) - Decimal(str(atr)) * Decimal(str(dyn_tp1)), "UP"
            )[0]
            recomputed_tp2 = _directional_tick_price(
                Decimal(str(actual_entry)) - Decimal(str(atr)) * Decimal(str(dyn_tp2)), "UP"
            )[0]

        post_fill_valid = True
        for name, val in [("stop", recomputed_stop), ("tp1", recomputed_tp1), ("tp2", recomputed_tp2)]:
            if not math.isfinite(val) or val <= 0:
                post_fill_valid = False
                log.critical(f"  🚨 {symbol}: POST-FILL {name}={val!r} is invalid")
                break

        if not post_fill_valid:
            _emergency_close_and_verify(
                deribit, trades, sig, symbol, signal, actual_entry,
                total_q, order_ids, "post-fill bracket invalid",
            )
            return False

        stop = recomputed_stop
        tp1  = recomputed_tp1
        tp2  = recomputed_tp2
        # The exchange's verified net position is authoritative after a
        # terminal entry order. Protect only this confirmed quantity, not the
        # originally requested amount.
        total_q = actual_pos_size
        qty_tp1, qty_tp2 = deribit.split_amount(symbol, total_q)
        sl_qty = total_q
        actual_risk_usd = round(abs(actual_entry - stop) * total_q, 8)
        estimated_round_trip_fees = round(
            ((actual_entry + stop) * total_q * float(taker_fee_rate)), 8
        )
        total_risk_with_fees = actual_risk_usd + estimated_round_trip_fees
        if (
            not all(math.isfinite(value) for value in (actual_risk_usd, estimated_round_trip_fees, total_risk_with_fees))
            or actual_risk_usd <= 0
            or total_risk_with_fees > budget_risk_usd * 1.000001
        ):
            log.critical(
                "  %s: POST-FILL RISK INVARIANT FAILED | stop_risk=%.8f fees=%.8f "
                "total=%.8f budget=%.8f USDC qty=%.12g",
                symbol, actual_risk_usd, estimated_round_trip_fees,
                total_risk_with_fees, budget_risk_usd, total_q,
            )
            _emergency_close_and_verify(
                deribit, trades, sig, symbol, signal, actual_entry,
                total_q, order_ids, "post-fill risk invariant failed",
            )
            return False

        # Persist the filled exposure and every deterministic protection label
        # before the first protective mutation. A runner crash after this point
        # leaves enough intent for read-only label reconciliation on restart.
        stop_label = _deterministic_order_label(sig, symbol, "STOP")
        tp_labels = {
            "tp1": _deterministic_order_label(sig, symbol, "TP1"),
            "tp2": _deterministic_order_label(sig, symbol, "TP2"),
        }
        tp_plan = [("TP1", qty_tp1, tp1, "tp1")]
        if qty_tp2 > 0:
            tp_plan.append(("TP2", qty_tp2, tp2, "tp2"))
        order_ids["stop_loss_label"] = stop_label
        order_submission_states = {
            "entry": "FILLED" if o_state == "filled" else "PARTIALLY_FILLED_TERMINAL",
            **{
                key: ("NOT_REQUIRED" if qty <= 0 else "NOT_SUBMITTED")
                for _, qty, _, key in tp_plan
            },
        }
        order_client_labels = {
            "entry": client_label,
            **{key: tp_labels[key] for _, qty, _, key in tp_plan if qty > 0},
        }
        lifecycle_record = {
            **_candidate_identity_fields(),
            "symbol": symbol, "signal": signal,
            "entry": actual_entry, "model_entry": model_entry,
            "execution_entry_ref": entry, "actual_entry": actual_entry,
            "stop": stop, "tp1": tp1, "tp2": tp2,
            "qty": total_q, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
            "risk_usd": actual_risk_usd, "initial_risk_usd": actual_risk_usd,
            "budget_risk_usd": budget_risk_usd, "balance_at_open": balance,
            "risk_amount": actual_risk_usd, "risk_currency": "USDC",
            "initial_risk_amount": actual_risk_usd, "budget_risk_amount": budget_risk_usd,
            "budget_risk_currency": "USDC", "balance_at_open_currency": "USDC",
            "estimated_round_trip_fees": estimated_round_trip_fees,
            "estimated_fees_currency": "USDC",
            "total_risk_with_fees": total_risk_with_fees,
            "total_risk_currency": "USDC",
            "quantity_source": "terminal_exchange_fill_and_verified_position",
            "instrument_id": instrument_spec.instrument_id,
            "risk_mult": final_risk_mult, "exit_mode": exit_mode,
            "order_ids": order_ids,
            "order_submission_states": order_submission_states,
            "order_client_labels": order_client_labels,
            "stop_loss_submission_state": "SUBMITTING",
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "tp1_hit": False, "tp2_hit": False, "closed": False,
            "confidence": sig["confidence"], "score": sig["score"],
            "reasons": sig.get("reasons", []), "tier": get_tier(symbol),
            "fg_override": sig.get("fg_override", False),
            "exchange": "deribit_testnet", "pnl_currency": "USDC",
            "pred_id": sig.get("pred_id"), "snapshot": sig.get("snapshot", {}),
            "logical_intent_id": entry_intent_record.get("logical_intent_id"),
            "entry_attempt_id": entry_intent_record.get("entry_attempt_id"),
            "entry_attempt_number": entry_intent_record.get("entry_attempt_number"),
            "entry_attempt": dict(entry_intent_record.get("entry_attempt", {})),
            "entry_attempt_history": [
                dict(item) for item in entry_intent_record.get("entry_attempt_history", [])
            ],
            "entry_order_contract": dict(entry_intent_record.get("entry_order_contract", {})),
            "stop_mult_used": effective_stop_mult,
            "manual_reconciliation_required": True,
            "protection_reconciliation_required": True,
            "protection_reconciliation_reason": "initial protection sequence in progress",
        }
        trades[symbol] = lifecycle_record
        if not _persist_trade_snapshot(symbol, lifecycle_record):
            log.critical("  %s: filled exposure could not be persisted before stop placement; attempting verified emergency close", symbol)
            _emergency_close_and_verify(
                deribit, trades, sig, symbol, signal, actual_entry,
                actual_pos_size, order_ids, "initial protection intent persistence failed",
            )
            return False

        # Stop protection is placed first as one reduce-only stop-market and
        # independently verified at the exchange. It is never blindly retried.
        try:
            stop_result = deribit.place_stop_loss(
                symbol, sl_side, sl_qty, stop, client_label=stop_label
            )
            stop_order = stop_result.get("order", stop_result) if isinstance(stop_result, dict) else {}
            stop_id = str(stop_order.get("order_id", "")).strip() if isinstance(stop_order, dict) else ""
            if not stop_id:
                raise OrderOutcomeUnknown("verified stop response omitted exchange order ID")
            _confirm_execution_gateway_stop(
                deribit, lifecycle_record, stop_client_label=stop_label,
                stop_order_id=stop_id, protected_quantity=sl_qty,
                evidence={"verification_source": "initial_stop_independent_exchange_readback"},
            )
            order_ids["stop_loss"] = stop_id
            lifecycle_record["stop_loss_submission_state"] = "ACKNOWLEDGED"
            lifecycle_record["manual_reconciliation_required"] = False
            lifecycle_record["protection_reconciliation_required"] = False
            lifecycle_record.pop("protection_reconciliation_reason", None)
            if not _persist_trade_snapshot(symbol, lifecycle_record):
                log.critical("  %s: verified stop could not be persisted locally; attempting verified emergency close", symbol)
                _emergency_close_and_verify(
                    deribit, trades, sig, symbol, signal, actual_entry,
                    actual_pos_size, order_ids, "verified stop acknowledgement persistence failed",
                )
                return False
            log.info("  ✓ Verified reduce-only stop-market %s qty=%s id=%s", symbol, sl_qty, stop_id)
        except OrderOutcomeUnknown as unknown_error:
            evidence = _reconcile_unknown_entry_by_label_read_only(
                deribit, symbol, stop_label
            )
            order_ids["stop_loss_label"] = stop_label
            matched_ids = evidence.get("matched_order_ids", [])
            if len(matched_ids) == 1:
                order_ids["stop_loss"] = matched_ids[0]
            elif matched_ids:
                order_ids["stop_loss_label_matches"] = matched_ids
            log.critical(
                "  %s: stop submission outcome UNKNOWN (%s); no second stop or flatten mutation permitted",
                symbol, type(unknown_error).__name__,
            )
            lifecycle_record["stop_loss_submission_state"] = "UNKNOWN"
            lifecycle_record["manual_reconciliation_required"] = True
            lifecycle_record["protection_reconciliation_required"] = True
            lifecycle_record["protection_reconciliation_reason"] = "initial stop outcome unknown"
            _persist_trade_snapshot(symbol, lifecycle_record)
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, actual_entry, actual_pos_size, order_ids,
                "stop-loss outcome unknown; read-only reconciliation and operator acknowledgement required",
                exchange_quantity_verified=True,
                reconciliation_evidence=evidence,
            )
            return False
        except Exception as stop_error:
            log.critical(
                "  %s: stop-market placement/verification failed (%s); attempting only verified emergency recovery",
                symbol, type(stop_error).__name__,
            )
            _emergency_close_and_verify(
                deribit, trades, sig, symbol, signal, actual_entry,
                actual_pos_size, order_ids, "stop-loss placement or verification failed",
            )
            return False

        tp_reconciliation_required = False
        for label, qty, price, key in tp_plan:
            if qty <= 0:
                continue
            lifecycle_record["order_submission_states"][key] = "SUBMITTING"
            lifecycle_record["manual_reconciliation_required"] = True
            lifecycle_record["protection_reconciliation_required"] = True
            lifecycle_record["protection_reconciliation_reason"] = f"{label} submission in progress"
            if not _persist_trade_snapshot(symbol, lifecycle_record):
                tp_reconciliation_required = True
                lifecycle_record["order_submission_states"][key] = "NOT_SUBMITTED"
                lifecycle_record["protection_reconciliation_reason"] = f"{label} pre-submit intent persistence failed; order not sent"
                log.critical("  %s: %s intent persistence failed; keeping verified stop, no TP request sent", symbol, label)
                break
            try:
                result = deribit.place_limit_order(
                    symbol, tp_side, qty, price, use_reduce_only=True,
                    client_label=tp_labels[key],
                )
                verified_tp = _verify_reduce_only_limit_ack(
                    deribit, symbol, tp_side, qty, price, result, tp_labels[key]
                )
                order_id = str(verified_tp.get("order_id", "")).strip()
                order_ids[key] = order_id
                order_ids[f"{key}_label"] = tp_labels[key]
                lifecycle_record["order_submission_states"][key] = "ACKNOWLEDGED"
                all_targets_resolved = all(
                    status in {"ACKNOWLEDGED", "NOT_REQUIRED", "NOT_SUBMITTED"}
                    for status in lifecycle_record["order_submission_states"].values()
                )
                if all_targets_resolved:
                    lifecycle_record["manual_reconciliation_required"] = False
                    lifecycle_record["protection_reconciliation_required"] = False
                    lifecycle_record.pop("protection_reconciliation_reason", None)
                if not _persist_trade_snapshot(symbol, lifecycle_record):
                    lifecycle_record["order_submission_states"][key] = "UNKNOWN"
                    tp_reconciliation_required = True
                    log.critical("  %s: %s acknowledged but local persistence failed; no retry", symbol, label)
                    break
                log.info("  ✓ %s reduce-only limit @ %s × %s id:%s", label, price, qty, order_id)
            except OrderOutcomeUnknown as unknown_error:
                tp_reconciliation_required = True
                order_ids[f"{key}_label"] = tp_labels[key]
                lifecycle_record["order_submission_states"][key] = "UNKNOWN"
                lifecycle_record["protection_reconciliation_reason"] = f"{label} outcome unknown"
                _persist_trade_snapshot(symbol, lifecycle_record)
                log.critical(
                    "  %s: %s order outcome UNKNOWN (%s); stop remains verified; no TP retry",
                    symbol, label, type(unknown_error).__name__,
                )
                break
            except Exception as tp_error:
                # Explicit rejection or preflight failure is not retried with
                # identical parameters. The already-verified stop remains.
                tp_reconciliation_required = True
                order_ids[f"{key}_label"] = tp_labels[key]
                lifecycle_record["order_submission_states"][key] = "REJECTED"
                lifecycle_record["protection_reconciliation_reason"] = f"{label} submission failed ({type(tp_error).__name__})"
                _persist_trade_snapshot(symbol, lifecycle_record)
                log.error(
                    "  %s: %s placement failed (%s); retaining verified stop and requiring manual TP review",
                    symbol, label, type(tp_error).__name__,
                )
                break

    except Exception as e:
        log.error(f"  Trade error {symbol}: {e}")
        return False

    record = {
        **lifecycle_record,
        **_candidate_identity_fields(),
        "symbol": symbol, "signal": signal,
        "entry": actual_entry,
        "model_entry": model_entry,
        "execution_entry_ref": entry,
        "actual_entry": actual_entry,
        "stop": stop, "tp1": tp1, "tp2": tp2,
        "qty": total_q, "qty_tp1": qty_tp1, "qty_tp2": qty_tp2,
        # Legacy numeric keys retained for dashboard/backward compatibility;
        # the explicit currency fields below define their denomination.
        "risk_usd": actual_risk_usd, "initial_risk_usd": actual_risk_usd,
        "budget_risk_usd": budget_risk_usd, "balance_at_open": balance,
        "risk_amount": actual_risk_usd, "risk_currency": "USDC",
        "initial_risk_amount": actual_risk_usd, "budget_risk_amount": budget_risk_usd,
        "budget_risk_currency": "USDC", "balance_at_open_currency": "USDC",
        "estimated_round_trip_fees": estimated_round_trip_fees,
        "estimated_fees_currency": "USDC",
        "total_risk_with_fees": total_risk_with_fees,
        "total_risk_currency": "USDC",
        "quantity_source": "terminal_exchange_fill_and_verified_position",
        "instrument_id": instrument_spec.instrument_id,
        "risk_mult": final_risk_mult, "exit_mode": exit_mode,
        "order_ids": order_ids,
        "order_submission_states": lifecycle_record.get("order_submission_states", {}),
        "order_client_labels": lifecycle_record.get("order_client_labels", {}),
        "stop_loss_submission_state": lifecycle_record.get("stop_loss_submission_state", "ACKNOWLEDGED"),
        "opened_at": lifecycle_record.get("opened_at") or datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False, "tp2_hit": False, "closed": False,
        "confidence": sig["confidence"], "score": sig["score"],
        "reasons": sig.get("reasons", []), "tier": get_tier(symbol),
        "fg_override": sig.get("fg_override", False),
        "exchange": "deribit_testnet",
        "risk_currency": "USDC",
        "pnl_currency": "USDC",
        "pred_id": sig.get("pred_id"),
        "snapshot": sig.get("snapshot", {}),
        "stop_mult_used": effective_stop_mult,
        "protection_reconciliation_required": bool(tp_reconciliation_required),
        "manual_reconciliation_required": bool(tp_reconciliation_required),
        "protection_reconciliation_reason": lifecycle_record.get("protection_reconciliation_reason") if tp_reconciliation_required else None,
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
    if not oid_str or oid_str in ("", "None"):
        raise ValueError("Tracked exchange order ID is missing")
    # Unknown order state must propagate to the enclosing management guard;
    # an unavailable read is not equivalent to a closed/empty order.
    return deribit.get_order(oid_str)

def _persist_stop_submission_intent(symbol: str, trade: dict, client_label: str) -> bool:
    """Persist a deterministic stop label before the exchange mutation."""
    trade.setdefault("order_ids", {})["stop_loss_label"] = str(client_label)
    trade["stop_loss_submission_state"] = "SUBMITTING"
    return _persist_trade_snapshot(symbol, trade)


def _cancel_order_and_verify(deribit, symbol: str, order_id: str) -> bool:
    """Cancel once, then require an independent open-order snapshot without that ID."""
    oid = str(order_id or "").strip()
    if not oid:
        return False
    try:
        deribit.cancel_order(oid)
    except Exception as exc:
        log.warning("  %s: cancel request for %s returned %s; checking exchange state", symbol, oid, type(exc).__name__)
    try:
        remaining = deribit.get_open_orders(symbol)
    except Exception as exc:
        log.critical("  %s: cannot verify cancellation of %s (%s)", symbol, oid, type(exc).__name__)
        return False
    if not isinstance(remaining, list):
        log.critical("  %s: malformed open-order snapshot while verifying cancellation of %s", symbol, oid)
        return False
    still_open = [row for row in remaining if isinstance(row, dict) and str(row.get("order_id", "")) == oid]
    if still_open:
        log.critical("  %s: order %s remains open after cancellation", symbol, oid)
        return False
    return True


def fp(o, fb):
    p = float(o.get("average_price") or o.get("last_price") or o.get("price") or 0)
    return p if p > 0 else fb

def _pnl(t, cp, ct):
    qty  = float(t["qty_tp1"] if ct == "tp1" else t["qty_tp2"] if ct == "tp2" else t["qty"])
    diff = (cp - float(t["entry"])) if t["signal"] == "BUY" else (float(t["entry"]) - cp)
    return round(diff * qty, 4)

def _close_record(t, cp, pnl, reason):
    append_history({
        **t,
        "close_price": cp,
        "pnl": pnl,
        "pnl_currency": t.get("pnl_currency"),
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "close_reason": reason,
    })


def _retain_emergency_recovery_record(
    trades: dict,
    sig: dict,
    symbol: str,
    signal: str,
    entry_price: float,
    fallback_qty: float,
    order_ids: dict,
    reason: str,
    *,
    exchange_quantity_verified: bool = True,
    reconciliation_evidence: dict = None,
) -> None:
    """Persist unresolved exposure without presenting requested size as a fill."""
    try:
        tracked_entry = float(entry_price)
    except (TypeError, ValueError, OverflowError):
        tracked_entry = 0.0
    if not math.isfinite(tracked_entry) or tracked_entry <= 0:
        try:
            tracked_entry = float(sig.get("entry", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            tracked_entry = 0.0
    try:
        tracked_qty = abs(float(fallback_qty or 0))
    except (TypeError, ValueError, OverflowError):
        tracked_qty = 0.0
    if not math.isfinite(tracked_qty):
        tracked_qty = 0.0

    existing = trades.get(symbol)
    if isinstance(existing, dict) and not existing.get("closed", False):
        record = existing
    else:
        record = {}

    record.update({
        "symbol": symbol,
        "signal": signal,
        "entry": tracked_entry,
        "model_entry": float(sig.get("entry", tracked_entry) or tracked_entry),
        "execution_entry_ref": tracked_entry,
        "actual_entry": tracked_entry,
        "stop": 0.0,
        "tp1": 0.0,
        "tp2": 0.0,
        "qty": tracked_qty,
        "qty_tp1": tracked_qty,
        "qty_tp2": 0.0,
        "risk_usd": 0.0,
        "initial_risk_usd": 0.0,
        "risk_currency": "USDC",
        "pnl_currency": "USDC",
        "budget_risk_usd": 0.0,
        "balance_at_open": 0.0,
        "risk_mult": 0.0,
        "exit_mode": "EMERGENCY_RECOVERY",
        "order_ids": dict(order_ids or {}),
        "opened_at": record.get("opened_at") or datetime.now(timezone.utc).isoformat(),
        "tp1_hit": False,
        "tp2_hit": False,
        "closed": False,
        "confidence": sig.get("confidence", 0),
        "score": sig.get("score", 0),
        "reasons": sig.get("reasons", []) if isinstance(sig.get("reasons", []), list) else [],
        "tier": sig.get("conf_tier", "emergency-recovery"),
        "fg_override": sig.get("fg_override", False),
        "pred_id": sig.get("pred_id"),
        "snapshot": sig.get("snapshot", {}) if isinstance(sig.get("snapshot", {}), dict) else {},
        "stop_mult_used": ATR_STOP_MULT,
        "emergency_recovery_required": True,
        "emergency_recovery_reason": reason,
        "exchange_quantity_verified": bool(exchange_quantity_verified),
        "quantity_source": "verified_exchange_position" if exchange_quantity_verified else "unknown_unreconciled",
        "reconciliation_evidence": reconciliation_evidence if isinstance(reconciliation_evidence, dict) else {},
        "manual_reconciliation_required": True,
    })
    if not exchange_quantity_verified:
        # Do not make requested quantity look like actual exposure. The
        # deterministic prediction/order IDs and exchange order IDs remain for
        # read-only reconciliation; qty stays zero until verified.
        record["qty"] = 0.0
        record["qty_tp1"] = 0.0
        record["qty_tp2"] = 0.0
    trades[symbol] = record
    expected_record = None
    try:
        write_json_mirrors(_trade_state_mirror_paths(), trades)
        persisted_state = read_json_mirrors(_trade_state_mirror_paths(), default=None)
        persisted = persisted_state.get(symbol) if isinstance(persisted_state, dict) else None
        expected_record = json.loads(canonical_json_bytes(record).decode("utf-8"))
    except Exception as persistence_error:
        persisted = None
        log.critical(
            "  %s: emergency recovery snapshot persistence failed (%s)",
            symbol, type(persistence_error).__name__,
        )
    if not isinstance(persisted, dict) or persisted != expected_record:
        log.critical(
            "  %s: emergency recovery tracking persistence could not be verified",
            symbol,
        )
    log.critical(
        "  %s: emergency recovery is unresolved (%s); tracking retained and manual exchange inspection required",
        symbol,
        reason,
    )
    _send(
        f"🚨 *MANUAL EXCHANGE INSPECTION REQUIRED — {symbol}*\n"
        f"Emergency recovery not verified: {reason}. Tracking record retained."
    )


def _record_verified_dust_residual(
    deribit: DeribitClient,
    trades: dict,
    sig: dict,
    symbol: str,
    signal: str,
    entry_price: float,
    order_ids: dict,
    first_signed_position: float,
    reason: str,
) -> bool:
    """Persist a bot-attributed, terminal, exchange-verified dust disposition.

    Returns True only when the residual has passed all classification checks
    and the complete local trade record has been durably mirrored. It never
    invents PnL and never classifies an unknown/unfilled entry outcome as dust.
    """
    try:
        from dust_residual_policy import classify_verified_dust_residual
        from capital_feasibility import resolve_active_capital_tier_policy

        entry_order_id = str((order_ids or {}).get("entry") or "").strip()
        prediction_id = str(sig.get("pred_id") or "").strip()
        prior = trades.get(symbol) if isinstance(trades, dict) else None
        if not entry_order_id or not prediction_id or not isinstance(prior, dict):
            raise ValueError("bot entry provenance is missing")
        if str(prior.get("pred_id") or "") != prediction_id:
            raise ValueError("local prediction identity does not match residual source")
        local_order_ids = prior.get("order_ids")
        if not isinstance(local_order_ids, dict) or str(local_order_ids.get("entry") or "") != entry_order_id:
            raise ValueError("entry order identity does not match the persisted trade record")
        if _has_unresolved_entry_attempt(prior):
            raise ValueError("entry attempt is not in a verified terminal disposition")
        if str(prior.get("entry_submission_state") or "").upper() not in {
            "FILLED", "PARTIALLY_FILLED_TERMINAL", "CANCELLED", "CANCELED"
        }:
            raise ValueError("local entry submission state is not terminal")
        entry_order = deribit.get_order(entry_order_id)
        if not isinstance(entry_order, dict) or str(entry_order.get("order_id") or "") != entry_order_id:
            raise ValueError("entry order state cannot be independently verified")
        entry_state = str(entry_order.get("order_state") or "").strip().lower()
        entry_filled = float(entry_order.get("filled_amount"))
        entry_amount = float(entry_order.get("amount"))
        if (
            not math.isfinite(entry_filled) or entry_filled <= 0
            or not math.isfinite(entry_amount) or entry_amount <= 0
            or entry_filled > entry_amount + max(1e-12, entry_amount * 1e-8)
        ):
            raise ValueError("entry order fill/amount is not internally consistent")

        first_orders = deribit.get_open_orders(symbol)
        if not isinstance(first_orders, list):
            raise ValueError("first open-order snapshot is malformed")
        time.sleep(0.25)
        second_signed = float(deribit.get_position_size(symbol))
        second_orders = deribit.get_open_orders(symbol)
        if not isinstance(second_orders, list):
            raise ValueError("second open-order snapshot is malformed")
        mark_price = float(deribit.get_mark_price(symbol))
        minimum = float(deribit.get_min_trade_amount(symbol))
        tier_policy = resolve_active_capital_tier_policy(config, environment="testnet")
        dust = classify_verified_dust_residual(
            symbol=symbol,
            signal=signal,
            first_signed_position=first_signed_position,
            second_signed_position=second_signed,
            min_trade_amount=minimum,
            mark_price=mark_price,
            first_open_orders=first_orders,
            second_open_orders=second_orders,
            entry_order_id=entry_order_id,
            entry_order_state=entry_state,
            entry_filled_quantity=entry_filled,
            prediction_id=prediction_id,
            max_position_notional_usdc=tier_policy["max_dust_position_notional_usdc"],
        )
        record = dict(prior)
        now = datetime.now(timezone.utc).isoformat()
        record.update({
            "symbol": symbol,
            "closed": True,
            "closed_at": now,
            "close_reason": "DUST_RESIDUAL_UNTRADEABLE",
            "exit_mode": "DUST_RESIDUAL_VERIFIED",
            "dust_residual": dust,
            "qty": float(dust["quantity"]),
            "qty_tp1": float(dust["quantity"]),
            "qty_tp2": 0.0,
            "exchange_quantity_verified": True,
            "quantity_source": "verified_exchange_position_dust_disposition",
            "stop_loss_submission_state": "DUST_UNPROTECTED",
            "emergency_recovery_required": False,
            "manual_reconciliation_required": False,
            "protection_reconciliation_required": False,
            "emergency_recovery_status": "VERIFIED_DUST_RESIDUAL",
            "emergency_recovery_reason": reason,
            "dust_reserve_currency": "USDC",
            "dust_reserve_per_position_usdc": str(tier_policy["dust_reserve_per_position_usdc"]),
        })
        # The residual is not assigned a fabricated close price, fee or PnL.
        if not _persist_trade_snapshot(symbol, record):
            raise ValueError("verified dust disposition could not be durably persisted")
        trades[symbol] = record
        log.warning(
            "  %s: verified bot-created dust residual qty=%s mark_notional=%s USDC; "
            "same-symbol entries remain blocked, unrelated symbols may proceed",
            symbol, dust["quantity"], dust["mark_notional_usdc"],
        )
        _send(
            f"ℹ️ *Verified dust residual — {symbol}*\n"
            f"Quantity {dust['quantity']} remains below exchange minimum; "
            "same-symbol entries are blocked, unrelated symbols remain eligible under portfolio reserve limits."
        )
        return True
    except Exception as exc:
        log.warning("  %s: dust disposition rejected; recovery remains blocked (%s)", symbol, type(exc).__name__)
        return False


def _emergency_close_and_verify(
    deribit: DeribitClient,
    trades: dict,
    sig: dict,
    symbol: str,
    signal: str,
    entry_price: float,
    fallback_qty: float,
    order_ids: dict,
    reason: str,
) -> bool:
    """Flatten only a freshly verified exchange position; never guess quantity.

    If the position or open-order snapshots are unavailable, preserve an
    unresolved recovery record and perform no mutation. A caller-supplied
    fallback quantity is metadata only and is never submitted to the exchange.
    """
    try:
        signed_position = float(deribit.get_position_size(symbol))
        if not math.isfinite(signed_position):
            raise ValueError("position quantity is non-finite")
    except Exception as exc:
        log.critical(
            "  %s: emergency position read unavailable (%s); refusing guessed-size flatten",
            symbol,
            type(exc).__name__,
        )
        _retain_emergency_recovery_record(
            trades, sig, symbol, signal, entry_price, 0.0, order_ids,
            f"{reason}; position snapshot unavailable",
            exchange_quantity_verified=False,
        )
        return False

    # Cancel first and verify the order book before deriving the final close
    # quantity. This avoids racing a resting TP/SL or unfilled entry order.
    if not _cancel_all_open_orders_for_symbol(deribit, symbol):
        log.critical("  %s: open-order cleanup is unverified; refusing emergency mutation", symbol)
        _retain_emergency_recovery_record(
            trades, sig, symbol, signal, entry_price, abs(signed_position), order_ids,
            f"{reason}; open-order cleanup unverified",
            exchange_quantity_verified=True,
        )
        return False

    # Re-read after cancellations; only this quantity may be used for flatten.
    try:
        signed_position = float(deribit.get_position_size(symbol))
        if not math.isfinite(signed_position):
            raise ValueError("position quantity is non-finite")
    except Exception as exc:
        log.critical(
            "  %s: post-cancel position read unavailable (%s); no market order sent",
            symbol,
            type(exc).__name__,
        )
        _retain_emergency_recovery_record(
            trades, sig, symbol, signal, entry_price, 0.0, order_ids,
            f"{reason}; post-cancel position snapshot unavailable",
            exchange_quantity_verified=False,
        )
        return False

    if abs(signed_position) > 1e-8:
        try:
            minimum = float(deribit.get_min_trade_amount(symbol))
        except Exception as exc:
            minimum = 0.0
            log.critical("  %s: minimum-lot read failed (%s); no flatten sent", symbol, type(exc).__name__)
        if not math.isfinite(minimum) or minimum <= 0:
            _retain_emergency_recovery_record(
                trades, sig, symbol, signal, entry_price, abs(signed_position), order_ids,
                f"{reason}; exchange minimum lot unavailable",
                exchange_quantity_verified=True,
            )
            return False
        if abs(signed_position) < minimum:
            # A sub-minimum amount cannot be safely closed with a guessed/up-rounded
            # market order. It may only be accepted later by the strict dust policy.
            log.warning("  %s: residual %.12g is below min lot %.12g; verifying dust disposition", symbol, abs(signed_position), minimum)
        else:
            close_side = "SELL" if signed_position > 0 else "BUY"
            try:
                close_qty = float(deribit.round_amount(symbol, abs(signed_position)))
            except Exception as exc:
                log.critical("  %s: position quantity rounding failed (%s); no flatten sent", symbol, type(exc).__name__)
                _retain_emergency_recovery_record(
                    trades, sig, symbol, signal, entry_price, abs(signed_position), order_ids,
                    f"{reason}; verified position could not be rounded",
                    exchange_quantity_verified=True,
                )
                return False
            if not math.isfinite(close_qty) or close_qty <= 0 or close_qty > abs(signed_position) + 1e-10:
                log.critical("  %s: rounded emergency quantity is invalid; no flatten sent", symbol)
                _retain_emergency_recovery_record(
                    trades, sig, symbol, signal, entry_price, abs(signed_position), order_ids,
                    f"{reason}; invalid rounded close quantity",
                    exchange_quantity_verified=True,
                )
                return False
            try:
                deribit.place_reduce_only_market_order(
                    symbol, close_side, close_qty,
                    client_label=_deterministic_order_label(sig, symbol, "EMERGENCY_FLATTEN"),
                )
            except Exception as exc:
                log.critical(
                    "  %s: one-shot reduce-only market flatten failed/unknown (%s); read-only verification only",
                    symbol,
                    type(exc).__name__,
                )

    flat_verified = _verify_actually_closed(deribit, symbol, tolerance=1e-12)
    orders_cleared = _cancel_all_open_orders_for_symbol(deribit, symbol)
    if flat_verified and orders_cleared:
        log.critical(
            "  %s: emergency recovery verified flat with no open orders (%s)",
            symbol,
            reason,
        )
        _send(
            f"✅ *Emergency recovery verified — {symbol}*\n"
            f"Exchange reports flat and no open orders: {reason}."
        )
        return True

    try:
        remaining = abs(float(deribit.get_position_size(symbol)))
        quantity_verified = math.isfinite(remaining)
    except Exception:
        remaining = 0.0
        quantity_verified = False
    if quantity_verified and remaining > 1e-12 and orders_cleared:
        try:
            first_dust_signed = float(deribit.get_position_size(symbol))
            if not math.isfinite(first_dust_signed):
                raise ValueError("first dust position read is non-finite")
        except Exception as dust_read_error:
            log.warning("  %s: first dust position read failed (%s); recovery remains blocked", symbol, type(dust_read_error).__name__)
            first_dust_signed = None
        if first_dust_signed is not None and _record_verified_dust_residual(
            deribit, trades, sig, symbol, signal, entry_price, order_ids,
            first_dust_signed, reason,
        ):
            # The position is not flat; False remains the emergency-close
            # result. The durable dust disposition clears global recovery flags
            # while the portfolio admission path reserves its exposure.
            return False
    _retain_emergency_recovery_record(
        trades, sig, symbol, signal, entry_price,
        remaining if quantity_verified else 0.0,
        order_ids,
        reason,
        exchange_quantity_verified=quantity_verified,
    )
    return False


def _replace_missing_orders(deribit: DeribitClient, symbol: str, trade: dict) -> bool:
    submission_states = trade.get("order_submission_states", {})
    if not isinstance(submission_states, dict):
        submission_states = {}
    unresolved_stop = trade.get("stop_loss_submission_state") in {"SUBMITTING", "UNKNOWN"}
    unresolved_targets = any(
        submission_states.get(key) in {"SUBMITTING", "UNKNOWN", "REJECTED"}
        for key in ("tp1", "tp2")
    )
    has_reconciliation_flags = bool(
        trade.get("manual_reconciliation_required")
        or trade.get("emergency_recovery_required")
        or trade.get("protection_reconciliation_required")
    )
    read_only_reconciliation = has_reconciliation_flags and (unresolved_stop or unresolved_targets)
    if has_reconciliation_flags and not read_only_reconciliation:
        log.critical("  %s: bracket repair blocked until manual reconciliation is acknowledged", symbol)
        return False
    # ── GUARD: Do not attempt to replace brackets if exchange position is flat ──
    try:
        actual_pos = abs(deribit.get_position_size(symbol))
        if actual_pos <= 0.0001:
            return False
    except Exception:
        return False

    oids   = trade.get("order_ids", {})
    signal = trade["signal"]
    entry  = float(trade["entry"])
    stop   = float(trade.get("stop", 0))
    tp1    = float(trade.get("tp1", 0))
    tp2    = float(trade.get("tp2", 0))
    qty    = min(float(trade.get("qty", 0)), actual_pos)
    qty_t1 = float(trade.get("qty_tp1", 0))
    qty_t2 = float(trade.get("qty_tp2", 0))
    dec    = 4 if entry < 10 else 2
    sl_side = "SELL" if signal == "BUY" else "BUY"
    tp_side = "SELL" if signal == "BUY" else "BUY"
    changed = False

    sl_oid  = str(oids.get("stop_loss", ""))
    has_sl  = bool(sl_oid and sl_oid not in ("", "None"))

    if not has_sl and not trade.get("closed") and qty > 0 and not (read_only_reconciliation and not unresolved_stop):
        if stop <= 0:
            try:
                live_p = deribit.get_live_price(symbol)
                ref_p = entry if entry > 0 else live_p
                raw15 = get_data(symbol, TIMEFRAME_ENTRY)
                fresh_atr = (float(raw15["close"].iloc[-1]) * 0.015) if raw15.empty else float(add_indicators(raw15)["atr"].iloc[-1])
                stop = deribit.round_price(symbol, ref_p - fresh_atr * ATR_STOP_MULT if signal == "BUY" else ref_p + fresh_atr * ATR_STOP_MULT)
                trade["stop"] = stop
                changed = True
            except Exception as heale:
                log.error(f"  Failed to heal stop for {symbol}: {heale}")

        if stop > 0:
            stop_label = str(oids.get("stop_loss_label") or _deterministic_order_label(trade, symbol, "STOP_REPAIR"))
            trade.setdefault("order_ids", {})["stop_loss_label"] = stop_label
            try:
                previous_state = trade.get("stop_loss_submission_state")
                if previous_state in {"SUBMITTING", "UNKNOWN"}:
                    matches = _order_label_matches(deribit, symbol, stop_label)
                    if len(matches) == 1:
                        oid = str(matches[0].get("order_id", "")).strip()
                        state = str(matches[0].get("order_state", "")).lower()
                        expected_stop_amount = float(matches[0].get("amount", 0) or 0)
                        expected_trigger = float(matches[0].get("trigger_price", 0) or 0)
                        verified_match = (
                            oid
                            and state in {"open", "untriggered"}
                            and str(matches[0].get("order_type", "")).lower() == "stop_market"
                            and str(matches[0].get("trigger", "")).lower() == "mark_price"
                            and matches[0].get("reduce_only") is True
                            and str(matches[0].get("direction", "")).lower() == sl_side.lower()
                            and math.isclose(expected_stop_amount, actual_pos, rel_tol=1e-8, abs_tol=max(1e-12, actual_pos * 1e-8))
                            and math.isclose(expected_trigger, stop, rel_tol=1e-8, abs_tol=1e-10)
                        )
                        if verified_match:
                            _confirm_execution_gateway_stop(
                                deribit, trade, stop_client_label=stop_label,
                                stop_order_id=oid, protected_quantity=expected_stop_amount,
                                evidence={"verification_source": "read_only_stop_label_reconciliation"},
                            )
                            trade["order_ids"]["stop_loss"] = oid
                            trade["stop_loss_submission_state"] = "RECONCILED"
                            states_map = trade.get("order_submission_states", {})
                            if not any(value in {"SUBMITTING", "UNKNOWN", "REJECTED"} for value in states_map.values()):
                                trade.pop("manual_reconciliation_required", None)
                                trade.pop("emergency_recovery_required", None)
                                trade.pop("protection_reconciliation_required", None)
                            changed = True
                        else:
                            raise OrderOutcomeUnknown("label matched a stop that is not resting")
                    else:
                        raise OrderOutcomeUnknown("previous stop submission not uniquely reconciled by label")
                elif previous_state == "REJECTED":
                    raise ProtectiveOrderVerificationError("previous stop submission was rejected; do not resend same parameters")
                else:
                    trade["stop_loss_submission_state"] = "SUBMITTING"
                    if not _persist_trade_snapshot(symbol, trade):
                        raise OrderOutcomeUnknown("stop pre-submit intent could not be persisted")
                    result = deribit.place_stop_loss(
                        symbol, sl_side, qty, stop, client_label=stop_label
                    )
                    order = result.get("order", result) if isinstance(result, dict) else {}
                    oid = str(order.get("order_id", "")).strip() if isinstance(order, dict) else ""
                    if not oid:
                        raise OrderOutcomeUnknown("stop acknowledgement omitted exchange order ID")
                    _confirm_execution_gateway_stop(
                        deribit, trade, stop_client_label=stop_label,
                        stop_order_id=oid, protected_quantity=qty,
                        evidence={"verification_source": "stop_repair_independent_exchange_readback"},
                    )
                    trade["order_ids"]["stop_loss"] = oid
                    trade["stop_loss_submission_state"] = "ACKNOWLEDGED"
                    changed = True
            except OrderOutcomeUnknown as exc:
                trade["stop_loss_submission_state"] = "UNKNOWN"
                trade["manual_reconciliation_required"] = True
                trade["emergency_recovery_required"] = True
                trade["emergency_recovery_reason"] = f"stop repair outcome unknown ({type(exc).__name__})"
                trade["reconciliation_evidence"] = {"client_label": stop_label, "read_only_required": True}
                _persist_trade_snapshot(symbol, trade)
                log.critical("  %s: stop repair UNKNOWN; no second stop or flatten request", symbol)
                return True
            except Exception as exc:
                # A confirmed rejection or a verified-but-invalid stop is a
                # protection failure. Attempt emergency handling only from a
                # fresh exchange position snapshot. Unknown outcomes took the
                # separate no-mutation/manual-reconciliation path above.
                recovery_trades = load_trades()
                recovery_trades[symbol] = trade
                emergency_result = _emergency_close_and_verify(
                    deribit,
                    recovery_trades,
                    trade,
                    symbol,
                    signal,
                    entry,
                    actual_pos,
                    dict(oids),
                    f"stop repair failed ({type(exc).__name__})",
                )
                trade["manual_reconciliation_required"] = True
                trade["emergency_recovery_required"] = True
                trade["emergency_recovery_reason"] = f"stop repair failed ({type(exc).__name__})"
                trade["emergency_flat_verified"] = bool(emergency_result)
                _persist_trade_snapshot(symbol, trade)
                log.critical("  %s: stop repair failed; same stop will not be retried (%s)", symbol, type(exc).__name__)
                return True

    if not has_sl and read_only_reconciliation and not unresolved_stop:
        log.critical("  %s: unresolved TP intent permits label reads only; stop submission remains blocked", symbol)

    tp1_oid  = str(oids.get("tp1", ""))
    tp2_oid  = str(oids.get("tp2", ""))

    if (tp1 <= 0 or tp2 <= 0) and qty > 0 and not trade.get("closed"):
        try:
            live_p = deribit.get_live_price(symbol)
            ref_p  = entry if entry > 0 else live_p
            raw15  = get_data(symbol, TIMEFRAME_ENTRY)
            fresh_atr = (float(raw15["close"].iloc[-1]) * 0.015) if raw15.empty else float(add_indicators(raw15)["atr"].iloc[-1])
            if tp1 <= 0:
                tp1 = deribit.round_price(symbol, ref_p + fresh_atr * ATR_TARGET1_MULT if signal == "BUY" else ref_p - fresh_atr * ATR_TARGET1_MULT)
                trade["tp1"] = tp1
                changed = True
            if tp2 <= 0 and qty_t2 > 0:
                tp2 = deribit.round_price(symbol, ref_p + fresh_atr * ATR_TARGET2_MULT if signal == "BUY" else ref_p - fresh_atr * ATR_TARGET2_MULT)
                trade["tp2"] = tp2
                changed = True
        except Exception as heale:
            log.error(f"  Failed to heal TP targets for {symbol}: {heale}")

    need_tp1 = (not tp1_oid or tp1_oid in ("", "None")) and tp1 > 0 and qty_t1 > 0
    reconcile_tp1 = submission_states.get("tp1") in {"SUBMITTING", "UNKNOWN", "REJECTED"}
    if (need_tp1 or reconcile_tp1) and not trade.get("tp1_hit") and not trade.get("closed"):
        if read_only_reconciliation and not reconcile_tp1:
            log.critical("  %s: unresolved state permits TP1 label reconciliation only; no new submission", symbol)
        else:
            oid = _place_tp_with_fallback(deribit, symbol, tp_side, qty_t1, tp1, "TP1", trade, "tp1", dec)
            if oid:
                trade.setdefault("order_ids", {})["tp1"] = oid
                changed = True

    need_tp2 = (not tp2_oid or tp2_oid in ("", "None")) and tp2 > 0 and qty_t2 > 0
    reconcile_tp2 = submission_states.get("tp2") in {"SUBMITTING", "UNKNOWN", "REJECTED"}
    if (need_tp2 or reconcile_tp2) and not trade.get("tp2_hit") and not trade.get("closed"):
        if read_only_reconciliation and not reconcile_tp2:
            log.critical("  %s: unresolved state permits TP2 label reconciliation only; no new submission", symbol)
        else:
            oid = _place_tp_with_fallback(deribit, symbol, tp_side, qty_t2, tp2, "TP2", trade, "tp2", dec)
            if oid:
                trade.setdefault("order_ids", {})["tp2"] = oid
                changed = True

    still_broken = (not trade.get("closed")) and (
        float(trade.get("stop", 0)) <= 0 or
        (float(trade.get("tp1", 0)) <= 0 and float(trade.get("tp2", 0)) <= 0) or
        not str(trade.get("order_ids", {}).get("stop_loss", "")).strip()
    )
    heal_attr = f"_heal_fail_streak_{symbol}"
    if still_broken:
        streak = getattr(_replace_missing_orders, heal_attr, 0) + 1
        setattr(_replace_missing_orders, heal_attr, streak)
        if streak >= 2:
            _send(f"🚨 *UNPROTECTED POSITION — {symbol}*\nBracket healing failed for {streak} consecutive scans.")
    else:
        setattr(_replace_missing_orders, heal_attr, 0)

    return changed


def check_open_trades(deribit: DeribitClient):
    trades = load_trades()
    if not trades: log.info("  No open trades"); return

    to_remove = []
    log.info(f"  Monitoring {len(trades)} trade(s)")

    live_positions = set()
    try:
        verified_positions = deribit.get_positions()
        for p in verified_positions:
            if float(p.get("size", 0)) != 0:
                inst = p.get("instrument_name", "")
                if not isinstance(inst, str) or not inst.endswith("_USDC-PERPETUAL"):
                    continue
                base = inst.split("_", 1)[0]
                symbol = f"{base}USDT"
                if deribit.get_instrument_name(symbol) != inst:
                    log.warning(
                        "  %s: position does not match the bot's linear instrument mapping; "
                        "not associating it with %s",
                        inst,
                        symbol,
                    )
                    continue
                live_positions.add(symbol)
    except Exception as e:
        log.error(
            "  Open-trade management skipped: live position snapshot unavailable (%s)",
            type(e).__name__,
        )
        return

    for symbol, trade in list(trades.items()):
        if trade.get("closed"):
            to_remove.append(symbol)
            continue
        submission_states = trade.get("order_submission_states", {})
        if not isinstance(submission_states, dict):
            submission_states = {}
        has_reconciliation_flags = bool(
            trade.get("emergency_recovery_required")
            or trade.get("manual_reconciliation_required")
            or trade.get("protection_reconciliation_required")
        )
        has_unresolved_intent = (
            trade.get("stop_loss_submission_state") in {"SUBMITTING", "UNKNOWN"}
            or any(submission_states.get(key) in {"SUBMITTING", "UNKNOWN", "REJECTED"} for key in ("tp1", "tp2"))
        )
        if has_reconciliation_flags and not has_unresolved_intent:
            log.critical(
                "  %s: automatic management/replacement paused; manual exchange reconciliation is required",
                symbol,
            )
            continue

        entry  = float(trade["entry"])
        stop   = float(trade.get("stop", 0))
        tp1_p  = float(trade.get("tp1",  0))
        tp2_p  = float(trade.get("tp2",  0))
        dec    = 4 if entry < 10 else 2
        signal = trade["signal"]
        oids   = trade.get("order_ids", {})

        # ── CODE 11030 FIX: Verify exchange existence BEFORE attempting order repair ──
        if live_positions and symbol not in live_positions and not trade.get("tp1_hit"):
            log.warning(f"  {symbol}: no live position found — ghost cleaner will handle")
            continue

        reconciliation_was_pending = has_reconciliation_flags
        if _replace_missing_orders(deribit, symbol, trade): save_trades(trades)
        if reconciliation_was_pending:
            log.info("  %s: reconciliation pass completed or remains pending; normal management resumes on the next scan", symbol)
            continue
        if (
            trade.get("emergency_recovery_required")
            or trade.get("manual_reconciliation_required")
            or trade.get("protection_reconciliation_required")
        ):
            log.critical("  %s: management paused until the persisted order outcome is reconciled", symbol)
            continue

        if trade.get("closed"):
            to_remove.append(symbol)
            continue

        try:
            live = deribit.get_live_price(symbol)
        except Exception as e: log.warning(f"  {symbol}: live price error — {e}"); continue
        if live <= 0: log.warning(f"  {symbol}: live price = 0 — skip"); continue

        risk_usd = float(trade.get("risk_usd", 0))
        if risk_usd > 0:
            try:
                close_side, mae_qty = _get_safe_close_info(deribit, symbol, trade)
            except Exception as position_error:
                log.error(
                    "  %s: MAE management skipped because position could not be verified (%s)",
                    symbol,
                    type(position_error).__name__,
                )
                continue
            mae_diff = (live - entry) if signal == "BUY" else (entry - live)
            mae_pnl  = round(mae_diff * mae_qty, 4)

            if mae_pnl < -(risk_usd * 3):
                log.warning(f"  🚨 {symbol}: MAE ${mae_pnl:.2f} > 3× risk ${risk_usd:.2f} — FORCE CLOSE")
                _cancel_all_open_orders_for_symbol(deribit, symbol)

                try:
                    if mae_qty > 0:
                        deribit.place_market_order(symbol, close_side, mae_qty, reduce_only=True)
                except Exception as e:
                    log.error("  MAE force-close %s order failed (%s)", symbol, type(e).__name__)
                    continue

                flat_verified = _verify_actually_closed(deribit, symbol)
                orders_cleared = _cancel_all_open_orders_for_symbol(deribit, symbol)
                if flat_verified and orders_cleared:
                    _close_record(trade, live, mae_pnl, "Max adverse excursion ❌")
                    _send(f"🚨 *FORCE CLOSE — {symbol}*\nLoss `{mae_pnl:+.4f}` exceeded 3× risk")
                    if mae_pnl < 0:
                        _add_cooldown(symbol, f"MAE close pnl={mae_pnl:+.4f}")
                        _record_outcome(symbol, won=False)
                    trade["closed"] = True
                    to_remove.append(symbol)
                    continue
                log.critical(
                    "  %s: MAE close or order cleanup unverified; retaining trade tracking",
                    symbol,
                )
                continue

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
                        try:
                            if not _cancel_order_and_verify(deribit, symbol, str(oids["stop_loss"])):
                                raise RuntimeError("existing stop cancellation could not be verified")
                            signed_after_tp1 = float(deribit.get_position_size(symbol))
                            if not math.isfinite(signed_after_tp1) or signed_after_tp1 == 0:
                                raise RuntimeError("post-TP1 position is not verifiable/non-zero")
                            expected_sign = 1 if signal == "BUY" else -1
                            if (1 if signed_after_tp1 > 0 else -1) != expected_sign:
                                raise RuntimeError("post-TP1 position direction mismatch")
                            remaining_qty = abs(signed_after_tp1)
                            sl_s = "SELL" if signal == "BUY" else "BUY"
                            be_label = _deterministic_order_label(trade, symbol, "BE_STOP_TP1")
                            if not _persist_stop_submission_intent(symbol, trade, be_label):
                                raise RuntimeError("break-even stop intent could not be persisted; submission blocked")
                            be = deribit.place_stop_loss(
                                symbol, sl_s, remaining_qty, entry, client_label=be_label
                            )
                            be_o = be.get("order", be) if isinstance(be, dict) else {}
                            nid = str(be_o.get("order_id", "")).strip() if isinstance(be_o, dict) else ""
                            if not nid:
                                raise OrderOutcomeUnknown("break-even stop acknowledgement omitted ID")
                            trade["order_ids"]["stop_loss"] = nid
                            trade["order_ids"]["stop_loss_label"] = be_label
                            trade["stop_loss_submission_state"] = "ACKNOWLEDGED"
                            trade["stop"] = entry
                            trade["qty"] = remaining_qty
                            if not _persist_trade_snapshot(symbol, trade):
                                trade["manual_reconciliation_required"] = True
                                trade["emergency_recovery_required"] = True
                                trade["emergency_recovery_reason"] = "break-even stop acknowledged but local persistence failed"
                                save_trades(trades)
                                continue
                            _send(f"🛡️ *{symbol} RISK-FREE* SL→entry `{entry:.{dec}f}`")
                        except OrderOutcomeUnknown as be_unknown:
                            trade["stop_loss_submission_state"] = "UNKNOWN"
                            trade["manual_reconciliation_required"] = True
                            trade["emergency_recovery_required"] = True
                            trade["emergency_recovery_reason"] = "break-even stop order outcome unknown"
                            save_trades(trades)
                            log.critical("  %s: break-even stop outcome unknown; no retry (%s)", symbol, type(be_unknown).__name__)
                            continue
                        except Exception as be_error:
                            log.critical("  %s: break-even stop replacement failed; attempting verified emergency close (%s)", symbol, type(be_error).__name__)
                            _emergency_close_and_verify(
                                deribit, trades, trade, symbol, signal, entry,
                                0.0,
                                dict(trade.get("order_ids", {})), "break-even stop replacement failed",
                            )
                            trade["manual_reconciliation_required"] = True
                            trade["emergency_recovery_required"] = True
                            trade["emergency_recovery_reason"] = "break-even stop replacement failed"
                            save_trades(trades)
                            continue

            if trade.get("tp1_hit") and not trade.get("tp2_hit") and oids.get("stop_loss"):
                halfway = (entry + tp2_p) / 2
                at_half = ((signal == "BUY"  and live >= halfway) or (signal == "SELL" and live <= halfway))
                sl_at_be = abs(float(trade.get("stop", 0)) - entry) < entry * 0.001
                if at_half and sl_at_be and float(trade.get("qty_tp2", 0)) > 0:
                    try:
                        if not _cancel_order_and_verify(deribit, symbol, str(oids["stop_loss"])):
                            raise RuntimeError("break-even stop cancellation could not be verified")
                        signed_position = float(deribit.get_position_size(symbol))
                        if not math.isfinite(signed_position) or signed_position == 0:
                            raise RuntimeError("trailing stop position is unavailable")
                        expected_sign = 1 if signal == "BUY" else -1
                        if (1 if signed_position > 0 else -1) != expected_sign:
                            raise RuntimeError("trailing stop position direction mismatch")
                        sl_s = "SELL" if signal == "BUY" else "BUY"
                        trail_label = _deterministic_order_label(trade, symbol, "TRAIL_STOP_TP1")
                        if not _persist_stop_submission_intent(symbol, trade, trail_label):
                            raise RuntimeError("trailing stop intent could not be persisted; submission blocked")
                        sl_r = deribit.place_stop_loss(
                            symbol, sl_s, abs(signed_position), tp1_p,
                            client_label=trail_label,
                        )
                        sl_o = sl_r.get("order", sl_r) if isinstance(sl_r, dict) else {}
                        nid = str(sl_o.get("order_id", "")).strip() if isinstance(sl_o, dict) else ""
                        if not nid:
                            raise OrderOutcomeUnknown("trailing stop acknowledgement omitted ID")
                        trade["order_ids"]["stop_loss"] = nid
                        trade["order_ids"]["stop_loss_label"] = trail_label
                        trade["stop_loss_submission_state"] = "ACKNOWLEDGED"
                        trade["stop"] = tp1_p
                        trade["qty"] = abs(signed_position)
                        if not _persist_trade_snapshot(symbol, trade):
                            trade["manual_reconciliation_required"] = True
                            trade["emergency_recovery_required"] = True
                            trade["emergency_recovery_reason"] = "trailing stop acknowledged but local persistence failed"
                            save_trades(trades)
                            continue
                        _send(f"🚀 *{symbol}* Trail SL→TP1 `{tp1_p:.{dec}f}` locked!")
                    except OrderOutcomeUnknown as unknown:
                        trade["stop_loss_submission_state"] = "UNKNOWN"
                        trade["manual_reconciliation_required"] = True
                        trade["emergency_recovery_required"] = True
                        trade["emergency_recovery_reason"] = "trailing stop order outcome unknown"
                        save_trades(trades)
                        log.critical("  %s: trailing stop outcome unknown; no retry (%s)", symbol, type(unknown).__name__)
                        continue
                    except Exception as exc:
                        log.critical("  %s: trailing stop replacement failed; attempting verified emergency close (%s)", symbol, type(exc).__name__)
                        _emergency_close_and_verify(
                            deribit, trades, trade, symbol, signal, entry,
                            0.0,
                            dict(trade.get("order_ids", {})), "trailing stop replacement failed",
                        )
                        trade["manual_reconciliation_required"] = True
                        trade["emergency_recovery_required"] = True
                        trade["emergency_recovery_reason"] = "trailing stop replacement failed"
                        save_trades(trades)
                        continue

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
                    method2 = ("order" if tp2_order_filled else "partial" if tp2_partial else "price-fallback")
                    fill2   = fp(o, tp2_p) if (tp2_order_filled or tp2_partial) else live
                    pnl2    = _pnl(trade, fill2, "tp2")

                    tp1_was_hit = bool(trade.get("tp1_hit"))
                    tp1_filled_during_check = False
                    tp1_leftover_pnl = 0.0
                    if not trade.get("tp1_hit"):
                        tp1_o = _safe_get_order(deribit, str(oids.get("tp1", "")))
                        tp1_really_filled = deribit.is_order_filled(tp1_o)
                        if tp1_really_filled:
                            trade["tp1_hit"] = True
                            tp1_filled_during_check = True
                        else:
                            qty_t1  = float(trade.get("qty_tp1", 0))
                            tp_side = "SELL" if signal == "BUY" else "BUY"
                            try:
                                if oids.get("tp1"):
                                    try: deribit.cancel_order(oids["tp1"])
                                    except Exception: pass
                                close_side_leftover = tp_side
                                remaining_real = abs(deribit.get_position_size(symbol))
                                leftover_qty = deribit.round_amount(symbol, min(qty_t1, remaining_real)) if remaining_real > 0 else 0
                                if leftover_qty > 0:
                                    deribit.place_market_order(symbol, close_side_leftover, leftover_qty, reduce_only=True)
                                    tp1_leftover_pnl = _pnl(trade, fill2, "tp1")
                            except Exception as le:
                                log.error(f"  🚨 {symbol}: leftover TP1 close failed: {le}")
                            trade["tp1_hit"] = True

                    if not _verify_actually_closed(deribit, symbol):
                        # TP2 implies an exit, but a residual may remain after a
                        # partial fill or failed TP1-leftover recovery. Retry only
                        # from a fresh, verified position quantity.
                        try:
                            residual_side, residual_qty = _get_safe_close_info(
                                deribit, symbol, trade
                            )
                            if residual_qty > 0:
                                deribit.place_market_order(
                                    symbol,
                                    residual_side,
                                    residual_qty,
                                    reduce_only=True,
                                )
                        except Exception as residual_error:
                            log.critical(
                                "  %s: TP2 residual close could not be verified (%s); "
                                "trade tracking retained",
                                symbol,
                                type(residual_error).__name__,
                            )

                        if not _verify_actually_closed(deribit, symbol):
                            if not tp1_was_hit and not tp1_filled_during_check:
                                trade["tp1_hit"] = False
                            log.critical(
                                "  %s: TP2 observed but exchange is not verified flat; "
                                "retaining trade tracking",
                                symbol,
                            )
                            continue

                    if not _cancel_all_open_orders_for_symbol(deribit, symbol):
                        log.critical(
                            "  %s: TP2 flatness seen but open-order cleanup unverified; retaining trade tracking",
                            symbol,
                        )
                        continue

                    pnl2 = round(pnl2 + tp1_leftover_pnl, 4)
                    trade["tp2_hit"] = True
                    trade["closed"]  = True
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
                        already_flat  = real_pos_size <= 1e-8
                    except Exception: already_flat = False
                    if already_flat: sl_hit = True

                sl_breached = stop > 0 and ((signal == "BUY" and live <= stop * 0.999) or (signal == "SELL" and live >= stop * 1.001))
                mark_price = deribit.get_mark_price(symbol)
                mark_breached = stop > 0 and (
                    (signal == "BUY"  and mark_price <= stop * 0.998) or
                    (signal == "SELL" and mark_price >= stop * 1.002)
                )

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
                    if _verify_actually_closed(deribit, symbol) and _cancel_all_open_orders_for_symbol(deribit, symbol):
                        trade["closed"] = True
                        fill = fp(sl_o, stop)
                        if fill == 0 or fill == stop:
                            fill = live if live > 0 else stop

                        pnl = _pnl(trade, fill, "sl")
                        lbl = ("BREAK-EVEN ⚖️" if abs(fill - entry) < entry * 0.002 else "STOPPED OUT ❌")
                        log.info(f"  ❌ SL {symbol} @ {fill:.{dec}f}  pnl≈{pnl:+.4f}")
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
                    else:
                        log.critical(
                            "  %s: stop-exit flatness or open-order cleanup unverified; retaining trade tracking",
                            symbol,
                        )

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

        # ── CODE 11030 FIX: Skip symbols that are already flat on Deribit ──
        try:
            pos_size = abs(deribit.get_position_size(symbol))
            if pos_size <= 0.0001:
                continue
        except Exception as position_error:
            log.error(
                "  %s: position read unavailable; stale-time exit deferred (%s)",
                symbol,
                type(position_error).__name__,
            )
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
            close_side, close_qty = _get_safe_close_info(deribit, symbol, trade)
            live = deribit.get_live_price(symbol)
            _cancel_all_open_orders_for_symbol(deribit, symbol)
            if close_qty > 0:
                try:
                    deribit.place_market_order(symbol, close_side, close_qty, reduce_only=True)
                except Exception as me:
                    log.warning(f"  Time-exit market order {symbol}: {me}")

            flat_verified = _verify_actually_closed(deribit, symbol)
            orders_cleared = _cancel_all_open_orders_for_symbol(deribit, symbol)
            if flat_verified and orders_cleared:
                close_price = live if live > 0 else float(trade["entry"])
                pnl = _pnl(trade, close_price, "sl")
                reason = f"Time exit ({age_h:.0f}h [{tier_label}])"
                _close_record(trade, close_price, pnl, reason)
                _send(f"⏰ *TIME EXIT ({tier_label}) — {symbol}*\n{age_h:.1f}h | PnL≈`{pnl:+.4f}`")
                to_remove.append(symbol)
            else:
                log.critical(
                    "  %s: stale-time exit flatness or order cleanup unverified; retaining trade tracking",
                    symbol,
                )
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
    try:
        verified_positions = deribit.get_positions()
    except Exception as e:
        log.error(
            "  Ghost-trade cleanup skipped: live position snapshot unavailable (%s)",
            type(e).__name__,
        )
        return
    for p in verified_positions:
        if float(p.get("size", 0)) != 0:
            inst = p.get("instrument_name", "")
            if not isinstance(inst, str) or not inst.endswith("_USDC-PERPETUAL"):
                continue
            base = inst.split("_", 1)[0]
            symbol = f"{base}USDT"
            if deribit.get_instrument_name(symbol) != inst:
                log.warning(
                    "  %s: position does not match the bot's linear instrument mapping; "
                    "not associating it with %s",
                    inst,
                    symbol,
                )
                continue
            live_pos[symbol] = True

    to_remove = []
    for symbol, trade in trades.items():
        if float(trade.get("stop", 0)) == 0 or float(trade.get("tp1", 0)) == 0:
            try:
                actual = abs(deribit.get_position_size(symbol))
                if actual > 0:
                    _replace_missing_orders(deribit, symbol, trade)
                    if (
                        trade.get("emergency_recovery_required")
                        or trade.get("manual_reconciliation_required")
                        or trade.get("protection_reconciliation_required")
                    ):
                        log.critical("  %s: corrupted-state recovery requires manual reconciliation; no further mutation", symbol)
                        save_trades(trades)
                        continue
                    if float(trade.get("stop", 0)) == 0:
                        _cancel_all_open_orders_for_symbol(deribit, symbol)
                        close_side = "SELL" if trade.get("signal") == "BUY" else "BUY"
                        deribit.place_market_order(symbol, close_side, actual, reduce_only=True)
                        flat_verified = _verify_actually_closed(deribit, symbol)
                        orders_cleared = _cancel_all_open_orders_for_symbol(deribit, symbol)
                        if flat_verified and orders_cleared:
                            _close_record(
                                trade,
                                float(trade.get("entry", 0)),
                                0.0,
                                "Corrupted state auto-flattened",
                            )
                            to_remove.append(symbol)
                        else:
                            log.critical(
                                "  %s: corrupted-state close unverified; retaining trade tracking",
                                symbol,
                            )
                    else:
                        log.warning(
                            "  %s: corrupted-state healing incomplete; tracking retained",
                            symbol,
                        )
                else:
                    flat_verified = _verify_actually_closed(deribit, symbol)
                    orders_cleared = _cancel_all_open_orders_for_symbol(deribit, symbol)
                    if flat_verified and orders_cleared:
                        to_remove.append(symbol)
                    else:
                        log.critical(
                            "  %s: no position observed but order cleanup/flatness unverified; retaining tracking",
                            symbol,
                        )
            except Exception as ge:
                log.error(f"  Ghost healing failed for {symbol} (%s); tracking retained", type(ge).__name__)
            continue
            
        if float(trade.get("score", 1)) == 0 and float(trade.get("confidence", 1)) == 0:
            try:
                actual = deribit.get_position_size(symbol)
            except Exception as e:
                log.error(
                    "  Broken ghost record %s retained: position read unavailable (%s)",
                    symbol,
                    type(e).__name__,
                )
                continue
            if abs(actual) > 0:
                try:
                    _cancel_all_open_orders_for_symbol(deribit, symbol)
                    close_side = "SELL" if actual > 0 else "BUY"
                    deribit.place_market_order(symbol, close_side, abs(actual), reduce_only=True)
                except Exception as close_error:
                    log.error(
                        "  Broken ghost record %s retained: reduce-only close failed (%s)",
                        symbol,
                        type(close_error).__name__,
                    )
                    continue

            flat_verified = _verify_actually_closed(deribit, symbol)
            orders_cleared = _cancel_all_open_orders_for_symbol(deribit, symbol)
            if not flat_verified or not orders_cleared:
                log.critical(
                    "  Broken ghost record %s retained: flatness or open-order cleanup unverified",
                    symbol,
                )
                continue
            _close_record(trade, float(trade.get("entry", 0)), 0.0, "Ghost — broken record")
            to_remove.append(symbol)
            continue
            
        if symbol not in live_pos:
            log.warning("  %s: position absent from verified snapshot; close PnL reconciliation required", symbol)
            try:
                fills = deribit.get_trade_history_for_instrument(
                    symbol, count=20, strict=True
                )
                if not isinstance(fills, list):
                    raise ValueError("Malformed exchange fill history")
            except Exception as exc:
                trade["reconciliation_status"] = "PENDING"
                trade["reconciliation_reason"] = "TRADE_HISTORY_UNAVAILABLE"
                trade["reconciliation_error_type"] = type(exc).__name__
                trade["last_reconciliation_at"] = datetime.now(timezone.utc).isoformat()
                save_trades(trades)
                log.warning(
                    "  %s: close reconciliation held; trade-history unavailable (%s)",
                    symbol, type(exc).__name__,
                )
                continue

            # A recent-fill page alone is not enough to prove that all close
            # quantity, fees and cashflows for this local record are present.
            # Keep the record open for a dedicated transaction-ledger matcher;
            # never fabricate a zero-PnL close from an empty/partial page.
            trade["reconciliation_status"] = "PENDING"
            trade["reconciliation_reason"] = (
                "AUTHORITATIVE_CLOSE_QUANTITY_AND_FEE_MATCH_REQUIRED"
                if fills else "NO_RECENT_FILLS_RETURNED"
            )
            trade["last_reconciliation_at"] = datetime.now(timezone.utc).isoformat()
            save_trades(trades)
            log.warning(
                "  %s: local record retained; %s; no PnL was inferred",
                symbol, trade["reconciliation_reason"],
            )
            continue

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
                    _send(f"💸 *FUNDING ALERT — {symbol}*\n{signal} paying `{abs(rate_pct):.3f}%` per 8h")
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
          f"📦 {qty} contracts · Real Risk: {risk:.2f} USDC · Risk Equity: {bal:.2f} USDC\n"
          f"━━━━━━━━━━━━━━━━━━━━")

VALID_EXECUTION_MODES = {
    "PREDICT_ONLY",
    "MANAGE_ONLY",
    "FULL",
}

# GitHub workflow_dispatch may explicitly request MANAGE_ONLY for
# controlled testnet validation. FULL is intentionally excluded.
MANUAL_OVERRIDE_EXECUTION_MODES = {
    "PREDICT_ONLY",
    "MANAGE_ONLY",
}


def _validate_execution_mode(mode: str, *, source: str) -> str:
    normalized = str(mode).strip().upper()

    if normalized not in VALID_EXECUTION_MODES:
        raise RuntimeError(
            f"Invalid {source}={normalized!r}; "
            f"allowed={sorted(VALID_EXECUTION_MODES)}"
        )

    return normalized


def get_execution_mode() -> str:
    """Resolve effective execution mode with a fail-closed policy.

    Rules:
      * Scheduled/non-manual GitHub Actions runs are forced to PREDICT_ONLY.
      * workflow_dispatch may select PREDICT_ONLY or MANAGE_ONLY.
      * Missing manual input defaults to PREDICT_ONLY.
      * FULL is not exposed through workflow_dispatch.
      * Invalid manual overrides fail closed.
      * Local execution continues to use config.EXECUTION_MODE.
    """
    in_github_actions = (
        os.getenv("GITHUB_ACTIONS", "").strip().lower() == "true"
    )

    github_event = os.getenv(
        "GITHUB_EVENT_NAME",
        "",
    ).strip().lower()

    if in_github_actions:

        if github_event == "workflow_dispatch":

            override = os.getenv(
                "EXECUTION_MODE_OVERRIDE",
                "",
            ).strip().upper()

            mode = override or "PREDICT_ONLY"

            if mode not in MANUAL_OVERRIDE_EXECUTION_MODES:
                raise RuntimeError(
                    f"Invalid EXECUTION_MODE_OVERRIDE={mode!r}; "
                    "manual workflow override allows only "
                    f"{sorted(MANUAL_OVERRIDE_EXECUTION_MODES)}"
                )

            return mode

        # Defense in depth:
        # scheduled, repository_dispatch, and every other non-manual
        # GitHub Actions invocation are forced to PREDICT_ONLY.
        return "PREDICT_ONLY"

    configured = str(
        getattr(
            config,
            "EXECUTION_MODE",
            "PREDICT_ONLY",
        )
    ).strip().upper()

    return _validate_execution_mode(
        configured,
        source="EXECUTION_MODE",
    )


# Deliberately false in this release. The durable intent coordinator and
# fill/protection recovery are implemented as primitives but are not yet wired
# around the legacy production submit path. Do not change this until that
# integration and its end-to-end fault tests are completed.
ORDER_LIFECYCLE_COORDINATOR_INTEGRATED = False


def _deribit_testnet_mutations_enabled() -> bool:
    """Fail-closed code, workflow, host, credential and acknowledgement gate."""
    requested = os.getenv("DERIBIT_TESTNET_MUTATIONS_ENABLED", "false").strip().lower() == "true"
    if not requested or not ORDER_LIFECYCLE_COORDINATOR_INTEGRATED:
        return False
    try:
        validate_deribit_testnet_mutation_environment()
    except EnvironmentGuardError as exc:
        log.error("Deribit Testnet mutation environment rejected (%s)", str(exc))
        return False
    return True


def _execution_allows_new_entries() -> bool:
    return get_execution_mode() == "FULL" and _deribit_testnet_mutations_enabled()


def _execution_allows_management() -> bool:
    return (
        get_execution_mode() in {"MANAGE_ONLY", "FULL"}
        and _deribit_testnet_mutations_enabled()
    )


def run_execution_scan():
    global _ACTIVE_DERIBIT_CLIENT
    log.info(f"\n{'═'*56}\nSCAN — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n{'═'*56}")

    if not _acquire_scan_lock():
        return
    try:
        _run_execution_scan_locked()
    finally:
        client = _ACTIVE_DERIBIT_CLIENT
        _ACTIVE_DERIBIT_CLIENT = None
        try:
            if client is not None:
                close_gateway = getattr(client, "close_execution_gateway", None)
                if callable(close_gateway):
                    close_gateway()
        finally:
            _release_scan_lock()

def _run_execution_scan_locked():
    global _ACTIVE_DERIBIT_CLIENT
    scan_started_at = datetime.now(timezone.utc).isoformat()
    execution_mode = get_execution_mode()

    _SCAN_LIVENESS.update({
        "scan_ran": False,
        "symbols_attempted": 0,
        "symbols_scored": 0,
        "predictions_saved": 0,
        "skip_reason": None,
    })

    save_json(SCAN_STATUS_FILE, {
        "phase": "started",
        "started_at": scan_started_at,
        "completed_at": None,
        "scan_ran": False,
        "symbols_attempted": 0,
        "symbols_scored": 0,
        "predictions_saved": 0,
        "execution_mode": execution_mode,
        "skip_reason": None,
    })

    # MANAGE_ONLY is intentionally independent of prediction scheduler gates.
    # But this release keeps all legacy exchange mutation paths blocked until
    # the persistent order coordinator is wired into submit/recovery/protection.
    if execution_mode == "MANAGE_ONLY" and not _deribit_testnet_mutations_enabled():
        reason = "DERIBIT_ORDER_COORDINATOR_NOT_INTEGRATED"
        _SCAN_LIVENESS["skip_reason"] = reason
        save_json(SCAN_STATUS_FILE, {
            "phase": "completed",
            "started_at": scan_started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "scan_ran": False,
            "symbols_attempted": 0,
            "symbols_scored": 0,
            "predictions_saved": 0,
            "execution_mode": execution_mode,
            "skip_reason": reason,
            "management_ran": False,
            "exchange_mutations_enabled": False,
        })
        return

    if execution_mode == "MANAGE_ONLY":
        run = True
        mode = {"label": "MANAGE_ONLY"}
        vol = {"status": "NORMAL"}
        reason = None
        pipeline = None
        thresholds = None
        risk_mult = 1.0
    else:
        run, mode, vol, reason = should_scan()

        if not run:
            log.info(f"  Scan skipped: {reason}")
            _SCAN_LIVENESS["skip_reason"] = str(reason)

            save_json(SCAN_STATUS_FILE, {
                "phase": "completed",
                "started_at": scan_started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "scan_ran": False,
                "symbols_attempted": 0,
                "symbols_scored": 0,
                "predictions_saved": 0,
                "execution_mode": execution_mode,
                "skip_reason": str(reason),
            })
            return

        pipeline = joblib.load(MODEL_FILE)
        thresholds = get_mode_thresholds(mode)
        risk_mult = get_effective_risk(mode, vol)

    deribit = None

    if _execution_allows_management():
        # A second validation occurs inside DeribitClient.__init__. The first
        # keeps credential lookup and error reporting explicit at the runner.
        mutation_credentials = validate_deribit_testnet_mutation_environment()
        deribit = DeribitClient(
            mutation_credentials.client_id,
            mutation_credentials.client_secret,
            testnet=True,
            auth_scope="trade:read_write account:read",
            allow_mutations=True,
            require_execution_gateway=True,
        )
        _ACTIVE_DERIBIT_CLIENT = deribit
        deribit.test_connection()

    _SCAN_LIVENESS["scan_ran"] = True
    _SCAN_LIVENESS["symbols_attempted"] = len(SYMBOLS)

    max_open_trades = int(getattr(config, "MAX_OPEN_TRADES", 8))
    max_same_dir    = int(getattr(config, "MAX_SAME_DIRECTION", 4))

    # Local ledger count is safe in every execution mode.
    # PREDICT_ONLY must not require exchange access just to report open count.
    open_count = len(
        [t for t in load_trades().values() if not t.get("closed", False)]
    )

    # ── CLAMP: Floor model thresholds against config.MIN_CONFIDENCE ──
    config_floor = float(getattr(config, "MIN_CONFIDENCE", 52.0))

    # MANAGE_ONLY has no prediction pipeline by design.
    if pipeline is None:
        rec_buy = config_floor
        rec_sell = config_floor
    else:
        rec_buy = max(
            config_floor,
            float(
                pipeline.get(
                    "recommended_threshold_buy",
                    pipeline.get("recommended_threshold", 0.40),
                )
            ) * 100.0,
        )
        rec_sell = max(
            config_floor,
            float(
                pipeline.get(
                    "recommended_threshold_sell",
                    pipeline.get("recommended_threshold", 0.45),
                )
            ) * 100.0,
        )

    log.info(
        f"  {mode['label']} | Active Targets: "
        f"BUY={rec_buy:.1f}% SELL={rec_sell:.1f}% "
        f"| score={thresholds['min_score'] if thresholds else 'N/A'} "
        f"| ADX={thresholds['min_adx'] if thresholds else 'N/A'} "
        f"| risk:{risk_mult:.2f}"
    )

    balance = 0.0
    if _execution_allows_management():
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
            # Normalize the entire snapshot before taking any action. If one row
            # is malformed, do not partially reconcile/flatten the account.
            normalized_positions = []
            for position in raw_positions:
                instrument = position.get("instrument_name", "")
                if not isinstance(instrument, str) or not instrument:
                    raise ValueError("position row has no valid instrument name")

                # This bot reconciles only mapped linear USDC perpetuals.
                # Never alias an inverse contract such as BTC-PERPETUAL to BTCUSDT.
                if not instrument.endswith("_USDC-PERPETUAL"):
                    log.warning(
                        "  [3c] unmanaged/non-linear position %s left untouched; "
                        "orphan reconciliation is restricted to linear USDC perpetuals",
                        instrument,
                    )
                    continue

                base = instrument.split("_", 1)[0]
                symbol = f"{base}USDT"
                if deribit.get_instrument_name(symbol) != instrument:
                    raise ValueError(
                        f"position instrument does not match bot symbol mapping: {instrument}"
                    )
                size = deribit.position_size_from_record(position)
                normalized_positions.append((symbol, size))
            fetch_succeeded = True
        except Exception as e:
            raw_positions = []
            normalized_positions = []
            fetch_succeeded = False
            log.warning(
                "  [3c] live position snapshot/normalization failed (%s); "
                "preserving orphan tracker and taking no flatten action",
                type(e).__name__,
            )

        for sym, size in normalized_positions:
            if abs(size) <= 0.0001:
                continue

            if sym not in tracked_symbols:
                active_orphans_this_scan.add(sym)
                first_seen = orphan_tracker.get(sym)

                if first_seen is None:
                    orphan_tracker[sym] = now_ts
                    log.warning(f"  ⚠️ [RECONCILER] Candidate orphan: {sym} (size={size})")
                    continue

                elapsed = now_ts - float(first_seen)
                if elapsed < ORPHAN_CONFIRM_SECONDS:
                    continue

                log.warning(f"  🚨 [RECONCILER] Confirmed orphan: {sym} — Flattening...")
                _cancel_all_open_orders_for_symbol(deribit, sym)

                flatten_side = "SELL" if size > 0 else "BUY"
                flatten_qty = deribit.round_amount(sym, abs(size))

                try:
                    if flatten_qty > 0:
                        deribit.place_market_order(symbol=sym, side=flatten_side, amount=flatten_qty, reduce_only=True)

                    flat_verified = _verify_actually_closed(deribit, sym)
                    orders_cleared = _cancel_all_open_orders_for_symbol(deribit, sym)
                    if flat_verified and orders_cleared:
                        orphan_tracker.pop(sym, None)
                        log.info(f"  ✓ [RECONCILER] Flattened orphan {sym}; flatness and order cleanup verified")
                        _send(f"🛡️ *[RECONCILER] Orphan Flattened — {sym}*")
                    else:
                        log.critical(
                            "  [RECONCILER] %s close/cleanup unverified; orphan tracker retained",
                            sym,
                        )
                except Exception as re_err:
                    log.error("  [RECONCILER] Flatten failed for %s (%s)", sym, type(re_err).__name__)

        if fetch_succeeded:
            for tracked_cand in list(orphan_tracker.keys()):
                if tracked_cand not in active_orphans_this_scan:
                    orphan_tracker.pop(tracked_cand, None)
            save_json(ORPHAN_TRACKER_FILE, orphan_tracker)
        
        balance = save_balance(deribit)

    if execution_mode == "MANAGE_ONLY":
        save_json(
            SCAN_STATUS_FILE,
            {
                "phase": "completed",
                "started_at": scan_started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "scan_ran": False,
                "symbols_attempted": 0,
                "symbols_scored": 0,
                "predictions_saved": 0,
                "execution_mode": execution_mode,
                "skip_reason": "MANAGE_ONLY_NO_NEW_ENTRIES",
                "management_ran": True,
            },
        )
        return

    log.info(f"\n[4] Scanning {len(SYMBOLS)} coins | Open:{open_count}/{max_open_trades} (Max Direction: {max_same_dir})")

    found = 0
    btc_momentum = check_btc_momentum()
    whale_flow = get_exchange_netflow("BTC")
    fng_data = check_fear_and_greed()
    vol_state = vol.get("status", "NORMAL")
    btc_df15_live = get_data("BTCUSDT", TIMEFRAME_ENTRY, canonical_observation=True)

    for symbol in SYMBOLS:
        log.info(f"\n  ── {symbol} ({get_tier(symbol)}) ──")
        sig = generate_signal(symbol, pipeline, thresholds, btc_momentum, whale_flow, fng_data, btc_df15_live=btc_df15_live)
        if sig is None: time.sleep(0.2); continue
        
        found += 1
        base_hurdle = rec_buy if sig["signal"] == "BUY" else rec_sell

        if execution_mode != "FULL":
            continue

        if execute_trade(
            deribit,
            sig,
            risk_mult,
            balance,
            vol_state,
            base_min_conf=base_hurdle,
        ):
            time.sleep(1.5)

    if _execution_allows_management():
        save_balance(deribit)

    log.info(
        f"\n{'═'*56}\nDONE — {found} signal(s) | "
        f"mode={execution_mode} | risk_equity={balance:.2f} USDC"
        f"\n{'═'*56}"
    )

    save_json(SCAN_STATUS_FILE, {
        "phase": "completed",
        "started_at": scan_started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scan_ran": bool(_SCAN_LIVENESS["scan_ran"]),
        "symbols_attempted": int(_SCAN_LIVENESS["symbols_attempted"]),
        "symbols_scored": int(_SCAN_LIVENESS["symbols_scored"]),
        "predictions_saved": int(_SCAN_LIVENESS["predictions_saved"]),
        "execution_mode": execution_mode,
        "skip_reason": _SCAN_LIVENESS.get("skip_reason"),
        "signals_found": int(found),
        "active_buy_conf": round(rec_buy, 1),
        "active_sell_conf": round(rec_sell, 1),
        "probation_offset": float(PROBATION_CONFIDENCE_OFFSET),
    })

if __name__ == "__main__":
    run_execution_scan()
