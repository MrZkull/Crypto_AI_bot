#!/usr/bin/env python3
# candidate_scanner.py — Prospective Shadow Candidate Scanner & Ledger Logger

import os
import sys
import json
import time
import math
import logging
import urllib.request
from pathlib import Path
from filelock import FileLock

import numpy as np
import pandas as pd
import joblib

from feature_engineering import add_indicators, ALL_FEATURES
from market_data_integrity import sanitize_closed_candles, merge_completed_htf
from execution_policy import build_candidate_config

try:
    from execution_policy import calculate_trade_brackets
except ImportError:
    def calculate_trade_brackets(entry: float, atr: float, side: str, vol_state: str,
                                 stop_mult: float, tp1_mult: float, tp2_mult: float) -> dict:
        side_norm = side.upper().strip()
        if side_norm == "BUY":
            return {
                "stop": entry - atr * stop_mult,
                "tp1":  entry + atr * tp1_mult,
                "tp2":  entry + atr * tp2_mult,
            }
        else:
            return {
                "stop": entry + atr * stop_mult,
                "tp1":  entry - atr * tp1_mult,
                "tp2":  entry - atr * tp2_mult,
            }

try:
    from config import SYMBOLS
except ImportError:
    SYMBOLS = [
        "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "NEARUSDT", "LTCUSDT",
        "UNIUSDT", "BCHUSDT", "DOTUSDT", "ALGOUSDT", "ENAUSDT", "DOGEUSDT",
        "TRUMPUSDT", "PUMPUSDT", "AAVEUSDT", "LINKUSDT", "SUIUSDT", "AVAXUSDT",
        "ADAUSDT", "TRXUSDT", "XLMUSDT", "ZECUSDT", "HBARUSDT", "CRVUSDT", "FILUSDT"
    ]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CANDIDATE_MODEL_FILE = Path("candidate_model.pkl")
CANDIDATE_META_FILE  = Path("candidate_meta_model.pkl")
CANDIDATE_MANIFEST   = Path("candidate_manifest.json")
EVENTS_LEDGER_FILE   = Path("candidate_events.jsonl")
STATE_FILE           = Path("candidate_state.json")


# ── Public Binance REST Ingestion ──────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int = 150) -> pd.DataFrame:
    """Fetches public market candles directly without requiring API credentials."""
    endpoints = [
        f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
    ]
    headers = {"User-Agent": "CryptoBot-Shadow-Scanner/2.0"}

    for url in endpoints:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    raw = json.loads(resp.read().decode())
                    rows = []
                    for k in raw:
                        rows.append({
                            "open_time": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "close_time": int(k[6]),
                            "taker_buy_base_vol": float(k[9]) if len(k) > 9 else 0.0
                        })
                    df = pd.DataFrame(rows)
                    if not df.empty:
                        return df.sort_values("open_time").reset_index(drop=True)
        except Exception:
            continue

    log.warning(f"Failed fetching {interval} candles for {symbol}")
    return pd.DataFrame()


# ── Strict HTF Alignment & Feature Construction ───────────────────────────

def _align_1h_to_15m(df1h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if df1h.empty or len(df1h) < 5 or df15.empty:
        return pd.DataFrame()
    h = merge_completed_htf(df15, df1h, ["rsi", "adx", "trend"], prefix="htf1h")
    return h.rename(columns={"htf1h_rsi": "rsi_1h", "htf1h_adx": "adx_1h", "htf1h_trend": "trend_1h"})


def _align_4h_to_15m(df4h: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if df4h.empty or len(df4h) < 5 or df15.empty:
        return pd.DataFrame()
    h = merge_completed_htf(df15, df4h, ["rsi", "trend"], prefix="htf4h")
    return h.rename(columns={"htf4h_rsi": "rsi_4h", "htf4h_trend": "trend_4h"})


def _align_btc_to_15m(btc_df15: pd.DataFrame, df15: pd.DataFrame) -> pd.DataFrame:
    if btc_df15 is None or btc_df15.empty or "close" not in btc_df15.columns:
        df15["btc_close"] = np.nan
        return df15
    try:
        out = merge_completed_htf(df15, btc_df15, ["close"], prefix="btc")
        return out.rename(columns={"btc_close": "btc_close"})
    except Exception:
        df15["btc_close"] = np.nan
        return df15


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "btc_close" in df.columns and df["btc_close"].notna().sum() > 30:
        btc_ret = df["btc_close"].pct_change()
        coin_ret = df["close"].pct_change()
        roll_cov = coin_ret.rolling(20, min_periods=10).cov(btc_ret)
        roll_var = btc_ret.rolling(20, min_periods=10).var()
        df["btc_corr_20"] = coin_ret.rolling(20, min_periods=10).corr(btc_ret)
        df["btc_beta_20"] = roll_cov / roll_var.replace(0, np.nan)
        df["btc_rel_strength"] = (df["close"].pct_change(6) - df["btc_close"].pct_change(6)) * 100
    else:
        df["btc_corr_20"] = 0.0
        df["btc_beta_20"] = 1.0
        df["btc_rel_strength"] = 0.0

    df["btc_corr_20"] = df["btc_corr_20"].fillna(0.0).clip(-1, 1)
    df["btc_beta_20"] = df["btc_beta_20"].fillna(1.0).clip(-5, 5)
    df["btc_rel_strength"] = df["btc_rel_strength"].fillna(0.0).clip(-50, 50)
    return df


def generate_live_features(symbol: str, btc_df15: pd.DataFrame) -> pd.DataFrame:
    """Reconstructs the full training feature matrix for the active symbol."""
    df15 = fetch_klines(symbol, "15m", limit=150)
    df1h = fetch_klines(symbol, "1h", limit=60)
    df4h = fetch_klines(symbol, "4h", limit=30)

    if df15.empty or len(df15) < 50:
        return pd.DataFrame()

    now_ms = int(time.time() * 1000)
    clean_15m = sanitize_closed_candles(df15, candle_duration_ms=15 * 60 * 1000, observation_time_ms=now_ms)
    df15 = pd.DataFrame(clean_15m)
    if df15.empty or len(df15) < 30:
        return pd.DataFrame()

    taker_col = None
    if "taker_buy_base_vol" in df15.columns:
        taker_col = df15[["open_time", "taker_buy_base_vol"]].copy()

    df15 = add_indicators(df15)
    if taker_col is not None and "taker_buy_base_vol" not in df15.columns:
        df15 = df15.merge(taker_col, on="open_time", how="left")

    if not df1h.empty:
        df1h = pd.DataFrame(sanitize_closed_candles(df1h, candle_duration_ms=60 * 60 * 1000, observation_time_ms=now_ms))
        if not df1h.empty:
            df1h_feat = add_indicators(df1h)
            df15 = _align_1h_to_15m(df1h_feat, df15)

    if not df4h.empty:
        df4h = pd.DataFrame(sanitize_closed_candles(df4h, candle_duration_ms=4 * 60 * 60 * 1000, observation_time_ms=now_ms))
        if not df4h.empty:
            df4h_feat = add_indicators(df4h)
            df15 = _align_4h_to_15m(df4h_feat, df15)

    df15 = _align_btc_to_15m(btc_df15, df15)
    df15 = _add_extra_features(df15)
    df15["symbol"] = symbol
    return df15


# ── Event Logging & Ledger State Tracking ──────────────────────────────────

def log_event(event_type: str, candidate_id: str, pred_id: str, data: dict):
    with FileLock("candidate_events.jsonl.lock"):
        with open(EVENTS_LEDGER_FILE, "a") as f:
            f.write(json.dumps({"event": event_type, "candidate_id": candidate_id, "pred_id": pred_id, **data}) + "\n")


def record_candidate_setup(
    manifest: dict,
    manifest_config: dict,
    pred_id: str,
    symbol: str,
    side: str,
    primary_selected: bool,
    meta_selected: bool,
    row: dict,
    simulated_entry: float,
    observation_time_ms: int
):
    if not primary_selected and meta_selected:
        raise ValueError(f"Meta-selected setup is not primary-selected: {pred_id}")

    if int(observation_time_ms) <= 0:
        raise ValueError(f"Invalid observation timestamp for {pred_id}")

    balance = float(manifest_config.get("evaluation_balance_usd", 10_000.0))
    atr = float(row.get("atr", 0.0))
    if atr <= 0.0 or np.isnan(atr):
        log.warning(f"Skipping setup {pred_id}: ATR <= 0")
        return

    brackets = calculate_trade_brackets(
        simulated_entry, atr, side, "NORMAL",
        float(manifest_config.get("ATR_STOP_MULT", 2.5)),
        float(manifest_config.get("ATR_TARGET1_MULT", 3.5)),
        float(manifest_config.get("ATR_TARGET2_MULT", 7.5))
    )

    risk_pct = float(manifest_config.get("RISK_PER_TRADE", 0.015))
    initial_risk_usd = balance * risk_pct
    stop_dist = abs(simulated_entry - brackets["stop"])
    qty = (initial_risk_usd / stop_dist) if stop_dist > 0 else 0.0

    event = {
        "candidate_id": manifest["candidate_id"],
        "pred_id": pred_id,
        "event": "SETUP_OBSERVED",

        "model_sha256": manifest["model_sha256"],
        "feature_schema_hash": manifest["feature_schema_hash"],
        "feature_code_hash": manifest["feature_code_hash"],
        "execution_policy_hash": manifest["execution_policy_hash"],
        "decision_policy_hash": manifest.get("decision_policy_hash", ""),
        "config_hash": manifest["config_hash"],

        "symbol": symbol,
        "side": side,
        "primary_selected": bool(primary_selected),
        "meta_selected": bool(meta_selected),

        "atr": atr,
        "simulated_entry": simulated_entry,
        "stop": brackets["stop"],
        "tp1": brackets["tp1"],
        "tp2": brackets["tp2"],
        "qty": qty,
        "initial_risk_usd": initial_risk_usd,

        "open_time": int(observation_time_ms),
        "entry_time_ms": int(observation_time_ms),
        "recorded_at_ms": int(time.time() * 1000),

        "entry_fee_usd": 0.0,
        "tp1_fee_usd": 0.0,
        "thesis_label": None,
        "net_r": None,

        "evidence_type": "EMPIRICAL_PROSPECTIVE"
    }

    log_event("SETUP_OBSERVED", manifest["candidate_id"], pred_id, event)

    with FileLock("candidate_state.json.lock"):
        state = {}
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as sf:
                    state = json.load(sf)
            except Exception:
                state = {}

        if pred_id in state:
            return

        event["state"] = "OPEN" if meta_selected else "COUNTERFACTUAL_OPEN"
        event["status"] = "ACTIVE" if meta_selected else "COUNTERFACTUAL_ACTIVE"
        event["last_funding_ts"] = int(observation_time_ms)
        event["funding_usd"] = 0.0
        event["realized_gross_pnl"] = 0.0

        state[pred_id] = event
        tmp_file = str(STATE_FILE) + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_file, STATE_FILE)


# ── Execution Scan Engine ──────────────────────────────────────────────────

def scan_and_evaluate_candidate():
    if not (CANDIDATE_MODEL_FILE.exists() and CANDIDATE_MANIFEST.exists()):
        log.info("Candidate model or manifest not present; skipping scan.")
        return

    with open(CANDIDATE_MANIFEST, "r") as f:
        manifest = json.load(f)

    if manifest.get("status") != "AWAITING_PROSPECTIVE_EVIDENCE":
        log.info(f"Candidate status is {manifest.get('status')}; skipping prospective scan.")
        return

    config = build_candidate_config()
    config["evaluation_balance_usd"] = 10_000.0

    primary_pipeline = joblib.load(CANDIDATE_MODEL_FILE)
    meta_pipeline = joblib.load(CANDIDATE_META_FILE) if CANDIDATE_META_FILE.exists() else None

    symbols = primary_pipeline.get("symbols", SYMBOLS)
    log.info(f"Scanning {len(symbols)} candidate pairs against {manifest.get('candidate_id')}...")

    btc_df15 = fetch_klines("BTCUSDT", "15m", limit=150)
    now_ms = int(time.time() * 1000)
    if not btc_df15.empty:
        btc_df15 = pd.DataFrame(sanitize_closed_candles(btc_df15, candle_duration_ms=15 * 60 * 1000, observation_time_ms=now_ms))

    for symbol in symbols:
        try:
            feats_df = generate_live_features(symbol, btc_df15)
            if feats_df.empty or len(feats_df) < 5:
                continue

            last_row = feats_df.iloc[-1]
            obs_time = int(last_row["open_time"])
            pred_id = f"{symbol}_{obs_time}"

            if STATE_FILE.exists():
                with open(STATE_FILE, "r") as sf:
                    try:
                        active_state = json.load(sf)
                        if pred_id in active_state:
                            continue
                    except Exception:
                        pass

            for f in primary_pipeline["all_features"]:
                if f not in feats_df.columns:
                    feats_df[f] = 0.0

            X = feats_df[primary_pipeline["all_features"]].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
            Xs = primary_pipeline["selector"].transform(X)

            probas = primary_pipeline["ensemble"].predict_proba(Xs)[0]
            classes = list(primary_pipeline["label_encoder"].classes_)

            buy_idx = classes.index("BUY") if "BUY" in classes else 0
            sell_idx = classes.index("SELL") if "SELL" in classes else 2

            p_buy = probas[buy_idx]
            p_sell = probas[sell_idx]

            thresh_buy = primary_pipeline.get("recommended_threshold_buy", 0.40)
            thresh_sell = primary_pipeline.get("recommended_threshold_sell", 0.45)

            side = None
            conf = 0.0
            if p_buy >= thresh_buy and p_buy > p_sell:
                side = "BUY"
                conf = p_buy
            elif p_sell >= thresh_sell and p_sell > p_buy:
                side = "SELL"
                conf = p_sell

            primary_selected = (side is not None)
            if not primary_selected:
                continue

            meta_selected = False
            if meta_pipeline and primary_selected:
                meta_thresh = meta_pipeline.get("recommended_meta_threshold", 0.50)
                side_code = 1.0 if side == "BUY" else -1.0
                meta_input = last_row.to_dict()
                meta_input["primary_side"] = side
                meta_input["primary_conf"] = conf
                meta_input["meta_primary_conf"] = conf
                meta_input["meta_primary_side_code"] = side_code

                rsi = float(meta_input.get("rsi", 50.0))
                macd = float(meta_input.get("macd_hist", 0.0))
                trend = float(meta_input.get("trend", 0.0))
                meta_input["meta_rsi_directional"] = (rsi - 50.0) * side_code
                meta_input["meta_macd_directional"] = macd * side_code
                meta_input["meta_trend_aligned"] = trend * side_code

                meta_df = pd.DataFrame([meta_input])
                for col in meta_pipeline["meta_features"]:
                    if col not in meta_df.columns:
                        meta_df[col] = 0.0

                X_meta = meta_df[meta_pipeline["meta_features"]].replace([np.inf, -np.inf], np.nan).fillna(0).values
                meta_proba = meta_pipeline["meta_ensemble"].predict_proba(X_meta)[0, meta_pipeline["meta_positive_idx"]]
                meta_selected = bool(meta_proba >= meta_thresh)

            record_candidate_setup(
                manifest=manifest,
                manifest_config=config,
                pred_id=pred_id,
                symbol=symbol,
                side=side,
                primary_selected=primary_selected,
                meta_selected=meta_selected,
                row=last_row.to_dict(),
                simulated_entry=float(last_row["close"]),
                observation_time_ms=obs_time
            )
            log.info(f"✅ Setup Recorded: {pred_id} | Side: {side} | Conf: {conf:.2f} | Meta-Approved: {meta_selected}")

        except Exception as e:
            log.error(f"Error evaluating candidate on {symbol}: {e}")


if __name__ == "__main__":
    scan_and_evaluate_candidate()
