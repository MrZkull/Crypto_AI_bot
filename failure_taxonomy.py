# failure_taxonomy.py — classifies WHY closed trades lost, proposes adaptations

import json, time, logging
from datetime import datetime, timezone
from pathlib import Path
import requests

import config
from config import ATR_STOP_MULT
from prediction_auditor import get_coin_calibration_hit_rate, load_json as _pa_load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

HISTORY_FILE      = "trade_history.json"
DOSSIER_FILE      = "coin_dossier.json"
PROPOSALS_FILE    = "adaptation_proposals.json"

BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]


def load_json(path, default):
    try:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path, data):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    Path(tmp).replace(path)


def fetch_klines_between(symbol, start_iso, end_iso):
    try:
        start_ms = int(datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp() * 1000)
        end_ms   = int(datetime.fromisoformat(end_iso.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return []
    for url in BINANCE_ENDPOINTS:
        try:
            r = requests.get(url, params={
                "symbol": symbol, "interval": "15m",
                "startTime": start_ms, "endTime": end_ms, "limit": 200
            }, timeout=10)
            if r.status_code == 200:
                raw = r.json()
                if raw:
                    return [{"high": float(k[2]), "low": float(k[3])} for k in raw]
        except Exception as e:
            log.debug(f"  kline fetch failed {symbol}: {e}")
    return []


def compute_trade_excursion(trade: dict):
    entry = float(trade.get("entry", 0) or 0)
    atr   = float(trade.get("snapshot", {}).get("entry_atr", 0) or 0)
    
    # Dynamic fallback to actual stop multiplier used
    if atr <= 0:
        stop = float(trade.get("stop", 0) or 0)
        stop_mult = float(trade.get("stop_mult_used", ATR_STOP_MULT))
        if stop > 0 and entry > 0:
            atr = abs(entry - stop) / stop_mult
            
    if entry <= 0 or atr <= 0:
        return None, None

    klines = fetch_klines_between(trade.get("symbol"), trade.get("opened_at", ""), trade.get("closed_at", ""))
    if not klines:
        return None, None

    highs = [k["high"] for k in klines]
    lows  = [k["low"] for k in klines]
    signal = trade.get("signal")

    if signal == "BUY":
        mfe = round((max(highs, default=entry) - entry) / atr, 2)
        mae = round((entry - min(lows, default=entry)) / atr, 2)
    elif signal == "SELL":
        mfe = round((entry - min(lows, default=entry)) / atr, 2)
        mae = round((max(highs, default=entry) - entry) / atr, 2)
    else:
        return None, None
    return mfe, mae


def count_same_day_peer_losses(symbol, signal, closed_at, history):
    try:
        day = closed_at[:10]
    except Exception:
        return 0
    return sum(
        1 for h in history
        if h.get("symbol") != symbol
        and h.get("signal") == signal
        and str(h.get("closed_at", ""))[:10] == day
        and float(h.get("pnl", 0)) < 0
    )


def classify_failure(trade, mfe_atr, mae_atr, same_day_peer_losses, calibration_hit_rate):
    if float(trade.get("pnl", 0)) >= 0:
        return None
    if mfe_atr is None or mae_atr is None:
        return "UNCLASSIFIED_NO_DATA"

    conf = float(trade.get("confidence", 0))

    if mfe_atr >= 2.5 and conf < 65:
        return "PREMATURE_WHIPSAW"

    if conf >= 65 and calibration_hit_rate < 0.45:
        return "MODEL_OVERCONFIDENT"

    if same_day_peer_losses >= 3:
        return "REGIME_CASCADE"

    if mae_atr <= 1.2:
        return "MOMENTUM_EXHAUSTION"

    return "UNCLASSIFIED"


ADAPTATION_RULES = {
    "PREMATURE_WHIPSAW": {
        "param": "atr_stop_mult_override",
        "suggested_change": "widen stop ~15% (capped) for this symbol",
        "min_occurrences": 3,
    },
    "MODEL_OVERCONFIDENT": {
        "param": "probation_offset_override",
        "suggested_change": "+5% confidence hurdle for this symbol",
        "min_occurrences": 3,
    },
}


def propose_adaptations(dossier):
    proposals = load_json(PROPOSALS_FILE, [])
    existing_keys = {(p["symbol"], p["tag"]) for p in proposals if p.get("status") == "pending"}

    for symbol, d in dossier.items():
        tags = d.get("tags", {})
        for tag, rule in ADAPTATION_RULES.items():
            count = tags.get(tag, 0)
            if count >= rule["min_occurrences"] and (symbol, tag) not in existing_keys:
                proposals.append({
                    "id": f"{symbol}_{tag}_{int(time.time())}",
                    "symbol": symbol,
                    "tag": tag,
                    "param": rule["param"],
                    "suggested_change": rule["suggested_change"],
                    "occurrences": count,
                    "status": "pending",
                    "proposed_at": datetime.now(timezone.utc).isoformat(),
                })
    save_json(PROPOSALS_FILE, proposals)
    return proposals


def run_failure_audit():
    history = load_json(HISTORY_FILE, [])
    dossier = load_json(DOSSIER_FILE, {})

    processed = 0
    for trade in history:
        if trade.get("failure_tagged"):
            continue
        if trade.get("signal") == "RECOVERED":
            continue
        if not trade.get("closed_at"):
            continue

        mfe_atr, mae_atr = compute_trade_excursion(trade)
        symbol = trade.get("symbol")
        same_day_losses = count_same_day_peer_losses(symbol, trade.get("signal"), trade.get("closed_at", ""), history)
        hit_rate = get_coin_calibration_hit_rate(symbol, float(trade.get("confidence", 0)), dossier)

        tag = classify_failure(trade, mfe_atr, mae_atr, same_day_losses, hit_rate)
        trade["failure_tagged"] = True
        trade["mfe_atr"] = mfe_atr
        trade["mae_atr"] = mae_atr
        trade["failure_tag"] = tag

        if tag and tag not in ("UNCLASSIFIED", "UNCLASSIFIED_NO_DATA"):
            if symbol not in dossier or not isinstance(dossier[symbol], dict):
                dossier[symbol] = {
                    "total_audited": 0, "total_correct": 0,
                    "disagreement_correct_sum": 0.0, "disagreement_incorrect_sum": 0.0,
                    "disagreement_correct_n": 0, "disagreement_incorrect_n": 0,
                    "by_confidence_bucket": {}, "tags": {}
                }
            dossier[symbol].setdefault("tags", {})
            dossier[symbol]["tags"][tag] = dossier[symbol]["tags"].get(tag, 0) + 1

        processed += 1
        time.sleep(0.1)

    save_json(HISTORY_FILE, history)
    save_json(DOSSIER_FILE, dossier)
    log.info(f"Failure audit: {processed} newly-closed trades tagged.")

    proposals = propose_adaptations(dossier)
    pending = [p for p in proposals if p["status"] == "pending"]
    log.info(f"Adaptation proposals: {len(pending)} pending review.")


if __name__ == "__main__":
    run_failure_audit()
    
