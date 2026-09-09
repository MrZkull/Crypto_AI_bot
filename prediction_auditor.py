# prediction_auditor.py — closes the loop on predictions.json

import json, time, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PREDICTIONS_FILE = "predictions.json"
DOSSIER_FILE     = "coin_dossier.json"
BAR_MINUTES      = 15
LOOKAHEAD_BARS   = 24
AUDIT_GRACE_BARS = 2
MAX_AUDIT_BATCH  = 300

ATR_STOP_MULT    = 2.5
ATR_TARGET1_MULT = 3.5

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


def fetch_klines_range(symbol, start_ms, end_ms, limit=40):
    for url in BINANCE_ENDPOINTS:
        try:
            r = requests.get(url, params={
                "symbol": symbol, "interval": "15m",
                "startTime": start_ms, "endTime": end_ms, "limit": limit
            }, timeout=10)
            if r.status_code == 200:
                raw = r.json()
                if raw:
                    return [{"high": float(k[2]), "low": float(k[3])} for k in raw]
        except Exception as e:
            log.debug(f"  kline fetch failed {symbol} via {url}: {e}")
    return []


def _window_closed(pred, now):
    try:
        gen_at = datetime.fromisoformat(pred["generated_at"].replace("Z", "+00:00"))
    except Exception:
        return False
    window_end = gen_at + timedelta(minutes=BAR_MINUTES * (pred.get("lookahead_bars", LOOKAHEAD_BARS) + AUDIT_GRACE_BARS))
    return now >= window_end


def audit_one(pred):
    """Replay the exact triple-barrier rule from make_targets() against real data."""
    symbol = pred["symbol"]
    sig    = pred["predicted_signal"]
    entry  = pred.get("entry_ref")
    atr    = pred.get("atr_ref")

    if not entry or not atr or atr <= 0:
        pred["ground_truth_label"] = "UNAUDITABLE"
        pred["correct"] = None
        pred["mfe_atr"] = pred["mae_atr"] = None
        pred["audited"] = True
        return pred

    try:
        gen_at = datetime.fromisoformat(pred["generated_at"].replace("Z", "+00:00"))
    except Exception:
        pred["ground_truth_label"] = "UNAUDITABLE"
        pred["correct"] = None
        pred["mfe_atr"] = pred["mae_atr"] = None
        pred["audited"] = True
        return pred

    lookahead = pred.get("lookahead_bars", LOOKAHEAD_BARS)
    start_ms  = int(gen_at.timestamp() * 1000)
    end_ms    = start_ms + BAR_MINUTES * 60 * 1000 * (lookahead + AUDIT_GRACE_BARS)

    klines = fetch_klines_range(symbol, start_ms, end_ms, limit=lookahead + AUDIT_GRACE_BARS)
    if not klines:
        pred["ground_truth_label"] = "UNAUDITABLE"
        pred["correct"] = None
        pred["mfe_atr"] = pred["mae_atr"] = None
        pred["audited"] = True
        return pred

    highs = [k["high"] for k in klines]
    lows  = [k["low"] for k in klines]

    buy_tp,  buy_sl  = entry + atr*ATR_TARGET1_MULT, entry - atr*ATR_STOP_MULT
    sell_tp, sell_sl = entry - atr*ATR_TARGET1_MULT, entry + atr*ATR_STOP_MULT

    buy_success = sell_success = False
    for h, l in zip(highs, lows):
        if l <= buy_sl: break
        if h >= buy_tp: buy_success = True; break
    for h, l in zip(highs, lows):
        if h >= sell_sl: break
        if l <= sell_tp: sell_success = True; break

    if sig == "NO_TRADE":
        if buy_success and not sell_success:
            ground_truth = "MISSED_BUY"
        elif sell_success and not buy_success:
            ground_truth = "MISSED_SELL"
        else:
            ground_truth = "NO_TRADE"
        pred["correct"] = (ground_truth == "NO_TRADE")
    else:
        if buy_success and not sell_success:
            ground_truth = "BUY"
        elif sell_success and not buy_success:
            ground_truth = "SELL"
        else:
            ground_truth = "NO_TRADE"
        pred["correct"] = (ground_truth == sig)

    pred["ground_truth_label"] = ground_truth

    if sig == "BUY":
        pred["mfe_atr"] = round((max(highs, default=entry) - entry) / atr, 2)
        pred["mae_atr"] = round((entry - min(lows,  default=entry)) / atr, 2)
    elif sig == "SELL":
        pred["mfe_atr"] = round((entry - min(lows,  default=entry)) / atr, 2)
        pred["mae_atr"] = round((max(highs, default=entry) - entry) / atr, 2)
    else:
        pred["mfe_atr"] = pred["mae_atr"] = None

    pred["audited"]    = True
    pred["audited_at"] = datetime.now(timezone.utc).isoformat()
    return pred


def update_coin_dossier(audited_preds):
    dossier = load_json(DOSSIER_FILE, {})
    for p in audited_preds:
        if p.get("correct") is None:
            continue
        sym = p["symbol"]
        
        # DEFENSIVE INITIALIZATION: Guarantees keys exist regardless of who touched dossier first
        if sym not in dossier or not isinstance(dossier[sym], dict):
            dossier[sym] = {}
            
        d = dossier[sym]
        d.setdefault("total_audited", 0)
        d.setdefault("total_correct", 0)
        d.setdefault("disagreement_correct_sum", 0.0)
        d.setdefault("disagreement_incorrect_sum", 0.0)
        d.setdefault("disagreement_correct_n", 0)
        d.setdefault("disagreement_incorrect_n", 0)
        d.setdefault("by_confidence_bucket", {})
        d.setdefault("tags", {})

        d["total_audited"] += 1
        d["total_correct"] += int(p["correct"])

        dv = p.get("ensemble_disagreement")
        if dv is not None:
            key = "disagreement_correct" if p["correct"] else "disagreement_incorrect"
            d[f"{key}_sum"] += dv
            d[f"{key}_n"]   += 1

        conf = p.get("confidence", 0)
        bucket = "45-55" if conf < 55 else "55-65" if conf < 65 else "65-75" if conf < 75 else "75+"
        b = d["by_confidence_bucket"].setdefault(bucket, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(p["correct"])

    save_json(DOSSIER_FILE, dossier)
    return dossier


def get_coin_calibration_hit_rate(symbol: str, confidence: float, dossier: dict) -> float:
    d = dossier.get(symbol)
    if not d:
        return 1.0
    bucket = "45-55" if confidence < 55 else "55-65" if confidence < 65 else "65-75" if confidence < 75 else "75+"
    b = d.get("by_confidence_bucket", {}).get(bucket)
    if not b or b.get("n", 0) < 5:
        return 1.0
    return b["correct"] / b["n"] if b["n"] > 0 else 1.0


def run_audit():
    preds = load_json(PREDICTIONS_FILE, [])
    if not preds:
        log.info("No predictions to audit.")
        return

    now = datetime.now(timezone.utc)
    pending = [p for p in preds if not p.get("audited") and _window_closed(p, now)]

    if not pending:
        log.info(f"No predictions ready for audit yet (total records: {len(preds)}).")
        return

    batch = pending[:MAX_AUDIT_BATCH]
    log.info(f"Auditing {len(batch)} of {len(pending)} eligible predictions...")

    for pred in batch:
        audit_one(pred)
        time.sleep(0.12)

    by_id = {p["pred_id"]: p for p in preds}
    for p in batch:
        by_id[p["pred_id"]] = p
    save_json(PREDICTIONS_FILE, list(by_id.values()))

    n_scored  = sum(1 for p in batch if p.get("correct") is not None)
    n_correct = sum(1 for p in batch if p.get("correct") is True)
    if n_scored:
        log.info(f"Audit complete: {n_correct}/{n_scored} correct ({n_correct/n_scored*100:.1f}%)")
    else:
        log.info("Audit complete: no auditable records this batch.")

    dossier = update_coin_dossier([p for p in batch if p.get("correct") is not None])
    log.info(f"coin_dossier.json updated — {len(dossier)} symbols tracked.")


if __name__ == "__main__":
    run_audit()
