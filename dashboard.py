# dashboard_api.py — Geo-restriction fix
#
# ROOT CAUSE: Binance/Bybit block Indian IPs and some Render.com IPs (451 error)
# FIX STRATEGY:
#   - Dashboard API NEVER calls the exchange directly
#   - Balance is written to balance.json by trade_executor.py (runs on GitHub Actions = US IP)
#   - Dashboard reads from local JSON files only
#   - Public Binance market data (prices) still works from anywhere
#   - All trading happens exclusively on GitHub Actions (US servers, no geo-block)

import os, json, time, threading, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)

TRADES_FILE  = "trades.json"
LOG_FILE     = "bot.log"
HISTORY_FILE = "trade_history.json"
SIGNALS_FILE = "signals.json"
BALANCE_FILE = "balance.json"   # ← written by trade_executor, read by dashboard

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ── File helpers ──────────────────────────────────────────────────
def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path) as f: return json.load(f)
    except Exception: pass
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Live prices via PUBLIC Binance API (no auth, no geo-block) ────
def get_live_prices(symbols):
    """Public endpoint — works from any country including India."""
    prices = {}
    if not symbols:
        return prices
    try:
        # Batch fetch — much faster than one-by-one
        sym_json = json.dumps(list(symbols))
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbols": sym_json},
            timeout=8
        )
        if r.ok:
            for item in r.json():
                prices[item["symbol"]] = float(item["price"])
            return prices
    except Exception:
        pass
    # Fallback: one by one
    for sym in symbols:
        try:
            r = requests.get("https://api.binance.com/api/v3/ticker/price",
                             params={"symbol": sym}, timeout=5)
            if r.ok:
                prices[sym] = float(r.json()["price"])
        except Exception:
            prices[sym] = None
    return prices


def enrich_trades(trades: dict, prices: dict) -> list:
    result = []
    for sym, t in trades.items():
        entry  = t.get("entry", 0)
        qty    = t.get("qty", 0)
        signal = t.get("signal", "BUY")
        live   = prices.get(sym)

        if live and entry and qty:
            if signal == "BUY":
                upnl = round((live - entry) * qty, 4)
                pct  = round((live - entry) / entry * 100, 2)
            else:
                upnl = round((entry - live) * qty, 4)
                pct  = round((entry - live) / entry * 100, 2)
        else:
            upnl, pct = 0, 0

        sl, tp1, tp2 = t.get("stop", 0), t.get("tp1", 0), t.get("tp2", 0)

        if signal == "BUY" and tp2 > entry > sl and live:
            prog = round(max(0, min(100, (live - entry) / (tp2 - entry) * 100)), 1)
        elif signal == "SELL" and tp2 < entry < sl and live:
            prog = round(max(0, min(100, (entry - live) / (entry - tp2) * 100)), 1)
        else:
            prog = 0

        result.append({
            **t,
            "live_price":     live,
            "unrealised_pnl": upnl,
            "pnl_pct":        pct,
            "progress":       prog,
            "status":         "TP1 hit" if t.get("tp1_hit") else "Open",
        })
    return result


# ══ API ROUTES ════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    trades  = load_json(TRADES_FILE, {})
    history = load_json(HISTORY_FILE, [])
    wins    = [h for h in history if (h.get("pnl") or 0) > 0]
    losses  = [h for h in history if (h.get("pnl") or 0) <= 0]
    total   = len(history)
    wr      = round(len(wins) / total * 100, 1) if total else 0
    totpnl  = round(sum(h.get("pnl") or 0 for h in history), 4)
    bal     = load_json(BALANCE_FILE, {})
    return jsonify({
        "ok":           True,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "open_trades":  len(trades),
        "max_trades":   3,
        "total_closed": total,
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     wr,
        "total_pnl":    totpnl,
        "model_acc":    73.1,
        "balance_usdt": bal.get("usdt", None),
        "balance_updated": bal.get("updated_at", None),
    })


@app.route("/api/balance")
def api_balance():
    """
    Reads balance.json written by trade_executor.py on GitHub Actions.
    Never calls the exchange directly (avoids geo-block).
    """
    bal = load_json(BALANCE_FILE, {})
    if not bal:
        return jsonify({
            "ok":    False,
            "usdt":  None,
            "note":  "Balance not yet fetched. Run a scan on GitHub Actions first.",
            "assets": [],
        })
    return jsonify({
        "ok":        True,
        "usdt":      bal.get("usdt", 0),
        "assets":    bal.get("assets", []),
        "other_count": len(bal.get("assets", [])),
        "updated_at": bal.get("updated_at"),
    })


@app.route("/api/trades/open")
def api_open():
    trades  = load_json(TRADES_FILE, {})
    symbols = list(trades.keys())
    prices  = get_live_prices(symbols)
    return jsonify(enrich_trades(trades, prices))


@app.route("/api/trades/history")
def api_history():
    return jsonify(load_json(HISTORY_FILE, [])[-100:])


@app.route("/api/signals")
def api_signals():
    return jsonify(load_json(SIGNALS_FILE, [])[-200:])


@app.route("/api/log")
def api_log():
    try:
        if Path(LOG_FILE).exists():
            lines = Path(LOG_FILE).read_text(errors="replace").splitlines()
            return jsonify({"lines": lines[-150:]})
    except Exception as e:
        pass
    return jsonify({"lines": ["Log file not found — check GitHub Actions artifacts"]})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """
    Cannot trigger scan from Render (geo-blocked).
    Direct user to GitHub Actions instead.
    """
    return jsonify({
        "ok":      False,
        "message": "Manual scans must be triggered from GitHub Actions (not from Render — geo-restricted).",
        "action":  "Go to your GitHub repo → Actions → Run workflow manually",
    }), 200


@app.route("/api/close_trade", methods=["POST"])
def api_close():
    """
    Cannot close trades from Render (geo-blocked).
    Returns instructions instead.
    """
    data   = request.get_json() or {}
    symbol = data.get("symbol", "?")
    return jsonify({
        "ok":      False,
        "message": f"Cannot close {symbol} from dashboard — Binance blocks Indian/Render IPs.",
        "action":  "Go to testnet.binance.vision → Open Orders → Cancel manually",
    }), 200


@app.route("/api/market")
def api_market():
    """Fetch 24h ticker for all 20 symbols — public API, no geo-block."""
    SYMBOLS = [
        "BTCUSDT","ETHUSDT","BNBUSDT",
        "SOLUSDT","AVAXUSDT","NEARUSDT","SUIUSDT","APTUSDT",
        "LINKUSDT","DOTUSDT","UNIUSDT","AAVEUSDT","XRPUSDT",
        "FETUSDT","RENDERUSDT","ADAUSDT","INJUSDT","ARBUSDT","OPUSDT","SEIUSDT",
    ]
    try:
        sym_json = json.dumps(SYMBOLS)
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/24hr",
            params={"symbols": sym_json},
            timeout=10
        )
        if r.ok:
            return jsonify({"ok": True, "data": r.json()})
    except Exception as e:
        pass
    return jsonify({"ok": False, "data": []})


@app.route("/api/fear_greed")
def api_fear_greed():
    """Fear & Greed index — no geo-block."""
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.ok:
            return jsonify({"ok": True, "data": r.json()["data"][0]})
    except Exception:
        pass
    return jsonify({"ok": False, "data": {}})


@app.route("/")
def root():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("dashboard_static", path)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard API → http://0.0.0.0:{port}")
    log.info("Note: Exchange calls run on GitHub Actions (US IP), not here")
    app.run(host="0.0.0.0", port=port, debug=False)
