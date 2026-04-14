import os, json, time, threading, logging, requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app  = Flask(__name__, static_folder="dashboard_static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log  = logging.getLogger(__name__)

GH_TOKEN  = os.getenv("GH_PAT_TOKEN", "")
GH_REPO   = os.getenv("GITHUB_REPO",  "Elliot14R/Crypto_AI_bot")
GH_BRANCH = os.getenv("GITHUB_BRANCH","main")

_cache      = {}
_cache_time = {}
CACHE_TTL   = 60   

def _gh_headers(raw=False):
    headers = {"Authorization": f"token {GH_TOKEN}"}
    if raw: headers["Accept"] = "application/vnd.github.v3.raw"
    else:   headers["Accept"] = "application/vnd.github.v3+json"
    return headers

def fetch_live_github_data(filename):
    """Bypasses caching and base64 limits using raw headers"""
    if not GH_TOKEN or not GH_REPO: return None
    try:
        url = f"https://api.github.com/repos/{GH_REPO}/contents/data/{filename}?ref={GH_BRANCH}&t={int(time.time())}"
        r = requests.get(url, headers=_gh_headers(raw=True), timeout=8)
        if r.status_code == 404: # Try root if not in data/
            url = f"https://api.github.com/repos/{GH_REPO}/contents/{filename}?ref={GH_BRANCH}&t={int(time.time())}"
            r = requests.get(url, headers=_gh_headers(raw=True), timeout=8)
            
        if r.status_code == 200:
            if filename.endswith(".json"):
                try: return r.json()
                except: return json.loads(r.text)
            return r.text
    except Exception as e: log.warning(f"GitHub raw fetch failed for {filename}: {e}")
    return None

def get_data(filename: str, default):
    now = time.time()
    if filename in _cache and now - _cache_time.get(filename, 0) < CACHE_TTL:
        return _cache[filename]

    gh_data = fetch_live_github_data(filename)
    if gh_data is not None:
        _cache[filename]      = gh_data
        _cache_time[filename] = now
        return gh_data

    # Fallback to local
    for p in [Path(filename), Path("data") / filename]:
        try:
            if p.exists(): return json.loads(p.read_text()) if filename.endswith('.json') else p.read_text()
        except Exception: pass
    return default

def force_refresh(filename: str): _cache_time[filename] = 0

@app.route("/")
def index(): return send_from_directory("dashboard_static", "index.html")

@app.route("/trading")
@app.route("/signals")
@app.route("/market")
@app.route("/open-trades")
@app.route("/history")
@app.route("/performance")
@app.route("/configuration")
def spa(): return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    try: return send_from_directory("dashboard_static", path)
    except Exception: return send_from_directory("dashboard_static", "index.html")

# ════════ FULL DASHBOARD STATUS (TOP BAR) ════════
@app.route("/api/status")
def api_status():
    history   = get_data("trade_history.json", [])
    signals   = get_data("signals.json",       [])
    scan_mode = get_data("scan_mode.json",     {})

    real      = [h for h in history if h.get("signal") != "RECOVERED"]
    wins      = [h for h in real if (h.get("pnl") or 0) > 0]
    total_pnl = sum(h.get("pnl", 0) for h in real)
    win_rate  = round(len(wins)/len(real)*100, 1) if real else 0

    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t_sigs    = [s for s in signals if s.get("generated_at","").startswith(today)]
    buys      = sum(1 for s in t_sigs if s.get("signal")=="BUY")
    sells     = sum(1 for s in t_sigs if s.get("signal")=="SELL")

    return jsonify({
        "win_rate":       win_rate,
        "wins":           len(wins),
        "losses":         len(real) - len(wins),
        "total_pnl":      round(total_pnl, 4),
        "total_trades":   len(real),
        "open_trades":    len(get_data("trades.json", {})),
        "max_trades":     3,
        "scan_mode":      scan_mode.get("mode", "active"),
        "mode_label":     scan_mode.get("mode","active").upper(),
        "min_confidence": scan_mode.get("min_confidence", 65),
        "min_score":      scan_mode.get("min_score", 2),
        "today_signals":  len(t_sigs),
        "today_buys":     buys,
        "today_sells":    sells,
        "model_accuracy": 73.1,
    })

# ════════ MULTI-COIN BALANCE FETCH ════════
@app.route("/api/balance")
def api_balance():
    try:
        cid, secret = os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET")
        if cid and secret:
            from deribit_client import DeribitClient
            client = DeribitClient(cid, secret)
            bals   = client.get_all_balances()
            total  = float(bals.get("USDC", {}).get("equity_usd", 0))
            
            # 🟢 FIX: Restored the loop that feeds the other coins (BTC, ETH, etc) to the UI
            assets_list = []
            for cur, info in bals.items():
                assets_list.append({
                    "asset": cur, 
                    "free": str(info.get("available", 0)), 
                    "total": str(info.get("equity_usd", 0))
                })

            return jsonify({
                "ok":True, "usdt": round(total,2), "equity": round(total,2),
                "assets": assets_list,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "exchange": "Deribit(by Coinbase) Testnet"
            })
    except Exception as e: log.warning(f"Live balance: {e}")
    return jsonify({"ok":True, "usdt": 0, "assets": []})

# ════════ GROUND-TRUTH OPEN TRADES ════════
@app.route("/api/trades/open")
def api_open_trades():
    force_refresh("trades.json")
    ai_memory = get_data("trades.json", {})
    formatted_trades = []

    try:
        cid, secret = os.getenv("DERIBIT_CLIENT_ID"), os.getenv("DERIBIT_CLIENT_SECRET")
        if cid and secret:
            from deribit_client import DeribitClient
            client = DeribitClient(cid, secret)
            positions = client.get_positions()

            for p in positions:
                qty = float(p.get("size", 0))
                if qty == 0: continue

                inst = p.get("instrument_name", "")
                base_coin = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                symbol = f"{base_coin}USDT"

                entry = float(p.get("average_price", 0))
                live = float(p.get("mark_price", 0))
                signal = "BUY" if qty > 0 else "SELL"

                upnl = float(p.get("floating_profit_loss_usd") or 0)
                if upnl == 0:
                    base_pnl = float(p.get("floating_profit_loss") or 0)
                    upnl = base_pnl * live if "USDC" not in inst else base_pnl
                pct = ((live - entry) / entry * 100) if signal == "BUY" else ((entry - live) / entry * 100)

                t_info = ai_memory.get(symbol, {})

                formatted_trades.append({
                    "symbol": symbol, "signal": signal, "entry": entry, "qty": abs(qty),
                    "live_price": round(live, 6), "unrealised": round(upnl, 4), "pnl_pct": round(pct, 2),
                    "stop": float(t_info.get("stop", 0) or 0), "tp1": float(t_info.get("tp1", 0) or 0),
                    "tp2": float(t_info.get("tp2", 0) or 0), "confidence": t_info.get("confidence", 0),
                    "score": t_info.get("score", 0), "progress": 50, "reasons": t_info.get("reasons", [])
                })
            return jsonify(formatted_trades)
    except Exception as e: log.error(f"Deribit API fetch failed: {e}")
    return jsonify([])

# ════════ HISTORY & SIGNALS ════════
@app.route("/api/trades/history")
def api_trade_history():
    history = get_data("trade_history.json", [])
    real    = [h for h in history if h.get("signal") != "RECOVERED"]
    return jsonify(list(reversed(real[-100:])))

@app.route("/api/signals")
def api_signals():
    signals = get_data("signals.json", [])
    return jsonify(list(reversed(signals[-100:])))

# ════════ FULL PERFORMANCE MATH ════════
@app.route("/api/performance")
def api_performance():
    # 🟢 FIX: Restored all the math for the Performance charts
    history = get_data("trade_history.json", [])
    real    = [h for h in history if h.get("signal") != "RECOVERED"]
    wins     = [h for h in real if (h.get("pnl") or 0) > 0]
    losses   = [h for h in real if (h.get("pnl") or 0) <= 0]
    tpnl     = sum(h.get("pnl", 0) for h in real)
    avg_win  = sum(h["pnl"] for h in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(h["pnl"] for h in losses) / len(losses) if losses else 0

    by_symbol = {}
    for h in real:
        sym = h.get("symbol", "?")
        if sym not in by_symbol: by_symbol[sym] = {"trades":0,"wins":0,"pnl":0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"]    += h.get("pnl", 0)
        if (h.get("pnl") or 0) > 0: by_symbol[sym]["wins"] += 1

    daily = {}
    for h in real:
        day = (h.get("closed_at") or h.get("opened_at",""))[:10]
        if day: daily[day] = round(daily.get(day, 0) + h.get("pnl", 0), 4)

    loss_total = sum(h["pnl"] for h in losses)
    pf = round(abs(sum(h["pnl"] for h in wins) / loss_total), 2) if loss_total != 0 else 0

    return jsonify({
        "total_trades":  len(real), "wins": len(wins), "losses": len(losses),
        "win_rate":      round(len(wins)/len(real)*100,1) if real else 0,
        "total_pnl":     round(tpnl, 4), "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4),
        "profit_factor": pf, "by_symbol": by_symbol, "daily_pnl": daily,
    })

# ════════ UTILITIES (LOGS, SCAN, ETC) ════════
@app.route("/api/log")
def api_log():
    force_refresh("bot.log")
    log_content = get_data("bot.log", "Loading logs...")
    if isinstance(log_content, str):
        lines = log_content.splitlines(keepends=True)[-200:]
        return jsonify({"log": "".join(lines), "lines": len(lines)})
    return jsonify({"log": "No logs found.", "lines": 0})

@app.route("/api/scan", methods=["POST"])
def api_scan():
    if not GH_TOKEN or not GH_REPO: return jsonify({"error": "GH_PAT_TOKEN or GITHUB_REPO not configured"}), 400
    for workflow in ["crypto_bot.yml", "crypto_bot.yaml", "main.yml"]:
        try:
            r = requests.post(f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{workflow}/dispatches", headers=_gh_headers(), json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}}, timeout=15)
            if r.status_code in (204, 200):
                return jsonify({"status": "triggered", "message": "Scan started — logs will appear in ~60 seconds"})
        except Exception as e: log.warning(f"Workflow {workflow}: {e}")
    return jsonify({"error": "Failed to trigger scan"}), 500

@app.route("/api/config")
def api_config(): return jsonify({"max_open_trades": 3, "exchange": "Deribit Testnet", "model_accuracy": 73.1})

@app.route("/api/sync")
def api_sync():
    for f in ["trades.json", "trade_history.json", "signals.json", "bot.log"]: force_refresh(f)
    return jsonify({"status": "synced"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
