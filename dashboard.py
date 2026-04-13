# dashboard.py — Final Production Version
import os, json, requests, base64, logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app   = Flask(__name__, static_folder="dashboard_static")
CORS(app)
log   = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Load environment variables with fallbacks
GH_TOKEN  = os.getenv("GH_PAT_TOKEN", "")
GH_REPO   = os.getenv("GITHUB_REPO", "Elliot14R/Crypto_AI_bot")
GH_BRANCH = os.getenv("GITHUB_BRANCH", "main")

# ── File helpers ──────────────────────────────────────────────────────

def load_json(filename, default):
    """Loads local JSON files as a fallback"""
    for p in [Path(filename), Path("data") / filename]:
        try:
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        except Exception:
            pass
    return default

def load_log(lines=200):
    for p in [Path("bot.log"), Path("data/bot.log")]:
        try:
            if p.exists():
                with open(p) as f:
                    return f.readlines()[-lines:]
        except Exception:
            pass
    return []

def fetch_live_github_data(filename):
    """Bypasses Render's stale disk and decodes the base64 file from GitHub API"""
    if not GH_REPO or not GH_TOKEN: 
        return {}
    
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Try the data/ folder first, then root
    for path in [f"data/{filename}", filename]:
        try:
            url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                data = r.json()
                # 🟢 THE FIX: Safely decode the base64 content
                content_raw = base64.b64decode(data["content"]).decode('utf-8')
                return json.loads(content_raw)
        except Exception as e:
            log.warning(f"Failed to fetch {filename} from {path}: {e}")
            continue
    return {}

# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/api/status")
def api_status():
    """Returns general bot health and open trade count"""
    history    = load_json("trade_history.json", [])
    # 🟢 FIX: Use live GitHub data for the trade count
    trades     = fetch_live_github_data("trades.json")
    scan_mode  = load_json("scan_mode.json", {})
    balance    = load_json("balance.json", {})

    real_hist  = [h for h in history if h.get("signal") != "RECOVERED"]
    wins       = [h for h in real_hist if (h.get("pnl") or 0) > 0]
    total_pnl  = sum(h.get("pnl", 0) for h in real_hist)
    win_rate   = round(len(wins)/len(real_hist)*100, 1) if real_hist else 0

    return jsonify({
        "win_rate":     win_rate,
        "total_pnl":    round(total_pnl, 4),
        "open_trades":  len(trades),
        "max_trades":   3,
        "scan_mode":    scan_mode.get("mode", "active"),
        "model_accuracy": 73.1,
        "balance":      balance.get("usdt", 0),
        "last_updated": balance.get("updated_at","")
    })

@app.route("/api/balance")
def api_balance():
    """Fetches LIVE balance directly from Deribit Testnet"""
    try:
        client_id = os.getenv("DERIBIT_CLIENT_ID")
        client_secret = os.getenv("DERIBIT_CLIENT_SECRET")
        if client_id and client_secret:
            from deribit_client import DeribitClient
            client = DeribitClient(client_id, client_secret)
            all_bals = client.get_all_balances()
            usdc_info = all_bals.get("USDC", {})
            main_val = float(usdc_info.get("equity_usd", 0))
            
            return jsonify({
                "ok": True,
                "usdt": round(main_val, 2),
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "exchange": "Deribit Testnet"
            })
    except Exception as e:
        log.warning(f"Live balance failed: {e}")
    return jsonify(load_json("balance.json", {}))

@app.route("/api/trades/open")
def api_open_trades():
    """Fetches LIVE positions from Deribit and stitches AI targets from GitHub"""
    try:
        local_trades = fetch_live_github_data("trades.json")
        client_id = os.getenv("DERIBIT_CLIENT_ID")
        client_secret = os.getenv("DERIBIT_CLIENT_SECRET")
        
        if client_id and client_secret:
            from deribit_client import DeribitClient
            client = DeribitClient(client_id, client_secret)
            positions = client.get_positions()
            formatted = []
            for p in positions:
                inst = p.get("instrument_name", "")
                base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                sym = f"{base}USDT"
                qty = float(p.get("size", 0))
                if qty == 0: continue
                
                entry = float(p.get("average_price", 0))
                live = float(p.get("mark_price", 0))
                side = "BUY" if qty > 0 else "SELL"
                pct = (live - entry) / entry * 100 if side == "BUY" else (entry - live) / entry * 100
                
                t_info = local_trades.get(sym, {})
                formatted.append({
                    "symbol": sym, "signal": side, "entry": entry, "qty": abs(qty),
                    "live_price": live, "pnl_pct": round(pct, 2),
                    "stop": t_info.get("stop", 0),
                    "tp1": t_info.get("tp1", 0),
                    "tp2": t_info.get("tp2", 0),
                    "confidence": t_info.get("confidence", 0),
                    "score": t_info.get("score", 0),
                    "status": "Live on Deribit"
                })
            return jsonify(formatted)
    except Exception as e:
        log.error(f"Open trades failed: {e}")
    return jsonify([])

@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Triggers the GitHub Action workflow confirmed in screenshots"""
    if not GH_TOKEN:
        return jsonify({"error": "GH_PAT_TOKEN missing"}), 400

    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # 🟢 THE FIX: Exact filename from your .github/workflows/ folder
    workflow_file = "crypto_bot.yml" 
    url = f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{workflow_file}/dispatches"
    
    try:
        r = requests.post(url, headers=headers, json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}}, timeout=10)
        if r.status_code in (204, 200):
            return jsonify({"status": "triggered", "message": "Scan started!"})
        else:
            return jsonify({"error": f"GitHub rejected: {r.text}"}), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/log")
def api_log():
    lines = load_log(200)
    return jsonify({"log": "".join(lines), "lines": len(lines)})

@app.route("/api/market")
def api_market():
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=5)
        if r.ok: return jsonify({i['symbol']: {"price": float(i['lastPrice'])} for i in r.json()})
    except: pass
    return jsonify({})

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("dashboard_static", path)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
