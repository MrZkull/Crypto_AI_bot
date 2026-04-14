import os, json, base64, threading, time, logging, requests
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
CACHE_TTL   = 120   

def _gh_headers():
    return {
        "Authorization": f"token {GH_TOKEN}",
        "Accept":        "application/vnd.github.v3+json",
    }

def fetch_from_github(filename: str) -> dict | list | None:
    if not GH_TOKEN or not GH_REPO:
        return None
    for path in [f"data/{filename}", filename]:
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GH_REPO}/contents/{path}",
                headers=_gh_headers(), timeout=8
            )
            if r.status_code == 200:
                content = base64.b64decode(r.json()["content"]).decode("utf-8")
                return json.loads(content)
        except Exception:
            pass
    return None

def get_data(filename: str, default):
    now = time.time()
    if filename in _cache and now - _cache_time.get(filename, 0) < CACHE_TTL:
        return _cache[filename]

    gh_data = fetch_from_github(filename)
    if gh_data is not None:
        _cache[filename]      = gh_data
        _cache_time[filename] = now
        try:
            Path(filename).write_text(json.dumps(gh_data, indent=2, default=str))
            Path("data").mkdir(exist_ok=True)
            (Path("data") / filename).write_text(json.dumps(gh_data, indent=2, default=str))
        except Exception:
            pass
        return gh_data

    for p in [Path(filename), Path("data") / filename]:
        try:
            if p.exists():
                data = json.loads(p.read_text())
                _cache[filename]      = data
                _cache_time[filename] = now
                return data
        except Exception:
            pass
    return default

def get_log_lines(n=200) -> str:
    for p in [Path("bot.log"), Path("data/bot.log")]:
        try:
            if p.exists():
                return "".join(p.read_text().splitlines(keepends=True)[-n:])
        except Exception:
            pass
    return ""

def force_refresh(filename: str):
    _cache_time[filename] = 0

def _background_sync():
    files = ["trades.json","trade_history.json","signals.json",
             "balance.json","scan_mode.json"]
    while True:
        try:
            for f in files:
                force_refresh(f)
                get_data(f, {} if f.endswith(".json") and "history" not in f
                         and "signals" not in f else [])
        except Exception:
            pass
        time.sleep(90)

threading.Thread(target=_background_sync, daemon=True).start()

@app.route("/")
def index():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/trading")
@app.route("/signals")
@app.route("/market")
@app.route("/open-trades")
@app.route("/history")
@app.route("/performance")
@app.route("/configuration")
def spa():
    return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    try:
        return send_from_directory("dashboard_static", path)
    except Exception:
        return send_from_directory("dashboard_static", "index.html")

@app.route("/api/status")
def api_status():
    force_refresh("trades.json")
    force_refresh("signals.json")
    force_refresh("trade_history.json")
    force_refresh("scan_mode.json")
    force_refresh("balance.json")

    trades    = get_data("trades.json",        {})
    history   = get_data("trade_history.json", [])
    signals   = get_data("signals.json",       [])
    scan_mode = get_data("scan_mode.json",     {})
    balance   = get_data("balance.json",       {})

    real      = [h for h in history if h.get("signal") != "RECOVERED"]
    wins      = [h for h in real if (h.get("pnl") or 0) > 0]
    total_pnl = sum(h.get("pnl", 0) for h in real)
    win_rate  = round(len(wins)/len(real)*100, 1) if real else 0

    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t_sigs    = [s for s in signals if s.get("generated_at","").startswith(today)]
    buys      = sum(1 for s in t_sigs if s.get("signal")=="BUY"  and not s.get("rejected"))
    sells     = sum(1 for s in t_sigs if s.get("signal")=="SELL" and not s.get("rejected"))

    return jsonify({
        "win_rate":       win_rate,
        "total_pnl":      round(total_pnl, 4),
        "wins":           len(wins),
        "losses":         len(real) - len(wins),
        "total_trades":   len(real),
        "open_trades":    len([t for t in trades.values() if not t.get("closed")]),
        "max_trades":     3,
        "scan_mode":      scan_mode.get("mode", "active"),
        "mode_label":     scan_mode.get("mode","active").upper(),
        "min_confidence": scan_mode.get("min_confidence", 50),
        "today_signals":  len(t_sigs),
        "today_buys":     buys,
        "today_sells":    sells,
        "model_accuracy": 73.1,
        "balance":        balance.get("usdt", 0),
        "exchange":       balance.get("exchange", "Deribit Testnet"),
        "last_updated":   balance.get("updated_at", ""),
        "open_positions": balance.get("open_positions", 0),
    })

@app.route("/api/balance")
def api_balance():
    force_refresh("balance.json")
    bal = get_data("balance.json", {})
    try:
        cid    = os.getenv("DERIBIT_CLIENT_ID","")
        secret = os.getenv("DERIBIT_CLIENT_SECRET","")
        if cid and secret:
            from deribit_client import DeribitClient
            client = DeribitClient(cid, secret)
            bals   = client.get_all_balances()
            total  = client.get_total_equity_usd()
            assets = [{"asset":c,"free":str(i.get("available",0)),"total":str(i.get("equity_usd",0))}
                      for c,i in bals.items()]
            result = {
                "ok":True,"usdt":round(total,2),"equity":round(total,2),
                "assets":assets,
                "updated_at":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "mode":"deribit_testnet","exchange":"Deribit(by Coinbase) Testnet",
            }
            _cache["balance.json"]      = result
            _cache_time["balance.json"] = time.time()
            return jsonify(result)
    except Exception as e:
        log.warning(f"Live balance: {e}")
    return jsonify({**bal,"ok":True})

@app.route("/api/trades/open")
def api_open_trades():
    force_refresh("trades.json")
    trades = get_data("trades.json", {})
    result = []

    symbols  = [s for s,t in trades.items() if not t.get("closed")]
    live_prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/price", timeout=6)
        if r.ok:
            live_prices = {item["symbol"]: float(item["price"])
                           for item in r.json()
                           if item["symbol"] in symbols}
    except Exception:
        pass

    for symbol, t in trades.items():
        if t.get("closed"):
            continue
        entry = float(t.get("entry", 0) or 0)
        live  = live_prices.get(symbol, 0.0)

        upnl    = 0.0
        pnl_pct = 0.0
        if live > 0 and entry > 0:
            qty = float(t.get("qty", 0) or 0)
            if t.get("signal") == "BUY":
                upnl    = (live - entry) * qty
                pnl_pct = (live - entry) / entry * 100
            else:
                upnl    = (entry - live) * qty
                pnl_pct = (entry - live) / entry * 100

        tp2      = float(t.get("tp2", 0) or 0)
        progress = 0.0
        if entry > 0 and tp2 > 0 and live > 0:
            total_dist = abs(tp2 - entry)
            moved_dist = abs(live - entry)
            if total_dist > 0:
                progress = min(100, max(0, moved_dist / total_dist * 100))

        result.append({
            **t,
            "symbol":        symbol,
            "live_price":    round(live, 6),
            "unrealised":    round(upnl, 4),
            "pnl_pct":       round(pnl_pct, 2),
            "progress":      round(progress, 1),
            "entry":         entry,
            "stop":          float(t.get("stop",  0) or 0),
            "tp1":           float(t.get("tp1",   0) or 0),
            "tp2":           float(t.get("tp2",   0) or 0),
            "confidence":    t.get("confidence",  0),
            "score":         t.get("score",       0),
            "reasons":       t.get("reasons",     []),
        })

    return jsonify(result)

@app.route("/api/trades/history")
def api_trade_history():
    force_refresh("trade_history.json")
    history = get_data("trade_history.json", [])
    real    = [h for h in history if h.get("signal") != "RECOVERED"]
    return jsonify(list(reversed(real[-100:])))

@app.route("/api/signals")
def api_signals():
    force_refresh("signals.json")
    signals  = get_data("signals.json", [])
    symbol   = request.args.get("symbol")
    sig_type = request.args.get("type")
    limit    = int(request.args.get("limit", 100))

    if symbol:
        signals = [s for s in signals if s.get("symbol") == symbol]
    if sig_type:
        signals = [s for s in signals if s.get("signal") == sig_type.upper()]

    return jsonify(list(reversed(signals[-limit:])))

@app.route("/api/log")
def api_log():
    return jsonify({"log": get_log_lines(200), "lines": 200})

@app.route("/api/market")
def api_market():
    symbols = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT",
        "XRPUSDT","LINKUSDT","NEARUSDT","DOTUSDT","ADAUSDT",
        "INJUSDT","ARBUSDT","OPUSDT","UNIUSDT","AAVEUSDT",
        "FETUSDT","RENDERUSDT","SEIUSDT","SUIUSDT","APTUSDT",
    ]
    prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=10)
        if r.ok:
            for item in r.json():
                if item["symbol"] in symbols:
                    prices[item["symbol"]] = {
                        "price":      float(item.get("lastPrice",          0)),
                        "change_24h": float(item.get("priceChangePercent", 0)),
                        "volume_24h": float(item.get("quoteVolume",        0)),
                        "high_24h":   float(item.get("highPrice",          0)),
                        "low_24h":    float(item.get("lowPrice",           0)),
                    }
    except Exception as e:
        log.warning(f"Market data: {e}")
    return jsonify(prices)

@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Trigger GitHub Actions workflow_dispatch with explicit error logging."""
    if not GH_TOKEN or not GH_REPO:
        return jsonify({"error": "GH_PAT_TOKEN or GITHUB_REPO not configured"}), 400

    errors = []
    for workflow in ["crypto_bot.yml", "crypto_bot.yaml", "main.yml"]:
        try:
            r = requests.post(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{workflow}/dispatches",
                headers={**_gh_headers(), "Content-Type": "application/json"},
                json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}},
                timeout=15
            )
            if r.status_code in (204, 200):
                log.info(f"✅ GitHub Actions triggered via {workflow}")
                for f in ["trades.json","balance.json","signals.json","trade_history.json","scan_mode.json"]:
                    force_refresh(f)
                return jsonify({
                    "status":   "triggered",
                    "workflow": workflow,
                    "message":  "Scan started — results appear in ~60 seconds",
                })
            else:
                # 🟢 NEW: Capture the exact reason GitHub rejected the workflow dispatch
                err_msg = f"{workflow} failed: {r.status_code} - {r.text}"
                log.error(err_msg)
                errors.append(err_msg)
        except Exception as e:
            log.warning(f"Workflow {workflow}: {e}")

    return jsonify({"error": "GitHub API rejected the request", "details": errors}), 500

@app.route("/api/performance")
def api_performance():
    force_refresh("trade_history.json")
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
        if sym not in by_symbol:
            by_symbol[sym] = {"trades":0,"wins":0,"pnl":0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"]    += h.get("pnl", 0)
        if (h.get("pnl") or 0) > 0:
            by_symbol[sym]["wins"] += 1

    daily = {}
    for h in real:
        day = (h.get("closed_at") or h.get("opened_at",""))[:10]
        if day:
            daily[day] = round(daily.get(day, 0) + h.get("pnl", 0), 4)

    loss_total = sum(h["pnl"] for h in losses)
    pf = 0
    if loss_total != 0:
        pf = round(abs(sum(h["pnl"] for h in wins) / loss_total), 2)

    return jsonify({
        "total_trades":  len(real),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins)/len(real)*100,1) if real else 0,
        "total_pnl":     round(tpnl, 4),
        "avg_win":       round(avg_win, 4),
        "avg_loss":      round(avg_loss, 4),
        "profit_factor": pf,
        "by_symbol":     by_symbol,
        "daily_pnl":     daily,
    })

@app.route("/api/config")
def api_config():
    return jsonify({
        "max_open_trades":        3,
        "risk_per_trade_pct":     2.0,
        "atr_stop_mult":          1.5,
        "atr_tp1_mult":           2.0,
        "atr_tp2_mult":           3.0,
        "min_confidence_active":  50,
        "min_confidence_quiet":   55,
        "min_score_active":       1,
        "min_score_quiet":        2,
        "min_adx":                15,
        "scan_interval_min":      15,
        "symbols":                20,
        "exchange":               "Deribit Testnet (USDC Linear Perpetuals)",
        "model_accuracy":         73.1,
    })

@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    data   = request.get_json() or {}
    symbol = data.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    force_refresh("trades.json")
    trades = get_data("trades.json", {})
    if symbol not in trades:
        return jsonify({"error": f"{symbol} not in open trades"}), 404

    trade = trades.pop(symbol)
    force_refresh("trade_history.json")
    history = get_data("trade_history.json", [])
    history.append({
        **trade,
        "close_price":  float(trade.get("entry", 0)),
        "pnl":          0.0,
        "closed_at":    datetime.now(timezone.utc).isoformat(),
        "close_reason": "Manual close via dashboard",
    })

    for p in [Path("trades.json"), Path("data/trades.json")]:
        try:
            p.parent.mkdir(exist_ok=True)
            with open(p,"w") as f: json.dump(trades, f, indent=2)
        except Exception: pass
    for p in [Path("trade_history.json"), Path("data/trade_history.json")]:
        try:
            p.parent.mkdir(exist_ok=True)
            with open(p,"w") as f: json.dump(history, f, indent=2, default=str)
        except Exception: pass

    force_refresh("trades.json")
    force_refresh("trade_history.json")

    return jsonify({
        "status":  "removed",
        "symbol":  symbol,
        "warning": "Also close position on Deribit testnet UI!",
    })

@app.route("/api/sync")
def api_sync():
    """Force refresh all caches from GitHub."""
    for f in ["trades.json","trade_history.json","signals.json",
              "balance.json","scan_mode.json"]:
        force_refresh(f)
    return jsonify({"status": "synced"})

@app.route("/health")
def health():
    return jsonify({"status":"ok","time":datetime.now(timezone.utc).isoformat()})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"CryptoBot Dashboard starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
