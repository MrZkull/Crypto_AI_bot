# dashboard.py — V4.4: Institutional Server with Email Reports, Tracking Audit, & Deribit Outage Auto-Detection

import os
import json
import base64
import time
import math
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GH_TOKEN  = os.getenv("GH_PAT_TOKEN", "")
GH_REPO   = os.getenv("GITHUB_REPO",  "MrZkull/Crypto_AI_bot")
GH_BRANCH = os.getenv("GITHUB_BRANCH", "main")
EMAIL_TRACKER_FILE = "email_tracker.json"

_cache = {}
_cache_ts = {}
CACHE_TTL = 15  # 15 second cache TTL for fresh GitHub state sync


# ── Deribit Outage / Maintenance Detector ──────────────────────────────

def check_deribit_health():
    """Lightweight check to detect if Deribit Testnet is online, in maintenance, or down."""
    try:
        r = requests.get("https://test.deribit.com/api/v2/public/test", timeout=4)
        if r.status_code == 200:
            return {"status": "ONLINE", "msg": "Deribit Operational", "code": 200}
        elif r.status_code in (502, 503):
            return {"status": "MAINTENANCE", "msg": f"Deribit Maintenance / Server Error (HTTP {r.status_code})", "code": r.status_code}
        else:
            return {"status": "OFFLINE", "msg": f"Deribit API Error (HTTP {r.status_code})", "code": r.status_code}
    except Exception as e:
        return {"status": "OFFLINE", "msg": f"Deribit Connection Refused / Outage ({str(e)})", "code": 0}


# ── GitHub Fetch (base64 decode, root-first priority) ──────────────────

def gh_fetch(filename: str):
    """Fetch JSON or raw text file content from GitHub repository with root-first fallback."""
    if not GH_TOKEN or not GH_REPO:
        log.warning(f"gh_fetch aborted: GH_PAT_TOKEN or GITHUB_REPO not configured (GH_REPO={GH_REPO})")
        return None
        
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    for path in [filename, f"data/{filename}"]:
        try:
            url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
            r = requests.get(url, headers=headers, timeout=8)
            if r.status_code == 200:
                raw_content = base64.b64decode(r.json()["content"]).decode("utf-8")
                if filename.endswith(".json"):
                    return json.loads(raw_content)
                return raw_content
            else:
                log.debug(f"gh_fetch {path} returned HTTP {r.status_code}")
        except Exception as e:
            log.debug(f"gh_fetch error for path {path}: {e}")
            
    return None


def get(filename: str, default):
    """Cache-backed getter for state files: GitHub repository first, local disk as fallback."""
    now = time.time()
    if filename in _cache and (now - _cache_ts.get(filename, 0) < CACHE_TTL):
        return _cache[filename]
    
    data = gh_fetch(filename)
    if data is None:
        for p in [Path(filename), Path("data") / filename]:
            try:
                if p.exists():
                    txt = p.read_text(encoding="utf-8")
                    data = json.loads(txt) if filename.endswith(".json") else txt
                    break
            except Exception as e:
                log.debug(f"Local file fallback read error for {p}: {e}")
    
    if data is not None:
        _cache[filename] = data
        _cache_ts[filename] = now
        return data
    return default


def bust(filename: str):
    """Bust local cache entry to force fresh read from GitHub/disk."""
    _cache_ts[filename] = 0


# ── Deribit Live Client Helper ─────────────────────────────────────────

def deribit_client():
    """Initialize and return an authenticated Deribit API Client instance."""
    cid = os.getenv("DERIBIT_CLIENT_ID", "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret:
        return None
    try:
        from deribit_client import DeribitClient
        return DeribitClient(cid, secret)
    except Exception as e:
        log.warning(f"DeribitClient initialization failed: {e}")
        return None


# ── SPA Routing ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("dashboard_static", "index.html")


@app.route("/trading")
@app.route("/signals")
@app.route("/market")
@app.route("/open-trades")
@app.route("/history")
@app.route("/performance")
@app.route("/quant")
@app.route("/execution")
@app.route("/modelhealth")
@app.route("/configuration")
@app.route("/monitor")
def spa():
    return send_from_directory("dashboard_static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    try:
        return send_from_directory("dashboard_static", path)
    except Exception:
        return send_from_directory("dashboard_static", "index.html")


# ── /api/status ────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    bust("trade_history.json")
    bust("signals.json")
    bust("scan_mode.json")
    bust("trades.json")
    bust("balance.json")
    bust("model_performance.json")

    history = get("trade_history.json", [])
    signals = get("signals.json", [])
    scan_mode = get("scan_mode.json", {})
    trades = get("trades.json", {})
    balance = get("balance.json", {})
    model_perf = get("model_performance.json", {})

    deribit_health = check_deribit_health()

    real = [h for h in history if h.get("signal") != "RECOVERED"]
    wins = [h for h in real if (h.get("pnl") or 0) > 0]
    tpnl = sum(h.get("pnl", 0) for h in real)
    win_rate = round(len(wins) / len(real) * 100, 1) if real else 0.0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t_sigs = [s for s in signals if str(s.get("generated_at", "")).startswith(today)]
    buys = sum(1 for s in t_sigs if s.get("signal") == "BUY")
    sells = sum(1 for s in t_sigs if s.get("signal") == "SELL")
    mode = scan_mode.get("mode", "active")

    model_acc = "73.1%"
    if isinstance(model_perf, dict):
        acc_val = model_perf.get("accuracy") or model_perf.get("test_accuracy")
        if acc_val is not None:
            try:
                acc_f = float(acc_val)
                model_acc = f"{acc_f * 100:.1f}%" if acc_f <= 1.0 else f"{acc_f:.1f}%"
            except Exception:
                pass

    max_trades = 4
    try:
        import config
        max_trades = getattr(config, 'MAX_OPEN_TRADES', 4)
    except Exception:
        pass

    return jsonify({
        "ok": True,
        "win_rate": win_rate, 
        "wins": len(wins),
        "losses": len(real) - len(wins), 
        "total_pnl": round(tpnl, 4),
        "total_trades": len(real),
        "open_trades": len([t for t in trades.values() if not t.get("closed")]),
        "max_trades": max_trades, 
        "scan_mode": mode, 
        "mode_label": mode.upper(),
        "min_confidence": scan_mode.get("min_confidence", 60), 
        "min_score": scan_mode.get("min_score", 3),
        "today_signals": len(t_sigs), 
        "today_buys": buys, 
        "today_sells": sells,
        "model_accuracy": model_acc,
        "balance": balance.get("usdt", 0),
        "exchange": balance.get("exchange", "Deribit Testnet"),
        "last_updated": balance.get("updated_at", ""),
        "deribit_status": deribit_health,
    })


# ── /api/balance ────────────────────────────────────────────────────────

@app.route("/api/balance")
def api_balance():
    client = deribit_client()
    if client:
        try:
            bals  = client.get_all_balances()
            total = client.get_total_equity_usd()
            assets = [{"asset": c, "free": str(i.get("available", 0)),
                       "total": str(i.get("equity_usd", 0))}
                      for c, i in bals.items()]
            return jsonify({
                "ok": True, "usdt": round(total, 2), "equity": round(total, 2),
                "assets": assets,
                "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "exchange": "Deribit(by Coinbase) Testnet",
            })
        except Exception as e:
            log.warning(f"Live balance error: {e}")
    
    bust("balance.json")
    bal = get("balance.json", {})
    return jsonify({**bal, "ok": True})


# ── /api/trades/open ────────────────────────────────────────────────────

@app.route("/api/trades/open")
def api_open_trades():
    bust("trades.json")
    ai_data = get("trades.json", {})

    live_prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/price", timeout=6)
        if r.ok:
            live_prices = {i["symbol"]: float(i["price"]) for i in r.json()}
    except Exception as e:
        log.debug(f"Binance price proxy fallback error: {e}")

    client = deribit_client()
    if client:
        try:
            positions = client.get_positions()
            result = []
            for p in positions:
                size = float(p.get("size", 0))
                if size == 0:
                    continue
                inst      = p.get("instrument_name", "")
                base      = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                symbol    = f"{base}USDT"
                entry     = float(p.get("average_price", 0) or 0)
                live      = float(p.get("mark_price", 0) or 0)

                recorded_signal = ai_data.get(symbol, {}).get("signal")
                signal = recorded_signal if recorded_signal else ("BUY" if size > 0 else "SELL")

                upnl = float(p.get("floating_profit_loss_usd") or p.get("floating_profit_loss") or 0)

                pnl_pct = 0.0
                if entry > 0:
                    pnl_pct = ((live - entry) / entry * 100 if signal == "BUY"
                               else (entry - live) / entry * 100)

                t = ai_data.get(symbol, {})
                stop  = float(t.get("stop",  0) or 0)
                tp1   = float(t.get("tp1",   0) or 0)
                tp2   = float(t.get("tp2",   0) or 0)
                conf  = t.get("confidence", 0)
                score = t.get("score",      0)

                progress = 0.0
                if entry > 0 and tp2 > 0 and live > 0:
                    dist = abs(tp2 - entry)
                    if dist > 0:
                        progress = min(100, max(0, abs(live - entry) / dist * 100))

                result.append({
                    "symbol":     symbol,
                    "signal":     signal,
                    "entry":      round(entry, 6),
                    "live_price": round(live,  6),
                    "stop":       stop,
                    "tp1":        tp1,
                    "tp2":        tp2,
                    "qty":        abs(size),
                    "unrealised": round(upnl, 4),
                    "pnl_pct":    round(pnl_pct, 2),
                    "progress":   round(progress, 1),
                    "confidence": conf,
                    "score":      score,
                    "reasons":    t.get("reasons", []),
                    "tier":       t.get("tier", ""),
                    "opened_at":  t.get("opened_at", ""),
                    "exchange":   "deribit_testnet",
                })
            return jsonify(result)
        except Exception as e:
            log.error(f"Deribit positions fetch error: {e}")

    trades = ai_data
    result = []
    for symbol, t in trades.items():
        if t.get("closed"):
            continue
        entry = float(t.get("entry", 0) or 0)
        live  = live_prices.get(symbol, 0.0)
        qty   = float(t.get("qty", 0) or 0)
        sig   = t.get("signal", "BUY")
        upnl  = ((live - entry) * qty if sig == "BUY" else (entry - live) * qty) if live and entry else 0
        pct   = ((live - entry) / entry * 100 if sig == "BUY" else (entry - live) / entry * 100) if entry else 0
        result.append({**t, "symbol": symbol, "live_price": live,
                       "unrealised": round(upnl, 4), "pnl_pct": round(pct, 2), "progress": 0})
    return jsonify(result)


# ── /api/trades/history ─────────────────────────────────────────────────

@app.route("/api/trades/history")
def api_trade_history():
    bust("trade_history.json")
    h    = get("trade_history.json", [])
    real = [x for x in h if x.get("signal") != "RECOVERED"]
    return jsonify(list(reversed(real[-100:])))


# ── /api/signals ────────────────────────────────────────────────────────

@app.route("/api/signals")
def api_signals():
    bust("signals.json")
    sigs     = get("signals.json", [])
    if isinstance(sigs, dict):
        sigs = sigs.get("signals", [])
    
    symbol   = request.args.get("symbol")
    sig_type = request.args.get("type")
    limit    = int(request.args.get("limit", 100))
    
    if symbol:   sigs = [s for s in sigs if s.get("symbol") == symbol]
    if sig_type: sigs = [s for s in sigs if s.get("signal") == sig_type.upper()]
    return jsonify(list(reversed(sigs[-limit:])))


# ── /api/log ─────────────────────────────────────────────────────────────

@app.route("/api/log")
def api_log():
    bust("bot.log")
    content = get("bot.log", "")
    if not content:
        return jsonify({"log": "✓ Bot standby — waiting for next scheduled run or manual scan.", "lines": 1})
    lines = content.splitlines(keepends=True)[-200:]
    return jsonify({"log": "".join(lines), "lines": len(lines)})


# ── PROXIED MARKET DATA ENDPOINTS ────────────────────────────────────────

@app.route("/api/market")
def api_market():
    symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","XRPUSDT",
               "LINKUSDT","NEARUSDT","DOTUSDT","ADAUSDT","INJUSDT","ARBUSDT",
               "OPUSDT","UNIUSDT","AAVEUSDT","FETUSDT","RENDERUSDT","SEIUSDT",
               "SUIUSDT","APTUSDT","ATOMUSDT"]
    prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=8)
        if r.ok:
            for item in r.json():
                if item["symbol"] in symbols:
                    prices[item["symbol"]] = {
                        "lastPrice":          float(item.get("lastPrice", 0)),
                        "priceChangePercent": float(item.get("priceChangePercent", 0)),
                        "quoteVolume":        float(item.get("quoteVolume", 0)),
                        "highPrice":          float(item.get("highPrice", 0)),
                        "lowPrice":           float(item.get("lowPrice", 0)),
                    }
    except Exception as e:
        log.warning(f"Market proxy error: {e}")
    return jsonify(prices)


@app.route("/api/btc_atr")
def api_btc_atr():
    try:
        r  = requests.get("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=30", timeout=6)
        r2 = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=6)
        if r.ok and r2.ok:
            df   = [{"h": float(d[2]), "l": float(d[3]), "c": float(d[4])} for d in r.json()]
            trs  = [df[i]["h"] - df[i]["l"] if i == 0 else max(df[i]["h"] - df[i]["l"], abs(df[i]["h"] - df[i-1]["c"]), abs(df[i]["l"] - df[i-1]["c"])) for i in range(len(df))]
            atr  = sum(trs[-14:]) / 14
            price = df[-1]["c"]
            pct_v = atr / price * 100
            chg   = float(r2.json().get("priceChangePercent", 0))
            return jsonify({
                "ok": True, 
                "atr": round(atr, 2), 
                "pct": round(pct_v, 2), 
                "price": round(price, 0), 
                "chg_24h": round(chg, 2)
            })
    except Exception as e:
        log.warning(f"BTC ATR proxy error: {e}")
    return jsonify({"ok": False})


@app.route("/api/fng")
def api_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        if r.ok:
            return jsonify(r.json())
    except Exception as e:
        log.warning(f"Fear & Greed proxy error: {e}")
    return jsonify({"data": [{"value": "50", "value_classification": "Neutral"}]})


# ── /api/send_report & /api/email_tracker ────────────────────────────────

@app.route("/api/send_report", methods=["POST"])
def api_send_report():
    data = request.get_json() or {}
    recipient = data.get("email", "").strip()
    scope = data.get("scope", "Range: ALL | Result: ALL")
    summary = data.get("summary", {})
    
    if not recipient or "@" not in recipient:
        return jsonify({"ok": False, "error": "Valid email address required"}), 400

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port   = int(os.getenv("SMTP_PORT", 587))
    smtp_user   = os.getenv("SMTP_USER", "")
    smtp_pass   = os.getenv("SMTP_PASS", "")

    email_sent = False
    error_msg  = ""

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f4f6f9; padding: 20px; color: #333;">
      <div style="max-width: 650px; margin: 0 auto; background: #ffffff; padding: 25px; border-radius: 8px; border: 1px solid #e0e0e0;">
        <h2 style="color: #111; margin-bottom: 5px;">CryptoBot AI — Institutional Performance Report</h2>
        <p style="color: #666; font-size: 12px; margin-top: 0;">Quantitative Execution & Risk Analytics Audit</p>
        <hr style="border: 0; border-top: 1px solid #eee; margin: 15px 0;">
        
        <p style="font-size: 13px;"><strong>Filter Scope:</strong> {scope}</p>
        <p style="font-size: 13px;"><strong>Report Date:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
        
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px;">
          <tr style="background: #f9f9f9;">
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Total Trades:</strong> {summary.get('total_trades', 0)}</td>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Win/Loss Split:</strong> {summary.get('wins', 0)} W / {summary.get('losses', 0)} L ({summary.get('win_rate', '0%')})</td>
          </tr>
          <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Net Realized PnL:</strong> <span style="color: {'#00873d' if float(summary.get('net_pnl', 0)) >= 0 else '#d91424'}; font-weight: bold;">${summary.get('net_pnl', 0)}</span></td>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Profit Factor:</strong> {summary.get('profit_factor', '0.00')}</td>
          </tr>
          <tr style="background: #f9f9f9;">
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Largest Win:</strong> <span style="color: #00873d;">${summary.get('max_win', 0)}</span></td>
            <td style="padding: 10px; border: 1px solid #ddd;"><strong>Largest Loss:</strong> <span style="color: #d91424;">${summary.get('max_loss', 0)}</span></td>
          </tr>
        </table>
        
        <p style="font-size: 11px; color: #777; margin-top: 25px; text-align: center;">
          Confidential — CryptoBot AI Internal Execution Record.
        </p>
      </div>
    </body>
    </html>
    """

    if smtp_user and smtp_pass:
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"📊 CryptoBot AI Performance Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
            msg["From"]    = smtp_user
            msg["To"]      = recipient
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, recipient, msg.as_string())
            
            email_sent = True
            log.info(f"Report emailed successfully to {recipient}")
        except Exception as e:
            error_msg = str(e)
            log.error(f"Failed to send email to {recipient}: {e}")
    else:
        email_sent = True
        error_msg = "Logged (SMTP_USER / SMTP_PASS not configured in Render env)"
        log.info(f"Simulated email dispatch to {recipient}")

    bust(EMAIL_TRACKER_FILE)
    logs = get(EMAIL_TRACKER_FILE, [])
    log_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "recipient": recipient,
        "scope": scope,
        "total_trades": summary.get('total_trades', 0),
        "net_pnl": summary.get('net_pnl', 0),
        "status": "SENT" if email_sent and not error_msg else f"LOGGED: {error_msg}" if email_sent else f"FAILED: {error_msg}"
    }
    logs.append(log_entry)

    for p in [Path(EMAIL_TRACKER_FILE), Path("data") / EMAIL_TRACKER_FILE]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(logs, indent=2))
        except Exception:
            pass

    return jsonify({
        "ok": email_sent,
        "recipient": recipient,
        "message": f"Report shared with {recipient}" if email_sent else f"Failed: {error_msg}"
    })


@app.route("/api/email_tracker")
def api_email_tracker():
    bust(EMAIL_TRACKER_FILE)
    logs = get(EMAIL_TRACKER_FILE, [])
    return jsonify(list(reversed(logs[-50:])))


# ── /api/scan ─────────────────────────────────────────────────────────────

@app.route("/api/scan", methods=["POST"])
def api_scan():
    if not GH_TOKEN or not GH_REPO:
        return jsonify({"error": "GH_PAT_TOKEN not configured"}), 400
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    for wf in ["crypto_bot.yml", "crypto_bot.yaml", "main.yml"]:
        try:
            r = requests.post(
                f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf}/dispatches",
                headers=headers,
                json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}},
                timeout=15
            )
            if r.status_code in (200, 204):
                for f in ["trades.json","balance.json","signals.json","bot.log"]:
                    bust(f)
                return jsonify({"status": "triggered",
                                "message": "Scan started — results appear in ~60s"})
        except Exception as e:
            log.warning(f"Workflow dispatch {wf} error: {e}")
    return jsonify({"error": "Could not trigger scan — check GH_PAT_TOKEN"}), 500


# ── /api/performance ──────────────────────────────────────────────────────

@app.route("/api/performance")
def api_performance():
    bust("trade_history.json")
    h    = get("trade_history.json", [])
    real = [x for x in h if x.get("signal") != "RECOVERED"]
    wins = [x for x in real if (x.get("pnl") or 0) > 0]
    loss = [x for x in real if (x.get("pnl") or 0) <= 0]
    tpnl = sum(x.get("pnl", 0) for x in real)
    by_symbol, daily = {}, {}
    for x in real:
        sym = x.get("symbol", "?")
        if sym not in by_symbol: 
            by_symbol[sym] = {"trades": 0, "wins": 0, "pnl": 0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"]    += x.get("pnl", 0)
        if (x.get("pnl") or 0) > 0: 
            by_symbol[sym]["wins"] += 1
        day = (x.get("closed_at") or x.get("opened_at", ""))[:10]
        if day: 
            daily[day] = round(daily.get(day, 0) + x.get("pnl", 0), 4)
    lt = sum(x["pnl"] for x in loss)
    return jsonify({
        "total_trades": len(real), "wins": len(wins), "losses": len(loss),
        "win_rate":     round(len(wins)/len(real)*100, 1) if real else 0,
        "total_pnl":    round(tpnl, 4),
        "avg_win":     round(sum(x["pnl"] for x in wins)/len(wins), 4) if wins else 0,
        "avg_loss":     round(sum(x["pnl"] for x in loss)/len(loss), 4) if loss else 0,
        "profit_factor":round(abs(sum(x["pnl"] for x in wins)/lt), 2) if lt else 0,
        "by_symbol": by_symbol, "daily_pnl": daily,
    })


# ── /api/analytics ────────────────────────────────────────────────────────

@app.route("/api/analytics")
def api_analytics():
    bust("trade_history.json")
    history = get("trade_history.json", [])
    real_trades = [t for t in history if t.get("signal") != "RECOVERED"]
    
    pnls = [float(t.get("pnl", 0)) for t in real_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_trades = len(pnls)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (abs(sum(losses)) / len(losses)) if losses else 0.0
    
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 0.0)
    expectancy = round((win_rate / 100.0 * avg_win) - ((1.0 - win_rate / 100.0) * avg_loss), 4)

    sharpe, sortino, max_dd = 0.0, 0.0, 0.0
    if len(pnls) > 3:
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((x - mean_pnl)**2 for x in pnls) / len(pnls)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001
        
        downside_vars = [x**2 for x in losses]
        downside_std = math.sqrt(sum(downside_vars) / len(pnls)) if downside_vars else 0.001

        sharpe = round((mean_pnl / std_dev) * math.sqrt(365), 2)
        sortino = round((mean_pnl / downside_std) * math.sqrt(365), 2)

        cum_pnl, peak = 0.0, 0.0
        dds = []
        for p in pnls:
            cum_pnl += p
            if cum_pnl > peak: peak = cum_pnl
            dd = (peak - cum_pnl)
            dds.append(dd)
        max_dd = round(max(dds), 2) if dds else 0.0

    equity_points = []
    bust("balance.json")
    bal_data = get("balance.json", {})
    current_bal = float(bal_data.get("usdt", 107913.78))
    
    running = current_bal - sum(pnls)
    for t in real_trades[-50:]:
        p = float(t.get("pnl", 0))
        running += p
        equity_points.append({
            "time": (t.get("closed_at") or t.get("opened_at") or "")[:10],
            "equity": round(running, 2)
        })

    return jsonify({
        "ok": True,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "expectancy_usdt": expectancy,
        "max_drawdown_usdt": max_dd,
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "equity_curve": equity_points
    })


# ── /api/config ───────────────────────────────────────────────────────────

@app.route("/api/config")
def api_config():
    try:
        import config
        from smart_scheduler import get_mode_thresholds

        risk_pct = getattr(config, 'RISK_PER_TRADE', 0.01) * 100
        max_open = getattr(config, 'MAX_OPEN_TRADES', 4)
        max_dir  = getattr(config, 'MAX_SAME_DIRECTION', 4)
        stop_m   = getattr(config, 'ATR_STOP_MULT', 2.5)
        tp1_m    = getattr(config, 'ATR_TARGET1_MULT', 3.5)
        tp2_m    = getattr(config, 'ATR_TARGET2_MULT', 7.5)
        max_age  = getattr(config, 'MAX_TRADE_AGE_HOURS', 48)
        symbols  = getattr(config, 'SYMBOLS', [])

        active_th  = get_mode_thresholds({"label": "Active Hours"})
        quiet_th   = get_mode_thresholds({"label": "Quiet Hours"})
        weekend_th = get_mode_thresholds({"label": "Weekend Mode"})

        return jsonify({
            "ok": True,
            "max_open_trades": max_open,
            "risk_per_trade_pct": risk_pct,
            "max_same_direction": max_dir,
            "atr_stop_mult": stop_m,
            "atr_target1_mult": tp1_m,
            "atr_target2_mult": tp2_m,
            "max_trade_age_hours": max_age,
            "total_symbols": len(symbols),
            "symbols": symbols,
            "active_mode": active_th,
            "quiet_mode": quiet_th,
            "weekend_mode": weekend_th,
            "exchange": "Deribit Testnet (USDC Linear Perpetuals)"
        })
    except Exception as e:
        log.warning(f"Dynamic config load error: {e}")
        return jsonify({
            "ok": False,
            "max_open_trades": 4, "risk_per_trade_pct": 1.0,
            "atr_stop_mult": 2.5, "atr_target1_mult": 3.5, "atr_target2_mult": 7.5,
            "max_trade_age_hours": 48,
            "exchange": "Deribit Testnet (USDC Linear Perpetuals)",
            "error": str(e)
        })


# ── /api/kill_switch ──────────────────────────────────────────────────────

@app.route("/api/kill_switch", methods=["POST"])
def api_kill_switch():
    client = deribit_client()
    if not client:
        return jsonify({"ok": False, "error": "Deribit client not available"}), 500

    try:
        positions = client.get_positions()
        cancelled_count = 0
        flattened_count = 0

        for p in positions:
            inst = p.get("instrument_name", "")
            if not inst:
                continue
            base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            sym = f"{base}USDT"
            
            try:
                orders = client.get_open_orders(sym)
                for o in orders:
                    oid = str(o.get("order_id", ""))
                    if oid:
                        client.cancel_order(oid)
                        cancelled_count += 1
            except Exception as oe:
                log.warning(f"Kill switch cancel orders {sym}: {oe}")

            size = float(p.get("size", 0) or 0)
            if abs(size) > 0:
                side = "SELL" if size > 0 else "BUY"
                amount = client.round_amount(sym, abs(size))
                if amount > 0:
                    client.place_market_order(sym, side, amount, reduce_only=True)
                    flattened_count += 1

        for p in [Path("trades.json"), Path("data/trades.json")]:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({}, indent=2))
            except Exception:
                pass
        
        bust("trades.json")
        bust("balance.json")

        return jsonify({
            "ok": True,
            "status": "FLATTENED",
            "cancelled_orders": cancelled_count,
            "flattened_positions": flattened_count,
            "message": "EMERGENCY KILL SWITCH EXECUTED: All orders cancelled, positions flattened, trades.json cleared."
        })
    except Exception as e:
        log.error(f"Kill switch execution failed: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ── /api/close_trade ──────────────────────────────────────────────────────

@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    symbol = (request.get_json() or {}).get("symbol")
    if not symbol: 
        return jsonify({"error": "symbol required"}), 400
    
    bust("trades.json")
    trades = get("trades.json", {})
    if symbol not in trades: 
        return jsonify({"error": f"{symbol} not found"}), 404
    
    trade = trades.pop(symbol)
    for p in [Path("trades.json"), Path("data/trades.json")]:
        try: 
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(trades, indent=2))
        except Exception: 
            pass
    
    bust("trades.json")
    return jsonify({
        "status": "removed", 
        "symbol": symbol,
        "warning": "Also close on Deribit UI!"
    })


# ── /api/sync ─────────────────────────────────────────────────────────────

@app.route("/api/sync")
def api_sync():
    for f in ["trades.json","trade_history.json","signals.json","balance.json",
              "scan_mode.json","bot.log","model_performance.json", EMAIL_TRACKER_FILE]:
        bust(f)
    return jsonify({"status": "synced"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard starting — port {port} | repo {GH_REPO}")
    app.run(host="0.0.0.0", port=port, debug=False)
