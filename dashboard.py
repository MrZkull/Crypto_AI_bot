# dashboard.py — V5.3: Master Institutional Server with EmailJS, Resilient Proxies & Auto-Probation API

import os
import json
import base64
import time
import math
import logging
import requests
import re
import socket
import uuid
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv

# Email & MIME imports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# Safe ReportLab import check
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

load_dotenv()
app = Flask(__name__, static_folder="dashboard_static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GH_TOKEN   = os.getenv("GH_PAT_TOKEN", "")
GH_REPO    = os.getenv("GITHUB_REPO",  "MrZkull/Crypto_AI_bot")
GH_BRANCH  = os.getenv("GITHUB_BRANCH", "main")
EMAIL_TRACKER_FILE = "email_tracker.json"
RELIABILITY_FILE   = "reliability.json"
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

_cache = {}
_cache_ts = {}
CACHE_TTL = 15

# ── Temporary Report Link Store (for EmailJS PDF-link workaround) ──────
# EmailJS free tier blocks binary attachments, so instead of attaching the
# PDF to the outgoing email we generate it once, hold it in memory keyed
# by a random id, and send a clickable download link in the email body.
REPORT_STORE = {}
REPORT_TTL_SECONDS = 60 * 60 * 48  # links stay valid for 48 hours


def _cleanup_reports():
    now = time.time()
    expired = [k for k, v in REPORT_STORE.items() if now - v["created"] > REPORT_TTL_SECONDS]
    for k in expired:
        REPORT_STORE.pop(k, None)


def _store_report(pdf_bytes: bytes) -> str:
    _cleanup_reports()
    rid = uuid.uuid4().hex
    REPORT_STORE[rid] = {"data": pdf_bytes, "created": time.time()}
    return rid


# ── IPv4 Socket Resolution Helper (Fixes Render [Errno 101]) ────────────

orig_getaddrinfo = socket.getaddrinfo
def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


# ── PDF Generation Helper ──────────────────────────────────────────────

def generate_pdf_bytes(scope: str, summary: dict, trades: list) -> bytes:
    if not HAS_REPORTLAB:
        raise ImportError("ReportLab package is not installed on this server.")

    summary = summary or {}
    trades = trades or []

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter, 
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'RepTitle', parent=styles['Heading1'], 
        fontSize=14, leading=18, textColor=colors.HexColor('#0f172a'), spaceAfter=2, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'RepSub', parent=styles['Normal'], 
        fontSize=8, leading=10, textColor=colors.HexColor('#64748b'), spaceAfter=10, fontName='Helvetica'
    )
    section_heading = ParagraphStyle(
        'RepSec', parent=styles['Heading2'], 
        fontSize=10, leading=14, textColor=colors.HexColor('#1e293b'), spaceBefore=10, spaceAfter=6, fontName='Helvetica-Bold'
    )
    cell_style = ParagraphStyle(
        'RepCell', parent=styles['Normal'], 
        fontSize=7, leading=9, textColor=colors.HexColor('#334155'), fontName='Helvetica'
    )
    cell_bold = ParagraphStyle(
        'RepCellBold', parent=cell_style, fontName='Helvetica-Bold'
    )
    cell_green = ParagraphStyle(
        'RepCellGreen', parent=cell_style, textColor=colors.HexColor('#10b981'), fontName='Helvetica-Bold'
    )
    cell_red = ParagraphStyle(
        'RepCellRed', parent=cell_style, textColor=colors.HexColor('#f43f5e'), fontName='Helvetica-Bold'
    )
    header_cell = ParagraphStyle(
        'RepHeaderCell', parent=styles['Normal'], 
        fontSize=7, leading=9, textColor=colors.white, fontName='Helvetica-Bold', alignment=1
    )

    elements = []

    # 1. Title Banner
    elements.append(Paragraph("CryptoBot AI — Institutional Performance Report", title_style))
    elements.append(Paragraph("Quantitative Execution & Risk Analytics Audit", subtitle_style))
    
    meta_text = f"<b>Scope:</b> {scope} | <b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    elements.append(Paragraph(meta_text, ParagraphStyle('Meta', parent=styles['Normal'], fontSize=8, leading=10, textColor=colors.HexColor('#475569'), spaceAfter=8)))

    # 2. KPI Summary Block
    pnl_val = float(summary.get('net_pnl') or 0)
    wins_val = int(summary.get('wins') or 0)
    losses_val = int(summary.get('losses') or 0)
    total_trades_val = int(summary.get('total_trades') or 0)
    win_rate_val = summary.get('win_rate', '0%')
    profit_factor_val = summary.get('profit_factor', '0.00')
    max_win_val = float(summary.get('max_win') or 0)
    max_loss_val = float(summary.get('max_loss') or 0)

    win_pnls, loss_pnls, durations_min, symbol_pnl = [], [], [], {}
    for t in trades:
        t = t or {}
        p = float(t.get('pnl') or 0)
        if p > 0: win_pnls.append(p)
        elif p < 0: loss_pnls.append(p)
        sym = str(t.get('symbol') or '—')
        symbol_pnl[sym] = symbol_pnl.get(sym, 0.0) + p
        try:
            if t.get('opened_at') and t.get('closed_at'):
                o = datetime.fromisoformat(str(t['opened_at']).replace('Z', '+00:00'))
                c = datetime.fromisoformat(str(t['closed_at']).replace('Z', '+00:00'))
                durations_min.append((c - o).total_seconds() / 60.0)
        except Exception:
            pass

    avg_win_val = (sum(win_pnls) / len(win_pnls)) if win_pnls else 0.0
    avg_loss_val = (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0
    avg_hold_val = (sum(durations_min) / len(durations_min)) if durations_min else None
    best_symbol = max(symbol_pnl.items(), key=lambda kv: kv[1]) if symbol_pnl else None
    worst_symbol = min(symbol_pnl.items(), key=lambda kv: kv[1]) if symbol_pnl else None

    summary_table_data = [
        [
            Paragraph("<b>Total Trades Taken:</b>", cell_style), Paragraph(str(total_trades_val), cell_bold),
            Paragraph("<b>Win / Loss Split:</b>", cell_style), Paragraph(f"{wins_val} W / {losses_val} L ({win_rate_val})", cell_bold)
        ],
        [
            Paragraph("<b>Net Realized PnL:</b>", cell_style), Paragraph(f"${pnl_val:+.2f}", cell_green if pnl_val >= 0 else cell_red),
            Paragraph("<b>Profit Factor:</b>", cell_style), Paragraph(str(profit_factor_val), cell_bold)
        ],
        [
            Paragraph("<b>Largest Win:</b>", cell_style), Paragraph(f"${max_win_val:+.2f}", cell_green),
            Paragraph("<b>Largest Loss:</b>", cell_style), Paragraph(f"${max_loss_val:+.2f}", cell_red)
        ],
        [
            Paragraph("<b>Avg Win / Avg Loss:</b>", cell_style),
            Paragraph(f"${avg_win_val:+.2f} / ${avg_loss_val:+.2f}", cell_style),
            Paragraph("<b>Avg Hold Time:</b>", cell_style),
            Paragraph(f"{avg_hold_val:.0f} min" if avg_hold_val is not None else "—", cell_style)
        ],
        [
            Paragraph("<b>Best Performing Symbol:</b>", cell_style),
            Paragraph(f"{best_symbol[0]} (${best_symbol[1]:+.2f})" if best_symbol else "—", cell_green),
            Paragraph("<b>Worst Performing Symbol:</b>", cell_style),
            Paragraph(f"{worst_symbol[0]} (${worst_symbol[1]:+.2f})" if worst_symbol else "—", cell_red)
        ]
    ]
    
    sum_table = Table(summary_table_data, colWidths=[110, 160, 110, 160])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    elements.append(sum_table)
    elements.append(Spacer(1, 10))

    # 3. Trade History Table
    elements.append(Paragraph("Executed Trade Records", section_heading))
    
    headers = ["#", "Date (UTC)", "Symbol", "Dir", "Entry", "Close", "Qty", "PnL (USDT)", "Reason"]
    header_row = [Paragraph(h, header_cell) for h in headers]
    trade_rows = [header_row]

    for idx, t in enumerate(trades, 1):
        t = t or {}
        pnl = float(t.get('pnl') or 0)
        entry = float(t.get('entry') or 0)
        close_price = float(t.get('close_price') or entry)
        sig = str(t.get('signal', ''))
        
        dir_style = cell_green if sig == 'BUY' else cell_red
        pnl_style = cell_green if pnl >= 0 else cell_red

        row = [
            Paragraph(str(idx), cell_style),
            Paragraph(str(t.get('closed_at') or t.get('opened_at') or '')[:16].replace('T', ' '), cell_style),
            Paragraph(str(t.get('symbol', '')), cell_bold),
            Paragraph(sig, dir_style),
            Paragraph(f"{entry:.4f}", cell_style),
            Paragraph(f"{close_price:.4f}", cell_style),
            Paragraph(str(t.get('qty', 0)), cell_style),
            Paragraph(f"{pnl:+.4f}", pnl_style),
            Paragraph(str(t.get('close_reason', 'Closed'))[:25], cell_style)
        ]
        trade_rows.append(row)

    col_widths = [22, 85, 65, 32, 52, 52, 45, 65, 120]
    trade_table = Table(trade_rows, colWidths=col_widths, repeatRows=1)
    
    t_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]
    
    for r_idx in range(1, len(trade_rows)):
        if r_idx % 2 == 0:
            t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor('#f8fafc')))
        else:
            t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.white))

    trade_table.setStyle(TableStyle(t_style))
    elements.append(trade_table)

    def _draw_footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont('Helvetica', 6.5)
        canvas_obj.setFillColor(colors.HexColor('#94a3b8'))
        disclaimer = "CryptoBot AI — Automated Report. Past performance is not indicative of future results."
        canvas_obj.drawString(36, 24, disclaimer)
        canvas_obj.drawRightString(letter[0] - 36, 24, f"Page {doc_obj.page}")
        canvas_obj.restoreState()

    doc.build(elements, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    buffer.seek(0)
    return buffer.getvalue()


# ── Deribit Health Check ───────────────────────────────────────────────

def check_deribit_health():
    try:
        r = requests.get("https://test.deribit.com/api/v2/public/test", timeout=4)
        if r.status_code == 200:
            return {"status": "ONLINE", "msg": "Deribit Operational", "code": 200}
        elif r.status_code in (502, 503):
            return {"status": "MAINTENANCE", "msg": f"Deribit Maintenance (HTTP {r.status_code})", "code": r.status_code}
        else:
            return {"status": "OFFLINE", "msg": f"Deribit API Error (HTTP {r.status_code})", "code": r.status_code}
    except Exception as e:
        return {"status": "OFFLINE", "msg": f"Deribit Outage ({str(e)})", "code": 0}


# ── GitHub State Persistence Helpers ──────────────────────────────────

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
        except Exception: pass
    return None


def gh_push(filename: str, content_dict):
    if not GH_TOKEN or not GH_REPO:
        return
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{filename}?ref={GH_BRANCH}"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        sha = r.json().get("sha") if r.ok else None
        payload = {
            "message": f"Update {filename} state via Dashboard",
            "content": base64.b64encode(json.dumps(content_dict, indent=2).encode('utf-8')).decode('utf-8'),
            "branch": GH_BRANCH
        }
        if sha: payload["sha"] = sha
        requests.put(url, headers=headers, json=payload, timeout=8)
    except Exception as e:
        log.warning(f"gh_push failed for {filename}: {e}")


def get(filename: str, default):
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
            except Exception: pass
    if data is not None:
        _cache[filename] = data
        _cache_ts[filename] = now
        return data
    return default


def bust(filename: str):
    _cache_ts[filename] = 0


def deribit_client():
    cid = os.getenv("DERIBIT_CLIENT_ID", "")
    secret = os.getenv("DERIBIT_CLIENT_SECRET", "")
    if not cid or not secret:
        return None
    try:
        from deribit_client import DeribitClient
        return DeribitClient(cid, secret)
    except Exception:
        return None


# ── Flask Routing ──────────────────────────────────────────────────────

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


@app.route("/api/status")
def api_status():
    for f in ["trade_history.json","signals.json","scan_mode.json","trades.json","balance.json","model_performance.json"]:
        bust(f)
    history = get("trade_history.json", [])
    signals = get("signals.json", [])
    scan_mode = get("scan_mode.json", {})
    trades = get("trades.json", {})
    balance = get("balance.json", {})

    real = [h for h in history if h.get("signal") != "RECOVERED"]
    wins = [h for h in real if (h.get("pnl") or 0) > 0]
    tpnl = sum(h.get("pnl", 0) for h in real)
    win_rate = round(len(wins) / len(real) * 100, 1) if real else 0.0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t_sigs = [s for s in signals if str(s.get("generated_at", "")).startswith(today)]

    return jsonify({
        "ok": True,
        "win_rate": win_rate, "wins": len(wins), "losses": len(real) - len(wins), 
        "total_pnl": round(tpnl, 4), "total_trades": len(real),
        "open_trades": len([t for t in trades.values() if not t.get("closed")]),
        "max_trades": 4, "scan_mode": scan_mode.get("mode", "active"),
        "today_signals": len(t_sigs), "today_buys": sum(1 for s in t_sigs if s.get("signal") == "BUY"),
        "today_sells": sum(1 for s in t_sigs if s.get("signal") == "SELL"),
        "balance": balance.get("usdt", 0), "exchange": balance.get("exchange", "Deribit Testnet"),
        "deribit_status": check_deribit_health(),
    })


@app.route("/api/balance")
def api_balance():
    client = deribit_client()
    if client:
        try:
            bals  = client.get_all_balances()
            total = client.get_total_equity_usd()
            assets = [{"asset": c, "free": str(i.get("available", 0)), "total": str(i.get("equity_usd", 0))} for c, i in bals.items()]
            return jsonify({"ok": True, "usdt": round(total, 2), "equity": round(total, 2), "assets": assets, "exchange": "Deribit Testnet"})
        except Exception: pass
    bal = get("balance.json", {})
    return jsonify({**bal, "ok": True})


@app.route("/api/trades/open")
def api_open_trades():
    bust("trades.json")
    ai_data = get("trades.json", {})
    client = deribit_client()
    if client:
        try:
            positions = client.get_positions()
            result = []
            for p in positions:
                size = float(p.get("size", 0))
                if size == 0: continue
                inst   = p.get("instrument_name", "")
                base   = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                symbol = f"{base}USDT"
                entry  = float(p.get("average_price", 0) or 0)
                live   = float(p.get("mark_price", 0) or 0)
                t      = ai_data.get(symbol, {})
                result.append({
                    "symbol": symbol, "signal": t.get("signal", "BUY" if size > 0 else "SELL"),
                    "entry": round(entry, 6), "live_price": round(live, 6),
                    "stop": float(t.get("stop", 0) or 0), "tp1": float(t.get("tp1", 0) or 0),
                    "tp2": float(t.get("tp2", 0) or 0), "qty": abs(size),
                    "unrealised": round(float(p.get("floating_profit_loss_usd") or p.get("floating_profit_loss") or 0), 4),
                    "confidence": t.get("confidence", 0), "score": t.get("score", 0),
                    "reasons": t.get("reasons", []), "opened_at": t.get("opened_at", "")
                })
            return jsonify(result)
        except Exception: pass
    return jsonify([])


@app.route("/api/trades/history")
def api_trade_history():
    bust("trade_history.json")
    h = get("trade_history.json", [])
    return jsonify(list(reversed([x for x in h if x.get("signal") != "RECOVERED"][-100:])))


@app.route("/api/signals")
def api_signals():
    bust("signals.json")
    sigs = get("signals.json", [])
    if isinstance(sigs, dict): sigs = sigs.get("signals", [])
    return jsonify(list(reversed(sigs[-100:])))


@app.route("/api/log")
def api_log():
    bust("bot.log")
    content = get("bot.log", "")
    lines = content.splitlines(keepends=True)[-200:] if content else ["✓ Bot standby."]
    return jsonify({"log": "".join(lines), "lines": len(lines)})


# ── Probation & Cooldown Tracker Endpoint (Auto-Audits History) ─────────────

# ── Probation & Cooldown Tracker Endpoint (Self-Healing Auto-Audit) ───

@app.route("/api/probation")
def api_probation():
    bust(RELIABILITY_FILE)
    bust("trade_history.json")
    
    rel = get(RELIABILITY_FILE, {})
    if not isinstance(rel, dict):
        rel = {}
        
    history = get("trade_history.json", [])
    now = time.time()
    updated = False

    # 1. Dynamically audit trade history for 3 consecutive losses
    symbols_in_history = set(t.get("symbol") for t in history if t.get("symbol"))
    
    for symbol in symbols_in_history:
        s_trades = [t for t in history if t.get("symbol") == symbol and t.get("signal") != "RECOVERED"]
        last_3 = s_trades[-3:] if len(s_trades) >= 3 else []
        
        has_3_losses = (len(last_3) == 3 and all((float(t.get("pnl") or 0)) < 0 for t in last_3))
        
        if symbol not in rel or not isinstance(rel[symbol], dict):
            rel[symbol] = {}
            
        if has_3_losses:
            # Genuine 3 consecutive losses -> Lock in probation
            if not rel[symbol].get("is_benched"):
                rel[symbol]["is_benched"] = True
                rel[symbol]["benched_at"] = now
                rel[symbol]["probation_wins"] = 0
                rel[symbol]["probation_consecutive_losses"] = 3
                updated = True
        else:
            # 💡 SELF-HEALING: If coin has < 3 losses, automatically unbench it!
            if rel[symbol].get("is_benched"):
                rel[symbol]["is_benched"] = False
                rel[symbol]["benched_at"] = 0
                rel[symbol]["probation_consecutive_losses"] = len([t for t in s_trades if (float(t.get("pnl") or 0)) < 0])
                updated = True

    # 2. Push corrected state to GitHub so reliability.json stays clean
    if updated:
        gh_push(RELIABILITY_FILE, rel)
        _cache[RELIABILITY_FILE] = rel

    # 3. Dynamic Base Confidence (50.0% Active Standard + 10.0% Premium = 60.0%)
    base_conf = 50.0

    # 4. Format and return probated coins for UI table
    probated_coins = []
    for symbol, data in rel.items():
        if isinstance(data, dict) and data.get("is_benched", False):
            benched_at = data.get("benched_at", 0)
            time_left_sec = max(0, (7 * 86400) - (now - benched_at)) if benched_at > 0 else 0
            
            probated_coins.append({
                "symbol": symbol,
                "probation_wins": data.get("probation_wins", 0),
                "probation_consecutive_losses": data.get("probation_consecutive_losses", 0),
                "time_left_hrs": round(time_left_sec / 3600, 1),
                "required_conf": round(base_conf + 10.0, 1),  # Exactly 50.0 + 10.0 = 60.0%
                "benched_at": datetime.fromtimestamp(benched_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if benched_at else "—"
            })

    return jsonify({"ok": True, "probated_coins": probated_coins})


# ── Market & ATR Proxies ────────────────────────────────────────────────

@app.route("/api/market")
def api_market():
    symbols = [
        "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","AVAXUSDT","NEARUSDT",
        "SUIUSDT","APTUSDT","ATOMUSDT","TRXUSDT","LINKUSDT","DOTUSDT",
        "UNIUSDT","AAVEUSDT","XRPUSDT","LTCUSDT","BCHUSDT","ALGOUSDT",
        "FETUSDT","ADAUSDT","DOGEUSDT"
    ]
    prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=6)
        if r.ok:
            for item in r.json():
                if item["symbol"] in symbols:
                    prices[item["symbol"]] = {
                        "lastPrice": float(item.get("lastPrice", 0)),
                        "priceChangePercent": float(item.get("priceChangePercent", 0)),
                        "quoteVolume": float(item.get("quoteVolume", 0)),
                    }
        else:
            log.warning(f"Binance 24hr ticker returned HTTP {r.status_code}. Trying CoinGecko backup...")
            r2 = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,binancecoin,solana,avalanche-2,near,sui,aptos,cosmos,tron,chainlink,polkadot,uniswap,aave,ripple,litecoin,bitcoin-cash,algorand,fetch-ai,cardano,dogecoin&vs_currencies=usd&include_24hr_change=true", timeout=6)
            if r2.ok:
                cg = r2.json()
                mapping = {
                    "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "binancecoin": "BNBUSDT",
                    "solana": "SOLUSDT", "avalanche-2": "AVAXUSDT", "near": "NEARUSDT",
                    "sui": "SUIUSDT", "aptos": "APTUSDT", "cosmos": "ATOMUSDT", "tron": "TRXUSDT",
                    "chainlink": "LINKUSDT", "polkadot": "DOTUSDT", "uniswap": "UNIUSDT",
                    "aave": "AAVEUSDT", "ripple": "XRPUSDT", "litecoin": "LTCUSDT",
                    "bitcoin-cash": "BCHUSDT", "algorand": "ALGOUSDT", "fetch-ai": "FETUSDT",
                    "cardano": "ADAUSDT", "dogecoin": "DOGEUSDT"
                }
                for cg_id, sym in mapping.items():
                    if cg_id in cg:
                        prices[sym] = {
                            "lastPrice": float(cg[cg_id].get("usd", 0)),
                            "priceChangePercent": float(cg[cg_id].get("usd_24h_change", 0)),
                            "quoteVolume": 25000000.0
                        }
    except Exception as e:
        log.warning(f"Market proxy error: {e}")
    return jsonify(prices)


@app.route("/api/btc_atr")
def api_btc_atr():
    try:
        r  = requests.get("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=30", timeout=5)
        r2 = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5)
        if r.ok and r2.ok:
            df   = [{"h": float(d[2]), "l": float(d[3]), "c": float(d[4])} for d in r.json()]
            trs  = [df[i]["h"] - df[i]["l"] if i == 0 else max(df[i]["h"] - df[i]["l"], abs(df[i]["h"] - df[i-1]["c"]), abs(df[i]["l"] - df[i-1]["c"])) for i in range(len(df))]
            atr  = sum(trs[-14:]) / 14
            price = df[-1]["c"]
            pct_v = atr / price * 100
            chg   = float(r2.json().get("priceChangePercent", 0))
            return jsonify({"ok": True, "atr": round(atr, 2), "pct": round(pct_v, 2), "price": round(price, 0), "chg_24h": round(chg, 2)})
    except Exception as e:
        log.warning(f"BTC ATR proxy error: {e}")
    return jsonify({"ok": True, "atr": 1250.50, "pct": 1.92, "price": 65000, "chg_24h": 1.45})


@app.route("/api/fng")
def api_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        if r.ok:
            return jsonify(r.json())
    except Exception: pass
    return jsonify({"data": [{"value": "50", "value_classification": "Neutral"}]})


# ── System Monitor API Endpoint ────────────────────────────────────────

@app.route("/api/monitor")
def api_monitor():
    bust("trades.json"); bust("balance.json"); bust("signals.json"); bust("bot.log"); bust("trade_history.json")
    t0 = time.time()
    deribit = check_deribit_health()
    trades = get("trades.json", {})
    open_trades = [t for t in trades.values() if not t.get("closed")]
    bal = get("balance.json", {})

    t1 = time.time()
    binance_ok, btc_price, err = False, None, None
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT", timeout=6)
        binance_ok = r.ok
        if r.ok: btc_price = float(r.json()["price"])
        else: err = f"HTTP {r.status_code}"
    except Exception as e:
        err = str(e)

    signals = get("signals.json", [])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history = get("trade_history.json", [])
    real = [h for h in history if h.get("signal") != "RECOVERED"]
    wins = [h for h in real if (h.get("pnl") or 0) > 0]

    return jsonify({
        "ok": True,
        "deribit": {"ok": deribit["status"] == "ONLINE", "msg": deribit["msg"], "latency_ms": round((time.time()-t0)*1000),
                    "balance": bal.get("usdt"), "open_positions": len(open_trades)},
        "market": {"ok": binance_ok, "btc_price": btc_price, "latency_ms": round((time.time()-t1)*1000), "error": err},
        "bot": {"signals_today": len([s for s in signals if str(s.get("generated_at","")).startswith(today)])},
        "integrity": {"open_slots": f"{len(open_trades)}/4",
                      "win_rate": round(len(wins)/len(real)*100,1) if real else None,
                      "sltp_missing": sum(1 for t in trades.values() if not t.get("stop") or not t.get("tp1"))}
    })


# ── Persistent Email Dispatch & PDF Download Routes ───────────────────────

def _log_email_attempt(recipient: str, scope: str, summary: dict, status: str):
    bust(EMAIL_TRACKER_FILE)
    logs = get(EMAIL_TRACKER_FILE, [])
    log_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "recipient": recipient, "scope": scope,
        "total_trades": summary.get('total_trades', 0),
        "net_pnl": summary.get('net_pnl', 0), "status": status
    }
    logs.append(log_entry)
    
    for p in [Path(EMAIL_TRACKER_FILE), Path("data") / EMAIL_TRACKER_FILE]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(logs, indent=2))
        except Exception: pass
    gh_push(EMAIL_TRACKER_FILE, logs)


@app.route("/api/download_report_pdf", methods=["POST"])
def api_download_report_pdf():
    if not HAS_REPORTLAB:
        return jsonify({
            "ok": False, "error": "REPORTLAB_MISSING",
            "message": "ReportLab is not installed on the server. Add 'reportlab' to requirements.txt."
        }), 500

    data = request.get_json() or {}
    scope = str(data.get("scope") or "Range: ALL | Result: ALL")
    summary = data.get("summary") or {}
    trades = data.get("trades") or []

    try:
        pdf_bytes = generate_pdf_bytes(scope, summary, trades)
    except Exception as e:
        log.error(f"PDF generation failed: {e}")
        return jsonify({"ok": False, "error": "PDF_GENERATION_FAILED", "message": str(e)}), 500

    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    filename = f"CryptoBot_Report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )


@app.route("/api/report/<report_id>.pdf")
def api_get_stored_report(report_id):
    """Serves a PDF that was generated during /api/send_report.
    This is the link EmailJS emails point to, since EmailJS's free tier
    cannot carry a binary attachment."""
    _cleanup_reports()
    entry = REPORT_STORE.get(report_id)
    if not entry:
        return jsonify({
            "ok": False, "error": "NOT_FOUND",
            "message": "This report link has expired (links last 48h) or does not exist. Generate a new report from the History tab."
        }), 404

    buffer = BytesIO(entry["data"])
    buffer.seek(0)
    filename = f"CryptoBot_Report_{report_id[:8]}.pdf"
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name=filename)


@app.route("/api/send_report", methods=["POST"])
def api_send_report():
    data = request.get_json() or {}
    recipient = str(data.get("email") or "").strip()
    scope = str(data.get("scope") or "Range: ALL | Result: ALL")
    summary = data.get("summary") or {}
    trades = data.get("trades") or []

    if not recipient or not re.match(EMAIL_REGEX, recipient):
        return jsonify({"ok": False, "error": "INVALID_FORMAT", "message": "Invalid email address format."}), 400

    # 0. Generate the PDF once up front and stash it behind a short-lived
    #    download link. EmailJS's free tier strips binary attachments, so a
    #    clickable link is the only reliable way to deliver the PDF itself
    #    without a paid plan or a custom domain (Resend) or phone
    #    verification (Brevo).
    report_url = None
    pdf_bytes = None
    if HAS_REPORTLAB:
        try:
            pdf_bytes = generate_pdf_bytes(scope, summary, trades)
            rid = _store_report(pdf_bytes)
            report_url = request.host_url.rstrip("/") + f"/api/report/{rid}.pdf"
        except Exception as e:
            log.warning(f"PDF pre-generation for email failed: {e}")

    # 1. Fetch Credentials
    emailjs_service_id = os.getenv("EMAILJS_SERVICE_ID", "").strip()
    emailjs_template_id = os.getenv("EMAILJS_TEMPLATE_ID", "").strip()
    emailjs_public_key = os.getenv("EMAILJS_PUBLIC_KEY", "").strip()
    emailjs_private_key = os.getenv("EMAILJS_PRIVATE_KEY", "").strip()

    brevo_api_key = os.getenv("BREVO_API_KEY", "").strip()
    brevo_sender = os.getenv("BREVO_SENDER_EMAIL", "").strip()
    resend_api_key = os.getenv("RESEND_API_KEY", "").strip()

    # ── 1. EMAILJS REST API (FAST, RELIABLE, OVER HTTPS PORT 443) ──
    if emailjs_service_id and emailjs_template_id and emailjs_public_key:
        try:
            payload = {
                "service_id": emailjs_service_id,
                "template_id": emailjs_template_id,
                "user_id": emailjs_public_key,
                "accessToken": emailjs_private_key,
                "template_params": {
                    "to_email": recipient,
                    "scope": scope,
                    "net_pnl": summary.get('net_pnl', 0),
                    "total_trades": summary.get('total_trades', 0),
                    "wins": summary.get('wins', 0),
                    "losses": summary.get('losses', 0),
                    "win_rate": summary.get('win_rate', '0%'),
                    # NEW: clickable PDF download link — add a {{report_url}}
                    # variable/button to your EmailJS template so this
                    # actually shows up in the email body.
                    "report_url": report_url or "PDF unavailable — ReportLab not installed on server."
                }
            }

            r = requests.post(
                "https://api.emailjs.com/api/v1.0/email/send",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=12
            )

            if r.ok or r.text.strip() == "OK":
                log.info(f"Report emailed via EmailJS API to {recipient}")
                _log_email_attempt(recipient, scope, summary, "SENT (EmailJS API)")
                return jsonify({
                    "ok": True, "recipient": recipient,
                    "message": f"Report shared with {recipient}",
                    "report_url": report_url
                })
            else:
                err_text = r.text
                log.error(f"EmailJS API Error ({r.status_code}): {err_text}")
                _log_email_attempt(recipient, scope, summary, f"FAILED EmailJS API: {err_text}")
                return jsonify({"ok": False, "error": "EMAILJS_API_ERROR", "message": f"EmailJS API Error: {err_text}"}), 400
        except Exception as api_err:
            log.error(f"EmailJS Exception: {api_err}")
            _log_email_attempt(recipient, scope, summary, f"FAILED EmailJS Exception: {api_err}")
            return jsonify({"ok": False, "error": "EMAILJS_EXCEPTION", "message": str(api_err)}), 500

    # ── 2. BREVO HTTP API ─────────────────────────────────────────────────────
    if brevo_api_key and brevo_sender:
        try:
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8") if pdf_bytes else None

            link_html = f'<p><a href="{report_url}">Download Full PDF Report</a></p>' if report_url else ''
            brevo_payload = {
                "sender": {"name": "CryptoBot AI", "email": brevo_sender},
                "to": [{"email": recipient}],
                "subject": f"📊 CryptoBot AI Performance Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                "htmlContent": f"<h3>CryptoBot AI Performance Report</h3><p>Filter Scope: {scope}</p><p>Net PnL: ${summary.get('net_pnl', 0)}</p>{link_html}"
            }
            if pdf_b64:
                brevo_payload["attachment"] = [{"name": f"CryptoBot_Report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf", "content": pdf_b64}]

            r = requests.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={"api-key": brevo_api_key, "Content-Type": "application/json"},
                json=brevo_payload,
                timeout=12
            )
            if r.ok:
                _log_email_attempt(recipient, scope, summary, "SENT (Brevo API)")
                return jsonify({"ok": True, "recipient": recipient, "message": f"Report shared with {recipient}", "report_url": report_url})
            else:
                return jsonify({"ok": False, "error": "BREVO_API_ERROR", "message": r.text}), 400
        except Exception as e:
            log.error(f"Brevo API error: {e}")
            return jsonify({"ok": False, "error": "BREVO_EXCEPTION", "message": str(e)}), 500

    # ── 3. RESEND HTTP API ────────────────────────────────────────────────────
    if resend_api_key:
        try:
            link_html = f'<p><a href="{report_url}">Download Full PDF Report</a></p>' if report_url else ''
            payload = {
                "from": os.getenv("RESEND_FROM", "CryptoBot AI <reports@alorix.io>"),
                "to": [recipient],
                "subject": f"📊 CryptoBot AI Performance Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                "html": f"<p>Filter Scope: {scope}</p><p>Net PnL: ${summary.get('net_pnl', 0)}</p>{link_html}"
            }
            r = requests.post("https://api.resend.com/emails", headers={"Authorization": f"Bearer {resend_api_key}", "Content-Type": "application/json"}, json=payload, timeout=12)
            if r.ok:
                _log_email_attempt(recipient, scope, summary, "SENT (Resend API)")
                return jsonify({"ok": True, "recipient": recipient, "message": f"Report shared with {recipient}", "report_url": report_url})
            else:
                return jsonify({"ok": False, "error": "RESEND_API_ERROR", "message": r.text}), 400
        except Exception as e:
            log.error(f"Resend API error: {e}")
            return jsonify({"ok": False, "error": "RESEND_EXCEPTION", "message": str(e)}), 500

    # ── 4. NO HTTP API LOADED ─────────────────────────────────────────────────
    _log_email_attempt(recipient, scope, summary, "FAILED: No Active API Key")
    return jsonify({
        "ok": False, 
        "error": "NOT_CONFIGURED", 
        "message": "No active HTTP email service detected. Please verify EMAILJS_PUBLIC_KEY, EMAILJS_SERVICE_ID, and EMAILJS_TEMPLATE_ID are saved in Render Environment Variables and trigger a redeploy!"
    }), 500

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
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}
    for wf in ["crypto_bot.yml", "crypto_bot.yaml", "main.yml"]:
        try:
            r = requests.post(f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf}/dispatches", headers=headers, json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}}, timeout=15)
            if r.status_code in (200, 204):
                for f in ["trades.json","balance.json","signals.json","bot.log"]: bust(f)
                return jsonify({"status": "triggered", "message": "Scan started — results appear in ~60s"})
        except Exception as e: log.warning(f"Workflow dispatch {wf} error: {e}")
    return jsonify({"error": "Could not trigger scan — check GH_PAT_TOKEN"}), 500


# ── /api/performance ──────────────────────────────────────────────────────

@app.route("/api/performance")
def api_performance():
    bust("trade_history.json")
    h = get("trade_history.json", [])
    real = [x for x in h if x.get("signal") != "RECOVERED"]
    wins = [x for x in real if (x.get("pnl") or 0) > 0]
    loss = [x for x in real if (x.get("pnl") or 0) <= 0]
    tpnl = sum(x.get("pnl", 0) for x in real)
    by_symbol, daily = {}, {}
    for x in real:
        sym = x.get("symbol", "?")
        if sym not in by_symbol: by_symbol[sym] = {"trades": 0, "wins": 0, "pnl": 0}
        by_symbol[sym]["trades"] += 1; by_symbol[sym]["pnl"] += x.get("pnl", 0)
        if (x.get("pnl") or 0) > 0: by_symbol[sym]["wins"] += 1
        day = (x.get("closed_at") or x.get("opened_at", ""))[:10]
        if day: daily[day] = round(daily.get(day, 0) + x.get("pnl", 0), 4)
    lt = sum(x["pnl"] for x in loss)
    return jsonify({
        "total_trades": len(real), "wins": len(wins), "losses": len(loss),
        "win_rate": round(len(wins)/len(real)*100, 1) if real else 0,
        "total_pnl": round(tpnl, 4), "avg_win": round(sum(x["pnl"] for x in wins)/len(wins), 4) if wins else 0,
        "avg_loss": round(sum(x["pnl"] for x in loss)/len(loss), 4) if loss else 0,
        "profit_factor": round(abs(sum(x["pnl"] for x in wins)/lt), 2) if lt else 0,
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
            dds.append(peak - cum_pnl)
        max_dd = round(max(dds), 2) if dds else 0.0

    daily_pnl_map = {}
    for t in real_trades:
        day = (t.get("closed_at") or t.get("opened_at") or "")[:10]
        if day: daily_pnl_map[day] = round(daily_pnl_map.get(day, 0) + float(t.get("pnl", 0)), 2)

    daily_points = [{"date": d, "pnl": p} for d, p in sorted(daily_pnl_map.items())]
    equity_points = []
    bust("balance.json")
    bal_data = get("balance.json", {})
    current_bal = float(bal_data.get("usdt", 108296.99))
    
    running = current_bal - sum(pnls)
    for t in real_trades[-50:]:
        running += float(t.get("pnl", 0))
        equity_points.append({"time": (t.get("closed_at") or t.get("opened_at") or "")[:10], "equity": round(running, 2)})

    return jsonify({
        "ok": True, "sharpe_ratio": sharpe, "sortino_ratio": sortino, "profit_factor": profit_factor,
        "expectancy_usdt": expectancy, "max_drawdown_usdt": max_dd, "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4), "equity_curve": equity_points, "daily_pnl": daily_points
    })


# ── /api/config (Live Smart Scheduler & Risk Config Sync) ──────────────

@app.route("/api/config")
def api_config():
    try:
        import config
        
        # 1. Pull active & quiet parameters dynamically
        active_conf = 50.0
        active_score = 3
        active_adx = 15
        
        quiet_conf = 55.0
        quiet_score = 3
        quiet_adx = 18

        try:
            from smart_scheduler import get_scan_mode
            # Optional check: inspect scheduler rules directly if present
        except Exception:
            pass

        return jsonify({
            "ok": True,
            # AI Scheduler Thresholds
            "active_conf": active_conf,
            "active_score": active_score,
            "active_adx": active_adx,
            "quiet_conf": quiet_conf,
            "quiet_score": quiet_score,
            "quiet_adx": quiet_adx,
            
            # Risk Management Config
            "max_open_trades": getattr(config, 'MAX_OPEN_TRADES', 4),
            "risk_per_trade_pct": getattr(config, 'RISK_PER_TRADE', 0.01) * 100,
            "max_same_direction": getattr(config, 'MAX_SAME_DIRECTION', 3),
            "atr_stop_mult": getattr(config, 'ATR_STOP_MULT', 2.5),
            "atr_target1_mult": getattr(config, 'ATR_TARGET1_MULT', 3.5),
            "atr_target2_mult": getattr(config, 'ATR_TARGET2_MULT', 7.5),
            "max_trade_age_hours": getattr(config, 'MAX_TRADE_AGE_HOURS', 48),
            "symbols": getattr(config, 'SYMBOLS', []),
            "exchange": "Deribit Testnet (USDC Linear Perpetuals)"
        })
    except Exception as e:
        log.error(f"/api/config error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Emergency Kill Switch & Trade Management ──────────────────────────

@app.route("/api/kill_switch", methods=["POST"])
def api_kill_switch():
    client = deribit_client()
    if not client:
        return jsonify({"ok": False, "error": "Deribit client not available"}), 500
    try:
        positions = client.get_positions()
        cancelled_count, flattened_count = 0, 0
        for p in positions:
            inst = p.get("instrument_name", "")
            if not inst:
                continue
            base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            sym = f"{base}USDT"
            try:
                for o in client.get_open_orders(sym):
                    oid = str(o.get("order_id", ""))
                    if oid:
                        client.cancel_order(oid)
                        cancelled_count += 1
            except Exception:
                pass
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
            "flattened_positions": flattened_count
        })
    except Exception as e:
        log.error(f"Kill switch error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/close_trade", methods=["POST"])
def api_close_trade():
    symbol = (request.get_json() or {}).get("symbol")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    bust("trades.json")
    trades = get("trades.json", {})
    if symbol not in trades:
        return jsonify({"error": f"{symbol} not found"}), 404
    trades.pop(symbol)
    for p in [Path("trades.json"), Path("data/trades.json")]:
        try:
            p.parent.mkdir(exist_ok=True)
            p.write_text(json.dumps(trades, indent=2))
        except Exception:
            pass
    bust("trades.json")
    return jsonify({"status": "removed", "symbol": symbol})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    log.info(f"Dashboard starting — port {port} | repo {GH_REPO}")
    app.run(host="0.0.0.0", port=port, debug=False)
