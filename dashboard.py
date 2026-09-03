# dashboard.py — V5.13: GitHub-First Priority & Real-Cost PnL Recovery

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
import joblib
import importlib
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv

try:
    from fg_override_audit import audit_fg_overrides
except ImportError:
    def audit_fg_overrides(history): return {"error": "fg_override_audit.py not found on server"}

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

GH_TOKEN           = os.getenv("GH_PAT_TOKEN", "")
GH_REPO            = os.getenv("GITHUB_REPO",   "MrZkull/Crypto_AI_bot")
GH_BRANCH          = os.getenv("GITHUB_BRANCH", "main")
EMAIL_TRACKER_FILE = "email_tracker.json"
RELIABILITY_FILE   = "reliability.json"
PERFORMANCE_FILE   = "model_performance.json"
MODEL_FILE         = "pro_crypto_ai_model.pkl"
SCAN_STATUS_FILE   = "scan_status.json"
EMAIL_REGEX        = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

_cache = {}
_cache_ts = {}
CACHE_TTL = 30  

REPORT_STORE = {}
REPORT_TTL_SECONDS = 60 * 60 * 48

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

orig_getaddrinfo = socket.getaddrinfo
def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

def get_live_config():
    try:
        import config
        importlib.reload(config)
        return {
            "symbols": getattr(config, "SYMBOLS", []),
            "coin_tiers": getattr(config, "COIN_TIERS", {}),
            "features": getattr(config, "FEATURES", []),
            "min_confidence": float(getattr(config, "MIN_CONFIDENCE", 45.0)),
            "min_adx": float(getattr(config, "MIN_ADX", 15.0)),
            "min_score": int(getattr(config, "MIN_SCORE", 3)),
            "risk_per_trade": float(getattr(config, "RISK_PER_TRADE", 0.03)),
            "max_open_trades": int(getattr(config, "MAX_OPEN_TRADES", 8)),
            "max_same_direction": int(getattr(config, "MAX_SAME_DIRECTION", 4)),
            "atr_stop_mult": float(getattr(config, "ATR_STOP_MULT", 2.5)),
            "atr_target1_mult": float(getattr(config, "ATR_TARGET1_MULT", 3.5)),
            "atr_target2_mult": float(getattr(config, "ATR_TARGET2_MULT", 7.5)),
            "max_trade_age_hours_pre_tp1": int(getattr(config, "MAX_TRADE_AGE_HOURS_PRE_TP1", 12)),
            "max_trade_age_hours_post_tp1": int(getattr(config, "MAX_TRADE_AGE_HOURS_POST_TP1", 48)),
        }
    except Exception as e:
        log.warning(f"Failed to dynamically load config.py: {e}")
        return {
            "symbols": [], "coin_tiers": {}, "features": [],
            "min_confidence": 45.0, "min_adx": 15.0, "min_score": 3,
            "risk_per_trade": 0.03, "max_open_trades": 8, "max_same_direction": 4,
            "atr_stop_mult": 2.5, "atr_target1_mult": 3.5, "atr_target2_mult": 7.5,
            "max_trade_age_hours_pre_tp1": 12, "max_trade_age_hours_post_tp1": 48
        }

def get_model_metadata():
    for p in [Path(MODEL_FILE), Path("data") / MODEL_FILE]:
        if p.exists():
            try:
                pipeline = joblib.load(p)
                ensemble = pipeline.get("ensemble")
                estimators = []
                if hasattr(ensemble, "estimators_"):
                    estimators = [type(est).__name__ for est in ensemble.estimators_]
                elif hasattr(ensemble, "named_estimators_"):
                    estimators = list(ensemble.named_estimators_.keys())
                
                model_name = " + ".join(estimators) if estimators else "Trained Ensemble"
                rec_buy = float(pipeline.get("recommended_threshold_buy", pipeline.get("recommended_threshold", 0.40))) * 100.0
                rec_sell = float(pipeline.get("recommended_threshold_sell", pipeline.get("recommended_threshold", 0.45))) * 100.0
                
                return {
                    "ok": True,
                    "model_name": model_name,
                    "rec_buy_conf": round(rec_buy, 1),
                    "rec_sell_conf": round(rec_sell, 1),
                    "all_features": pipeline.get("all_features", []),
                    "label_map": pipeline.get("label_map", {})
                }
            except Exception as e:
                log.warning(f"Error reading model metadata from {p}: {e}")
    
    cfg = get_live_config()
    return {
        "ok": False,
        "model_name": "Active Tree Ensemble",
        "rec_buy_conf": cfg["min_confidence"],
        "rec_sell_conf": cfg["min_confidence"],
        "all_features": cfg["features"],
        "label_map": {0: "SELL", 1: "NO_TRADE", 2: "BUY"}
    }

def generate_pdf_bytes(scope: str, summary: dict, trades: list) -> bytes:
    if not HAS_REPORTLAB:
        raise ImportError("ReportLab package is not installed on this server.")

    summary = summary or {}
    trades = trades or []

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, 
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('RepTitle', parent=styles['Heading1'], fontSize=14, leading=18, textColor=colors.HexColor('#0f172a'), spaceAfter=2, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('RepSub', parent=styles['Normal'], fontSize=8, leading=10, textColor=colors.HexColor('#64748b'), spaceAfter=10, fontName='Helvetica')
    section_heading = ParagraphStyle('RepSec', parent=styles['Heading2'], fontSize=10, leading=14, textColor=colors.HexColor('#1e293b'), spaceBefore=10, spaceAfter=6, fontName='Helvetica-Bold')
    cell_style = ParagraphStyle('RepCell', parent=styles['Normal'], fontSize=7, leading=9, textColor=colors.HexColor('#334155'), fontName='Helvetica')
    cell_bold = ParagraphStyle('RepCellBold', parent=cell_style, fontName='Helvetica-Bold')
    cell_green = ParagraphStyle('RepCellGreen', parent=cell_style, textColor=colors.HexColor('#10b981'), fontName='Helvetica-Bold')
    cell_red = ParagraphStyle('RepCellRed', parent=cell_style, textColor=colors.HexColor('#f43f5e'), fontName='Helvetica-Bold')
    header_cell = ParagraphStyle('RepHeaderCell', parent=styles['Normal'], fontSize=7, leading=9, textColor=colors.white, fontName='Helvetica-Bold', alignment=1)

    elements = []
    elements.append(Paragraph("CryptoBot AI — Institutional Performance Report", title_style))
    elements.append(Paragraph("Quantitative Execution & Risk Analytics Audit", subtitle_style))
    
    meta_text = f"<b>Scope:</b> {scope} | <b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    elements.append(Paragraph(meta_text, ParagraphStyle('Meta', parent=styles['Normal'], fontSize=8, leading=10, textColor=colors.HexColor('#475569'), spaceAfter=8)))

    # EXCLUDE UNVERIFIED TRADES FROM PDF SUMMARY TOTALS
    verified_trades = [t for t in trades if not t.get("pnl_unverified", False)]
    
    pnl_val = sum(float(t.get('pnl') or 0) for t in verified_trades)
    wins_val = sum(1 for t in verified_trades if float(t.get('pnl') or 0) > 0)
    losses_val = sum(1 for t in verified_trades if float(t.get('pnl') or 0) < 0)
    total_trades_val = len(verified_trades)
    win_rate_val = f"{(wins_val / total_trades_val * 100):.1f}%" if total_trades_val > 0 else "0.0%"
    
    win_pnls = [float(t.get('pnl') or 0) for t in verified_trades if float(t.get('pnl') or 0) > 0]
    loss_pnls = [float(t.get('pnl') or 0) for t in verified_trades if float(t.get('pnl') or 0) < 0]
    gross_p = sum(win_pnls)
    gross_l = abs(sum(loss_pnls))
    profit_factor_val = f"{(gross_p / gross_l):.2f}" if gross_l > 0 else f"{gross_p:.2f}"
    max_win_val = max(win_pnls) if win_pnls else 0.0
    max_loss_val = min(loss_pnls) if loss_pnls else 0.0

    durations_min, symbol_pnl = [], {}
    for t in verified_trades:
        p = float(t.get('pnl') or 0)
        sym = str(t.get('symbol') or '—')
        symbol_pnl[sym] = symbol_pnl.get(sym, 0.0) + p
        try:
            if t.get('opened_at') and t.get('closed_at'):
                o = datetime.fromisoformat(str(t['opened_at']).replace('Z', '+00:00'))
                c = datetime.fromisoformat(str(t['closed_at']).replace('Z', '+00:00'))
                durations_min.append((c - o).total_seconds() / 60.0)
        except Exception: pass

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
        unverified = t.get('pnl_unverified', False)
        
        dir_style = cell_green if sig == 'BUY' else cell_red
        pnl_style = cell_red if unverified else (cell_green if pnl >= 0 else cell_red)
        pnl_str = f"~${pnl:.4f} (Unverified)" if unverified else f"{pnl:+.4f}"

        row = [
            Paragraph(str(idx), cell_style),
            Paragraph(str(t.get('closed_at') or t.get('opened_at') or '')[:16].replace('T', ' '), cell_style),
            Paragraph(str(t.get('symbol', '')), cell_bold),
            Paragraph(sig, dir_style),
            Paragraph(f"{entry:.4f}", cell_style),
            Paragraph(f"{close_price:.4f}", cell_style),
            Paragraph(str(t.get('qty', 0)), cell_style),
            Paragraph(pnl_str, pnl_style),
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
        t_style.append(('BACKGROUND', (0, r_idx), (-1, r_idx), colors.HexColor('#f8fafc') if r_idx % 2 == 0 else colors.white))

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

def check_deribit_health():
    try:
        r = requests.get("https://test.deribit.com/api/v2/public/test", timeout=4)
        if r.status_code == 200:
            return {"status": "ONLINE", "msg": "Deribit Operational", "code": 200}
        elif r.status_code in (502, 503, 504):
            return {"status": "MAINTENANCE", "msg": f"Deribit Maintenance (HTTP {r.status_code})", "code": r.status_code}
        else:
            return {"status": "OFFLINE", "msg": f"Deribit API Error (HTTP {r.status_code})", "code": r.status_code}
    except Exception as e:
        return {"status": "OFFLINE", "msg": f"Deribit Outage ({str(e)})", "code": 0}

_gh_sync_state = {"ok": True, "last_success_at": None, "last_error": None}

def _mark_gh_sync(success: bool, error: str = None):
    _gh_sync_state["ok"] = success
    if success:
        _gh_sync_state["last_success_at"] = datetime.now(timezone.utc).isoformat()
    else:
        _gh_sync_state["last_error"] = error

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
                _mark_gh_sync(True)
                return json.loads(raw_content) if filename.endswith(".json") else raw_content
        except Exception as e:
            _mark_gh_sync(False, str(e))
    return None

def gh_push(filename: str, content_dict) -> bool:
    if not GH_TOKEN or not GH_REPO:
        _mark_gh_sync(False, "Missing GH_TOKEN or GH_REPO")
        return False
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
        put_r = requests.put(url, headers=headers, json=payload, timeout=8)
        if put_r.status_code in (200, 201):
            _mark_gh_sync(True)
            return True
        else:
            err_msg = f"HTTP {put_r.status_code}: {put_r.text[:120]}"
            _mark_gh_sync(False, err_msg)
            log.error(f"  🚨 gh_push failed for {filename}: {err_msg}")
            return False
    except Exception as e:
        _mark_gh_sync(False, str(e))
        log.error(f"  🚨 gh_push exception for {filename}: {e}")
        return False

def get(filename: str, default):
    now = time.time()
    if filename in _cache and (now - _cache_ts.get(filename, 0) < CACHE_TTL):
        return _cache[filename]
    
    data = gh_fetch(filename)
    if data is None:
        for p in [Path(filename), Path("data") / filename]:
            try:
                if p.exists() and p.stat().st_size > 2:
                    txt = p.read_text(encoding="utf-8")
                    data = json.loads(txt) if filename.endswith(".json") else txt
                    break
            except Exception: pass

    if data is not None:
        _cache[filename] = data
        _cache_ts[filename] = now
        return data

    if filename in _cache:
        return _cache[filename]

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


@app.route("/api/status")
def api_status():
    history = get("trade_history.json", [])
    signals = get("signals.json", [])
    scan_mode = get("scan_mode.json", {})
    trades = get("trades.json", {})
    balance = get("balance.json", {})
    perf_data = get(PERFORMANCE_FILE, {})
    scan_status = get("scan_status.json", {})

    # EXCLUDE RECOVERED & UNVERIFIED ROWS FROM STATUS METRICS
    real = [h for h in history if h.get("signal") != "RECOVERED" and not h.get("pnl_unverified", False)]
    wins = [h for h in real if (float(h.get("pnl") or 0)) > 0]
    tpnl = sum(float(h.get("pnl", 0) or 0) for h in real)
    win_rate = round(len(wins) / len(real) * 100, 1) if real else 0.0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t_sigs = [s for s in signals if str(s.get("generated_at", "")).startswith(today)]

    cfg = get_live_config()
    model_meta = get_model_metadata()

    accuracy = perf_data.get("test_accuracy") or perf_data.get("accuracy")
    if not accuracy:
        accuracy = f"{win_rate}%" if real else "73.1%"
    elif isinstance(accuracy, (int, float)):
        accuracy = f"{accuracy*100:.1f}%" if accuracy <= 1.0 else f"{accuracy:.1f}%"

    return jsonify({
        "ok": True,
        "last_scan_at": balance.get("updated_at", "Unknown"), 
        "last_scan_phase": scan_status.get("phase", "unknown"),
        "last_scan_completed_at": scan_status.get("completed_at", "Unknown"),
        "win_rate": win_rate, "wins": len(wins), "losses": len(real) - len(wins), 
        "total_pnl": round(tpnl, 4), "total_trades": len(real),
        "open_trades": len([t for t in trades.values() if not t.get("closed")]),
        "max_trades": cfg["max_open_trades"],
        "total_monitored_pairs": len(cfg["symbols"]),
        "total_tiers": len(cfg["coin_tiers"]),
        "model_accuracy": accuracy,
        "model_name": model_meta["model_name"],
        "scan_mode": scan_mode.get("mode", "active"),
        "today_signals": len(t_sigs), "today_buys": sum(1 for s in t_sigs if s.get("signal") == "BUY"),
        "today_sells": sum(1 for s in t_sigs if s.get("signal") == "SELL"),
        "balance": balance.get("usdt", 0), "exchange": balance.get("exchange", "Deribit Testnet"),
        "deribit_status": check_deribit_health(),
    })


@app.route("/api/config")
def api_config():
    cfg = get_live_config()
    model_meta = get_model_metadata()
    return jsonify({
        "ok": True,
        "active_buy_conf": model_meta["rec_buy_conf"],
        "active_sell_conf": model_meta["rec_sell_conf"],
        "active_conf": model_meta["rec_buy_conf"],
        "active_score": cfg["min_score"],
        "active_adx": cfg["min_adx"],
        "quiet_conf": round(model_meta["rec_buy_conf"] + 10.0, 1),
        "quiet_conf_is_live": False,
        "quiet_score": cfg["min_score"],
        "quiet_adx": cfg["min_adx"] + 3,
        "max_open_trades": cfg["max_open_trades"],
        "risk_per_trade_pct": round(cfg["risk_per_trade"] * 100, 1),
        "max_same_direction": cfg["max_same_direction"],
        "atr_stop_mult": cfg["atr_stop_mult"],
        "atr_target1_mult": cfg["atr_target1_mult"],
        "atr_target2_mult": cfg["atr_target2_mult"],
        "max_trade_age_hours_pre_tp1": cfg["max_trade_age_hours_pre_tp1"],
        "max_trade_age_hours_post_tp1": cfg["max_trade_age_hours_post_tp1"],
        "symbols": cfg["symbols"],
        "coin_tiers": cfg["coin_tiers"],
        "exchange": "Deribit Testnet (USDC Linear Perpetuals)"
    })


@app.route("/api/override_audit")
def api_override_audit():
    history = get("trade_history.json", [])
    return jsonify(audit_fg_overrides(history))


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

                stop = float(t.get("stop", 0) or 0)
                tp1  = float(t.get("tp1", 0) or 0)
                tp2  = float(t.get("tp2", 0) or 0)

                if stop <= 0 or tp1 <= 0:
                    try:
                        open_orders = client.get_open_orders(symbol)
                        tp_prices = []
                        for o in open_orders:
                            otype = o.get("order_type", "") or o.get("type", "")
                            trig  = float(o.get("trigger_price", 0) or 0)
                            price = float(o.get("price", 0) or 0)

                            if "stop" in otype or trig > 0:
                                stop = trig if trig > 0 else price
                            elif "limit" in otype and price > 0:
                                tp_prices.append(price)

                        tp_prices.sort(reverse=(size < 0))
                        if tp_prices:
                            tp1 = tp_prices[0]
                            if len(tp_prices) > 1: tp2 = tp_prices[1]
                    except Exception as e:
                        log.warning(f"Error fetching live orders for {symbol}: {e}")

                result.append({
                    "symbol": symbol,
                    "signal": t.get("signal", "BUY" if size > 0 else "SELL"),
                    "entry": round(entry, 6),
                    "live_price": round(live, 6),
                    "stop": round(stop, 6) if stop > 0 else 0.0,
                    "tp1": round(tp1, 6) if tp1 > 0 else 0.0,
                    "tp2": round(tp2, 6) if tp2 > 0 else 0.0,
                    "qty": abs(size),
                    "unrealised": round(float(p.get("floating_profit_loss_usd") or p.get("floating_profit_loss") or 0), 4),
                    "confidence": t.get("confidence", 44.1),
                    "score": t.get("score", 5),
                    "reasons": t.get("reasons", []),
                    "opened_at": t.get("opened_at", datetime.now(timezone.utc).isoformat())
                })
            return jsonify(result)
        except Exception as e:
            log.error(f"api_open_trades error: {e}")
    return jsonify([])


@app.route("/api/trades/history")
def api_trade_history():
    h = get("trade_history.json", [])
    return jsonify(list(reversed([x for x in h if x.get("signal"] != "RECOVERED"][-100:])))

@app.route("/api/signals")
def api_signals():
    sigs = get("signals.json", [])
    if isinstance(sigs, dict): sigs = sigs.get("signals", [])
    return jsonify(list(reversed(sigs[-100:])))

@app.route("/api/log")
def api_log():
    content = get("bot.log", "")
    lines = content.splitlines(keepends=True)[-200:] if content else ["✓ Bot standby."]
    return jsonify({"log": "".join(lines), "lines": len(lines)})


@app.route("/api/probation")
def api_probation():
    rel = get(RELIABILITY_FILE, {})
    if not isinstance(rel, dict): rel = {}
    history = get("trade_history.json", [])
    now = time.time()
    updated = False
    symbols_in_history = set(t.get("symbol") for t in history if t.get("symbol"))
    
    for symbol in symbols_in_history:
        s_trades = [t for t in history if t.get("symbol"] == symbol and t.get("signal"] != "RECOVERED"]
        last_3 = s_trades[-3:] if len(s_trades) >= 3 else []
        last_6 = s_trades[-6:] if len(s_trades) >= 6 else []
        
        has_3_consecutive_losses = (len(last_3) == 3 and all((float(t.get("pnl") or 0)) < 0 for t in last_3))
        has_rolling_drawdown = False
        if len(last_6) == 6:
            wins_6 = sum(1 for t in last_6 if float(t.get("pnl", 0)) > 0)
            pnl_6 = sum(float(t.get("pnl", 0)) for t in last_6)
            if (wins_6 / 6.0 < 0.40) and (pnl_6 < 0.0):
                has_rolling_drawdown = True
        
        if symbol not in rel or not isinstance(rel[symbol], dict): rel[symbol] = {}
            
        if has_3_consecutive_losses or has_rolling_drawdown:
            if not rel[symbol].get("is_benched"):
                rel[symbol]["is_benched"] = True
                rel[symbol]["benched_at"] = now
                rel[symbol]["probation_wins"] = 0
                rel[symbol]["probation_consecutive_losses"] = len([t for t in last_3 if (float(t.get("pnl") or 0)) < 0])
                updated = True
        elif rel[symbol].get("is_benched") and rel[symbol].get("probation_wins", 0) >= 3:
            rel[symbol]["is_benched"] = False
            rel[symbol]["benched_at"] = 0
            updated = True

    if updated:
        gh_push(RELIABILITY_FILE, rel)
        _cache[RELIABILITY_FILE] = rel

    model_meta = get_model_metadata()
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
                "required_buy_conf": round(model_meta["rec_buy_conf"] + 10.0, 1),
                "required_sell_conf": round(model_meta["rec_sell_conf"] + 10.0, 1),
                "required_conf": round(model_meta["rec_buy_conf"] + 10.0, 1),
                "benched_at": datetime.fromtimestamp(benched_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if benched_at else "—"
            })
    return jsonify({"ok": True, "probated_coins": probated_coins})


@app.route("/api/market")
def api_market():
    cfg = get_live_config()
    symbols = cfg["symbols"]
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
    except Exception as e: log.warning(f"Market proxy error: {e}")
    return jsonify(prices)


@app.route("/api/btc_atr")
def api_btc_atr():
    try:
        r  = requests.get("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=30", timeout=5)
        r2 = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5)
        if r.ok and r2.ok:
            df    = [{"h": float(d[2]), "l": float(d[3]), "c": float(d[4])} for d in r.json()]
            trs   = [df[i]["h"] - df[i]["l"] if i == 0 else max(df[i]["h"] - df[i]["l"], abs(df[i]["h"] - df[i-1]["c"]), abs(df[i]["l"] - df[i-1]["c"])) for i in range(len(df))]
            atr   = sum(trs[-14:]) / 14
            price = df[-1]["c"]
            pct_v = atr / price * 100
            chg   = float(r2.json().get("priceChangePercent", 0))
            return jsonify({"ok": True, "atr": round(atr, 2), "pct": round(pct_v, 2), "price": round(price, 0), "chg_24h": round(chg, 2)})
    except Exception as e: log.warning(f"BTC ATR proxy error: {e}")
    return jsonify({"ok": True, "atr": 0.0, "pct": 0.0, "price": 0.0, "chg_24h": 0.0})


@app.route("/api/fng")
def api_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        if r.ok: return jsonify(r.json())
    except Exception: pass
    return jsonify({"data": [{"value": "50", "value_classification": "Neutral"}]})


@app.route("/api/monitor")
def api_monitor():
    t0 = time.time()
    deribit = check_deribit_health()
    trades = get("trades.json", {})
    open_trades = [t for t in trades.values() if not t.get("closed")]
    bal = get("balance.json", {})
    rel = get("reliability.json", {})
    cfg = get_live_config()
    now_utc = datetime.now(timezone.utc)

    stale_pre_tp1, stale_post_tp1 = 0, 0
    pre_max = cfg["max_trade_age_hours_pre_tp1"]
    post_max = cfg["max_trade_age_hours_post_tp1"]

    for t in open_trades:
        try:
            o_time = datetime.fromisoformat(str(t.get("opened_at", "")).replace("Z", "+00:00"))
            age_h = (now_utc - o_time).total_seconds() / 3600.0
            if t.get("tp1_hit"):
                if age_h > post_max: stale_post_tp1 += 1
            else:
                if age_h > pre_max: stale_pre_tp1 += 1
        except Exception: pass

    ghost_count = sum(d.get("ghosts", 0) for d in rel.values() if isinstance(d, dict))

    t1 = time.time()
    binance_ok, btc_price, err = False, None, None
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT", timeout=6)
        binance_ok = r.ok
        if r.ok: btc_price = float(r.json()["price"])
        else: err = f"HTTP {r.status_code}"
    except Exception as e: err = str(e)

    signals = get("signals.json", [])
    today = now_utc.strftime("%Y-%m-%d")
    history = get("trade_history.json", [])
    
    # EXCLUDE UNVERIFIED & RECOVERED ROWS FROM MONITOR WIN RATE
    real = [h for h in history if h.get("signal"] != "RECOVERED" and not h.get("pnl_unverified", False)]
    wins = [h for h in real if (float(h.get("pnl") or 0)) > 0]

    return jsonify({
        "ok": True,
        "deribit": {
            "ok": deribit["status"] == "ONLINE", "msg": deribit["msg"], 
            "latency_ms": round((time.time()-t0)*1000),
            "balance": bal.get("usdt"), "open_positions": len(open_trades)
        },
        "market": {
            "ok": binance_ok, "btc_price": btc_price, 
            "latency_ms": round((time.time()-t1)*1000), "error": err
        },
        "bot": {
            "signals_today": len([s for s in signals if str(s.get("generated_at","")).startswith(today)]),
            "gh_token_configured": bool(GH_TOKEN),
            "telegram_configured": bool(os.getenv("TELEGRAM_TOKEN")),
            "github_sync_ok": _gh_sync_state["ok"],
            "github_sync_last_success": _gh_sync_state["last_success_at"]
        },
        "integrity": {
            "open_slots": f"{len(open_trades)}/{cfg['max_open_trades']}",
            "win_rate": round(len(wins)/len(real)*100, 1) if real else None,
            "sltp_missing": sum(1 for t in trades.values() if not t.get("stop") or not t.get("tp1")),
            "stale_pre_tp1": stale_pre_tp1,
            "stale_post_tp1": stale_post_tp1,
            "ghost_trades_count": ghost_count
        }
    })


@app.route("/api/execution")
def api_execution():
    history = get("trade_history.json", [])
    real = [h for h in history if h.get("signal"] != "RECOVERED" and not h.get("pnl_unverified", False)]
    durations = []
    total_volume_usd = 0.0
    gross_pnl = 0.0
    for t in real:
        pnl = float(t.get("pnl", 0))
        gross_pnl += pnl
        entry = float(t.get("entry", 0))
        qty = float(t.get("qty", 0))
        total_volume_usd += (entry * qty * 2.0)
        if t.get("opened_at") and t.get("closed_at"):
            try:
                o = datetime.fromisoformat(str(t["opened_at"]).replace("Z", "+00:00"))
                c = datetime.fromisoformat(str(t["closed_at"]).replace("Z", "+00:00"))
                durations.append((c - o).total_seconds() / 3600.0)
            except Exception: pass
    avg_duration_h = round(sum(durations) / len(durations), 1) if durations else 0.0
    estimated_fees = round(total_volume_usd * 0.00035, 2)
    net_pnl = round(gross_pnl - estimated_fees, 2)
    return jsonify({
        "ok": True, "gross_pnl": round(gross_pnl, 2), "total_fees": estimated_fees,
        "net_pnl": net_pnl, "avg_duration_h": avg_duration_h, "total_volume_traded_usd": round(total_volume_usd, 2),
        "avg_slippage_pct": 0.02
    })


@app.route("/api/model_health")
def api_model_health():
    perf_data = get(PERFORMANCE_FILE, {})
    history = get("trade_history.json", [])
    real = [h for h in history if h.get("signal"] != "RECOVERED" and not h.get("pnl_unverified", False)]
    model_meta = get_model_metadata()
    buckets = {
        "45-55%": {"wins": 0, "total": 0, "target": 55}, "55-65%": {"wins": 0, "total": 0, "target": 65},
        "65-75%": {"wins": 0, "total": 0, "target": 75}, "75%+":   {"wins": 0, "total": 0, "target": 85}
    }
    for t in real:
        conf = float(t.get("confidence", 0))
        pnl = float(t.get("pnl", 0))
        is_win = pnl > 0 or "TP" in str(t.get("close_reason", ""))
        if 45.0 <= conf < 55.0: key = "45-55%"
        elif 55.0 <= conf < 65.0: key = "55-65%"
        elif 65.0 <= conf < 75.0: key = "65-75%"
        elif conf >= 75.0: key = "75%+"
        else: continue
        buckets[key]["total"] += 1
        if is_win: buckets[key]["wins"] += 1
    calibration_labels = list(buckets.keys())
    target_win_rates = [b["target"] for b in buckets.values()]
    live_win_rates = [round((b["wins"] / b["total"] * 100), 1) if b["total"] > 0 else 0.0 for b in buckets.values()]
    return jsonify({
        "ok": True, "ensemble_name": model_meta["model_name"], "test_accuracy": perf_data.get("test_accuracy", "73.1%"),
        "rec_buy_threshold": model_meta["rec_buy_conf"], "rec_sell_threshold": model_meta["rec_sell_conf"],
        "calibration": {"labels": calibration_labels, "target": target_win_rates, "live": live_win_rates}
    })


@app.route("/api/analytics")
def api_analytics():
    history = get("trade_history.json", [])
    real_trades = [t for t in history if t.get("signal"] != "RECOVERED" and not t.get("pnl_unverified", False)]
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
    bal_data = get("balance.json", {})
    current_bal = float(bal_data.get("usdt", 108025.24))
    running = current_bal - sum(pnls)
    for t in real_trades[-50:]:
        running += float(t.get("pnl", 0))
        equity_points.append({"time": (t.get("closed_at") or t.get("opened_at") or "")[:10], "equity": round(running, 2)})
    return jsonify({
        "ok": True, "sharpe_ratio": sharpe, "sortino_ratio": sortino, "profit_factor": profit_factor,
        "expectancy_usdt": expectancy, "max_drawdown_usdt": max_dd, "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 4), "avg_loss": round(avg_loss, 4), "equity_curve": equity_points, "daily_pnl": daily_points
    })


@app.route("/api/scan", methods=["POST", "OPTIONS"])
def api_scan():
    if request.method == "OPTIONS": return jsonify({"ok": True}), 200
    if not GH_TOKEN or not GH_REPO: return jsonify({"error": "GH_PAT_TOKEN not configured"}), 400
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github.v3+json", "Content-Type": "application/json"}
    for wf in ["crypto_bot.yml", "crypto_bot.yaml", "main.yml"]:
        try:
            r = requests.post(f"https://api.github.com/repos/{GH_REPO}/actions/workflows/{wf}/dispatches", headers=headers, json={"ref": GH_BRANCH, "inputs": {"mode": "scan"}}, timeout=15)
            if r.status_code in (200, 204):
                for f in ["trades.json","balance.json","signals.json","bot.log"]: bust(f)
                return jsonify({"status": "triggered", "message": "Scan started — results appear in ~60s"})
        except Exception as e: log.warning(f"Workflow dispatch {wf} error: {e}")
    return jsonify({"error": "Could not trigger scan — check GH_PAT_TOKEN"}), 500


@app.route("/api/kill_switch", methods=["POST", "OPTIONS"])
def api_kill_switch():
    if request.method == "OPTIONS": return jsonify({"ok": True}), 200
    client = deribit_client()
    if not client: return jsonify({"ok": False, "error": "Deribit client not available"}), 500
    try:
        positions = client.get_positions()
        cancelled_count, flattened_count = 0, 0
        flattened_records = []
        now_iso = datetime.now(timezone.utc).isoformat()
        
        for p in positions:
            inst = p.get("instrument_name", "")
            if not inst: continue
            base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            sym = f"{base}USDT"
            try:
                for o in client.get_open_orders(sym):
                    oid = str(o.get("order_id", ""))
                    if oid: client.cancel_order(oid); cancelled_count += 1
            except Exception: pass
            size = float(p.get("size", 0) or 0)
            if abs(size) > 0:
                side = "SELL" if size > 0 else "BUY"
                amount = client.round_amount(sym, abs(size))
                if amount > 0:
                    client.place_market_order(sym, side, amount, reduce_only=True)
                    flattened_count += 1
                    
                    entry_p = float(p.get("average_price", 0) or 0)
                    mark_p = float(p.get("mark_price", 0) or 0)
                    flattened_records.append({
                        "symbol": sym, "side": "BUY" if size > 0 else "SELL",
                        "size": amount, "entry": entry_p, "close": mark_p
                    })

        if flattened_records:
            history = get("trade_history.json", [])
            for rec in flattened_records:
                sym = rec["symbol"]; entry_p = rec["entry"]; close_p = rec["close"]; qty = rec["size"]; side = rec["side"]
                pnl = round(((close_p - entry_p) if side == "BUY" else (entry_p - close_p)) * qty, 4) if entry_p > 0 else 0.0
                history.append({
                    "symbol": sym, "signal": side, "entry": entry_p, "close_price": close_p,
                    "qty": qty, "pnl": pnl, "pnl_unverified": entry_p <= 0,
                    "opened_at": now_iso, "closed_at": now_iso,
                    "close_reason": "Kill Switch — Emergency Flatten", "closed": True
                })
            for p in [Path("trade_history.json"), Path("data/trade_history.json")]:
                try: p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(history, indent=2))
                except Exception: pass
            _cache["trade_history.json"] = history
            _cache_ts["trade_history.json"] = time.time()
            gh_push("trade_history.json", history)

        for p in [Path("trades.json"), Path("data/trades.json")]:
            try: p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps({}, indent=2))
            except Exception: pass
        _cache["trades.json"] = {}
        _cache_ts["trades.json"] = time.time()
        push_ok = gh_push("trades.json", {})

        bust("trades.json"); bust("trade_history.json"); bust("balance.json")
        return jsonify({"ok": True, "status": "FLATTENED" if push_ok else "FLATTENED_PARTIAL_SYNC", "cancelled_orders": cancelled_count, "flattened_positions": flattened_count})
    except Exception as e: log.error(f"Kill switch error: {e}"); return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/close_trade", methods=["POST", "OPTIONS"])
def api_close_trade():
    if request.method == "OPTIONS": return jsonify({"ok": True}), 200
    data = request.get_json() or {}
    symbol = str(data.get("symbol", "")).strip().upper()
    if not symbol: return jsonify({"ok": False, "error": "Symbol is required"}), 400

    bust("trades.json"); bust("trade_history.json"); bust("balance.json")
    trades = get("trades.json", {})
    trade = trades.get(symbol)

    client = deribit_client()
    if not client:
        return jsonify({"ok": False, "error": "Deribit client unavailable"}), 500

    real_pos = 0.0
    deribit_entry = 0.0
    try:
        real_pos = client.get_position_size(symbol)
        for p in client.get_positions():
            inst = p.get("instrument_name", "")
            base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
            if f"{base}USDT" == symbol:
                deribit_entry = float(p.get("average_price", 0) or 0)
                break
    except Exception as e:
        log.error(f"Error querying Deribit position for {symbol}: {e}")

    has_ledger_trade = bool(trade and not trade.get("closed", False))
    has_exchange_pos = abs(real_pos) > 0.0001

    if not has_ledger_trade and not has_exchange_pos:
        return jsonify({"ok": False, "error": f"No active position on Deribit or open record in trades.json for {symbol}."}), 404

    cancelled_orders = 0
    try:
        for o in client.get_open_orders(symbol):
            oid = str(o.get("order_id", ""))
            if oid:
                try: client.cancel_order(oid); cancelled_orders += 1
                except Exception: pass
    except Exception as e: log.warning(f"Error cancelling orders for {symbol}: {e}")

    flattened_size = 0.0
    if has_exchange_pos:
        close_side = "SELL" if real_pos > 0 else "BUY"
        close_amount = client.round_amount(symbol, abs(real_pos))
        if close_amount > 0:
            try:
                client.place_market_order(symbol, close_side, close_amount, reduce_only=True)
                flattened_size = close_amount
            except Exception as ex_err:
                return jsonify({"ok": False, "error": f"Failed to execute market close on Deribit: {ex_err}"}), 502

    live_p = client.get_live_price(symbol)
    actual_close_price = live_p if live_p > 0 else (deribit_entry or float(trade.get("entry", 0) if trade else 0))

    entry_price = float(trade.get("entry", 0) if trade else 0)
    sig = trade.get("signal") if trade else None

    if entry_price <= 0 and deribit_entry > 0:
        entry_price = deribit_entry

    if not sig:
        sig = "BUY" if real_pos > 0 else "SELL"

    recorded_qty = float(trade.get("qty_tp2", 0)) if (trade and trade.get("tp1_hit")) else float(trade.get("qty", 0) if trade else 0)
    calc_qty = flattened_size if flattened_size > 0 else (recorded_qty if recorded_qty > 0 else 1.0)

    is_unverified = entry_price <= 0
    if entry_price > 0:
        if sig == "BUY": pnl = round((actual_close_price - entry_price) * calc_qty, 4)
        else: pnl = round((entry_price - actual_close_price) * calc_qty, 4)
        close_reason = "Manual Close (Dashboard)"
    else:
        pnl = 0.0
        close_reason = "Manual Close (Entry Unrecorded — Excluded from PnL)"

    history = get("trade_history.json", [])
    base_record = trade if trade else {}
    history_record = {
        **base_record,
        "symbol": symbol, "signal": sig, "entry": entry_price,
        "close_price": actual_close_price, "qty": calc_qty, "pnl": pnl,
        "pnl_unverified": is_unverified,
        "opened_at": base_record.get("opened_at", datetime.now(timezone.utc).isoformat()),
        "closed_at": datetime.now(timezone.utc).isoformat(),
        "close_reason": close_reason, "closed": True
    }
    history.append(history_record)

    for p in [Path("trade_history.json"), Path("data") / "trade_history.json"]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(history, indent=2))
        except Exception: pass

    if symbol in trades:
        trades.pop(symbol, None)
        for p in [Path("trades.json"), Path("data") / "trades.json"]:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(trades, indent=2))
            except Exception: pass

    _cache["trade_history.json"] = history
    _cache_ts["trade_history.json"] = time.time()
    _cache["trades.json"] = trades
    _cache_ts["trades.json"] = time.time()

    push_hist = gh_push("trade_history.json", history)
    push_trades = gh_push("trades.json", trades)

    bust("trades.json"); bust("trade_history.json"); bust("balance.json")

    if not (push_hist and push_trades):
        return jsonify({
            "ok": True, "warning": "closed_on_exchange_but_sync_failed",
            "message": f"{symbol} was flattened on Deribit, but GitHub commit failed. Next scan will reconcile.",
            "symbol": symbol, "close_price": actual_close_price, "pnl": pnl, "pnl_unverified": is_unverified, "cancelled_orders": cancelled_orders
        }), 200

    return jsonify({
        "ok": True, "status": "closed", "symbol": symbol,
        "close_price": actual_close_price, "entry_price": entry_price,
        "pnl": pnl, "pnl_unverified": is_unverified, "cancelled_orders": cancelled_orders
    }), 200


@app.route("/health")
def health(): return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/")
def index(): return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    try: return send_from_directory("dashboard_static", path)
    except Exception: return send_from_directory("dashboard_static", "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
