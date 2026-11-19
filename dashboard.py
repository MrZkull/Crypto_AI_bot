# dashboard.py — V5.21: Unified State Pipeline, Memory-Safe Telemetry & Execution Mode Integration

import os
import json
import base64
import time
import math
import logging
import hmac
import requests
import re
import socket
import uuid
import importlib
import threading
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from flask import Flask, Response, jsonify, request, send_from_directory, send_file
from dotenv import load_dotenv

orig_getaddrinfo = socket.getaddrinfo
def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = ipv4_only_getaddrinfo

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# All dashboard routes are private except the minimal health endpoint used by
# the hosting platform. Configure these values in the Render service settings.
# The browser uses HTTP Basic Auth; credentials are never shipped to JavaScript.
_AUTH_REALM = "CryptoBot AI Dashboard"
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# This release is a monitoring surface only. No dashboard write control can be
# enabled by a Render environment-variable mistake. A future control plane must
# be a separately reviewed service integrated with the execution coordinator.
DASHBOARD_MUTATIONS_ENABLED = False


def _dashboard_read_only_response(operation: str):
    """Defence-in-depth response for every non-PDF dashboard mutation route."""
    return jsonify({
        "ok": False,
        "error": "dashboard_read_only",
        "operation": str(operation),
        "message": "This dashboard is monitoring-only; exchange and state mutations are disabled.",
    }), 403


def _canonical_origin(value: str, *, allow_path: bool = False):
    """Return a canonical HTTPS origin (or loopback HTTP origin), else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        if not parsed.scheme or not parsed.netloc:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        if not allow_path and (
            parsed.path not in ("", "/") or parsed.query or parsed.fragment
        ):
            return None
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host or scheme not in {"https", "http"}:
            return None
        if scheme != "https" and not (
            scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}
        ):
            return None
        port = parsed.port
        if port is not None and not (1 <= port <= 65535):
            return None
    except ValueError:
        return None

    if port is None or (scheme == "https" and port == 443) or (
        scheme == "http" and port == 80
    ):
        authority = f"[{host}]" if ":" in host else host
    else:
        authority_host = f"[{host}]" if ":" in host else host
        authority = f"{authority_host}:{port}"
    return f"{scheme}://{authority}"


def _dashboard_auth_settings():
    """Load and validate auth configuration on each request (fail closed)."""
    username = os.getenv("DASHBOARD_AUTH_USERNAME", "").strip()
    password = os.getenv("DASHBOARD_AUTH_PASSWORD", "")
    raw_origin = os.getenv("DASHBOARD_PUBLIC_ORIGIN", "").strip()
    origin = _canonical_origin(raw_origin, allow_path=False)
    if not username or len(username) > 128:
        return None
    if len(password) < 20 or len(password) > 1024:
        return None
    if origin is None:
        return None
    return username, password, origin


def _unauthorized_response():
    response = Response("Authentication required.", status=401, mimetype="text/plain")
    response.headers["WWW-Authenticate"] = (
        f'Basic realm="{_AUTH_REALM}", charset="UTF-8"'
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Authorization"
    return response


@app.before_request
def enforce_dashboard_auth():
    # The public health route reports readiness but never returns private state.
    if request.endpoint == "health":
        return None

    settings = _dashboard_auth_settings()
    if settings is None:
        # Missing or weak credentials/origin must never result in an open dashboard.
        return jsonify({
            "ok": False,
            "error": "dashboard_auth_not_configured",
            "message": "Dashboard authentication is not configured securely.",
        }), 503

    expected_user, expected_password, expected_origin = settings
    auth = request.authorization
    supplied_user = getattr(auth, "username", "") or ""
    supplied_password = getattr(auth, "password", "") or ""
    user_ok = hmac.compare_digest(
        supplied_user.encode("utf-8"), expected_user.encode("utf-8")
    )
    password_ok = hmac.compare_digest(
        supplied_password.encode("utf-8"), expected_password.encode("utf-8")
    )
    if not (user_ok and password_ok):
        return _unauthorized_response()

    # Basic Auth alone can be vulnerable to browser-driven cross-site actions.
    # Validate every unsafe method before any route handler can run.
    if request.method in _UNSAFE_METHODS:
        origin_header = request.headers.get("Origin", "")
        if origin_header:
            supplied_origin = _canonical_origin(origin_header, allow_path=False)
        else:
            # Some clients omit Origin on same-origin requests; permit a same-origin
            # Referer only. Requests with neither header fail closed.
            supplied_origin = _canonical_origin(
                request.headers.get("Referer", ""), allow_path=True
            )
        if supplied_origin != expected_origin:
            return jsonify({
                "ok": False,
                "error": "cross_origin_request_rejected",
            }), 403

    # Hard read-only boundary for the entire dashboard, independent of Render
    # configuration. The only POST exceptions generate in-memory PDFs; they do
    # not mutate trading, account, proposal, probation, or cooldown state.
    if (
        request.method in _UNSAFE_METHODS
        and request.endpoint not in {
            "api_download_report_pdf",
            "api_download_paper_report_pdf",
        }
        and not DASHBOARD_MUTATIONS_ENABLED
    ):
        return jsonify({
            "ok": False,
            "error": "dashboard_read_only",
            "message": "This dashboard is monitoring-only; use the separately approved control plane for operations.",
        }), 403

    return None


@app.after_request
def add_dashboard_security_headers(response):
    if request.endpoint != "health":
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Authorization"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
    return response

GH_READ_TOKEN      = os.getenv("GH_READ_TOKEN", "")
GH_REPO            = os.getenv("GITHUB_REPO",   "MrZkull/Crypto_AI_bot")
GH_BRANCH          = os.getenv("GITHUB_BRANCH", "main")
EMAIL_TRACKER_FILE = "email_tracker.json"
RELIABILITY_FILE   = "reliability.json"
COOLDOWN_FILE      = "cooldown.json"
PERFORMANCE_FILE   = "model_performance.json"
SCAN_STATUS_FILE   = "scan_status.json"
PREDICTIONS_FILE   = "predictions.json"
OVERRIDES_FILE     = "adaptation_overrides.json"
DOSSIER_FILE       = "coin_dossier.json"
PROPOSALS_FILE     = "adaptation_proposals.json"
EMAIL_REGEX        = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

_cache = {}
_cache_ts = {}
CACHE_TTL = 25

_DERIBIT_CLIENT = None
_DERIBIT_CLIENT_KEY = None
_DERIBIT_CLIENT_LOCK = threading.Lock()
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

def get_live_config():
    try:
        import config
        importlib.reload(config)
        return {
            "symbols": getattr(config, "SYMBOLS", []),
            "coin_tiers": getattr(config, "COIN_TIERS", {}),
            "features": getattr(config, "FEATURES", []),
            "execution_mode": getattr(config, "EXECUTION_MODE", "PREDICT_ONLY"),
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
            "probation_offset": float(getattr(config, "PROBATION_CONFIDENCE_OFFSET", 10.0)),
        }
    except Exception as e:
        log.warning(f"Failed to dynamically load config.py: {e}")
        return {
            "symbols": [], "coin_tiers": {}, "features": [],
            "execution_mode": "PREDICT_ONLY",
            "min_confidence": 45.0, "min_adx": 15.0, "min_score": 3,
            "risk_per_trade": 0.03, "max_open_trades": 8, "max_same_direction": 4,
            "atr_stop_mult": 2.5, "atr_target1_mult": 3.5, "atr_target2_mult": 7.5,
            "max_trade_age_hours_pre_tp1": 12, "max_trade_age_hours_post_tp1": 48,
            "probation_offset": 10.0
        }


def _currency_safe_pnl(trades):
    """Aggregate PnL only when every record has one finite, declared currency."""
    if not trades:
        return 0.0, "USDC"
    currencies = []
    values = []
    for item in trades:
        if not isinstance(item, dict):
            return None, None
        currency = str(item.get("pnl_currency", "")).strip().upper()
        if not currency:
            return None, None
        raw = item.get("pnl")
        if raw is None or isinstance(raw, bool):
            return None, None
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            return None, None
        if not math.isfinite(value):
            return None, None
        currencies.append(currency)
        values.append(value)
    if len(set(currencies)) != 1:
        return None, None
    return round(sum(values), 4), currencies[0]


def get_model_metadata():
    perf = get(PERFORMANCE_FILE, {})
    scan_status = get(SCAN_STATUS_FILE, {})
    cfg = get_live_config()

    if isinstance(perf, dict) and perf.get("test_accuracy"):
        return {
            "ok": True,
            "model_name": perf.get("model_name", "Trained Ensemble (v2.0)"),
            "rec_buy_conf": float(perf.get("recommended_threshold_buy", 40.0)),
            "rec_sell_conf": float(perf.get("recommended_threshold_sell", 45.0)),
            "all_features": perf.get("all_features", cfg["features"]),
            "label_map": {0: "SELL", 1: "NO_TRADE", 2: "BUY"}
        }

    live_buy = scan_status.get("active_buy_conf")
    live_sell = scan_status.get("active_sell_conf")
    base_buy = float(live_buy) if live_buy is not None else cfg["min_confidence"]
    base_sell = float(live_sell) if live_sell is not None else cfg["min_confidence"]

    return {
        "ok": True,
        "model_name": "Trained Ensemble (v2.0)",
        "rec_buy_conf": round(base_buy, 1),
        "rec_sell_conf": round(base_sell, 1),
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

    verified_trades = [t for t in trades if isinstance(t, dict) and not t.get("pnl_unverified", False)]
    pnl_val, pnl_currency = _currency_safe_pnl(verified_trades)
    pnl_valid = pnl_val is not None
    pnl_unit = pnl_currency if pnl_valid else "currency unverified"

    wins_val = sum(1 for t in verified_trades if float(t.get('pnl') or 0) > 0)
    losses_val = sum(1 for t in verified_trades if float(t.get('pnl') or 0) < 0)
    total_trades_val = len(verified_trades)
    win_rate_val = f"{(wins_val / total_trades_val * 100):.1f}%" if total_trades_val > 0 else "0.0%"

    if pnl_valid:
        win_pnls = [float(t.get('pnl') or 0) for t in verified_trades if float(t.get('pnl') or 0) > 0]
        loss_pnls = [float(t.get('pnl') or 0) for t in verified_trades if float(t.get('pnl') or 0) < 0]
        gross_p = sum(win_pnls)
        gross_l = abs(sum(loss_pnls))
        profit_factor_val = f"{(gross_p / gross_l):.2f}" if gross_l > 0 else f"{gross_p:.2f}"
        max_win_val = max(win_pnls) if win_pnls else 0.0
        max_loss_val = min(loss_pnls) if loss_pnls else 0.0
        avg_win_val = (sum(win_pnls) / len(win_pnls)) if win_pnls else 0.0
        avg_loss_val = (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0.0
        currency_pnl_text = f"{pnl_val:+.2f} {pnl_unit}"
        currency_win_text = f"{max_win_val:+.2f} {pnl_unit}"
        currency_loss_text = f"{max_loss_val:+.2f} {pnl_unit}"
        currency_avg_text = f"{avg_win_val:+.2f} / {avg_loss_val:+.2f} {pnl_unit}"
    else:
        profit_factor_val = "N/A — mixed/unverified currency"
        max_win_val = max_loss_val = 0.0
        avg_win_val = avg_loss_val = 0.0
        currency_pnl_text = "N/A — mixed/unverified currency"
        currency_win_text = currency_loss_text = "N/A"
        currency_avg_text = "N/A — mixed/unverified currency"

    durations_min, symbol_pnl = [], {}
    for t in verified_trades:
        if pnl_valid:
            p = float(t.get('pnl') or 0)
            sym = str(t.get('symbol') or '—')
            symbol_pnl[sym] = symbol_pnl.get(sym, 0.0) + p
        try:
            if t.get('opened_at') and t.get('closed_at'):
                o = datetime.fromisoformat(str(t['opened_at']).replace('Z', '+00:00'))
                c = datetime.fromisoformat(str(t['closed_at']).replace('Z', '+00:00'))
                durations_min.append((c - o).total_seconds() / 60.0)
        except Exception: pass

    avg_hold_val = (sum(durations_min) / len(durations_min)) if durations_min else None
    best_symbol = max(symbol_pnl.items(), key=lambda kv: kv[1]) if symbol_pnl else None
    worst_symbol = min(symbol_pnl.items(), key=lambda kv: kv[1]) if symbol_pnl else None

    summary_table_data = [
        [
            Paragraph("<b>Total Trades Taken:</b>", cell_style), Paragraph(str(total_trades_val), cell_bold),
            Paragraph("<b>Win / Loss Split:</b>", cell_style), Paragraph(f"{wins_val} W / {losses_val} L ({win_rate_val})", cell_bold)
        ],
        [
            Paragraph("<b>Net Realized PnL:</b>", cell_style), Paragraph(currency_pnl_text, cell_green if pnl_valid and pnl_val >= 0 else cell_style),
            Paragraph("<b>Profit Factor:</b>", cell_style), Paragraph(str(profit_factor_val), cell_bold)
        ],
        [
            Paragraph("<b>Largest Win:</b>", cell_style), Paragraph(currency_win_text, cell_green if pnl_valid else cell_style),
            Paragraph("<b>Largest Loss:</b>", cell_style), Paragraph(currency_loss_text, cell_red if pnl_valid else cell_style)
        ],
        [
            Paragraph("<b>Avg Win / Avg Loss:</b>", cell_style),
            Paragraph(currency_avg_text, cell_style),
            Paragraph("<b>Avg Hold Time:</b>", cell_style),
            Paragraph(f"{avg_hold_val:.0f} min" if avg_hold_val is not None else "—", cell_style)
        ],
        [
            Paragraph("<b>Best Performing Symbol:</b>", cell_style),
            Paragraph(f"{best_symbol[0]} ({best_symbol[1]:+.2f} {pnl_unit})" if best_symbol else "—", cell_green),
            Paragraph("<b>Worst Performing Symbol:</b>", cell_style),
            Paragraph(f"{worst_symbol[0]} ({worst_symbol[1]:+.2f} {pnl_unit})" if worst_symbol else "—", cell_red)
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
    headers = ["#", "Date (UTC)", "Symbol", "Dir", "Entry", "Close", "Qty", "PnL (currency)", "Reason"]
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
        row_currency = str(t.get("pnl_currency") or "currency unverified").upper()
        pnl_str = f"{pnl:+.4f} {row_currency} (unverified)" if unverified else f"{pnl:+.4f} {row_currency}"

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
        r = requests.get("https://test.deribit.com/api/v2/public/test", timeout=3)
        if r.status_code == 200:
            return {"status": "ONLINE", "msg": "Deribit Operational", "code": 200}
        return {"status": "MAINTENANCE", "msg": f"Deribit Code {r.status_code}", "code": r.status_code}
    except Exception as e:
        return {"status": "OFFLINE", "msg": "Deribit Unreachable", "code": 0}

_gh_sync_state = {"ok": True, "last_success_at": None, "last_error": None}

def gh_fetch(filename: str):
    if not GH_READ_TOKEN or not GH_REPO:
        return None
    headers = {"Authorization": f"token {GH_READ_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    for path in [filename, f"data/{filename}"]:
        try:
            url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}?ref={GH_BRANCH}"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                raw_content = base64.b64decode(r.json()["content"]).decode("utf-8")
                _gh_sync_state["ok"] = True
                _gh_sync_state["last_success_at"] = datetime.now(timezone.utc).isoformat()
                if filename == "bot.log":
                    return "\n".join(raw_content.splitlines()[-400:])
                return json.loads(raw_content) if filename.endswith(".json") else raw_content
            elif r.status_code in (403, 429):
                _gh_sync_state["ok"] = False
                _gh_sync_state["last_error"] = f"GitHub API Rate Limit ({r.status_code})"
                break
        except Exception as e:
            _gh_sync_state["ok"] = False
            _gh_sync_state["last_error"] = str(e)
    return None

def gh_push(filename: str, content_dict, max_retries: int = 2) -> bool:
    """Dashboard release is read-only: refuse all GitHub content writes."""
    log.warning("Dashboard GitHub write refused in monitoring-only mode: %s", filename)
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

    return _cache.get(filename, default)

def bust(filename: str):
    _cache_ts[filename] = 0

def deribit_client():
    '''Return a strictly Testnet client authenticated with dashboard read-only credentials.

    The Render dashboard intentionally does not fall back to generic execution
    credentials. Its exchange key must be separately created with read-only
    permissions in the Deribit Testnet account.
    '''
    global _DERIBIT_CLIENT, _DERIBIT_CLIENT_KEY
    cid = os.getenv("DERIBIT_TESTNET_READONLY_CLIENT_ID", "").strip()
    secret = os.getenv("DERIBIT_TESTNET_READONLY_CLIENT_SECRET", "")
    if not cid or not secret:
        return None

    client_key = (cid, secret, "testnet-readonly")
    with _DERIBIT_CLIENT_LOCK:
        if _DERIBIT_CLIENT is not None and _DERIBIT_CLIENT_KEY == client_key:
            return _DERIBIT_CLIENT
        try:
            from deribit_client import DeribitClient, TESTNET_BASE
            client = DeribitClient(cid, secret, testnet=True, auth_scope="trade:read account:read")
            if getattr(client, "base", None) != TESTNET_BASE:
                raise RuntimeError("Dashboard Deribit client is not pinned to the Testnet host")
            _DERIBIT_CLIENT = client
            _DERIBIT_CLIENT_KEY = client_key
            return client
        except Exception as e:
            log.warning("Deribit Testnet read-only client initialization failed: %s", type(e).__name__)
            return None

def _log_email_attempt(recipient: str, scope: str, summary: dict, status: str):
    bust(EMAIL_TRACKER_FILE)
    logs = get(EMAIL_TRACKER_FILE, [])
    if not isinstance(logs, list): logs = []
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
    _cache[EMAIL_TRACKER_FILE] = logs
    _cache_ts[EMAIL_TRACKER_FILE] = time.time()
    gh_push(EMAIL_TRACKER_FILE, logs)

@app.route("/api/dashboard_state", methods=["GET"])
def api_dashboard_state():
    """Single aggregated state endpoint to eliminate client burst requests."""
    try:
        cfg = get_live_config()
        scan_status = get("scan_status.json", {})
        history = get("trade_history.json", [])
        signals = get("signals.json", [])
        trades = get("trades.json", {})
        balance = get("balance.json", {})
        perf_data = get(PERFORMANCE_FILE, {})
        rel = get(RELIABILITY_FILE, {})
        cd = get(COOLDOWN_FILE, {})
        model_meta = get_model_metadata()

        real_trades = [h for h in history if h.get("signal") != "RECOVERED" and not h.get("pnl_unverified", False)]
        wins = [h for h in real_trades if float(h.get("pnl") or 0) > 0]
        tpnl, pnl_currency = _currency_safe_pnl(real_trades)
        win_rate = round(len(wins) / len(real_trades) * 100, 1) if real_trades else 0.0

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        t_sigs = [s for s in signals if str(s.get("generated_at", "")).startswith(today)]

        accuracy = perf_data.get("test_accuracy") or perf_data.get("accuracy") or f"{win_rate}%"
        if isinstance(accuracy, (int, float)):
            accuracy = f"{accuracy*100:.1f}%" if accuracy <= 1.0 else f"{accuracy:.1f}%"

        open_trades_list = []
        client = deribit_client()
        if client and cfg["execution_mode"] != "PREDICT_ONLY":
            try:
                positions = client.get_positions()
                for p in positions:
                    size = float(p.get("size", 0))
                    if abs(size) <= 0.0005: continue
                    inst = p.get("instrument_name", "")
                    base = inst.split("_")[0] if "_" in inst else inst.split("-")[0]
                    sym = f"{base}USDT"
                    t = trades.get(sym, {})
                    open_trades_list.append({
                        "symbol": sym,
                        "signal": t.get("signal", "BUY" if size > 0 else "SELL"),
                        "entry": round(float(p.get("average_price", 0) or 0), 4),
                        "live_price": round(float(p.get("mark_price", 0) or 0), 4),
                        "stop": float(t.get("stop", 0) or 0),
                        "tp1": float(t.get("tp1", 0) or 0),
                        "tp2": float(t.get("tp2", 0) or 0),
                        "qty": abs(size),
                        "unrealised": round(float(p.get("floating_profit_loss_usd") or 0), 4),
                        "confidence": t.get("confidence", 50.0),
                        "score": t.get("score", 5),
                        "opened_at": t.get("opened_at", datetime.now(timezone.utc).isoformat())
                    })
            except Exception as e:
                log.warning(f"Exchange position fetch failed: {e}")

        now = time.time()
        base_offset = float(scan_status.get("probation_offset", cfg["probation_offset"]))
        probated_coins = []
        for symbol, data in rel.items():
            if isinstance(data, dict) and data.get("is_benched", False):
                b_at = data.get("benched_at", 0)
                t_left = max(0, (7 * 86400) - (now - b_at)) if b_at > 0 else 0
                probated_coins.append({
                    "symbol": symbol,
                    "probation_wins": data.get("probation_wins", 0),
                    "probation_consecutive_losses": data.get("probation_consecutive_losses", 0),
                    "time_left_hrs": round(t_left / 3600, 1),
                    "required_buy_conf": round(model_meta["rec_buy_conf"] + base_offset, 1),
                    "required_sell_conf": round(model_meta["rec_sell_conf"] + base_offset, 1),
                    "required_conf": round(model_meta["rec_buy_conf"] + base_offset, 1)
                })

        active_cds = []
        for symbol, entry in cd.items():
            if isinstance(entry, dict):
                rem = float(entry.get("blocked_until", 0) or 0) - now
                if rem > 0:
                    active_cds.append({
                        "symbol": symbol, "reason": entry.get("reason", "—"),
                        "blocked_at": entry.get("blocked_at", "—"), "time_left_hrs": round(rem / 3600, 2)
                    })

        return jsonify({
            "ok": True,
            "execution_mode": cfg.get("execution_mode", "PREDICT_ONLY"),
            "status": {
                "ok": True,
                "execution_mode": cfg.get("execution_mode", "PREDICT_ONLY"),
                "last_scan_phase": scan_status.get("phase", "idle"),
                "last_scan_completed_at": scan_status.get("completed_at", "Unknown"),
                "win_rate": win_rate, "wins": len(wins), "losses": len(real_trades) - len(wins),
                "total_pnl": tpnl, "pnl_currency": pnl_currency,
                "total_trades": len(real_trades),
                "open_trades": len(open_trades_list),
                "max_trades": cfg["max_open_trades"],
                "total_monitored_pairs": len(cfg["symbols"]),
                "total_tiers": len(cfg["coin_tiers"]),
                "model_accuracy": accuracy,
                "model_name": model_meta["model_name"],
                "today_signals": len(t_sigs),
                "today_buys": sum(1 for s in t_sigs if s.get("signal") == "BUY"),
                "today_sells": sum(1 for s in t_sigs if s.get("signal") == "SELL"),
                "deribit_status": check_deribit_health(),
            },
            "balance": {
                "ok": True,
                "usdt": (
                    round(float(balance.get("account_equity_usd")), 2)
                    if str(balance.get("equity_currency", "")).upper() == "USD"
                    and _dashboard_number(balance.get("account_equity_usd")) is not None
                    else None
                ),
                "equity": (
                    round(float(balance.get("account_equity_usd")), 2)
                    if str(balance.get("equity_currency", "")).upper() == "USD"
                    and _dashboard_number(balance.get("account_equity_usd")) is not None
                    else None
                ),
                "equity_currency": balance.get("equity_currency"),
                "risk_equity": _dashboard_number(balance.get("risk_equity")),
                "risk_equity_currency": balance.get("risk_equity_currency"),
                "assets": balance.get("assets", [])
            },
            "trades": open_trades_list,
            "signals": list(reversed(signals[-50:])) if isinstance(signals, list) else [],
            "config": {
                "ok": True,
                "execution_mode": cfg.get("execution_mode", "PREDICT_ONLY"),
                "active_buy_conf": model_meta["rec_buy_conf"],
                "active_sell_conf": model_meta["rec_sell_conf"],
                "active_score": cfg["min_score"],
                "active_adx": cfg["min_adx"],
                "max_open_trades": cfg["max_open_trades"],
                "risk_per_trade_pct": round(cfg["risk_per_trade"] * 100, 1),
                "max_same_direction": cfg["max_same_direction"],
                "atr_stop_mult": cfg["atr_stop_mult"],
                "atr_target1_mult": cfg["atr_target1_mult"],
                "atr_target2_mult": cfg["atr_target2_mult"],
                "max_trade_age_hours_pre_tp1": cfg["max_trade_age_hours_pre_tp1"],
                "symbols": cfg["symbols"],
                "coin_tiers": cfg["coin_tiers"],
            },
            "probation": {"ok": True, "probated_coins": probated_coins},
            "cooldown": {"ok": True, "cooldowns": active_cds},
            "liveness": {
                "execution_mode": scan_status.get("execution_mode", cfg.get("execution_mode", "PREDICT_ONLY")),
                "scan_ran": scan_status.get("scan_ran", False),
                "symbols_attempted": scan_status.get("symbols_attempted", 0),
                "symbols_scored": scan_status.get("symbols_scored", 0),
                "predictions_saved": scan_status.get("predictions_saved", 0),
                "feature_validation_rejects": scan_status.get("feature_validation_rejects", 0),
                "skip_reason": scan_status.get("skip_reason"),
            }
        })
    except Exception as e:
        log.error(f"Error in api_dashboard_state: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/send_report", methods=["POST", "OPTIONS"])
def api_send_report():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_send_report")

@app.route("/api/report/<report_id>.pdf")
def api_get_stored_report(report_id):
    _cleanup_reports()
    entry = REPORT_STORE.get(report_id)
    if not entry:
        return jsonify({"ok": False, "error": "NOT_FOUND", "message": "Report expired or not found."}), 404
    buffer = BytesIO(entry["data"])
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name=f"CryptoBot_Report_{report_id[:8]}.pdf")

@app.route("/api/email_tracker", methods=["GET"])
def api_email_tracker():
    logs = get(EMAIL_TRACKER_FILE, [])
    if not isinstance(logs, list): logs = []
    return jsonify(list(reversed(logs[-50:])))

@app.route("/api/download_report_pdf", methods=["POST", "OPTIONS"])
def api_download_report_pdf():
    if request.method == "OPTIONS": return jsonify({"ok": True}), 200
    if not HAS_REPORTLAB:
        return jsonify({"ok": False, "error": "REPORTLAB_MISSING", "message": "ReportLab not installed."}), 500

    data = request.get_json(silent=True) or {}
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
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name=f"CryptoBot_Report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.pdf")

@app.route("/api/status")
def api_status():
    history = get("trade_history.json", [])
    signals = get("signals.json", [])
    trades = get("trades.json", {})
    balance = get("balance.json", {})
    perf_data = get(PERFORMANCE_FILE, {})
    scan_status = get("scan_status.json", {})
    cfg = get_live_config()
    model_meta = get_model_metadata()

    real = [h for h in history if h.get("signal") != "RECOVERED" and not h.get("pnl_unverified", False)]
    wins = [h for h in real if float(h.get("pnl") or 0) > 0]
    tpnl, pnl_currency = _currency_safe_pnl(real)
    win_rate = round(len(wins) / len(real) * 100, 1) if real else 0.0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    t_sigs = [s for s in signals if str(s.get("generated_at", "")).startswith(today)]

    accuracy = perf_data.get("test_accuracy") or perf_data.get("accuracy") or f"{win_rate}%"
    if isinstance(accuracy, (int, float)):
        accuracy = f"{accuracy*100:.1f}%" if accuracy <= 1.0 else f"{accuracy:.1f}%"

    return jsonify({
        "ok": True,
        "execution_mode": cfg.get("execution_mode", "PREDICT_ONLY"),
        "last_scan_phase": scan_status.get("phase", "unknown"),
        "last_scan_completed_at": scan_status.get("completed_at", "Unknown"),
        "win_rate": win_rate, "wins": len(wins), "losses": len(real) - len(wins),
        "total_pnl": tpnl, "pnl_currency": pnl_currency,
        "total_trades": len(real),
        "open_trades": len([t for t in trades.values() if not t.get("closed")]),
        "max_trades": cfg["max_open_trades"],
        "total_monitored_pairs": len(cfg["symbols"]),
        "total_tiers": len(cfg["coin_tiers"]),
        "model_accuracy": accuracy,
        "model_name": model_meta["model_name"],
        "today_signals": len(t_sigs),
        "today_buys": sum(1 for s in t_sigs if s.get("signal") == "BUY"),
        "today_sells": sum(1 for s in t_sigs if s.get("signal") == "SELL"),
        "deribit_status": check_deribit_health()
    })

@app.route("/api/config")
def api_config():
    cfg = get_live_config()
    model_meta = get_model_metadata()
    return jsonify({
        "ok": True,
        "execution_mode": cfg.get("execution_mode", "PREDICT_ONLY"),
        "active_buy_conf": model_meta["rec_buy_conf"],
        "active_sell_conf": model_meta["rec_sell_conf"],
        "active_score": cfg["min_score"],
        "active_adx": cfg["min_adx"],
        "max_open_trades": cfg["max_open_trades"],
        "risk_per_trade_pct": round(cfg["risk_per_trade"] * 100, 1),
        "max_same_direction": cfg["max_same_direction"],
        "atr_stop_mult": cfg["atr_stop_mult"],
        "atr_target1_mult": cfg["atr_target1_mult"],
        "atr_target2_mult": cfg["atr_target2_mult"],
        "max_trade_age_hours_pre_tp1": cfg["max_trade_age_hours_pre_tp1"],
        "symbols": cfg["symbols"],
        "coin_tiers": cfg["coin_tiers"]
    })

@app.route("/api/dashboard/mode")
def api_dashboard_mode():
    return jsonify({
        "ok": True,
        "read_only": True,
        "execution_controls_enabled": False,
        "control_plane": "separate_approved_execution_service",
    })


def _dashboard_account_snapshot(client):
    response = client.get_account_summaries()
    if not isinstance(response, dict) or not isinstance(response.get("summaries"), list):
        raise ValueError("Malformed account summaries")

    rows = response["summaries"]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Account summaries are empty or malformed")

    assets = []
    currencies = set()
    margin_models = set()
    usd_totals = []
    usdc_equity = None
    segregated_models = {"segregated_sm", "segregated_pm"}
    cross_models = {"cross_sm", "cross_pm"}
    for row in rows:
        currency = str(row.get("currency", "")).strip().upper()
        if not currency or currency in currencies:
            raise ValueError("Account summaries contain missing or duplicate currencies")
        currencies.add(currency)
        margin_model = str(row.get("margin_model", "")).strip().lower()
        if margin_model not in segregated_models | cross_models:
            raise ValueError(f"Unknown or missing margin_model for {currency}")
        margin_models.add(margin_model)
        native_equity = _dashboard_number(row.get("equity"))
        available = _dashboard_number(row.get("available_funds"))
        if native_equity is None:
            raise ValueError(f"Account equity is missing for {currency}")
        assets.append({
            "asset": currency,
            "available": available,
            "equity": native_equity,
            "unit": currency,
        })
        if currency == "USDC":
            usdc_equity = native_equity
        # Only the explicit cross margin models expose a comparable account-wide
        # USD amount. In segregated models this field may be a per-currency USD
        # conversion and must not be treated as an account aggregate.
        candidate = _dashboard_number(row.get("total_equity_usd"))
        if margin_model in cross_models and candidate is not None:
            usd_totals.append(candidate)

    if len(margin_models) != 1:
        raise ValueError("Account summaries disagree on active margin_model")
    active_margin_model = next(iter(margin_models))
    account_equity_usd = None
    if active_margin_model in cross_models:
        if not usd_totals:
            raise ValueError("Cross-collateral account has no total_equity_usd")
        canonical = usd_totals[0]
        if any(not math.isclose(value, canonical, rel_tol=1e-9, abs_tol=1e-7) for value in usd_totals[1:]):
            raise ValueError("Deribit returned inconsistent account-wide USD equity values")
        account_equity_usd = canonical

    return {
        "assets": assets,
        "account_equity_usd": account_equity_usd,
        "account_equity_native": usdc_equity,
        "account_equity_currency": "USDC" if usdc_equity is not None else None,
        "margin_model": active_margin_model,
        "balance_snapshot_complete": True,
        "balance_currency_count": len(assets),
    }


@app.route("/api/testnet/overview")
def api_testnet_overview():
    """Read-only, host-pinned snapshot for the Deribit Testnet dashboard panel."""
    checked_at = datetime.now(timezone.utc).isoformat()
    client = deribit_client()
    if client is None:
        return jsonify({
            "ok": False, "environment": "TESTNET", "connection": "NOT_CONFIGURED",
            "read_only": True, "checked_at": checked_at,
            "error": "testnet_readonly_credentials_missing",
        }), 503

    try:
        from deribit_client import TESTNET_BASE, SYMBOL_MAP
        instrument_to_symbol = {item["instrument"]: logical for logical, item in SYMBOL_MAP.items()}
        if getattr(client, "base", None) != TESTNET_BASE:
            raise RuntimeError("Deribit client host is not the pinned Testnet endpoint")

        balance_snapshot = _dashboard_account_snapshot(client)
        balances = balance_snapshot["assets"]
        equity = balance_snapshot["account_equity_usd"]
        positions_raw = client.get_positions()  # strict complete snapshot; errors fail this endpoint
        if not isinstance(positions_raw, list):
            raise ValueError("Malformed position snapshot")

        positions = []
        position_instruments = []
        order_query_errors = []
        order_checks = 0
        max_order_checks = 4
        for position in positions_raw:
            instrument = str(position.get("instrument_name", ""))
            if not instrument:
                raise ValueError("Position is missing instrument_name")
            contracts = float(position.get("size", 0) or 0)
            quantity = float(position.get("size_currency", contracts) or 0)
            if abs(contracts) <= 0.0005 and abs(quantity) <= 0.0005:
                continue
            symbol = instrument_to_symbol.get(instrument, instrument)
            direction = str(position.get("direction", "unknown")).lower()
            stop_status = "UNKNOWN"
            open_order_count = None
            if order_checks < max_order_checks:
                order_checks += 1
                try:
                    orders = client.get_open_orders_by_instrument(instrument)
                    open_order_count = len(orders)
                    stop_found = any(
                        isinstance(order, dict)
                        and str(order.get("order_type", order.get("type", ""))).lower() == "stop_market"
                        and str(order.get("trigger", "")).lower() == "mark_price"
                        and order.get("reduce_only") is True
                        and str(order.get("order_state", "open")).lower() in {"open", "untriggered"}
                        and _dashboard_number(order.get("trigger_price")) is not None
                        and _dashboard_number(order.get("trigger_price")) > 0
                        for order in orders
                    )
                    # Presence is not proof that the stop quantity fully covers the position.
                    stop_status = "PRESENT" if stop_found else "MISSING"
                except Exception as order_exc:
                    stop_status = "UNKNOWN"
                    order_query_errors.append({"instrument_name": instrument, "error": type(order_exc).__name__})
            elif order_checks >= max_order_checks:
                stop_status = "NOT_CHECKED_SCOPE_LIMIT"
            position_instruments.append(instrument)
            positions.append({
                "symbol": symbol,
                "instrument_name": instrument,
                "direction": direction,
                "quantity": abs(quantity),
                "contracts": abs(contracts),
                "average_price": _dashboard_number(position.get("average_price")),
                "mark_price": _dashboard_number(position.get("mark_price")),
                "unrealized_pnl_usd": _dashboard_number(position.get("floating_profit_loss_usd")),
                "stop_status": stop_status,
                "open_orders_count": open_order_count,
            })

        journal = get("trade_history.json", [])
        if not isinstance(journal, list):
            journal = []
        fill_instruments = []
        for instrument in position_instruments:
            if instrument not in fill_instruments:
                fill_instruments.append(instrument)
        for record in reversed(journal):
            if not isinstance(record, dict):
                continue
            logical_symbol = str(record.get("symbol", "")).strip().upper()
            mapping = SYMBOL_MAP.get(logical_symbol)
            instrument = mapping.get("instrument") if isinstance(mapping, dict) else None
            if instrument and instrument not in fill_instruments:
                fill_instruments.append(instrument)
            if len(fill_instruments) >= 4:
                break

        fills = []
        fill_errors = []
        for instrument in fill_instruments[:4]:
            symbol = instrument_to_symbol.get(instrument, instrument)
            try:
                rows = client.get_trade_history_by_instrument_name(instrument, count=10, strict=True)
                if not isinstance(rows, list):
                    raise ValueError("Malformed exchange fills response")
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    raw_ts = row.get("timestamp")
                    timestamp_utc = None
                    try:
                        if raw_ts is not None:
                            ts = float(raw_ts)
                            if ts > 10_000_000_000:
                                ts /= 1000.0
                            timestamp_utc = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    except (TypeError, ValueError, OverflowError, OSError):
                        timestamp_utc = None
                    fills.append({
                        "symbol": symbol,
                        "instrument_name": instrument,
                        "trade_id": row.get("trade_id"),
                        "order_id": row.get("order_id"),
                        "direction": row.get("direction"),
                        "amount": _dashboard_number(row.get("amount")),
                        "price": _dashboard_number(row.get("price")),
                        "fee": _dashboard_number(row.get("fee")),
                        "fee_currency": row.get("fee_currency"),
                        "timestamp": raw_ts,
                        "timestamp_utc": timestamp_utc,
                    })
            except Exception as exc:
                fill_errors.append({"instrument_name": instrument, "error": type(exc).__name__})

        fills.sort(key=lambda item: _dashboard_number(item.get("timestamp")) or 0.0, reverse=True)
        assets = balances
        return jsonify({
            "ok": True,
            "environment": "TESTNET",
            "connection": "AUTHENTICATED",
            "verified_host": "test.deribit.com",
            "read_only": True,
            "checked_at": checked_at,
            "account_equity_usd": round(equity, 4) if equity is not None else None,
            "account_equity_native": balance_snapshot["account_equity_native"],
            "account_equity_currency": balance_snapshot["account_equity_currency"],
            "margin_model": balance_snapshot["margin_model"],
            "balance_snapshot_complete": balance_snapshot["balance_snapshot_complete"],
            "balance_currency_count": balance_snapshot["balance_currency_count"],
            "balances": assets,
            "positions": positions,
            "order_query_errors": order_query_errors,
            "order_check_scope_limit": max_order_checks,
            "order_checks_completed": order_checks,
            "fills": fills[:100],
            "fills_scope_symbols": [instrument_to_symbol.get(item, item) for item in fill_instruments[:4]],
            "fills_scope_instruments": fill_instruments[:4],
            "fills_complete_for_scope": not fill_errors,
            "fill_query_errors": fill_errors,
            "bot_journal": {
                "source": "GitHub trade_history.json",
                "record_count": len(journal),
                "recent_records": journal[-10:],
            },
        })
    except Exception as exc:
        log.error("Testnet overview failed closed: %s", type(exc).__name__)
        return jsonify({
            "ok": False, "environment": "TESTNET", "connection": "UNVERIFIED",
            "read_only": True, "checked_at": checked_at,
            "error": "testnet_private_snapshot_failed",
        }), 503


@app.route("/api/balance")
def api_balance():
    client = deribit_client()
    if client is None:
        return jsonify({
            "ok": False,
            "exchange": "Deribit Testnet",
            "source": "unavailable",
            "error": "testnet_readonly_credentials_missing",
        }), 503
    try:
        from deribit_client import TESTNET_BASE
        if getattr(client, "base", None) != TESTNET_BASE:
            raise RuntimeError("Deribit host verification failed")
        snapshot = _dashboard_account_snapshot(client)
        total_usd = snapshot["account_equity_usd"]
        assets = [
            {
                "asset": item["asset"],
                "free": str(item["available"] if item["available"] is not None else "N/A"),
                "total": str(item["equity"]),
                "unit": item["unit"],
            }
            for item in snapshot["assets"]
        ]
        return jsonify({
            "ok": True,
            "usdt": round(total_usd, 2) if total_usd is not None else None,
            "equity": round(total_usd, 2) if total_usd is not None else None,
            "equity_currency": "USD" if total_usd is not None else None,
            "native_equity": snapshot["account_equity_native"],
            "native_equity_currency": snapshot["account_equity_currency"],
            "margin_model": snapshot["margin_model"],
            "assets": assets,
            "exchange": "Deribit Testnet",
            "source": "deribit_private_get_account_summaries",
            "balance_snapshot_complete": snapshot["balance_snapshot_complete"],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        log.warning("Testnet balance query failed: %s", type(exc).__name__)
        return jsonify({
            "ok": False, "exchange": "Deribit Testnet", "source": "unavailable",
            "error": "testnet_balance_query_failed",
        }), 503

@app.route("/api/trades/open")
def api_open_trades():
    ai_data = get("trades.json", {})
    # Preserve legacy metadata when the root ledger is empty.
    if not isinstance(ai_data, dict) or not ai_data:
        legacy_data = get("data/trades.json", {})
        ai_data = legacy_data if isinstance(legacy_data, dict) else {}
    client = deribit_client()
    if client is None:
        return jsonify({"ok": False, "error": "testnet_positions_unavailable",
                        "message": "Exchange positions are not verified; this is not a flat-account confirmation."}), 503
    try:
        positions = client.get_positions()
        from deribit_client import SYMBOL_MAP
        instrument_to_symbol = {item["instrument"]: logical for logical, item in SYMBOL_MAP.items()}
        result = []
        for position in positions:
            contracts = float(position.get("size", 0) or 0)
            quantity = float(position.get("size_currency", contracts) or 0)
            if abs(contracts) <= 0.0005 and abs(quantity) <= 0.0005:
                continue
            instrument = str(position.get("instrument_name", ""))
            symbol = instrument_to_symbol.get(instrument, instrument)
            local = ai_data.get(symbol, {}) if symbol != instrument and isinstance(ai_data, dict) else {}
            direction = str(position.get("direction", "")).lower()
            result.append({
                "symbol": symbol,
                "signal": local.get("signal", "BUY" if direction == "buy" else "SELL"),
                "entry": round(float(position.get("average_price", 0) or 0), 4),
                "live_price": round(float(position.get("mark_price", 0) or 0), 4),
                "stop": float(local.get("stop", 0) or 0),
                "tp1": float(local.get("tp1", 0) or 0),
                "tp2": float(local.get("tp2", 0) or 0),
                "qty": abs(quantity),
                "unrealised": round(float(position.get("floating_profit_loss_usd") or 0), 4),
                "confidence": local.get("confidence"),
                "score": local.get("score"),
                "opened_at": local.get("opened_at"),
                "source": "deribit_testnet_positions",
            })
        return jsonify(result)
    except Exception as exc:
        log.error("api_open_trades exchange snapshot failed: %s", type(exc).__name__)
        return jsonify({"ok": False, "error": "testnet_positions_unavailable",
                        "message": "Exchange positions are not verified; this is not a flat-account confirmation."}), 503

@app.route("/api/trades/heal", methods=["POST", "OPTIONS"])
def api_trades_heal():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_trades_heal")

@app.route("/api/trades/history")
def api_trade_history():
    h = get("trade_history.json", [])
    if not isinstance(h, list): h = []
    return jsonify(list(reversed([x for x in h if x.get("signal") != "RECOVERED"][-100:])))

@app.route("/api/signals")
def api_signals():
    sigs = get("signals.json", [])
    if isinstance(sigs, dict): sigs = sigs.get("signals", [])
    if not isinstance(sigs, list): sigs = []
    return jsonify(sigs[:100])

@app.route("/api/probation")
def api_probation():
    rel = get(RELIABILITY_FILE, {})
    cfg = get_live_config()
    model_meta = get_model_metadata()
    now = time.time()
    probated_coins = []
    for symbol, data in rel.items():
        if isinstance(data, dict) and data.get("is_benched", False):
            b_at = data.get("benched_at", 0)
            t_left = max(0, (7 * 86400) - (now - b_at)) if b_at > 0 else 0
            probated_coins.append({
                "symbol": symbol, "probation_wins": data.get("probation_wins", 0),
                "probation_consecutive_losses": data.get("probation_consecutive_losses", 0),
                "time_left_hrs": round(t_left / 3600, 1),
                "required_buy_conf": round(model_meta["rec_buy_conf"] + 10.0, 1),
                "required_sell_conf": round(model_meta["rec_sell_conf"] + 10.0, 1),
                "required_conf": round(model_meta["rec_buy_conf"] + 10.0, 1)
            })
    return jsonify({"ok": True, "probated_coins": probated_coins})

@app.route("/api/probation/release", methods=["POST", "OPTIONS"])
def api_probation_release():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_probation_release")

@app.route("/api/probation/bench", methods=["POST", "OPTIONS"])
def api_probation_bench():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_probation_bench")

@app.route("/api/cooldown")
def api_cooldown():
    cd = get(COOLDOWN_FILE, {})
    now = time.time()
    active = []
    if isinstance(cd, dict):
        for symbol, entry in cd.items():
            if isinstance(entry, dict):
                rem = float(entry.get("blocked_until", 0) or 0) - now
                if rem > 0:
                    active.append({
                        "symbol": symbol, "reason": entry.get("reason", "—"),
                        "blocked_at": entry.get("blocked_at", "—"), "time_left_hrs": round(rem / 3600, 2)
                    })
    return jsonify({"ok": True, "cooldowns": active})

@app.route("/api/cooldown/release", methods=["POST", "OPTIONS"])
def api_cooldown_release():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_cooldown_release")

@app.route("/api/predictions")
def api_predictions():
    preds = get(PREDICTIONS_FILE, [])
    if not isinstance(preds, list): preds = []
    audited = [p for p in preds if p.get("audited")]
    scored = [p for p in audited if p.get("correct") is not None]
    correct = [p for p in audited if p.get("correct") is True]
    disagreements = [p.get("ensemble_disagreement") for p in preds if p.get("ensemble_disagreement") is not None]
    return jsonify({
        "ok": True, "total_predictions": len(preds),
        "unaudited_count": len(preds) - len(audited), "audited_count": len(audited),
        "overall_hit_rate": round(len(correct) / len(scored) * 100, 1) if scored else None,
        "avg_ensemble_disagreement": round(sum(disagreements) / len(disagreements), 4) if disagreements else None,
        "recent": list(reversed(preds[-100:])),
    })

@app.route("/api/dossier")
def api_dossier():
    dossier = get(DOSSIER_FILE, {})
    rows = []
    for symbol, d in dossier.items():
        total = d.get("total_audited", 0)
        correct = d.get("total_correct", 0)
        rows.append({
            "symbol": symbol, "total_audited": total,
            "hit_rate": round(correct / total * 100, 1) if total else None,
            "avg_disagreement_when_correct": round(d.get("disagreement_correct_sum", 0.0) / d.get("disagreement_correct_n", 1), 4) if d.get("disagreement_correct_n") else None,
            "avg_disagreement_when_incorrect": round(d.get("disagreement_incorrect_sum", 0.0) / d.get("disagreement_incorrect_n", 1), 4) if d.get("disagreement_incorrect_n") else None,
            "tags": d.get("tags", {}),
        })
    return jsonify({"ok": True, "coins": rows})

@app.route("/api/adaptation_proposals")
def api_adaptation_proposals():
    status = request.args.get("status")
    proposals = get(PROPOSALS_FILE, [])
    if status: proposals = [p for p in proposals if p.get("status") == status]
    return jsonify({"ok": True, "proposals": list(reversed(proposals))})

@app.route("/api/adaptation_proposals/<proposal_id>/approve", methods=["POST", "OPTIONS"])
def api_approve_proposal(proposal_id):
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_approve_proposal")

@app.route("/api/adaptation_proposals/<proposal_id>/reject", methods=["POST", "OPTIONS"])
def api_reject_proposal(proposal_id):
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_reject_proposal")

@app.route("/api/market")
def api_market():
    cfg = get_live_config()
    prices = {}
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=5)
        if r.ok:
            for item in r.json():
                if item["symbol"] in cfg["symbols"]:
                    prices[item["symbol"]] = {
                        "lastPrice": float(item.get("lastPrice", 0)),
                        "priceChangePercent": float(item.get("priceChangePercent", 0)),
                        "quoteVolume": float(item.get("quoteVolume", 0)),
                    }
    except Exception: pass
    return jsonify(prices)

@app.route("/api/btc_atr")
def api_btc_atr():
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=30", timeout=4)
        r2 = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=4)
        if r.ok and r2.ok:
            df = [{"h": float(d[2]), "l": float(d[3]), "c": float(d[4])} for d in r.json()]
            trs = [df[i]["h"] - df[i]["l"] if i == 0 else max(df[i]["h"] - df[i]["l"], abs(df[i]["h"] - df[i-1]["c"]), abs(df[i]["l"] - df[i-1]["c"])) for i in range(len(df))]
            atr = sum(trs[-14:]) / 14
            price = df[-1]["c"]
            pct_v = atr / price * 100
            chg = float(r2.json().get("priceChangePercent", 0))
            return jsonify({"ok": True, "atr": round(atr, 2), "pct": round(pct_v, 2), "price": round(price, 0), "chg_24h": round(chg, 2)})
    except Exception: pass
    return jsonify({"ok": True, "atr": 0.0, "pct": 0.0, "price": 0.0, "chg_24h": 0.0})

@app.route("/api/fng")
def api_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
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
    signals = get("signals.json", [])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return jsonify({
        "ok": True,
        "deribit": {
            "ok": deribit["status"] == "ONLINE",
            "msg": deribit["msg"],
            "balance": (
                _dashboard_number(bal.get("account_equity_usd"))
                if str(bal.get("equity_currency", "")).upper() == "USD"
                else None
            ),
            "balance_currency": "USD" if str(bal.get("equity_currency", "")).upper() == "USD" else None,
            "risk_equity": _dashboard_number(bal.get("risk_equity")),
            "risk_equity_currency": bal.get("risk_equity_currency"),
            "open_positions": len(open_trades),
        },
        "market": {"ok": True, "btc_price": 0, "latency_ms": 50},
        "bot": {"signals_today": len([s for s in signals if str(s.get("generated_at","")).startswith(today)]), "gh_read_token_configured": bool(GH_READ_TOKEN)},
        "integrity": {"open_slots": f"{len(open_trades)}/{cfg['max_open_trades']}", "sltp_missing": sum(1 for t in trades.values() if not t.get("stop") or not t.get("tp1"))}
    })

def _dashboard_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _measured_value(record, keys):
    for key in keys:
        if key in record:
            value = _dashboard_number(record.get(key))
            if value is not None:
                return value
    return None


@app.route("/api/execution")
def api_execution():
    history = get("trade_history.json", [])
    if not isinstance(history, list):
        history = []
    completed = [
        item for item in history
        if isinstance(item, dict)
        and item.get("signal") != "RECOVERED"
        and not item.get("pnl_unverified", False)
        and (item.get("closed") is True or item.get("closed_at"))
    ]
    gross_pnl, pnl_currency = _currency_safe_pnl(completed)
    pnl_valid = gross_pnl is not None

    fee_values = []
    fee_currencies = []
    for item in completed:
        if item.get("fees_usd") is not None or item.get("fee_usd") is not None or item.get("commission_usd") is not None:
            fee = _measured_value(item, ("fees_usd", "fee_usd", "commission_usd"))
            fee_currency = "USD"
        else:
            fee = _measured_value(item, ("fees", "fee", "commission"))
            fee_currency = str(item.get("fee_currency", "")).strip().upper()
        fee_values.append(fee)
        fee_currencies.append(fee_currency)
    fees_complete = (
        bool(completed)
        and pnl_valid
        and all(value is not None for value in fee_values)
        and all(currency == pnl_currency for currency in fee_currencies)
    )
    total_fees = round(sum(value for value in fee_values if value is not None), 4) if fees_complete else None

    durations = []
    for item in completed:
        try:
            opened = datetime.fromisoformat(str(item["opened_at"]).replace("Z", "+00:00"))
            closed = datetime.fromisoformat(str(item["closed_at"]).replace("Z", "+00:00"))
            duration = (closed - opened).total_seconds() / 3600.0
            if math.isfinite(duration) and duration >= 0:
                durations.append(duration)
        except (KeyError, TypeError, ValueError):
            continue

    volume_values = [_measured_value(item, ("volume_traded_usd", "notional_usd", "volume_usd")) for item in completed]
    volume_complete = bool(completed) and all(value is not None for value in volume_values)
    slip_values = [_measured_value(item, ("slippage_pct", "avg_slippage_pct")) for item in completed]
    slippage_complete = bool(completed) and all(value is not None for value in slip_values)

    return jsonify({
        "ok": True,
        "gross_pnl": gross_pnl if pnl_valid else None,
        "pnl_currency": pnl_currency if pnl_valid else None,
        "total_fees": total_fees,
        "fee_currency": pnl_currency if fees_complete else None,
        "net_pnl": round(gross_pnl - total_fees, 4) if fees_complete and gross_pnl is not None else None,
        "avg_duration_h": round(sum(durations) / len(durations), 4) if durations else None,
        "total_volume_traded_usd": round(sum(volume_values), 4) if volume_complete else None,
        "avg_slippage_pct": round(sum(slip_values) / len(slip_values), 6) if slippage_complete else None,
        "measured_trade_count": len(completed),
        "fee_coverage_complete": fees_complete,
        "volume_coverage_complete": volume_complete,
        "slippage_coverage_complete": slippage_complete,
        "source": "trade_history.json; unavailable fields are null, not estimates",
    })


@app.route("/api/model_health")
def api_model_health():
    model_meta = get_model_metadata()
    performance = get(PERFORMANCE_FILE, {})
    if not isinstance(performance, dict):
        performance = {}
    accuracy = performance.get("test_accuracy", performance.get("accuracy"))
    if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool):
        accuracy = f"{float(accuracy):.1f}%"
    if not accuracy:
        accuracy = "N/A"
    calibration = performance.get("calibration")
    if not (
        isinstance(calibration, dict)
        and isinstance(calibration.get("labels"), list)
        and isinstance(calibration.get("target"), list)
        and isinstance(calibration.get("live"), list)
        and len(calibration["labels"]) == len(calibration["target"]) == len(calibration["live"])
    ):
        calibration = None
    return jsonify({
        "ok": True,
        "ensemble_name": model_meta["model_name"],
        "test_accuracy": accuracy,
        "rec_buy_threshold": model_meta["rec_buy_conf"],
        "rec_sell_threshold": model_meta["rec_sell_conf"],
        "calibration": calibration,
        "accuracy_source": "model_performance.json" if accuracy != "N/A" else "unavailable",
    })

@app.route("/api/analytics")
def api_analytics():
    history = get("trade_history.json", [])
    if not isinstance(history, list):
        history = []
    real_trades = [
        item for item in history
        if isinstance(item, dict)
        and item.get("signal") != "RECOVERED"
        and not item.get("pnl_unverified", False)
    ]
    pnl_total, pnl_currency = _currency_safe_pnl(real_trades)
    pnl_valid = pnl_total is not None
    pnls = [float(item.get("pnl", 0) or 0) for item in real_trades] if pnl_valid else []
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value <= 0]
    total_trades = len(real_trades)
    win_rate = (len(wins) / total_trades * 100) if pnl_valid and total_trades else None
    avg_win = (sum(wins) / len(wins)) if wins else None
    avg_loss = (abs(sum(losses)) / len(losses)) if losses else None
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (
        round(gross_profit / gross_loss, 2) if gross_loss > 0
        else (round(gross_profit, 2) if gross_profit > 0 else 0.0)
    ) if pnl_valid else None
    expectancy = (
        round((win_rate / 100.0 * (avg_win or 0.0)) -
              ((1.0 - win_rate / 100.0) * (avg_loss or 0.0)), 4)
        if pnl_valid and win_rate is not None else None
    )

    sharpe = sortino = max_dd = None
    if pnl_valid and len(pnls) > 3:
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((value - mean_pnl) ** 2 for value in pnls) / len(pnls)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001
        downside_vars = [value ** 2 for value in losses]
        downside_std = math.sqrt(sum(downside_vars) / len(pnls)) if downside_vars else 0.001
        sharpe = round((mean_pnl / std_dev) * math.sqrt(365), 2)
        sortino = round((mean_pnl / downside_std) * math.sqrt(365), 2)
        cumulative = peak = 0.0
        drawdowns = []
        for value in pnls:
            cumulative += value
            peak = max(peak, cumulative)
            drawdowns.append(peak - cumulative)
        max_dd = round(max(drawdowns), 4) if drawdowns else 0.0

    daily_pnl_map = {}
    if pnl_valid:
        for item in real_trades:
            day = (item.get("closed_at") or item.get("opened_at") or "")[:10]
            if day:
                daily_pnl_map[day] = round(
                    daily_pnl_map.get(day, 0.0) + float(item.get("pnl", 0) or 0), 4
                )
    daily_points = [
        {"date": day, "pnl": value, "currency": pnl_currency}
        for day, value in sorted(daily_pnl_map.items())
    ]

    bal_data = get("balance.json", {})
    risk_currency = str(bal_data.get("risk_equity_currency", "")).upper()
    current_bal = _dashboard_number(bal_data.get("risk_equity"))
    equity_points = []
    curve_currency_valid = (
        pnl_valid
        and pnl_currency == risk_currency
        and risk_currency in {"USDC", "USD"}
        and current_bal is not None
        and current_bal > 0
    )
    if curve_currency_valid:
        running = current_bal - sum(float(item.get("pnl", 0) or 0) for item in real_trades)
        for item in real_trades[-50:]:
            running += float(item.get("pnl", 0) or 0)
            time_str = (item.get("closed_at") or item.get("opened_at") or "")[:10]
            equity_points.append({
                "time": time_str,
                "equity": round(running, 4),
                "currency": risk_currency,
            })

    return jsonify({
        "ok": True,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
        "pnl_currency": pnl_currency if pnl_valid else None,
        "pnl_metrics_available": pnl_valid,
        # Deprecated aliases retained for older clients; they are not USD/USDT claims.
        "expectancy_usdt": expectancy,
        "max_drawdown_usdt": max_dd,
        "win_rate": round(win_rate, 1) if win_rate is not None else None,
        "avg_win": round(avg_win, 4) if avg_win is not None else None,
        "avg_loss": round(avg_loss, 4) if avg_loss is not None else None,
        "equity_curve": equity_points,
        "equity_curve_currency": risk_currency if curve_currency_valid else None,
        "equity_curve_available": curve_currency_valid,
        "daily_pnl": daily_points,
    })

@app.route("/api/log")
def api_log():
    content = get("bot.log", "")
    lines = content.splitlines(keepends=True)[-200:] if content else ["✓ Bot standby."]
    return jsonify({"log": "".join(lines), "lines": len(lines)})

@app.route("/api/scan", methods=["POST", "OPTIONS"])
def api_scan():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_scan")

@app.route("/api/kill_switch", methods=["POST", "OPTIONS"])
def api_kill_switch():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_kill_switch")

@app.route("/api/close_trade", methods=["POST", "OPTIONS"])
def api_close_trade():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    return _dashboard_read_only_response("api_close_trade")

@app.route("/api/paper/state")
def api_paper_state():
    # Read-only Stage 2E paper ledger endpoint; never accesses the exchange.
    try:
        state = get("paper_state.json", {})
        if not isinstance(state, dict):
            state = {}

        raw_positions = state.get("positions", {})
        if not isinstance(raw_positions, dict):
            raw_positions = {}

        predictions = get(PREDICTIONS_FILE, [])
        if not isinstance(predictions, list):
            predictions = []
        predictions_by_id = {
            str(item.get("pred_id")): item
            for item in predictions
            if isinstance(item, dict) and item.get("pred_id") not in (None, "")
        }

        positions = []
        enrich_fields = (
            "confidence",
            "rsi_15m",
            "adx_15m",
            "ensemble_disagreement",
            "high_disagreement",
            "score",
        )
        for pred_id, raw_position in raw_positions.items():
            if not isinstance(raw_position, dict):
                continue
            position = dict(raw_position)
            position.setdefault("pred_id", str(pred_id))
            prediction = predictions_by_id.get(str(pred_id), {})
            position.setdefault(
                "side",
                prediction.get("predicted_signal") or prediction.get("signal"),
            )
            for field in enrich_fields:
                if position.get(field) is None and prediction.get(field) is not None:
                    position[field] = prediction[field]
            positions.append(position)

        positions.sort(
            key=lambda item: int(item.get("opened_at_ms") or item.get("open_time") or 0),
            reverse=True,
        )

        raw_decisions = state.get("decisions", {})
        if not isinstance(raw_decisions, dict):
            raw_decisions = {}
        decisions = []
        for pred_id, raw_decision in raw_decisions.items():
            if not isinstance(raw_decision, dict):
                continue
            decision = dict(raw_decision)
            decision.setdefault("pred_id", str(pred_id))
            prediction = predictions_by_id.get(str(pred_id), {})
            decision.setdefault("symbol", prediction.get("symbol") or prediction.get("logical_symbol"))
            decision.setdefault("side", prediction.get("predicted_signal") or prediction.get("signal"))
            decision.setdefault("generated_at", prediction.get("generated_at"))
            decisions.append(decision)
        decisions.sort(key=lambda item: int(item.get("timestamp_ms") or 0), reverse=True)

        raw_events = get("paper_events.jsonl", "")
        if isinstance(raw_events, list):
            event_lines = raw_events
        elif isinstance(raw_events, str):
            event_lines = raw_events.splitlines()
        else:
            event_lines = []

        event_count = sum(1 for line in event_lines if str(line).strip())
        events = []
        # Parse only a bounded tail; the append-only source file is never mutated.
        for line in event_lines[-500:]:
            if isinstance(line, dict):
                event = line
            else:
                try:
                    event = json.loads(str(line))
                except (TypeError, json.JSONDecodeError):
                    continue
            if isinstance(event, dict):
                events.append(event)
        events = events[-100:][::-1]

        mode = str(state.get("execution_mode") or "PAPER_ONLY")
        active_count = sum(1 for item in positions if item.get("status") == "ACTIVE")
        closed_count = sum(1 for item in positions if item.get("status") == "CLOSED")

        return jsonify({
            "ok": True,
            "paper_only": mode == "PAPER_ONLY",
            "execution_mode": mode,
            "policy_version": state.get("policy_version", "stage2e-v1"),
            "schema_version": state.get("schema_version"),
            "initial_balance_usd": state.get("initial_balance_usd", 10000.0),
            "risk_per_trade": state.get("risk_per_trade"),
            "positions": positions,
            "decisions": decisions[:100],
            "events": events,
            "portfolio": {
                "active_positions": active_count,
                "closed_positions": closed_count,
                "decision_count": len(decisions),
                "event_count": event_count,
                "max_open_positions": 8,
                "max_same_direction": 4,
            },
        })
    except Exception:
        log.exception("Stage 2E paper state endpoint failed")
        return jsonify({"ok": False, "error": "Paper state is temporarily unavailable"}), 500


def generate_paper_pdf_bytes(snapshot: dict) -> bytes:
    # Separate Stage 2E template. Existing History PDF is intentionally untouched.
    if not HAS_REPORTLAB:
        raise ImportError("ReportLab package is not installed on this server.")
    snapshot = snapshot or {}
    positions = [p for p in snapshot.get("positions", []) if isinstance(p, dict)]
    decisions = [d for d in snapshot.get("decisions", []) if isinstance(d, dict)]
    events = [e for e in snapshot.get("events", []) if isinstance(e, dict)]
    portfolio = snapshot.get("portfolio") or {}

    def num(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def pnl(p):
        # Stage 2E stores the verified close amount as net_pnl_usd.
        for key in ("net_pnl_usd", "net_pnl", "pnl", "realized_pnl"):
            if p.get(key) is not None:
                try:
                    return float(p[key])
                except (TypeError, ValueError):
                    pass

        # net_r is only converted when a real resolved R value and matching
        # initial risk are present. Ambiguous outcomes remain unverified.
        if p.get("net_r") is not None and p.get("initial_risk_usd") is not None:
            try:
                if p.get("exit_reason") == "AMBIGUOUS_BARRIER":
                    return None
                return float(p["net_r"]) * float(p["initial_risk_usd"])
            except (TypeError, ValueError):
                pass
        return None

    active = [p for p in positions if p.get("status") == "ACTIVE"]
    closed = [p for p in positions if p.get("status") == "CLOSED"]
    values = [pnl(p) for p in closed]
    values = [v for v in values if v is not None]
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v < 0]
    realized = sum(values)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    win_rate = (len(wins) / len(values) * 100.0) if values else 0.0
    pf = gross_profit / gross_loss if gross_loss else None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=(letter[1], letter[0]), rightMargin=28, leftMargin=28,
        topMargin=30, bottomMargin=34
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Stage2ETitle", parent=styles["Heading1"], fontSize=15, leading=18,
        textColor=colors.HexColor("#0891b2"), spaceAfter=3,
        fontName="Helvetica-Bold"
    )
    sub = ParagraphStyle(
        "Stage2ESub", parent=styles["Normal"], fontSize=8, leading=10,
        textColor=colors.HexColor("#64748b"), spaceAfter=8
    )
    sec = ParagraphStyle(
        "Stage2ESec", parent=styles["Heading2"], fontSize=10, leading=13,
        textColor=colors.HexColor("#0f172a"), spaceBefore=9, spaceAfter=5,
        fontName="Helvetica-Bold"
    )
    cell = ParagraphStyle(
        "Stage2ECell", parent=styles["Normal"], fontSize=6.5, leading=8,
        textColor=colors.HexColor("#334155")
    )
    bold = ParagraphStyle("Stage2EBold", parent=cell, fontName="Helvetica-Bold")
    green = ParagraphStyle(
        "Stage2EGreen", parent=cell, textColor=colors.HexColor("#059669"),
        fontName="Helvetica-Bold"
    )
    red = ParagraphStyle(
        "Stage2ERed", parent=cell, textColor=colors.HexColor("#dc2626"),
        fontName="Helvetica-Bold"
    )
    head = ParagraphStyle(
        "Stage2EHead", parent=cell, textColor=colors.white,
        fontName="Helvetica-Bold", alignment=1
    )

    mode = str(snapshot.get("execution_mode") or "PAPER_ONLY")
    paper_only = bool(snapshot.get("paper_only"))
    policy = str(snapshot.get("policy_version") or "stage2e-v1")
    schema = str(snapshot.get("schema_version") or "—")
    initial = num(snapshot.get("initial_balance_usd"))

    elements = [
        Paragraph("CryptoBot AI — Stage 2E Paper Trading Report", title),
        Paragraph(
            "Paper-only operational, performance & lifecycle audit. "
            "This is a separate template from the existing History report.",
            sub
        ),
        Paragraph(
            f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
            f"| <b>Mode:</b> {mode} | <b>Paper-only:</b> "
            f"{'PASS' if paper_only else 'WARNING'}",
            cell
        ),
        Paragraph(
            f"<b>Policy:</b> {policy} | <b>Schema:</b> {schema} | "
            f"<b>Starting virtual balance:</b> {initial:.2f} USDT",
            cell
        )
    ]

    summary = [
        ["Metric", "Value", "Metric", "Value"],
        ["Active positions", str(len(active)), "Closed positions", str(len(closed))],
        ["Verified closes", str(len(values)), "Win rate", f"{win_rate:.1f}%"],
        ["Realized paper PnL", f"{realized:+.4f} USDT",
         "Profit factor", f"{pf:.2f}" if pf is not None else "—"],
        ["Largest win", f"{max(wins):+.4f} USDT" if wins else "—",
         "Largest loss", f"{min(losses):+.4f} USDT" if losses else "—"],
        ["Entry decisions", str(len(decisions)),
         "Lifecycle events", str(len(events))]
    ]
    table = Table(
        [[Paragraph(str(x), head if r == 0 else cell) for x in row]
         for r, row in enumerate(summary)],
        colWidths=[110, 135, 110, 135], repeatRows=1
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ("PADDING", (0, 0), (-1, -1), 4)
    ]))
    elements += [Spacer(1, 7), Paragraph("Paper Portfolio & Safety Summary", sec), table]

    def add_table(title_text, headers, rows, widths, header_color="#0f172a"):
        elements.append(Paragraph(title_text, sec))
        all_rows = [[Paragraph(str(x), head) for x in headers]]
        all_rows.extend(rows)
        if len(all_rows) == 1:
            all_rows.append([Paragraph("No records available.", cell)] + [Paragraph("", cell)] * (len(headers)-1))
        t = Table(all_rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 3),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        elements.append(t)

    active_rows = []
    quotes = snapshot.get("market_quotes") if isinstance(snapshot.get("market_quotes"), dict) else {}
    for p in active:
        symbol = str(p.get("symbol") or p.get("logical_symbol") or "")
        quote = quotes.get(symbol, {}) if isinstance(quotes.get(symbol, {}), dict) else {}
        mark_price = num(quote.get("lastPrice"))
        entry_price = num(p.get("entry_price", p.get("simulated_entry", p.get("entry_ref"))))
        remaining_qty = num(p.get("remaining_qty", p.get("qty")))
        side = str(p.get("side") or "").upper()
        estimated_open_pnl = None
        if mark_price > 0 and entry_price > 0 and remaining_qty > 0 and side in ("BUY", "SELL"):
            estimated_open_pnl = ((mark_price-entry_price) if side == "BUY" else (entry_price-mark_price)) * remaining_qty
            estimated_open_pnl -= num(p.get("entry_fee_usd")) + num(p.get("tp1_fee_usd"))
            estimated_open_pnl -= mark_price * remaining_qty * 0.0006
            estimated_open_pnl += num(p.get("tp1_realized_gross_pnl"))
        active_rows.append([
            Paragraph(str(p.get("symbol") or p.get("logical_symbol") or "—"), bold),
            Paragraph(str(p.get("side") or "—"), cell),
            Paragraph("PAPER ONLY", cell),
            Paragraph(f'{mark_price:.6f}' if mark_price > 0 else "—", cell),
            Paragraph(f'{estimated_open_pnl:+.4f}' if estimated_open_pnl is not None else "—", green if estimated_open_pnl is not None and estimated_open_pnl >= 0 else red if estimated_open_pnl is not None else cell),
            Paragraph(f'{num(p.get("entry_price") or p.get("simulated_entry") or p.get("entry_ref")):.6f}', cell),
            Paragraph(f'{num(p.get("qty")):.6f}', cell),
            Paragraph(f'{num(p.get("remaining_qty", p.get("qty"))):.6f}', cell),
            Paragraph(f'{num(p.get("stop", p.get("stop_loss"))):.6f}', cell),
            Paragraph(f'{num(p.get("tp1")):.6f}', cell),
            Paragraph(f'{num(p.get("tp2")):.6f}', cell),
            Paragraph(f'{num(p.get("confidence")):.2f}%' if p.get("confidence") is not None else "—", cell),
            Paragraph(f'{num(p.get("adx_15m")):.2f}' if p.get("adx_15m") is not None else "—", cell),
            Paragraph(f'{num(p.get("initial_risk_usd", p.get("initial_risk"))):.4f}', cell)
        ])
    add_table(
        "Open Paper Positions",
        ["Symbol","Side","Status","Mark","Open PnL*","Entry","Qty","Remaining","SL","TP1","TP2","Confidence","ADX","Risk"],
        active_rows, [46,30,44,50,60,50,42,45,43,43,43,52,36,45], "#0891b2"
    )

    closed_rows = []
    for i, p in enumerate(closed[:100], 1):
        value = pnl(p)
        style = green if value is not None and value >= 0 else red if value is not None else cell
        closed_rows.append([
            Paragraph(str(i), cell),
            Paragraph(str(p.get("symbol") or p.get("logical_symbol") or "—"), bold),
            Paragraph(str(p.get("side") or "—"), cell),
            Paragraph(str(p.get("exit_reason") or p.get("state") or p.get("status") or "—")[:26], cell),
            Paragraph(f'{num(p.get("entry_price")):.6f}', cell),
            Paragraph(f'{num(p.get("exit_price")):.6f}' if p.get("exit_price") is not None else "—", cell),
            Paragraph(f"{value:+.4f}" if value is not None else "AMBIGUOUS / UNVERIFIED", style),
            Paragraph(f'{num(p.get("net_r")):.4f}R' if p.get("net_r") is not None else "—", cell),
            Paragraph(f'{num(p.get("confidence")):.2f}%' if p.get("confidence") is not None else "—", cell),
            Paragraph(f'{num(p.get("adx_15m")):.2f}' if p.get("adx_15m") is not None else "—", cell)
        ])
    add_table(
        "Closed Paper Positions & Outcomes",
        ["#","Symbol","Side","Exit reason","Entry","Exit","Net PnL","Net R","Confidence","ADX"],
        closed_rows, [18,52,36,72,52,52,62,45,58,40]
    )

    decision_rows = []
    for d in decisions[:100]:
        decision_rows.append([
            Paragraph(str(d.get("status") or "UNKNOWN"), cell),
            Paragraph(str(d.get("symbol") or "—"), bold),
            Paragraph(str(d.get("side") or "—"), cell),
            Paragraph(str(d.get("timestamp_ms") or d.get("generated_at") or "—"), cell),
            Paragraph(str(d.get("pred_id") or "—")[:28], cell),
            Paragraph(str(d.get("reason") or d.get("state") or d.get("exit_reason") or "—")[:80], cell)
        ])
    add_table(
        "Entry Decisions / Signal Handling",
        ["Status","Symbol","Side","Time UTC","Prediction ID","Decision detail"],
        decision_rows, [65,55,40,95,90,145], "#475569"
    )

    reconciliation = snapshot.get("reconciliation") or {}
    evidence = [
        f"<b>Execution mode:</b> {mode}",
        f"<b>Paper-only flag:</b> {'PASS' if paper_only else 'WARNING'}",
        f"<b>Reconciliation:</b> {reconciliation.get('ok', 'not present')}",
        f"<b>Active / closed:</b> {len(active)} / {len(closed)}",
        f"<b>Max open / same direction:</b> {portfolio.get('max_open_positions', 8)} / {portfolio.get('max_same_direction', 4)}",
        f"<b>Risk per trade:</b> {num(portfolio.get('risk_per_trade')):.4f}",
        f"<b>Initial virtual balance:</b> {initial:.2f} USDT",
        "<b>Safety:</b> simulated paper positions; this report does not prove exchange execution."
    ]
    elements += [Paragraph("Lifecycle & Reconciliation Evidence", sec),
                 Paragraph("<br/>".join(evidence), cell)]

    event_rows = []
    for e in events[:100]:
        event_rows.append([
            Paragraph(str(e.get("event") or e.get("type") or "EVENT"), cell),
            Paragraph(str(e.get("symbol") or "—"), bold),
            Paragraph(str(e.get("side") or "—"), cell),
            Paragraph(str(e.get("reason") or e.get("state") or e.get("exit_reason") or "—")[:50], cell),
            Paragraph(str(e.get("timestamp_ms") or e.get("candle_close_time") or e.get("open_time") or "—"), cell),
            Paragraph(str(e.get("pred_id") or "—")[:28], cell)
        ])
    add_table(
        "Recent Paper Lifecycle Events",
        ["Event","Symbol","Side","Reason / State","Time UTC","Prediction ID"],
        event_rows, [100,55,40,130,95,70]
    )

    def footer(canvas_obj, doc_obj):
        canvas_obj.saveState()
        canvas_obj.setFont("Helvetica", 6.5)
        canvas_obj.setFillColor(colors.HexColor("#94a3b8"))
        canvas_obj.drawString(
            28, 20,
            "CryptoBot AI — Stage 2E PAPER ONLY. Open PnL is indicative using latest spot quotes; no exchange execution is represented."
        )
        canvas_obj.drawRightString(doc_obj.pagesize[0] - 28, 20, f"Page {doc_obj.page}")
        canvas_obj.restoreState()

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)
    buffer.seek(0)
    return buffer.getvalue()


@app.route("/api/download_paper_report_pdf", methods=["POST", "OPTIONS"])
def api_download_paper_report_pdf():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    if not HAS_REPORTLAB:
        return jsonify({
            "ok": False, "error": "REPORTLAB_MISSING",
            "message": "ReportLab not installed on this server."
        }), 500
    data = request.get_json(silent=True) or {}
    try:
        pdf_bytes = generate_paper_pdf_bytes(data)
    except Exception as exc:
        log.error(f"Stage 2E paper PDF generation failed: {exc}")
        return jsonify({
            "ok": False, "error": "PAPER_PDF_GENERATION_FAILED",
            "message": str(exc)
        }), 500
    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    filename = (
        "CryptoBot_Stage2E_Paper_Report_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.pdf"
    )
    return send_file(
        buffer, mimetype="application/pdf",
        as_attachment=True, download_name=filename
    )

@app.route("/health")
def health():
    ready = _dashboard_auth_settings() is not None
    return jsonify({
        "status": "ok" if ready else "not_ready",
        "time": datetime.now(timezone.utc).isoformat(),
    }), (200 if ready else 503)

@app.route("/")
def index(): return send_from_directory("dashboard_static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    try: return send_from_directory("dashboard_static", path)
    except Exception: return send_from_directory("dashboard_static", "index.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
