# telegram_alert.py - Sends alerts to your phone

import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

def send_message(text, parse_mode="Markdown", silent=False):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured - check your .env file")
        return False
    url     = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID, 
        "text": text, 
        "parse_mode": parse_mode,
        "disable_notification": silent
    }
    try:
        requests.post(url, data=payload, timeout=10)
        return True
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def send_signal(symbol, signal, confidence, entry, stop, t1, t2, reasons, score):
    emoji  = "🟢" if signal == "BUY" else "🔴"
    stars  = "⭐" * min(score, 5)
    sl_pct = abs((stop - entry) / entry * 100)
    t1_pct = abs((t1   - entry) / entry * 100)
    t2_pct = abs((t2   - entry) / entry * 100)
    dec    = 4 if entry < 10 else 2
    fp     = lambda v: f"{v:,.{dec}f}"
    reason_lines = "\n".join([f"  - {r}" for r in reasons])

    msg = (
        f"🤖 *CryptoBot AI Signal*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{emoji} *{signal}  —  {symbol}* {stars}\n"
        f"🎯 Confidence: *{confidence:.1f}%*\n\n"
        f"⚡ *ENTRY:* `{fp(entry)}`\n"
        f"🛑 *STOP LOSS:* `{fp(stop)}`  (-{sl_pct:.1f}%)\n"
        f"🎯 *TARGET 1:* `{fp(t1)}`  (+{t1_pct:.1f}%)\n"
        f"🎯 *TARGET 2:* `{fp(t2)}`  (+{t2_pct:.1f}%)\n"
        f"⚖️  *R:R:* 1:{(t1_pct/sl_pct):.1f}\n\n"
        f"📊 *Why:*\n{reason_lines}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Not financial advice. Always use stop loss._"
    )
    return send_message(msg)

def send_open_trade(sym, sig, conf, score, entry, stop, tp1, tp2, qty, q1, q2, risk, bal, is_probation=False, tier=""):
    emoji = "🟢" if sig == "BUY" else "🔴"
    d = 4 if entry < 10 else 2
    
    sl_pct  = abs((stop - entry) / entry * 100)
    tp1_pct = abs((tp1 - entry) / entry * 100)
    tp2_pct = abs((tp2 - entry) / entry * 100)
    
    rr_tp1 = (tp1_pct / sl_pct) if sl_pct > 0 else 3.5
    rr_tp2 = (tp2_pct / sl_pct) if sl_pct > 0 else 7.5

    prob_tag = "\n🔒 *PROBATION TRADE* (+10.0% Conf Premium)" if is_probation else ""

    text = (
        f"⚡ *CryptoBot AI — Position Opened*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} *{sig} — {sym}* ({tier})\n"
        f"🎯 *AI Conf:* `{conf:.1f}%`  |  ⭐ *Score:* `{score}/6`{prob_tag}\n\n"
        f"📊 *Execution Details:*\n"
        f"• Entry:  `{entry:,.{d}f}`\n"
        f"• Size:   `{qty} contracts` (${risk:.2f} risk)\n"
        f"• Margin: `10x Leverage` (Deribit Testnet)\n\n"
        f"🛡️ *Risk & Target Levels:*\n"
        f"🛑 *SL:*  `{stop:,.{d}f}` (-{sl_pct:.1f}%)\n"
        f"🎯 *TP1:* `{tp1:,.{d}f}` (+{tp1_pct:.1f}%) × {q1} `[{rr_tp1:.1f}R]`\n"
        f"🎯 *TP2:* `{tp2:,.{d}f}` (+{tp2_pct:.2f}%) × {q2} `[{rr_tp2:.1f}R]`\n\n"
        f"💰 *Account Equity:* `${bal:,.2f} USDT`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    return send_message(text, silent=False)

def send_startup():
    from config import SYMBOLS
    try:
        from news_sentiment import get_market_conditions
        market = get_market_conditions()
        fg     = market.get("fear_greed", 50)
        cond   = market.get("condition", "Neutral")
        adv    = market.get("advice", "Trade safely.")
        lbl    = market.get("label", "Neutral")
    except ImportError:
        fg = 50; cond = "Neutral"; adv = "Trade carefully."; lbl = "Neutral"

    coins  = "  ".join([s.replace("USDT","") for s in SYMBOLS])
    emoji  = "😱" if fg < 25 else "😨" if fg < 45 else "😐" if fg < 55 else "😊" if fg < 75 else "🤑"
    
    send_message(
        f"🚀 *CryptoBot AI Started*\n"
        f"Monitoring: `{coins}`\n\n"
        f"🌍 *Market Conditions*\n"
        f"Fear & Greed: {fg} {emoji} — {lbl}\n"
        f"Condition: {cond}\n"
        f"_{adv}_"
    )
