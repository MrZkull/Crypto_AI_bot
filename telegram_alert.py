# telegram_alert.py - Sends alerts to your phone

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)


BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")



def send_message(text, parse_mode="Markdown"):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured - check your .env file")
        return False
    url     = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": parse_mode}
    try:
        r = requests.post(url, data=payload, timeout=10)
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
        f"⚡ *ENTRY:*      `{fp(entry)}`\n"
        f"🛑 *STOP LOSS:*  `{fp(stop)}`  (-{sl_pct:.1f}%)\n"
        f"🎯 *TARGET 1:*   `{fp(t1)}`  (+{t1_pct:.1f}%)\n"
        f"🎯 *TARGET 2:*   `{fp(t2)}`  (+{t2_pct:.1f}%)\n"
        f"⚖️  *R:R:* 1:{(t1_pct/sl_pct):.1f}\n\n"
        f"📊 *Why:*\n{reason_lines}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"_Not financial advice. Always use stop loss._"
    )
    return send_message(msg)


def send_startup():
    from config import SYMBOLS
    from news_sentiment import get_market_conditions
    coins  = "  ".join([s.replace("USDT","") for s in SYMBOLS])
    market = get_market_conditions()
    fg     = market["fear_greed"]
    emoji  = "😱" if fg < 25 else "😨" if fg < 45 else "😐" if fg < 55 else "😊" if fg < 75 else "🤑"
    send_message(
        f"🚀 *CryptoBot AI Started*\n"
        f"Monitoring: `{coins}`\n\n"
        f"🌍 *Market Conditions*\n"
        f"Fear & Greed: {fg} {emoji} — {market['label']}\n"
        f"Condition: {market['condition']}\n"
        f"_{market['advice']}_"
    )