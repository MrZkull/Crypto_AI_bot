# live_scanner.py - Live scanner that runs every 15 minutes

import requests
import pandas as pd
import joblib
import time
import schedule
import logging
from datetime import datetime
from feature_engineering import add_indicators
from telegram_alert import send_signal, send_startup
from news_sentiment import get_news_sentiment, get_market_conditions
from config import (
    SYMBOLS, FEATURES, TIMEFRAME_ENTRY, TIMEFRAME_CONFIRM,
    TIMEFRAME_TREND, MIN_CONFIDENCE, MIN_ADX, MIN_SCORE,
    ATR_STOP_MULT, ATR_TARGET1_MULT, ATR_TARGET2_MULT,
    MODEL_FILE, LIVE_LIMIT, SCAN_INTERVAL_MIN, LOG_FILE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

model        = None
last_signals = {}


def load_model():
    global model
    try:
        model = joblib.load(MODEL_FILE)
        log.info(f"Model loaded: {MODEL_FILE}")
    except FileNotFoundError:
        log.error(f"Model not found! Run train_model.py first.")
        raise


def get_data(symbol, interval):
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": LIVE_LIMIT}
    resp   = requests.get(url, params=params, timeout=15)
    df     = pd.DataFrame(resp.json()).iloc[:, :6]
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def quality_score(row_entry, row_confirm, row_trend, signal, confidence):
    score   = 0
    reasons = []

    # 1. AI confidence
    if confidence >= 75:
        score += 1
        reasons.append(f"High AI confidence ({confidence:.0f}%)")
    elif confidence >= 65:
        reasons.append(f"AI confidence ({confidence:.0f}%)")

    # 2. ADX trend strength
    adx = row_entry.get("adx", 0)
    if adx > 25:
        score += 1
        reasons.append(f"Strong trend (ADX {adx:.0f})")
    elif adx > 20:
        score += 1
        reasons.append(f"Moderate trend (ADX {adx:.0f})")

    # 3. RSI zone
    rsi = row_entry.get("rsi", 50)
    if signal == "BUY" and rsi < 40:
        score += 1
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif signal == "SELL" and rsi > 60:
        score += 1
        reasons.append(f"RSI overbought ({rsi:.0f})")

    # 4. EMA alignment on entry timeframe
    e20  = row_entry.get("ema20",  0)
    e50  = row_entry.get("ema50",  0)
    e200 = row_entry.get("ema200", 0)
    if signal == "BUY" and e20 > e50 > e200:
        score += 1
        reasons.append("EMA uptrend (EMA20 > EMA50 > EMA200)")
    elif signal == "SELL" and e20 < e50 < e200:
        score += 1
        reasons.append("EMA downtrend (EMA20 < EMA50 < EMA200)")

    # 5. Confirmation timeframe
    if signal == "BUY" and row_confirm.get("ema20", 0) > row_confirm.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_CONFIRM} EMA confirms uptrend")
    elif signal == "SELL" and row_confirm.get("ema20", 0) < row_confirm.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_CONFIRM} EMA confirms downtrend")

    # 6. Trend timeframe
    if signal == "BUY" and row_trend.get("ema20", 0) > row_trend.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_TREND} EMA confirms uptrend")
    elif signal == "SELL" and row_trend.get("ema20", 0) < row_trend.get("ema50", 0):
        score += 1
        reasons.append(f"{TIMEFRAME_TREND} EMA confirms downtrend")

    return score, reasons


def scan_symbol(symbol):
    try:
        # Fetch all three timeframes
        df_entry   = add_indicators(get_data(symbol, TIMEFRAME_ENTRY))
        df_confirm = add_indicators(get_data(symbol, TIMEFRAME_CONFIRM))
        df_trend   = add_indicators(get_data(symbol, TIMEFRAME_TREND))

        if df_entry.empty or len(df_entry) < 10:
            return

        row_entry   = df_entry.iloc[-1]
        row_confirm = df_confirm.iloc[-1] if not df_confirm.empty else pd.Series(dtype=float)
        row_trend   = df_trend.iloc[-1]   if not df_trend.empty   else pd.Series(dtype=float)

        # Check all required features exist
        missing = [f for f in FEATURES if f not in df_entry.columns]
        if missing:
            log.warning(f"  {symbol}: Missing features {missing}")
            return

        # AI prediction
        X    = pd.DataFrame([row_entry[FEATURES].values], columns=FEATURES)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        labels     = {0: "BUY", 1: "SELL", 2: "NO_TRADE"}
        signal     = labels[pred]
        confidence = round(float(max(prob)) * 100, 1)

        # Basic filters
        if signal == "NO_TRADE" or confidence < MIN_CONFIDENCE:
            log.info(f"  {symbol}: {signal} {confidence}% — skipped (low confidence)")
            return

        adx_val = float(row_entry.get("adx", 0))
        if adx_val < MIN_ADX:
            log.info(f"  {symbol}: ADX {adx_val:.0f} too low — skipped")
            return

        # Quality scoring
        score, reasons = quality_score(
            row_entry, row_confirm, row_trend, signal, confidence
        )

        if score < MIN_SCORE:
            log.info(f"  {symbol}: Score {score}/6 too low — skipped")
            return

        # Avoid duplicate signals
        if last_signals.get(symbol) == signal:
            log.info(f"  {symbol}: Same signal as last time — skipped")
            return
        last_signals[symbol] = signal

        # Calculate entry, stop loss, targets using ATR
        entry = float(row_entry["close"])
        atr   = float(row_entry["atr"])
        dec   = 4 if entry < 10 else 2

        if signal == "BUY":
            stop = round(entry - atr * ATR_STOP_MULT,    dec)
            t1   = round(entry + atr * ATR_TARGET1_MULT, dec)
            t2   = round(entry + atr * ATR_TARGET2_MULT, dec)
        else:
            stop = round(entry + atr * ATR_STOP_MULT,    dec)
            t1   = round(entry - atr * ATR_TARGET1_MULT, dec)
            t2   = round(entry - atr * ATR_TARGET2_MULT, dec)

        # News sentiment and global market check
        try:
            market = get_market_conditions()
            news   = get_news_sentiment(symbol)

            if not market["trade_ok"]:
                log.info(f"  {symbol}: Market warning — {market['condition']}")
                reasons.append(f"Market: {market['condition']} — trade carefully")

            if news["sentiment"] == "BULLISH" and signal == "BUY":
                score += 1
                reasons.append(f"News BULLISH (score: {news['score']:+d})")
            elif news["sentiment"] == "BEARISH" and signal == "SELL":
                score += 1
                reasons.append(f"News BEARISH (score: {news['score']:+d})")
            elif news["sentiment"] == "BULLISH" and signal == "SELL":
                reasons.append("News BULLISH but signal SELL — caution")
            elif news["sentiment"] == "BEARISH" and signal == "BUY":
                reasons.append("News BEARISH but signal BUY — caution")

            fg = market.get("fear_greed", 50)
            reasons.append(f"Fear & Greed: {fg} — {market.get('label', '')}")

        except Exception as news_err:
            log.warning(f"  {symbol}: News check failed — {news_err}")

        # Send Telegram alert
        log.info(f"  SIGNAL: {symbol} {signal} | {confidence}% | score {score}/6")
        send_signal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            entry=round(entry, dec),
            stop=stop,
            t1=t1,
            t2=t2,
            reasons=reasons,
            score=score,
        )

    except Exception as e:
        log.error(f"  {symbol}: Error — {e}")


def run_scan():
    now = datetime.now().strftime("%H:%M:%S")
    log.info(f"\n{'─'*50}")
    log.info(f"Scanning {len(SYMBOLS)} coins at {now}")
    log.info(f"{'─'*50}")
    for symbol in SYMBOLS:
        scan_symbol(symbol)
        time.sleep(0.5)


def main():
    load_model()
    send_startup()
    run_scan()
    schedule.every(SCAN_INTERVAL_MIN).minutes.do(run_scan)
    log.info(f"\nRunning every {SCAN_INTERVAL_MIN} min. Press Ctrl+C to stop.\n")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()