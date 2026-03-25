# news_sentiment.py
# Fetches crypto news and scores it as positive/negative
# Uses free CryptoPanic API — sign up at cryptopanic.com/api

import requests
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(".env"), override=True)

CRYPTOPANIC_KEY = os.getenv("CRYPTOPANIC_KEY", "")

# Positive words = price likely to go UP
BULLISH_WORDS = [
    "surge", "rally", "bullish", "breakout", "adoption",
    "partnership", "upgrade", "launch", "record", "high",
    "buy", "growth", "approval", "etf", "institutional",
    "moon", "pump", "rise", "gains", "positive", "boost"
]

# Negative words = price likely to go DOWN
BEARISH_WORDS = [
    "crash", "dump", "bearish", "ban", "hack", "lawsuit",
    "sell", "drop", "fear", "loss", "regulation", "fraud",
    "scam", "war", "crisis", "inflation", "recession",
    "bubble", "collapse", "fine", "investigation", "decline"
]


def get_news_sentiment(coin_symbol: str) -> dict:
    """
    Returns sentiment score for a coin based on latest news.
    Score:  > 0 = bullish  |  < 0 = bearish  |  0 = neutral
    """
    coin = coin_symbol.replace("USDT", "").lower()

    # Try CryptoPanic API first (best source)
    if CRYPTOPANIC_KEY:
        try:
            url    = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": CRYPTOPANIC_KEY,
                "currencies": coin.upper(),
                "filter":     "hot",
                "public":     "true",
            }
            resp  = requests.get(url, params=params, timeout=10)
            data  = resp.json()
            posts = data.get("results", [])

            score = 0
            for post in posts[:10]:  # check latest 10 news items
                title = post.get("title", "").lower()
                for word in BULLISH_WORDS:
                    if word in title:
                        score += 1
                for word in BEARISH_WORDS:
                    if word in title:
                        score -= 1

            return {
                "coin":      coin.upper(),
                "score":     score,
                "sentiment": "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL",
                "source":    "CryptoPanic",
                "articles":  len(posts),
            }
        except Exception as e:
            print(f"  CryptoPanic error: {e}")

    # Fallback — use free RSS news scan
    try:
        url  = f"https://cryptopanic.com/news/{coin}/"
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
        text = resp.text.lower()

        bull = sum(text.count(w) for w in BULLISH_WORDS)
        bear = sum(text.count(w) for w in BEARISH_WORDS)
        score = bull - bear

        return {
            "coin":      coin.upper(),
            "score":     score,
            "sentiment": "BULLISH" if score > 2 else "BEARISH" if score < -2 else "NEUTRAL",
            "source":    "Web scan",
        }
    except Exception as e:
        return {"coin": coin.upper(), "score": 0, "sentiment": "NEUTRAL", "source": "Error"}


def get_global_sentiment() -> dict:
    """
    Checks global market conditions — Fear & Greed Index.
    Score 0-100:  0=Extreme Fear  50=Neutral  100=Extreme Greed
    Best to BUY when score < 25 (extreme fear = opportunity)
    Avoid buying when score > 75 (extreme greed = risky)
    """
    try:
        url  = "https://api.alternative.me/fng/"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        val  = int(data["data"][0]["value"])
        txt  = data["data"][0]["value_classification"]

        # War/crisis detection from description
        warning = ""
        if val < 20:
            warning = "EXTREME FEAR — possible buying opportunity"
        elif val > 80:
            warning = "EXTREME GREED — high risk, consider reducing size"

        return {
            "score":   val,
            "label":   txt,
            "warning": warning,
            "action":  "BUY_ZONE" if val < 30 else "SELL_ZONE" if val > 75 else "NEUTRAL",
        }
    except Exception as e:
        return {"score": 50, "label": "Unknown", "warning": "", "action": "NEUTRAL"}


def get_market_conditions() -> dict:
    """
    Combines Fear & Greed + BTC dominance to judge overall market.
    """
    fg    = get_global_sentiment()
    score = fg["score"]

    # Market condition based on fear/greed
    if score < 25:
        condition = "CRISIS / EXTREME FEAR"
        advice    = "Market is very fearful — historically good time to buy"
    elif score < 45:
        condition = "BEARISH / FEAR"
        advice    = "Market is fearful — be cautious, wait for confirmation"
    elif score < 55:
        condition = "NEUTRAL"
        advice    = "Market is balanced — follow individual coin signals"
    elif score < 75:
        condition = "BULLISH / GREED"
        advice    = "Market is greedy — good for riding trends, use tight SL"
    else:
        condition = "EXTREME GREED"
        advice    = "Market is extremely greedy — high risk of reversal, reduce size"

    return {
        "fear_greed":  score,
        "label":       fg["label"],
        "condition":   condition,
        "advice":      advice,
        "trade_ok":    25 < score < 75,  # safest trading zone
    }


if __name__ == "__main__":
    print("\n🌍 Global Market Conditions")
    print("─" * 40)
    market = get_market_conditions()
    print(f"Fear & Greed Index: {market['fear_greed']} — {market['label']}")
    print(f"Condition: {market['condition']}")
    print(f"Advice: {market['advice']}")
    print(f"Safe to trade: {'YES' if market['trade_ok'] else 'CAUTION'}")

    print("\n📰 Coin News Sentiment")
    print("─" * 40)
    for coin in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "SUIUSDT"]:
        s = get_news_sentiment(coin)
        print(f"{s['coin']:6s}  {s['sentiment']:8s}  score: {s['score']:+d}")
