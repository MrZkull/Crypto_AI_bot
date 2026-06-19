# whale_tracker.py — Tier 1 Institutional Netflow Tracker

import requests, logging, time
log = logging.getLogger(__name__)

def get_exchange_netflow(symbol: str = "BTC") -> dict:
    """
    Exchange netflow: positive = inflow (selling), negative = outflow (accumulation).
    Source: Coinglass free API
    """
    try:
        r = requests.get(
            "https://open-api.coinglass.com/public/v2/indicator/exchange_net_position_change",
            params={"symbol": symbol, "time_type": "h4", "limit": 3},
            timeout=10
        )
        # Coinglass occasionally requires a free API key header depending on rate limits. 
        # If you get 401s, add headers={"coinglassSecret": "YOUR_KEY"}
        data = r.json().get("data", [])
        
        if not data:
            return {"netflow": 0, "bias": None, "score_mod": 0, "message": "No netflow data"}

        recent = float(data[-1].get("netInflow", 0) or 0)

        # >$50M inflow = whale selling (Exchange balances increasing)
        if recent > 50_000_000:     
            return {
                "netflow": recent, 
                "bias": "SELL", 
                "score_mod": 1,
                "message": f"🐋 ${recent/1e6:.0f}M exchange inflow — SELL pressure"
            }
        # >$50M outflow = whale accumulating (Exchange balances decreasing)
        elif recent < -50_000_000:  
            return {
                "netflow": recent, 
                "bias": "BUY", 
                "score_mod": 1,
                "message": f"🐋 ${abs(recent)/1e6:.0f}M exchange outflow — Accumulation"
            }
        else:
            return {
                "netflow": recent, 
                "bias": None, 
                "score_mod": 0,
                "message": f"Netflow ${recent/1e6:.1f}M — Neutral"
            }
            
    except Exception as e:
        log.warning(f"Exchange netflow check failed: {e}")
        return {"netflow": 0, "bias": None, "score_mod": 0, "message": "Netflow unavailable"}

