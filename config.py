# config.py — V12.2: Multi-Tier Architecture (Testnet & Mainnet Scalable)

# ── File Paths ────────────────────────────────────────────────────────
RAW_DATA_FILE = "data/crypto_historical_15m_expanded.csv"
FEATURES_FILE = "data/crypto_features_15m_expanded.csv"
MODEL_FILE    = "pro_crypto_ai_model.pkl"
LOG_FILE      = "bot.log"

# ── Features List ─────────────────────────────────────────────────────
FEATURES = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d',
    'adx', 'cci', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
    'ema20', 'ema50', 'sma200', 'volume_ratio', 'price_change',
    'rsi_1h', 'adx_1h', 'trend_1h'
]

# ── Coin Tiers & Allocations ──────────────────────────────────────────

# 👑 Tier 1: Majors & Benchmarks
TIER_BIG3 = [
    "ETHUSDT",
    "BTCUSDT",
    "BNBUSDT",
]

# 🏆 Tier 2: Core Proven Winners
TIER_PROVEN = [
    "XRPUSDT",
    "ALGOUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "BCHUSDT",
]

# 💎 Tier 3: Micro-Granular Gems (Sub-Dollar Min Lots on Deribit)
TIER_MICRO_GEMS = [
    "ENAUSDT",
    "AAVEUSDT",
    "HYPEUSDT",
    "DOGEUSDT",
]

# 🏛 Tier 4: Deep Liquidity Altcoins
TIER_MAJORS_ALTS = [
    "LINKUSDT",
    "AVAXUSDT",
    "SOLUSDT",
    "ADAUSDT",
]

# ⚠️ Tier 5: Scaling Reserve
TIER_EXPANSION_RESERVE = [
    "TRXUSDT",
    "SUIUSDT",
    "APTUSDT",
    "ATOMUSDT",
    "FETUSDT",
    "RENDERUSDT",
]

# Active Universe
SYMBOLS = TIER_BIG3 + TIER_PROVEN + TIER_MICRO_GEMS + TIER_MAJORS_ALTS

COIN_TIERS = {
    "big3":       {"label": "👑 Majors",               "coins": TIER_BIG3},
    "proven":     {"label": "🏆 Proven Winners",       "coins": TIER_PROVEN},
    "micro_gems": {"label": "💎 Micro-Granular Gems",   "coins": TIER_MICRO_GEMS},
    "majors_alt": {"label": "🏛 Deep Liquidity Alts",   "coins": TIER_MAJORS_ALTS},
    "reserve":    {"label": "🔒 Scaling Reserve",      "coins": TIER_EXPANSION_RESERVE},
}

def get_tier(symbol: str) -> str:
    for t in COIN_TIERS.values():
        if symbol in t["coins"]:
            return t["label"]
    return "Unknown"

# ── Timeframes & Scheduling ───────────────────────────────────────────
TIMEFRAME_ENTRY   = "15m"
TIMEFRAME_CONFIRM = "1h"
TIMEFRAME_TREND   = "4h"
LIVE_LIMIT        = 300
SCAN_INTERVAL_MIN = 15

# ── Strategy & Filter Baselines ───────────────────────────────────────
MIN_CONFIDENCE    = 40.0  # Fallback baseline if model has no threshold
MIN_ADX           = 15.0  # Minimum trend strength
MIN_SCORE         = 3     # Minimum score (out of 6)

# ── Risk Management Parameters ────────────────────────────────────────
RISK_PER_TRADE     = 0.03   # 3.0% of equity
MAX_OPEN_TRADES    = 2      # Max 2 concurrent positions
MAX_SAME_DIRECTION = 2      # Max 2 BUY or 2 SELL
ATR_STOP_MULT      = 2.5    # SL = entry ± 2.5 × ATR
ATR_TARGET1_MULT   = 3.5    # TP1 = entry ± 3.5 × ATR (50% exit)
ATR_TARGET2_MULT   = 7.5    # TP2 = entry ± 7.5 × ATR (50% runner)

# ── Stale Position Circuit Breaker ────────────────────────────────────
MAX_TRADE_AGE_HOURS = 48
