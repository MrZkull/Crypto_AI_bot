# config.py — V12.1: Multi-Tier Architecture (Testnet & Mainnet Scalable)

# ── File Paths ────────────────────────────────────────────────────────
RAW_DATA_FILE = "data/crypto_historical_15m_expanded.csv"
FEATURES_FILE = "data/crypto_features_15m_expanded.csv"
MODEL_FILE    = "pro_crypto_ai_model.pkl"
LOG_FILE      = "bot.log"

# ── Features List (20 Institutional Technical Features) ───────────────
FEATURES = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d',
    'adx', 'cci', 'atr', 'bb_upper', 'bb_middle', 'bb_lower',
    'ema20', 'ema50', 'sma200', 'volume_ratio', 'price_change',
    'rsi_1h', 'adx_1h', 'trend_1h'
]

# ── Coin Tiers & Scalable Allocations ─────────────────────────────────

# 👑 Tier 1: Majors & Benchmarks
# ETH unlocked (0.0001 min lot); BTC & BNB preserved for ₹5,000+ capital
TIER_BIG3 = [
    "ETHUSDT",
    # "BTCUSDT",  # 🔒 UNCOMMENT WHEN CAPITAL >= ₹5,000 ($58+) (Min Lot: 0.0001 BTC ≈ $6.30)
    # "BNBUSDT",  # 🔒 UNCOMMENT WHEN CAPITAL >= ₹5,000 (Min Lot: 0.001 BNB)
]

# 🏆 Tier 2: Core Proven Testnet Winners (High Win Rate + Granular)
TIER_PROVEN = [
    "XRPUSDT",   # Testnet: +$371.69 (75% WR)
    "ALGOUSDT",  # Testnet: +$96.14 (100% WR)
    "NEARUSDT",  # Testnet: +$158.93 (80% WR)
    "DOTUSDT",   # Testnet: +$95.67 (80% WR)
    "LTCUSDT",   # Testnet: +$119.20 (75% WR)
    "UNIUSDT",   # Testnet: +$89.26 (67% WR)
]

# 💎 Tier 3: Micro-Granular Gems (Sub-Dollar Min Lots on Deribit)
TIER_MICRO_GEMS = [
    "ENAUSDT",   # Deribit Linear Perp (Min Lot: 1.0 ENA ≈ $0.08)
    "AAVEUSDT",  # Deribit Linear Perp (Min Lot: 0.01 AAVE ≈ $0.86)
    "HYPEUSDT",  # Deribit Linear Perp (Min Lot: 0.01 HYPE ≈ $0.20)
    "DOGEUSDT",  # Deribit Linear Perp (Min Lot: 1.0 DOGE ≈ $0.08)
]

# 🏛 Tier 4: Deep Liquidity Altcoins
TIER_MAJORS_ALTS = [
    "LINKUSDT",  # Deribit Linear Perp (Min Lot: 0.01 LINK ≈ $0.09)
    "AVAXUSDT",  # Deribit Linear Perp (Min Lot: 0.001 AVAX ≈ $0.02)
    "BCHUSDT",   # Deribit Linear Perp (Min Lot: 0.001 BCH ≈ $0.20)
]

# ⚠️ Tier 5: Preserved Altcoins (Commented for ₹500, easily enabled later)
TIER_EXPANSION_RESERVE = [
    # "SOLUSDT",   # 🔒 High intraday whipsaws on 15m; re-test when Capital >= ₹5,000
    # "ADAUSDT",   # 🔒 Low testnet momentum (25% WR); re-test when Capital >= ₹5,000
    # "TRXUSDT",   # 🔒 Flat volatility / spread friction; re-test when Capital >= ₹5,000
    # "SUIUSDT",   # 🔒 Preserved for future Deribit listing additions
    # "APTUSDT",   # 🔒 Preserved for future Deribit listing additions
    # "ATOMUSDT",  # 🔒 Preserved for future Deribit listing additions
    # "FETUSDT",   # 🔒 Preserved for future Deribit listing additions
    # "RENDERUSDT",# 🔒 Preserved for future Deribit listing additions
]

# Active Universe for Testnet & Phase-1 Mainnet (14 Coins)
SYMBOLS = TIER_BIG3 + TIER_PROVEN + TIER_MICRO_GEMS + TIER_MAJORS_ALTS

COIN_TIERS = {
    "big3":       {"label": "👑 Majors",                 "coins": TIER_BIG3},
    "proven":     {"label": "🏆 Proven Winners",         "coins": TIER_PROVEN},
    "micro_gems": {"label": "💎 Micro-Granular Gems",     "coins": TIER_MICRO_GEMS},
    "majors_alt": {"label": "🏛 Deep Liquidity Alts",     "coins": TIER_MAJORS_ALTS},
    "reserve":    {"label": "🔒 Scaling Reserve (₹5k+)", "coins": TIER_EXPANSION_RESERVE},
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

# ── AI & Strategy Filters ─────────────────────────────────────────────
MIN_CONFIDENCE    = 55.0  # Base AI confidence threshold
MIN_ADX           = 20.0  # Minimum trend strength
MIN_SCORE         = 2     # Minimum quality score (out of 6)

# ── Risk Management Parameters ────────────────────────────────────────
RISK_PER_TRADE     = 0.03   # 3.0% of equity (~$0.174 / ₹15 on ₹500 capital)
MAX_OPEN_TRADES    = 2      # Max 2 concurrent positions (preserves micro-capital margin)
MAX_SAME_DIRECTION = 2      # Max 2 BUY or 2 SELL
ATR_STOP_MULT      = 2.5    # SL = entry ± 2.5 × ATR
ATR_TARGET1_MULT   = 3.5    # TP1 = entry ± 3.5 × ATR (50% position exit)
ATR_TARGET2_MULT   = 7.5    # TP2 = entry ± 7.5 × ATR (50% runner exit)

# ── Stale Position Circuit Breaker ────────────────────────────────────
MAX_TRADE_AGE_HOURS = 48    # Auto-flatten positions stagnant for > 48h
