# config.py — V12.4: Multi-Tier Architecture (Active + Tier-Categorized Reserves)

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

# ── Coin Tiers & Allocations ──────────────────────────────────────────

# 👑 Tier 1: Majors & Benchmarks (Min Lot < $1.00; BTC preserved for Capital >= $100)
TIER_BIG3 = [
    "ETHUSDT",   # Deribit Min Lot: 0.0001 ETH ≈ $0.25 (Vol: $773M)
    "BNBUSDT",   # Deribit Min Lot: 0.001 BNB  ≈ $0.70 (Vol: $121M)
    "SOLUSDT",   # Deribit Min Lot: 0.001 SOL  ≈ $0.10 (Vol: $363M)
    # "BTCUSDT", # 🔒 Min Lot: 0.0001 BTC ≈ $7.75 (Enable when Capital >= $100 / ₹8,500)
]

# 🏆 Tier 2: Core Proven Winners (High Win Rate + Clean Execution)
TIER_PROVEN = [
    "XRPUSDT",   # Deribit Min Lot: 1.0 XRP   ≈ $1.52 (Vol: $397M)
    "NEARUSDT",  # Deribit Min Lot: 0.1 NEAR  ≈ $0.20 (Vol: $48M)
    "LTCUSDT",   # Deribit Min Lot: 0.01 LTC  ≈ $0.53 (Vol: $23M)
    "UNIUSDT",   # Deribit Min Lot: 0.01 UNI  ≈ $0.05 (Vol: $41M)
    "BCHUSDT",   # Deribit Min Lot: 0.001 BCH ≈ $0.28 (Vol: $14M)
    "DOTUSDT",   # Deribit Min Lot: 0.1 DOT   ≈ $0.09 (Vol: $6.5M)
    "ALGOUSDT",  # Deribit Min Lot: 1.0 ALGO  ≈ $0.09 (Vol: $2.3M)
]

# 💎 Tier 3: Micro-Granular Gems (Sub-Dime Min Lots on Deribit | Perfect Dual TP)
TIER_MICRO_GEMS = [
    "ENAUSDT",   # Deribit Min Lot: 1.0 ENA     ≈ $0.17 (Vol: $98M)
    "DOGEUSDT",  # Deribit Min Lot: 1.0 DOGE    ≈ $0.09 (Vol: $114M)
    "TRUMPUSDT", # Deribit Min Lot: 0.01 TRUMP  ≈ $0.03 (Vol: $124M)
    "PUMPUSDT",  # Deribit Min Lot: 1.0 PUMP    ≈ $0.01 (Vol: $92M)
    "AAVEUSDT",  # Deribit Min Lot: 0.01 AAVE   ≈ $1.41 (Vol: $39M)
    # "HYPEUSDT", # 🔒 Preserved for re-evaluation once Binance spot candle feed is live
]

# 🏛 Tier 4: Deep Liquidity Altcoins (Vol > $20M)
TIER_MAJORS_ALTS = [
    "LINKUSDT",  # Deribit Min Lot: 0.01 LINK   ≈ $0.12 (Vol: $55M)
    "SUIUSDT",   # Deribit Min Lot: 0.1 SUI     ≈ $0.08 (Vol: $69M)
    "AVAXUSDT",  # Deribit Min Lot: 0.001 AVAX  ≈ $0.01 (Vol: $20M)
    "ADAUSDT",   # Deribit Min Lot: 1.0 ADA     ≈ $0.23 (Vol: $38M)
    "TRXUSDT",   # Deribit Min Lot: 1.0 TRX     ≈ $0.34 (Vol: $26M)
    # "APTUSDT",  # 🔒 Low Historical Precision (33% WR)
    # "ATOMUSDT", # 🔒 Low Trend Strength (<15 ADX)
    # "FETUSDT",  # 🔒 High Spread / Chop Friction
]

# 🧪 Tier 5: Testnet Incubation Lab (New High-Potential Additions)
TIER_TEST_LAB = [
    "ZECUSDT",    # High Volume Privacy/L1 ($263M | Min Lot $0.85)
    "TAOUSDT",    # AI Momentum Benchmark ($22M | Min Lot $0.23)
    "XLMUSDT",    # Payment Sector Sync ($27M | Min Lot $0.20)
    "HBARUSDT",   # Enterprise L1 Micro Lot ($11M | Min Lot $0.08)
    "PENDLEUSDT", # High-Beta DeFi Swings ($6M | Min Lot $0.20)
    "WIFUSDT",    # Solana Meme High Volatility ($3.3M | Min Lot $0.02)
    "CRVUSDT",    # DeFi Mean-Reversion ($7M | Min Lot $0.34)
    "RENDERUSDT", # DePIN / GPU Narrative ($4.2M | Min Lot $0.15)
    # "PAXGUSDT", # 🔒 Min Lot: 0.0001 PAXG ≈ $0.46 (Gold Peg / Low Volatility)
    # "FILUSDT",  # 🔒 Deribit Min Lot: 0.1 FIL ≈ $0.08 (Vol: $5.9M)
    # "JUPUSDT",  # 🔒 Deribit Min Lot: 0.1 JUP ≈ $0.02 (Vol: $2.1M)
]

# Active Universe (28 Total Pairs for Testnet Screening)
SYMBOLS = TIER_BIG3 + TIER_PROVEN + TIER_MICRO_GEMS + TIER_MAJORS_ALTS + TIER_TEST_LAB

COIN_TIERS = {
    "big3":       {"label": "👑 Majors",               "coins": TIER_BIG3},
    "proven":     {"label": "🏆 Proven Winners",       "coins": TIER_PROVEN},
    "micro_gems": {"label": "💎 Micro-Granular Gems",   "coins": TIER_MICRO_GEMS},
    "majors_alt": {"label": "🏛 Deep Liquidity Alts",   "coins": TIER_MAJORS_ALTS},
    "test_lab":   {"label": "🧪 Testnet Lab",          "coins": TIER_TEST_LAB},
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
MIN_CONFIDENCE    = 45.0
MIN_ADX           = 15.0
MIN_SCORE         = 3

# ── Risk Management Parameters ────────────────────────────────────────
RISK_PER_TRADE     = 0.03   # 3.0% of equity
MAX_OPEN_TRADES    = 2      # Max 2 concurrent positions
MAX_SAME_DIRECTION = 2      # Max 2 BUY or 2 SELL
ATR_STOP_MULT      = 2.5    # SL = entry ± 2.5 × ATR
ATR_TARGET1_MULT   = 3.5    # TP1 = entry ± 3.5 × ATR (50% position exit)
ATR_TARGET2_MULT   = 7.5    # TP2 = entry ± 7.5 × ATR (50% runner exit)

# ── Stale Position Circuit Breaker ────────────────────────────────────
MAX_TRADE_AGE_HOURS = 48
