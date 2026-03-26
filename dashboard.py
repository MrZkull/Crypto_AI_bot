import streamlit as st
import pandas as pd
import requests
import json
import time
from datetime import datetime
import plotly.graph_objects as go

# ─── CONFIGURATION ───
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/Elliot14R/Crypto_AI_bot/main"
st.set_page_config(page_title="CryptoBot AI Pro", page_icon="🤖", layout="wide")

# ─── PRO TERMINAL CSS ───
st.markdown("""
    <style>
    .stApp { background-color: #0D1117; color: #C9D1D9; }
    /* Replit-style Signal Rows */
    .signal-row {
        background: #161B22; border: 1px solid #30363D; border-radius: 6px;
        padding: 12px 20px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;
    }
    .badge { padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    .buy-bg { color: #3FB950; background: rgba(63, 185, 80, 0.1); border: 1px solid rgba(63, 185, 80, 0.3); }
    .sell-bg { color: #F85149; background: rgba(248, 81, 73, 0.1); border: 1px solid rgba(248, 81, 73, 0.3); }
    .star-gold { color: #E3B341; }
    .metric-box { background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px; text-align: left; }
    .price-card { background: #161B22; border: 1px solid #30363D; border-radius: 10px; padding: 12px; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ─── DATA ENGINES ───
def fetch_github_data(filename):
    try:
        r = requests.get(f"{GITHUB_RAW_BASE}/{filename}")
        return r.json() if r.status_code == 200 else {}
    except: return {}

def get_market_data():
    try:
        r = requests.get("https://api.binance.us/api/v3/ticker/24hr").json()
        return pd.DataFrame(r)
    except: return pd.DataFrame()

# ─── SIDEBAR (REPLICATING REPLIT NAVIGATION) ───
with st.sidebar:
    st.markdown("### 🤖 CryptoBot AI")
    menu = st.radio("Navigation", ["Dashboard", "Signals", "Market", "Paper Trading", "Configuration"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("MONITORED COINS")
    symbols = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "SUI", "DOT", "LINK", "MATIC", "NEAR", "APT", "ARB"]
    st.markdown("".join([f"<span style='background:#21262d; padding:2px 5px; margin:2px; border-radius:4px; font-size:10px; border:1px solid #30363d;'>{s}</span>" for s in symbols]), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("🟢 **LIVE** &nbsp; <span style='font-size:10px; color:#8B949E;'>00:15:59</span>", unsafe_allow_html=True)

# ─── PAGE: DASHBOARD ───
if menu == "Dashboard":
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Signals", "10", "All time")
    m2.metric("Buy Signals", "6", "Long positions")
    m3.metric("Sell Signals", "4", "Short positions")
    m4.metric("Win Rate", "70%", "Verified signals")

    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("📈 Recent Signals")
        # Header Row for Signals
        st.markdown("<div style='display:flex; color:#8B949E; font-size:12px; margin-bottom:5px; padding:0 20px;'><span>Asset</span><span style='margin-left:80px;'>Entry</span><span style='margin-left:120px;'>Stop Loss</span><span style='margin-left:120px;'>Target 1</span></div>", unsafe_allow_html=True)
        
        # Display Signal Row (Mocking your Replit UI logic)
        mock_data = [{"sym": "BTC", "type": "BUY", "conf": 78.4, "price": 83420.50, "sl": 81850.00, "tp": 85810.50, "rr": "1:1.5"}]
        for s in mock_data:
            st.markdown(f"""
            <div class="signal-row">
                <div style="width:120px;"><span class="badge buy-bg">↗ {s['type']}</span> <b>{s['sym']}</b> <small style='color:#8B949E'>{s['conf']}%</small></div>
                <div style="width:100px;">{s['price']:,}</div>
                <div style="width:100px; color:#F85149;">{s['sl']:,}</div>
                <div style="width:100px; color:#3FB950;">{s['tp']:,}</div>
                <div style="width:80px;" class="star-gold">⭐⭐⭐⭐⭐</div>
            </div>
            """, unsafe_allow_html=True)

    with col_side:
        st.subheader("🔥 Market Sentiment")
        st.markdown("<div style='background:#161B22; border:1px solid #30363D; border-radius:10px; padding:20px; text-align:center;'><h1 style='color:#F85149; margin:0;'>10</h1><p style='color:#F85149; font-weight:bold;'>EXTREME FEAR</p><small style='color:#8B949E;'>Historically a buying opportunity</small></div>", unsafe_allow_html=True)

# ─── PAGE: SIGNALS (HISTORY & FILTERS) ───
elif menu == "Signals":
    st.header("⚡ Signal History")
    f1, f2, f3 = st.columns(3)
    f1.multiselect("Filter Symbol", symbols, default=["BTC", "ETH"])
    f2.selectbox("Type", ["All", "BUY", "SELL"])
    f3.slider("Min Confidence", 60, 100, 75)
    
    # Signal History Table (Replicating your Signal View)
    st.markdown("---")
    st.info("AI-generated trading signals from 15m scanner")
    # Using a list of rows to replicate the "Star" rating view
    st.table(pd.DataFrame({
        "Time": ["1h ago", "2h ago"], "Asset": ["BTC", "ETH"], "Signal": ["BUY", "BUY"], "Conf": ["78.4%", "71.2%"], "R:R": ["1:1.5", "1:2.0"], "Rank": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐"]
    }))

# ─── PAGE: MARKET (TILES & GAINERS) ───
elif menu == "Market":
    st.header("🌐 Market Overview")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Gainers", "0", "of 15 pairs")
    t2.metric("Losers", "15", "of 15 pairs", delta_color="inverse")
    t3.metric("Best 24h", "BTC", "-2.78%")
    t4.metric("Worst 24h", "ADA", "-5.6%")

    df_m = get_market_data()
    if not df_m.empty:
        df_m = df_m[df_m['symbol'].isin([s+"USDT" for s in symbols])]
        cols = st.columns(3)
        for i, row in enumerate(df_m.head(6).to_dict('records')):
            with cols[i % 3]:
                change = float(row['priceChangePercent'])
                st.markdown(f"""
                <div class="price-card">
                    <div style="display:flex; justify-content:space-between;">
                        <b>{row['symbol'].replace('USDT','')}</b>
                        <span style="color:{'#3FB950' if change > 0 else '#F85149'}">{change}%</span>
                    </div>
                    <div style="font-size:20px; font-weight:bold; margin:10px 0;">${float(row['lastPrice']):,}</div>
                    <div style="color:#8B949E; font-size:11px;">Vol: {float(row['volume']):,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

# ─── PAGE: PAPER TRADING (EQUITY & TRADES) ───
elif menu == "Paper Trading":
    st.header("📉 Paper Trading")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Virtual Balance", "$10,000.00", "Started at $1,000")
    p2.metric("Total P&L", "+$0.00", "0.00% return")
    p3.metric("Win Rate", "0%", "0W / 0L")
    p4.metric("Total Trades", "0", "0 open")

    # Equity Curve Mockup
    fig = go.Figure(go.Scatter(x=[1,2,3,4], y=[1000, 1050, 1020, 1100], line=dict(color='#3FB950', width=3)))
    fig.update_layout(title="Equity Curve", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#8B949E", height=300)
    st.plotly_chart(fig, use_container_width=True)

# ─── PAGE: CONFIGURATION (FULL REPLIT SPECS) ───
elif menu == "Configuration":
    st.header("⚙️ Bot Configuration")
    
    with st.expander("🧠 AI Ensemble Model", expanded=True):
        st.write("Trained on 6 months of 15-m OHLCV data. Ensemble: LightGBM (3x), XGBoost (2x), Random Forest (1x).")
        st.markdown("<span class='badge' style='background:#238636'>Active Configuration</span>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🕒 Scanner Settings")
        st.json({"Entry TF": "15m", "Confirm TF": "1h", "Trend TF": "4h", "Scan Interval": "15 min"})
        st.subheader("🛡️ Risk Management")
        st.json({"Stop Multiplier": "1.5x ATR", "Target 1": "2x ATR", "Target 2": "3x ATR", "Risk/Trade": "1%"})
    
    with c2:
        st.subheader("🎯 Signal Filters")
        st.json({"Min Confidence": "75%", "Min ADX": 25, "Min Score": "3 / 6"})
        st.subheader("🧬 Feature Engineering")
        st.write("14 groups • 50 total features (EMA Slopes, RSI Momentum, MACD, ATR Volatility, etc.)")
