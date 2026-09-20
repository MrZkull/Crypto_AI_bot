"""Fail-closed portfolio risk admission helpers. These do not model Deribit liquidation."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite

@dataclass(frozen=True)
class RiskLimits:
    max_open: int = 8
    max_same_direction: int = 4
    max_sector: int = 3
    max_gross_notional_pct: float = 0.50
    max_trade_risk_pct: float = 0.03

SECTORS = {
    "Majors":{"ETHUSDT","BNBUSDT","SOLUSDT"},
    "Proven_L1_L2":{"XRPUSDT","NEARUSDT","LTCUSDT","BCHUSDT","DOTUSDT","ALGOUSDT","ADAUSDT","TRXUSDT","XLMUSDT"},
    "DeFi_Infrastructure":{"UNIUSDT","AAVEUSDT","LINKUSDT","CRVUSDT"},
    "High_Beta_Ecosystem":{"ENAUSDT","DOGEUSDT","TRUMPUSDT","PUMPUSDT","SUIUSDT","AVAXUSDT","ZECUSDT","HBARUSDT","FILUSDT"},
}

def sector(symbol):
    for name, coins in SECTORS.items():
        if symbol in coins: return name
    return "OTHER"

def admit(open_trades:list[dict], symbol:str, side:str, balance:float, notional:float, risk_usd:float, limits=RiskLimits()) -> tuple[bool,str]:
    if not all(isfinite(float(x)) for x in (balance,notional,risk_usd)) or balance<=0 or notional<0 or risk_usd<0:
        return False,"INVALID_RISK_INPUT"
    live=[t for t in open_trades if not t.get("closed",False)]
    if len(live)>=limits.max_open:return False,"MAX_OPEN_TRADES"
    if sum(t.get("signal")==side for t in live)>=limits.max_same_direction:return False,"MAX_SAME_DIRECTION"
    if any(t.get("symbol")==symbol for t in live):return False,"SYMBOL_ALREADY_OPEN"
    if sum(1 for t in live if sector(t.get("symbol",""))==sector(symbol))>=limits.max_sector:return False,"MAX_SECTOR"
    if risk_usd > balance*limits.max_trade_risk_pct*(1+1e-12):return False,"MAX_TRADE_RISK"
    if notional > balance*limits.max_gross_notional_pct*(1+1e-12):return False,"MAX_GROSS_NOTIONAL"
    return True,"OK"
