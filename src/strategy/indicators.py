import pandas as pd
import ta

def compute_vwap(df):
    """Compute the Volume Weighted Average Price (VWAP)"""
    vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    return vwap

def compute_adx(df, period=14):
    """Compute the Average Directional Index (ADX)"""
    adx = ta.ADX(df['high'], df['low'], df['close'], timeperiod=period)
    return adx