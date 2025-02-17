import pandas as pd
import numpy as np

def compute_vwap(df):
    """
    Compute the Volume Weighted Average Price (VWAP)
    """
    vwap = df['close'].copy()
    vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    return vwap

def compute_adx(df, period=14):
    """
    Compute the Average Directional Index (ADX)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range calculation
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1)
    tr = tr.max(axis=1)

    # Directional movement
    plus_dm = high.diff()
    minus_dm = low.diff()

    # Smoothing
    tr_smooth = tr.rolling(window=period).sum()
    plus_dm_smooth = plus_dm.rolling(window=period).sum()
    minus_dm_smooth = minus_dm.rolling(window=period).sum()

    # Directional indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    # ADX calculation
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    return adx
