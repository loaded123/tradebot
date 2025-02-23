# src/strategy/indicators.py

import pandas as pd
import pandas_ta as ta
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def compute_vwap(df):
    """Compute Volume Weighted Average Price (VWAP) using pandas-ta."""
    try:
        return ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    except Exception as e:
        logging.error(f"VWAP computation failed: {e}")
        return pd.Series([0] * len(df), index=df.index)

def compute_adx(df, period=20):
    """Compute Average Directional Index (ADX) using pandas-ta."""
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=period)
        return adx_df[f'ADX_{period}']
    except Exception as e:
        logging.error(f"ADX computation failed: {e}")
        return pd.Series([0] * len(df), index=df.index)

# Placeholder for additional indicators
def compute_bollinger_bands(df, period=20, std_dev=2):
    """Compute Bollinger Bands (future enhancement)."""
    try:
        bbands = ta.bbands(df['close'], length=period, std=std_dev)
        return bbands  # Returns 'BBL_<period>_<std>', 'BBM_<period>_<std>', 'BBU_<period>_<std>'
    except Exception as e:
        logging.error(f"Bollinger Bands computation failed: {e}")
        return pd.DataFrame(index=df.index)
    
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range (ATR)."""
    tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
    return tr.rolling(window=window).mean()

if __name__ == "__main__":
    # Test with dummy data
    dummy_df = pd.DataFrame({
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range("2024-01-01", periods=5, freq="H"))
    
    logging.info(f"VWAP:\n{compute_vwap(dummy_df)}")
    logging.info(f"ADX:\n{compute_adx(dummy_df, period=5)}")