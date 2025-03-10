# src/strategy/indicators.py
import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
import requests
from textblob import TextBlob

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def compute_vwap(df):
    """Compute Volume Weighted Average Price (VWAP) using pandas-ta with fallback."""
    try:
        return ta.vwap(df['high'], df['low'], df['close'], df['volume']).fillna(78877.88)  # Default BTC price
    except Exception as e:
        logging.error(f"VWAP computation failed: {e}")
        return pd.Series([78877.88] * len(df), index=df.index)

def compute_adx(df, period=20):
    """Compute Average Directional Index (ADX) using pandas-ta with fallback."""
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=period)
        return adx_df[f'ADX_{period}'].fillna(10.0)  # Default ADX value
    except Exception as e:
        logging.error(f"ADX computation failed: {e}")
        return pd.Series([10.0] * len(df), index=df.index)

def compute_bollinger_bands(df, period=20, std_dev=2):
    """Compute Bollinger Bands and breakout signals for high-frequency trading."""
    try:
        bbands = ta.bbands(df['close'], length=period, std=std_dev)
        df['bb_upper'] = bbands[f'BBU_{period}_{std_dev}'].fillna(method='bfill')
        df['bb_middle'] = bbands[f'BBM_{period}_{std_dev}'].fillna(method='bfill')
        df['bb_lower'] = bbands[f'BBL_{period}_{std_dev}'].fillna(method='bfill')
        
        # Generate breakout signals (e.g., price crossing upper/lower bands)
        df['bb_breakout'] = 0
        df.loc[df['close'] > df['bb_upper'], 'bb_breakout'] = 1  # Bullish breakout
        df.loc[df['close'] < df['bb_lower'], 'bb_breakout'] = -1  # Bearish breakout
        return df[['bb_upper', 'bb_middle', 'bb_lower', 'bb_breakout']]
    except Exception as e:
        logging.error(f"Bollinger Bands computation failed: {e}")
        return pd.DataFrame({'bb_upper': [79367.5] * len(df), 'bb_middle': [78877.88] * len(df), 
                            'bb_lower': [78186.98] * len(df), 'bb_breakout': [0] * len(df)}, index=df.index)

def calculate_rsi(prices, window=14, volatility_threshold=0.02):
    """Calculate RSI with dynamic window based on volatility."""
    price_volatility = prices.pct_change().rolling(window=24).std().fillna(0.0)  # 24-hour rolling std
    dynamic_window = window
    if price_volatility.iloc[-1] > volatility_threshold:
        dynamic_window = max(5, int(window * 0.5))  # Faster response in high volatility
    elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
        dynamic_window = min(30, int(window * 1.5))  # Slower response in low volatility
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=dynamic_window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=dynamic_window, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    return pd.Series(100 - (100 / (1 + rs)), index=prices.index).fillna(50.0)  # Default RSI

def calculate_macd(prices, fast=12, slow=26, signal=9, volatility_threshold=0.02):
    """Calculate MACD with dynamic periods based on volatility."""
    price_volatility = prices.pct_change().rolling(window=24).std().fillna(0.0)
    dynamic_fast = fast
    dynamic_slow = slow
    dynamic_signal = signal
    if price_volatility.iloc[-1] > volatility_threshold:
        dynamic_fast = max(6, int(fast * 0.5))  # Faster response
        dynamic_slow = max(12, int(slow * 0.5))
        dynamic_signal = max(5, int(signal * 0.5))
    elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
        dynamic_fast = min(24, int(fast * 1.5))
        dynamic_slow = min(52, int(slow * 1.5))
        dynamic_signal = min(18, int(signal * 1.5))
    
    exp1 = prices.ewm(span=dynamic_fast, adjust=False).mean()
    exp2 = prices.ewm(span=dynamic_slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=dynamic_signal, adjust=False).mean()
    return macd.fillna(0.0), signal_line.fillna(0.0)  # Default values

def calculate_atr(high, low, close, period=14, volatility_threshold=0.02, min_periods: int = 1):
    """Calculate Average True Range (ATR) with dynamic window size based on volatility and minimum periods."""
    price_volatility = close.pct_change().rolling(window=24).std().fillna(0.0)  # 24-hour rolling std
    dynamic_window = period
    if price_volatility.iloc[-1] > volatility_threshold:
        dynamic_window = max(5, int(period * 0.5))  # Reduce window in high volatility
    elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
        dynamic_window = min(30, int(period * 1.5))  # Increase window in low volatility
    
    tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
    atr = tr.rolling(window=dynamic_window, min_periods=min_periods).mean().bfill().fillna(500.0)  # Use min_periods and backfill
    return atr

def calculate_sentiment_score(symbol="BTC/USD", source="twitter"):
    """Calculate market sentiment score from Twitter/X or news (simulated with fallback)."""
    try:
        # Mock API call (replace with actual API like Twitter/X or Reddit)
        # Note: Free X API tier does not support tweet search; using simulated data
        response = {"tweets": [{"text": "Bitcoin is rising!"} for _ in range(10)] + [{"text": "Bitcoin is falling!"} for _ in range(5)]}
        tweets = response.get('tweets', [])
        sentiment_scores = [TextBlob(tweet['text']).sentiment.polarity for tweet in tweets]
        return np.mean(sentiment_scores) if sentiment_scores else 0.0
    except Exception as e:
        logging.error(f"Sentiment score computation failed: {e}")
        return 0.0

def get_onchain_metrics(symbol="BTC"):
    """Fetch on-chain metrics (e.g., whale transactions, hash rate) (simulated with fallback)."""
    try:
        # Mock API call (replace with Glassnode/CryptoQuant API)
        # Simulated data since real API integration is pending
        response = {"whale_transactions": 5, "hash_rate": 200.0}  # Example values
        metrics = response
        whale_moves = metrics.get('whale_transactions', 0)
        hash_rate = metrics.get('hash_rate', 0)
        return {'whale_moves': whale_moves, 'hash_rate': hash_rate}
    except Exception as e:
        logging.error(f"On-chain metrics computation failed: {e}")
        return {'whale_moves': 0, 'hash_rate': 0}

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
    logging.info(f"Bollinger Bands:\n{compute_bollinger_bands(dummy_df)}")
    logging.info(f"RSI:\n{calculate_rsi(dummy_df['close'])}")
    logging.info(f"MACD:\n{calculate_macd(dummy_df['close'])}")
    logging.info(f"ATR:\n{calculate_atr(dummy_df['high'], dummy_df['low'], dummy_df['close'])}")
    logging.info(f"Sentiment Score: {calculate_sentiment_score()}")
    logging.info(f"On-Chain Metrics: {get_onchain_metrics()}")