# src/strategy/market_regime.py
import pandas as pd
import logging
import numpy as np
from src.strategy.indicators import calculate_rsi, calculate_macd, compute_adx

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def detect_market_regime(df: pd.DataFrame, window: int = 1440, log_interval: int = 100) -> str:
    """
    Detect market regime based on price movement, volatility, RSI, MACD, ADX, volume trends, and SMA crossovers.
    Enhanced with trend strength (ADX) and volume analysis for better classification.

    Args:
        df (pd.DataFrame): DataFrame with required columns ('close', 'volume', etc.).
        window (int): Lookback window for calculating metrics (default: 1440 hours ~ 60 days).
        log_interval (int): Log only every Nth detection to reduce verbosity.

    Returns:
        str: Detected market regime (e.g., 'Bullish High Volatility', 'Neutral').
    """
    try:
        required_columns = ['close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.warning("DataFrame missing required columns. Defaulting to Neutral regime.")
            return 'Neutral'

        # Ensure sufficient data
        if len(df) < window:
            logging.warning(f"Insufficient data for market regime detection: {len(df)} rows, need at least {window}")
            return 'Neutral'

        recent_data = df.tail(window)

        # Calculate metrics with proper initialization and fallbacks
        returns = recent_data['close'].pct_change().mean()
        if pd.isna(returns) or len(recent_data['close'].dropna()) < 2:
            returns = 0.0

        volatility = recent_data['close'].pct_change().rolling(window=24, min_periods=1).std().mean() * np.sqrt(window)
        if pd.isna(volatility) or volatility == 0:
            volatility = recent_data['close'].pct_change().std() * np.sqrt(window) if len(recent_data['close']) > 1 else 0.0

        vol_mean = df['close'].pct_change().rolling(window=window, min_periods=1).std().mean() * np.sqrt(window)
        if pd.isna(vol_mean):
            vol_mean = volatility

        rsi = calculate_rsi(recent_data['close']).iloc[-1]
        if pd.isna(rsi):
            rsi = 50.0  # Neutral RSI

        macd, macd_signal = calculate_macd(recent_data['close'])
        macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0
        macd_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0.0

        adx = compute_adx(recent_data, period=14).iloc[-1]
        if pd.isna(adx):
            adx = 10.0  # Default ADX (below trend threshold)

        volume_trend = (recent_data['volume'].iloc[-1] - recent_data['volume'].iloc[0]) / (recent_data['volume'].iloc[0] + 1e-10)

        # Add SMA crossovers for trend detection
        sma_50 = recent_data['close'].rolling(window=50, min_periods=1).mean().iloc[-1]
        sma_200 = recent_data['close'].rolling(window=200, min_periods=1).mean().iloc[-1]
        trend_strength = 0
        if sma_50 > sma_200 and recent_data['close'].iloc[-1] > sma_50:
            trend_strength = 1  # Bullish trend
        elif sma_50 < sma_200 and recent_data['close'].iloc[-1] < sma_50:
            trend_strength = -1  # Bearish trend

        # Log only every log_interval detections
        detect_market_regime.counter = getattr(detect_market_regime, 'counter', 0) + 1
        if detect_market_regime.counter % log_interval == 0:
            logging.debug(f"Returns: {returns:.6f}, Volatility: {volatility:.4f}, RSI: {rsi:.2f}, MACD: {macd:.6f}, "
                          f"ADX: {adx:.2f}, Volume Trend: {volume_trend:.4f}, Vol Mean: {vol_mean:.4f}, "
                          f"SMA Trend: {trend_strength}")
            logging.info(f"Detected market regime: Returns={returns:.6f}, RSI={rsi:.2f}, MACD={macd:.6f}, "
                         f"ADX={adx:.2f}, Volume Trend={volume_trend:.4f}, Volatility={volatility:.4f}")

        if pd.isna(returns) or pd.isna(volatility) or pd.isna(rsi) or pd.isna(macd) or pd.isna(adx):
            logging.warning("NaN detected in regime calculation, returning default regime")
            return 'Neutral'

        # Enhanced regime detection with lower ADX threshold (20) and trend strength
        if (returns > 0.00005 and rsi > 40 and macd >= macd_signal and adx > 20 and volume_trend > 0) or trend_strength == 1:
            regime = 'Bullish'
        elif (returns < -0.0001 and rsi < 60 and macd <= macd_signal and adx > 20 and volume_trend < 0) or trend_strength == -1:
            regime = 'Bearish'
        else:
            regime = 'Neutral'

        # Volatility modifier
        if volatility >= vol_mean * 1.5:
            regime += ' High Volatility'
        else:
            regime += ' Low Volatility'

        # Log the detected regime (sampled)
        if detect_market_regime.counter % log_interval == 0:
            logging.info(f"Detected market regime: Returns={returns:.6f}, RSI={rsi:.2f}, MACD={macd:.6f}, "
                         f"ADX={adx:.2f}, Volume Trend={volume_trend:.4f}, Volatility={volatility:.4f}, "
                         f"Trend Strength={trend_strength} -> {regime}")

        return regime

    except Exception as e:
        logging.error(f"Error detecting market regime: {e}")
        return 'Neutral'

if __name__ == "__main__":
    # Dummy test
    dummy_df = pd.DataFrame({
        'close': [100, 101] + [102] * 10 + [101] * 12,  # Added 'close' for volatility fallback
        'volume': [1000] * 12 + [900] * 12
    }, index=pd.date_range("2024-01-01", periods=24, freq="H"))
    regime = detect_market_regime(dummy_df)
    print(f"Detected regime: {regime}")