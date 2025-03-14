# src/strategy/market_regime.py
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def detect_market_regime(df, window=24):
    """
    Detect market regime based on price movement, volatility, RSI, MACD, ADX, and volume trends.
    Enhanced with trend strength (ADX) and volume analysis for better classification.
    """
    try:
        required_columns = ['returns', 'price_volatility', 'momentum_rsi', 'trend_macd', 'adx', 'volume']
        if not all(col in df.columns for col in required_columns):
            logging.warning("DataFrame missing required columns. Defaulting to Neutral regime.")
            return 'Neutral'

        recent_data = df.tail(window)
        returns = recent_data['returns'].mean()
        volatility = recent_data['price_volatility'].mean()
        vol_mean = df['price_volatility'].mean()
        rsi = recent_data['momentum_rsi'].mean()
        macd = recent_data['trend_macd'].mean()
        adx = recent_data['adx'].mean()
        volume_trend = (recent_data['volume'].iloc[-1] - recent_data['volume'].iloc[0]) / recent_data['volume'].iloc[0] if recent_data['volume'].iloc[0] != 0 else 0

        logging.debug(f"Returns: {returns:.6f}, Volatility: {volatility:.4f}, RSI: {rsi:.2f}, MACD: {macd:.6f}, ADX: {adx:.2f}, Volume Trend: {volume_trend:.4f}, Vol Mean: {vol_mean:.4f}")

        if pd.isna(returns) or pd.isna(volatility) or pd.isna(rsi) or pd.isna(macd) or pd.isna(adx):
            logging.warning("NaN detected in regime calculation, returning default regime")
            return 'Neutral'

        # Enhanced regime detection with ADX and volume
        if returns > 0.00005 and rsi > 40 and macd >= 0 and adx > 25 and volume_trend > 0:
            regime = 'Bullish'
        elif returns < -0.0001 and rsi < 60 and macd <= 0 and adx > 25 and volume_trend < 0:
            regime = 'Bearish'
        else:
            regime = 'Neutral'

        # Volatility modifier
        if volatility >= vol_mean * 1.5:
            regime += ' High Volatility'
        else:
            regime += ' Low Volatility'

        logging.info(f"Detected market regime: Returns={returns:.6f}, RSI={rsi:.2f}, MACD={macd:.6f}, ADX={adx:.2f}, Volume Trend={volume_trend:.4f}, Volatility={volatility:.4f} -> {regime}")
        return regime

    except Exception as e:
        logging.error(f"Error detecting market regime: {e}")
        return 'Neutral'

if __name__ == "__main__":
    # Dummy test
    dummy_df = pd.DataFrame({
        'returns': [0.0001] * 12 + [-0.0001] * 12,
        'price_volatility': [0.02] * 12 + [0.05] * 12,
        'momentum_rsi': [60] * 12 + [40] * 12,
        'trend_macd': [0.001] * 12 + [-0.001] * 12,
        'adx': [30] * 12 + [30] * 12,
        'volume': [1000] * 12 + [900] * 12
    }, index=pd.date_range("2024-01-01", periods=24, freq="H"))
    regime = detect_market_regime(dummy_df)
    print(f"Detected regime: {regime}")