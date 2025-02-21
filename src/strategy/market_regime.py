# src/strategy/market_regime.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def detect_market_regime(df, window=60):
    """
    Detect market regime based on price movement and volatility.

    Args:
        df (pd.DataFrame): DataFrame with 'returns', 'price_volatility' columns
        window (int): Lookback period for metrics
    
    Returns:
        str: Market regime
    """
    try:
        if 'returns' not in df.columns or 'price_volatility' not in df.columns:
            raise ValueError("DataFrame missing 'returns' or 'price_volatility' columns")
        
        returns = df['returns'].rolling(window).mean().iloc[-1]
        volatility = df['price_volatility'].rolling(window).mean().iloc[-1]
        vol_mean = df['price_volatility'].mean()
        
        logging.debug(f"Returns: {returns}, Volatility: {volatility}, Vol Mean: {vol_mean}")
        
        if pd.isna(returns) or pd.isna(volatility):
            logging.warning("NaN detected in regime calculation, returning default regime")
            return 'Neutral'
        
        if returns > 0 and volatility < vol_mean:
            return 'Bullish Low Volatility'
        elif returns > 0 and volatility >= vol_mean:
            return 'Bullish High Volatility'
        elif returns < 0 and volatility < vol_mean:
            return 'Bearish Low Volatility'
        else:
            return 'Bearish High Volatility'
    
    except Exception as e:
        logging.error(f"Error detecting market regime: {e}")
        return 'Neutral'

if __name__ == "__main__":
    # Dummy test
    dummy_df = pd.DataFrame({
        'returns': [0.01] * 30 + [-0.01] * 30,
        'price_volatility': [0.02] * 30 + [0.05] * 30
    }, index=pd.date_range("2024-01-01", periods=60, freq="H"))
    regime = detect_market_regime(dummy_df)
    print(f"Detected regime: {regime}")