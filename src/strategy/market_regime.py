# src/strategy/market_regime.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def detect_market_regime(df, window=12):  # Reduced to 12 hours
    """
    Detect market regime based on price movement and volatility.
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
        
        if returns > 0.00005 and volatility < vol_mean:  # More sensitive threshold
            return 'Bullish Low Volatility'
        elif returns > 0.00005 and volatility >= vol_mean:
            return 'Bullish High Volatility'
        elif returns < -0.00005 and volatility < vol_mean:  # More sensitive threshold
            return 'Bearish Low Volatility'
        elif returns < -0.00005 and volatility >= vol_mean:
            return 'Bearish High Volatility'
        else:
            return 'Neutral'
    
    except Exception as e:
        logging.error(f"Error detecting market regime: {e}")
        return 'Neutral'

if __name__ == "__main__":
    # Dummy test
    dummy_df = pd.DataFrame({
        'returns': [0.01] * 6 + [-0.01] * 6,
        'price_volatility': [0.02] * 6 + [0.05] * 6
    }, index=pd.date_range("2024-01-01", periods=12, freq="H"))
    regime = detect_market_regime(dummy_df)
    print(f"Detected regime: {regime}")