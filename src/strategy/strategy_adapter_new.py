# src/strategy/strategy_adapter_new.py
import pandas as pd
from src.data.data_preprocessor import FEATURE_COLUMNS
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

def calculate_market_regime(df, window=50):
    """Estimate market regime using a rolling linear regression slope."""
    X = np.arange(window).reshape(-1, 1)
    df['regime'] = df['close'].rolling(window).apply(
        lambda x: LinearRegression().fit(X, x).coef_[0], raw=True
    )
    return df

def adapt_strategy_parameters(df, window=30):
    """Adapt strategy parameters based on recent market volatility and regime."""
    df = calculate_market_regime(df, window)
    volatility = df['price_volatility'].rolling(window=window).mean().iloc[-1]
    try:
        bollinger_bandwidth = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle']
    except KeyError:
        logging.debug("Missing 'bollinger_middle', using 'sma_20' as proxy")
        bollinger_bandwidth = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma_20']
    market_regime = df['regime'].iloc[-1]

    params = {
        'rsi_threshold': 70 if volatility > df['price_volatility'].mean() else 60,
        'macd_fast': 12 if market_regime > 0 else 10,
        'macd_slow': 26 if market_regime > 0 else 20,
        'atr_multiplier': 3 if bollinger_bandwidth.iloc[-1] > bollinger_bandwidth.mean() else 2,
        'max_risk_pct': 0.015 if volatility > df['price_volatility'].mean() else 0.02
    }
    return params