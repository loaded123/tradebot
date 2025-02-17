import pandas as pd
from src.data.data_preprocessor import FEATURE_COLUMNS
from sklearn.linear_model import LinearRegression

def calculate_market_regime(df, window=50):
    """
    Estimate market regime using a rolling linear regression slope.
    """
    df['regime'] = df['close'].rolling(window).apply(lambda x: LinearRegression().fit(range(window), x).coef_[0], raw=True)
    return df

def adapt_strategy_parameters(df, window=30):
    """
    Adapt strategy parameters based on recent market volatility and regime.

    :param df: DataFrame with preprocessed data
    :param window: Rolling window for calculating volatility
    :return: Dictionary of adapted parameters
    """
    df = calculate_market_regime(df, window)

    volatility = df['price_volatility'].rolling(window=window).mean().iloc[-1]
    bollinger_bandwidth = (df['upper_band'] - df['lower_band']) / df['middle_band']
    market_regime = df['regime'].iloc[-1]

    params = {
        'rsi_threshold': 70 if volatility > df['price_volatility'].mean() else 60,
        'macd_fast': 12 if market_regime > 0 else 10,
        'macd_slow': 26 if market_regime > 0 else 20,
        'atr_multiplier': 3 if bollinger_bandwidth.iloc[-1] > bollinger_bandwidth.mean() else 2,
        'max_risk_pct': 0.015 if volatility > df['price_volatility'].mean() else 0.02
    }
    
    return params
