# Add to src/strategy/market_regime.py

def detect_market_regime(df, window=60):
    """
    Detect market regime based on price movement characteristics.

    :param df: DataFrame with price data
    :param window: Window for calculating regime metrics
    :return: String indicating the current market regime
    """
    returns = df['returns'].rolling(window).mean().iloc[-1]
    volatility = df['price_volatility'].rolling(window).mean().iloc[-1]
    
    if returns > 0 and volatility < df['price_volatility'].mean():
        return 'Bullish Low Volatility'
    elif returns > 0 and volatility > df['price_volatility'].mean():
        return 'Bullish High Volatility'
    elif returns < 0 and volatility < df['price_volatility'].mean():
        return 'Bearish Low Volatility'
    else:
        return 'Bearish High Volatility'

# Usage in backtest_visualizer.py
async def main():
        
    # Before generating signals
    regime = detect_market_regime(preprocessed_data)
    if regime == 'Bullish Low Volatility':
        # Adjust strategy for a trending market with less risk
        adapted_params = {'rsi_threshold': 60, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 1.5}
    elif regime == 'Bullish High Volatility':
        # Strategy for a volatile bull market
        adapted_params = {'rsi_threshold': 70, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.5}
    elif regime == 'Bearish Low Volatility':
        # Strategy for a bear market with low volatility
        adapted_params = {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2}
    else:
        # Strategy for a volatile bear market
        adapted_params = {'rsi_threshold': 40, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3}
    
    signal_data = generate_signals(scaled_df, model, feature_columns, feature_scaler, target_scaler, **adapted_params)