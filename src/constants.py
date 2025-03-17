# src/constants.py
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'volume', 'returns', 'log_returns',
    'price_volatility', 'sma_20', 'atr', 'vwap', 'adx',
    'momentum_rsi', 'trend_macd', 'ema_50',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_breakout',
    'dist_to_poc', 'dist_to_hvn_upper', 'dist_to_hvn_lower',
    'dist_to_lvn_upper', 'dist_to_lvn_lower',
    'luxalgo_signal', 'trendspider_signal', 'metastock_slope',
    'days_to_next_halving', 'days_since_last_halving', 'garch_volatility',
    'volume_normalized', 'hour_of_day', 'day_of_week',
    'smrt_scalping_signal'  
]

# Flag to toggle between simulated and real data
SIMULATE_INDICATORS = True

# Toggles for new features
USE_LUXALGO_SIGNALS = True
USE_TRENDSPIDER_PATTERNS = True
USE_HASSONLINE_ARBITRAGE = False  # Disabled by default until live trading
USE_SMRT_SCALPING = True
USE_METASTOCK_TREND_SLOPE = True

# Feature weights for signal combination
WEIGHT_LUXALGO = 0.4  # Increased from 0.2
WEIGHT_TRENDSPIDER = 0.2  # Increased from 0.15
WEIGHT_SMRT_SCALPING = 0.2  # Increased from 0.15
WEIGHT_METASTOCK = 0.1
WEIGHT_MODEL_CONFIDENCE = 0.1  # Reduced from 0.4