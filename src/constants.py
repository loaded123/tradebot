# src/constants.py
"""
This module defines constants used across the trading bot, including feature columns,
model parameters, configuration flags, and feature weights. These constants control which
features are computed, which indicators are enabled, simulation settings, and the weighting
of signals for combination.

Key Integrations:
- **src.data.data_preprocessor.preprocess_data**: Uses FEATURE_COLUMNS to determine
  which features and indicators to compute during preprocessing.
  - Note: 'luxalgo_signal' is now computed using SuperTrend, 'trendspider_signal' using
    Ichimoku Cloud, and 'metastock_slope' using EMA Slope.
- **src.strategy.indicators**: Uses SIMULATE_INDICATORS to toggle simulated vs. real data usage
  for sentiment and on-chain metrics.
- **src.strategy.signal_generator.generate_signals**: References FEATURE_COLUMNS for
  sequence creation, uses SEQ_LENGTH, BATCH_SIZE, NUM_SAMPLES for model inference, and
  CAPITAL for position sizing. HALVING_DATES integrates with time_utils for halving cycle
  adjustments.
- **src.strategy.signal_filter**: Utilizes WEIGHT_LUXALGO, WEIGHT_TRENDSPIDER, WEIGHT_SMRT_SCALPING,
  WEIGHT_METASTOCK, and WEIGHT_MODEL_CONFIDENCE to combine signals.
  - Note: Weights now correspond to new indicators (e.g., WEIGHT_LUXALGO for SuperTrend signals).

Future Considerations:
- Add validation to ensure FEATURE_COLUMNS aligns with downstream requirements (e.g., signal filtering).
- Consider adding a config file or environment variables for runtime configuration.
- Evaluate the impact of feature weights on signal accuracy and adjust dynamically based on backtest results.
"""

import pandas as pd

# Feature columns to be computed and used in the pipeline
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'volume', 'returns', 'log_returns', 'price_volatility', 'sma_20', 'atr', 'atr_normalized', 'vwap',
    'adx', 'momentum_rsi', 'trend_macd', 'macd_signal', 'macd_histogram', 'ema_50', 'bb_upper', 'bb_middle', 'bb_lower',
    'bb_breakout', 'bb_bandwidth', 'dist_to_poc', 'dist_to_hvn_upper', 'dist_to_hvn_lower', 'dist_to_lvn_upper',
    'dist_to_lvn_lower', 'luxalgo_signal', 'trendspider_signal', 'metastock_slope', 'metastock_slope_normalized',
    'days_to_next_halving', 'days_since_last_halving', 'volume_normalized', 'hour_of_day', 'day_of_week', 'month',
    'day_of_month', 'quarter', 'smrt_scalping_signal', 'garch_volatility', 'stoch_k', 'stoch_d', 'stoch_signal',
    'dist_to_fib_236', 'dist_to_fib_382', 'dist_to_fib_618'
]

DEFAULT_OPEN = 78877.88
DEFAULT_CLOSE = 78877.88
DEFAULT_HIGH = 79367.5
DEFAULT_LOW = 78186.98
DEFAULT_VOLUME = 1000.0

# Flag to toggle simulated data
SIMULATE_INDICATORS = True

# Toggles for features
USE_HASSONLINE_ARBITRAGE = False
USE_SMRT_SCALPING = False

# Feature weights
WEIGHT_LUXALGO = 0.4
WEIGHT_TRENDSPIDER = 0.2
WEIGHT_SMRT_SCALPING = 0.2
WEIGHT_METASTOCK = 0.1
WEIGHT_MODEL_CONFIDENCE = 0.1

# Model hyperparameters
SEQ_LENGTH = 24
BATCH_SIZE = 4000
NUM_SAMPLES = 5

# Default capital
CAPITAL = 17396.68

# Halving dates
HALVING_DATES = [
    pd.Timestamp("2012-11-28").tz_localize('UTC'),
    pd.Timestamp("2016-07-09").tz_localize('UTC'),
    pd.Timestamp("2020-05-11").tz_localize('UTC'),
    pd.Timestamp("2024-04-19").tz_localize('UTC'),
    pd.Timestamp("2028-03-15").tz_localize('UTC'),
]