# src/data/data_preprocessor.py
"""
This module preprocesses raw market data by computing technical indicators, market regime,
volatility clustering, and normalized features for training and backtesting.

Key Integrations:
- **src.data.data_fetcher.fetch_historical_data**: Provides raw OHLCV data.
- **src.strategy.signal_filter.smrt_scalping_signals**: Generates SMRT scalping signals.
- **src.utils.time_utils**: Adds halving-related features.
- **src.constants.FEATURE_COLUMNS**: Defines the feature set to compute.
- **src.models.train_transformer_model**: Uses preprocessed data for training.
- **src.strategy.signal_generator**: Uses scaled features for signal generation.

Future Considerations:
- Optimize GARCH computation for large datasets.
- Add multi-timeframe features (e.g., 4h, daily).
- Implement feature selection to reduce dimensionality.

Dependencies:
- pandas
- pandas-ta
- numpy
- arch (for GARCH)
- src.strategy.signal_filter
- src.utils.time_utils
- src.constants
- src.strategy.indicators
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import List
from src.strategy.signal_filter import smrt_scalping_signals
from src.utils.time_utils import calculate_days_to_next_halving, calculate_days_since_last_halving
from src.constants import FEATURE_COLUMNS, USE_SMRT_SCALPING
from src.strategy.indicators import (
    calculate_atr, compute_vwap, compute_adx, calculate_rsi, calculate_macd,
    compute_bollinger_bands, calculate_vpvr, calculate_stochastic_oscillator,
    calculate_fibonacci_levels
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('src.data.data_preprocessor')

def compute_supertrend(df: pd.DataFrame, length: int = 7, multiplier: float = 2.0) -> pd.Series:
    """
    Compute SuperTrend indicator with optimized parameters.
    """
    try:
        supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=length, multiplier=multiplier)
        signal = supertrend[f'SUPERTd_{length}_{multiplier}']
        signal = signal.fillna(0).astype(int)
        return signal.rename('luxalgo_signal')
    except Exception as e:
        logger.error(f"Error computing SuperTrend: {e}")
        return pd.Series(0, index=df.index, name='luxalgo_signal')

def compute_ichimoku_signals(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.Series:
    """
    Compute Ichimoku Cloud signals with optimized parameters.
    """
    try:
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], tenkan=tenkan, kijun=kijun, senkou=senkou)[0]
        signal = pd.Series(0, index=df.index, name='trendspider_signal')
        signal[df['close'] > ichimoku['ISA_9']] = 1  # Bullish
        signal[df['close'] < ichimoku['ISB_26']] = -1  # Bearish
        return signal
    except Exception as e:
        logger.error(f"Error computing Ichimoku signals: {e}")
        return pd.Series(0, index=df.index, name='trendspider_signal')

def compute_ema_slope(df: pd.DataFrame, ema_period: int = 20, slope_window: int = 5) -> pd.DataFrame:
    """
    Compute EMA slope with optimized parameters and normalization.
    """
    try:
        ema = df['close'].ewm(span=ema_period, adjust=False).mean()
        slope = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            start_idx = max(0, i - slope_window + 1)
            window = ema.iloc[start_idx:i+1]
            if len(window) >= 2:
                x = np.arange(len(window))
                y = window.values
                slope.iloc[i] = np.polyfit(x, y, 1)[0]
            else:
                slope.iloc[i] = 0.0
        slope_normalized = slope / df['close']
        return pd.DataFrame({
            'metastock_slope': slope,
            'metastock_slope_normalized': slope_normalized
        }, index=df.index)
    except Exception as e:
        logger.error(f"Error computing EMA slope: {e}")
        return pd.DataFrame({
            'metastock_slope': [0.0] * len(df),
            'metastock_slope_normalized': [0.0] * len(df)
        }, index=df.index)

def compute_garch_volatility(returns: pd.Series, window: int = 24) -> pd.Series:
    """
    Compute GARCH(1,1) volatility (replaced with EWMA for performance).
    """
    try:
        ewma_vol = returns.ewm(span=window, adjust=False).std()
        return ewma_vol.fillna(0.0).rename('garch_volatility')
    except Exception as e:
        logger.error(f"Error computing GARCH volatility: {e}")
        return pd.Series(0.0, index=returns.index, name='garch_volatility')

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw market data by computing features and signals.
    """
    logger.info("Starting data preprocessing...")
    
    # Check for required columns (excluding timestamp since it's the index)
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    if data.empty:
        logger.warning("Empty DataFrame passed to preprocess_data")
        return data

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    df = data.copy()

    # Remove duplicates in the index and sort
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)

    # Validate and correct close prices
    earliest_valid_close = df['close'].dropna().iloc[0] if not df['close'].dropna().empty else 14.54
    valid_range_min = max(0.01, earliest_valid_close * 0.1)
    valid_range_max = 200200
    if (df['close'] <= 0).any() or (df['close'] < valid_range_min).any() or (df['close'] > valid_range_max).any():
        logger.warning(f"Close prices contain invalid values. Correcting to nearest valid value or {earliest_valid_close:.2f}.")
        df['close'] = df['close'].apply(
            lambda x: df['close'].iloc[max(df.index.get_loc(df.index[df['close'].notna()].get_loc(x, method='nearest')), 0)]
            if pd.isna(x) or x <= 0 or x < valid_range_min or x > valid_range_max else x
        )

    # Basic features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_volatility'] = df['log_returns'].rolling(window=24, min_periods=1).std().fillna(0.0)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['volume_normalized'] = (df['volume'] - df['volume'].rolling(window=24, min_periods=1).mean()) / df['volume'].rolling(window=24, min_periods=1).std()

    # Time-based features using the index
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['quarter'] = df.index.quarter
    df['days_to_next_halving'] = df.index.map(lambda x: calculate_days_to_next_halving(x)[0])
    df['days_since_last_halving'] = df.index.map(calculate_days_since_last_halving)

    # Advanced features
    df['target'] = df['log_returns'].shift(-1).fillna(0.0)
    df['garch_volatility'] = compute_garch_volatility(df['log_returns'])
    df['luxalgo_signal'] = compute_supertrend(df, length=7, multiplier=2.0)
    df['trendspider_signal'] = compute_ichimoku_signals(df, tenkan=9, kijun=26, senkou=52)
    ema_slope = compute_ema_slope(df, ema_period=20, slope_window=5)
    df = df.join(ema_slope)

    # Compute additional indicators from indicators.py
    df['atr'] = calculate_atr(df, period=14)
    df['atr_normalized'] = df['atr'] / df['close']
    df['vwap'] = compute_vwap(df)
    df['adx'] = compute_adx(df, period=14)
    df['momentum_rsi'] = calculate_rsi(df['close'])
    macd, macd_signal = calculate_macd(df['close'])
    df['trend_macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd - macd_signal
    bb_upper, bb_middle, bb_lower, bb_breakout, bb_bandwidth = compute_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_lower
    df['bb_breakout'] = bb_breakout
    df['bb_bandwidth'] = bb_bandwidth
    vpvr_metrics = calculate_vpvr(df)
    df['dist_to_poc'] = vpvr_metrics['dist_to_poc']
    df['dist_to_hvn_upper'] = vpvr_metrics['dist_to_hvn_upper']
    df['dist_to_hvn_lower'] = vpvr_metrics['dist_to_hvn_lower']
    df['dist_to_lvn_upper'] = vpvr_metrics['dist_to_lvn_upper']
    df['dist_to_lvn_lower'] = vpvr_metrics['dist_to_lvn_lower']
    stoch_k, stoch_d, stoch_signal = calculate_stochastic_oscillator(df)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['stoch_signal'] = stoch_signal
    fib_metrics = calculate_fibonacci_levels(df)
    df['dist_to_fib_236'] = fib_metrics['dist_to_fib_236']
    df['dist_to_fib_382'] = fib_metrics['dist_to_fib_382']
    df['dist_to_fib_618'] = fib_metrics['dist_to_fib_618']

    # SMRT Scalping signals
    if USE_SMRT_SCALPING:
        try:
            df['smrt_scalping_signal'] = smrt_scalping_signals(df, atr_multiplier=1.0)
        except Exception as e:
            logger.error(f"Failed to compute SMRT scalping signals: {e}")
            df['smrt_scalping_signal'] = 0
    else:
        logger.info("SMRT scalping signals disabled (USE_SMRT_SCALPING=False), setting smrt_scalping_signal to 0")
        df['smrt_scalping_signal'] = 0

    # Market regime (simplified)
    df['market_regime'] = pd.cut(df['price_volatility'], bins=[0, 0.01, 0.03, np.inf], labels=['Low', 'Medium', 'High'], include_lowest=True)
    # Add 'Neutral' to the categories before filling NaNs
    df['market_regime'] = df['market_regime'].astype('category')
    df['market_regime'] = df['market_regime'].cat.add_categories(['Neutral'])
    df['market_regime'] = df['market_regime'].fillna('Neutral')

    # Ensure all FEATURE_COLUMNS are present
    for col in FEATURE_COLUMNS:
        if col not in df.columns and col != 'market_regime':
            logger.warning(f"Column {col} missing from preprocessed data, filling with 0")
            df[col] = 0.0

    # Fill NaNs for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    logger.info(f"Preprocessed data shape: {df.shape}")
    logger.info(f"Preprocessed columns: {list(df.columns)}")
    return df

def scale_features(data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Scale features to [0, 1] range.

    Args:
        data (pd.DataFrame): Preprocessed data
        feature_columns (List[str]): Columns to scale

    Returns:
        pd.DataFrame: Scaled DataFrame
    """
    scaled_df = data.copy()
    for col in feature_columns:
        if col in scaled_df.columns and col != 'market_regime':
            min_val = scaled_df[col].min()
            max_val = scaled_df[col].max()
            if max_val != min_val:
                scaled_df[col] = (scaled_df[col] - min_val) / (max_val - min_val)
            else:
                scaled_df[col] = 0
            logger.info(f"Feature {col}: min={min_val:.4f}, max={max_val:.4f}, mean={scaled_df[col].mean():.4f}")
        else:
            logger.warning(f"Feature {col} not found in DataFrame or is categorical, skipping scaling")
    return scaled_df