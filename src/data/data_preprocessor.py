# src/data/data_preprocessor.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict
import ta  # Technical Analysis library
from src.constants import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

def calculate_vpvr(data: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume', num_bins: int = 100) -> Dict[str, pd.Series]:
    """Calculate Volume Profile Visible Range (VPVR) and distances to key levels."""
    prices = data[price_col]
    volumes = data[volume_col]
    
    # Create price bins
    price_range = prices.max() - prices.min()
    bin_size = price_range / num_bins
    bins = np.arange(prices.min(), prices.max() + bin_size, bin_size)
    
    # Calculate histogram of volume at each price level
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    
    # Find Point of Control (POC) - price with highest volume
    poc_idx = np.argmax(hist)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
    
    # Find High Volume Nodes (HVN) and Low Volume Nodes (LVN)
    volume_threshold = np.percentile(hist, 75)  # Top 25% of volume for HVN
    hvn_mask = hist > volume_threshold
    lvn_mask = hist < np.percentile(hist, 25)  # Bottom 25% for LVN
    
    hvn_prices = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(hvn_mask)) if hvn_mask[i]]
    lvn_prices = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(lvn_mask)) if lvn_mask[i]]
    
    hvn_upper = max(hvn_prices) if hvn_prices else poc_price
    hvn_lower = min(hvn_prices) if hvn_prices else poc_price
    lvn_upper = max(lvn_prices) if lvn_prices else poc_price
    lvn_lower = min(lvn_prices) if lvn_prices else poc_price
    
    # Calculate distances
    dist_to_poc = prices - poc_price
    dist_to_hvn_upper = prices - hvn_upper
    dist_to_hvn_lower = prices - hvn_lower
    dist_to_lvn_upper = prices - lvn_upper
    dist_to_lvn_lower = prices - lvn_lower
    
    return {
        'dist_to_poc': dist_to_poc,
        'dist_to_hvn_upper': dist_to_hvn_upper,
        'dist_to_hvn_lower': dist_to_hvn_lower,
        'dist_to_lvn_upper': dist_to_lvn_upper,
        'dist_to_lvn_lower': dist_to_lvn_lower
    }

def generate_luxalgo_signals(data: pd.DataFrame, trend_threshold: float = 0.01, reversal_threshold: float = 0.005) -> pd.Series:
    """Generate mock LuxAlgo trend/reversal signals based on price movements."""
    signals = pd.Series(0, index=data.index, dtype=int)
    returns = data['close'].pct_change()
    
    # Trend signals
    signals[returns > trend_threshold] = 1  # Uptrend
    signals[returns < -trend_threshold] = -1  # Downtrend
    
    # Reversal signals (mocked based on RSI overbought/oversold)
    rsi = ta.momentum.RSIIndicator(data['close']).rsi()
    signals[(rsi > 70) & (returns < -reversal_threshold)] = -1  # Potential reversal down
    signals[(rsi < 30) & (returns > reversal_threshold)] = 1   # Potential reversal up
    
    return signals

def generate_trendspider_signals(data: pd.DataFrame, pattern_window: int = 10) -> pd.Series:
    """Generate mock TrendSpider pattern signals based on candlestick patterns."""
    signals = pd.Series(0, index=data.index, dtype=int)
    returns = data['close'].pct_change()
    
    # Mock pattern detection (e.g., higher highs/higher lows for bullish, lower highs/lower lows for bearish)
    rolling_high = data['high'].rolling(window=pattern_window).max()
    rolling_low = data['low'].rolling(window=pattern_window).min()
    signals[(rolling_high > rolling_high.shift(1)) & (rolling_low > rolling_low.shift(1))] = 1  # Bullish pattern
    signals[(rolling_high < rolling_high.shift(1)) & (rolling_low < rolling_low.shift(1))] = -1  # Bearish pattern
    
    return signals

def generate_metastock_slope(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Generate MetaStock trend slope using a linear regression slope over a window."""
    def linreg_slope(series):
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series, 1)
        return slope
    
    return data['close'].rolling(window=window).apply(linreg_slope, raw=True).fillna(0)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw market data for model input, including technical indicators and new signals.
    
    Args:
        data (pd.DataFrame): Raw market data with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with all required features
    """
    logger.info("Starting data preprocessing...")
    
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        missing = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    processed_df = data.copy()
    
    # Calculate basic features
    processed_df['returns'] = processed_df['close'].pct_change()
    processed_df['log_returns'] = np.log1p(processed_df['returns'])
    processed_df['price_volatility'] = processed_df['close'].rolling(window=20).std()
    
    # Simple Moving Average (SMA)
    processed_df['sma_20'] = processed_df['close'].rolling(window=20).mean()
    
    # Average True Range (ATR)
    processed_df['atr'] = ta.volatility.AverageTrueRange(
        high=processed_df['high'],
        low=processed_df['low'],
        close=processed_df['close'],
        window=14
    ).average_true_range()
    
    # Volume Weighted Average Price (VWAP)
    typical_price = (processed_df['high'] + processed_df['low'] + processed_df['close']) / 3
    processed_df['vwap'] = (typical_price * processed_df['volume']).cumsum() / processed_df['volume'].cumsum()
    
    # Average Directional Index (ADX)
    processed_df['adx'] = ta.trend.ADXIndicator(
        high=processed_df['high'],
        low=processed_df['low'],
        close=processed_df['close'],
        window=14
    ).adx()
    if processed_df['adx'].isna().sum() > 0:
        logger.warning(f"ADX contains {processed_df['adx'].isna().sum()} NaN values, filling with default")
        processed_df['adx'] = processed_df['adx'].fillna(0)
    
    # Momentum (RSI)
    processed_df['momentum_rsi'] = ta.momentum.RSIIndicator(close=processed_df['close'], window=14).rsi()
    
    # Trend (MACD)
    macd = ta.trend.MACD(close=processed_df['close'], window_slow=26, window_fast=12)
    processed_df['trend_macd'] = macd.macd()
    
    # Exponential Moving Average (EMA)
    processed_df['ema_50'] = processed_df['close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=processed_df['close'], window=20, window_dev=2)
    processed_df['bollinger_upper'] = bb.bollinger_hband()
    processed_df['bollinger_lower'] = bb.bollinger_lband()
    processed_df['bollinger_middle'] = bb.bollinger_mavg()  # Corrected from bollinger_mband to bollinger_mavg
    
    # VPVR features
    vpvr_features = calculate_vpvr(processed_df)
    for key, value in vpvr_features.items():
        processed_df[key] = value
    
    # LuxAlgo signals
    processed_df['luxalgo_signal'] = generate_luxalgo_signals(processed_df)
    logger.info(f"Generated LuxAlgo signals: {processed_df['luxalgo_signal'].value_counts().to_dict()}")
    
    # TrendSpider signals
    processed_df['trendspider_signal'] = generate_trendspider_signals(processed_df)
    logger.info(f"Generated TrendSpider signals: {processed_df['trendspider_signal'].value_counts().to_dict()}")
    
    # MetaStock trend slope
    processed_df['metastock_slope'] = generate_metastock_slope(processed_df)
    
    # Handle missing values
    processed_df = processed_df.fillna(0)
    
    # Ensure all required columns are present
    for col in FEATURE_COLUMNS:
        if col not in processed_df.columns:
            logger.warning(f"Feature column {col} not computed, adding with zeros")
            processed_df[col] = 0
    
    logger.info(f"Preprocessed data shape: {processed_df.shape}")
    logger.info(f"Preprocessed columns: {list(processed_df.columns)}")
    return processed_df

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
        if col in scaled_df.columns:
            min_val = scaled_df[col].min()
            max_val = scaled_df[col].max()
            if max_val != min_val:
                scaled_df[col] = (scaled_df[col] - min_val) / (max_val - min_val)
            else:
                scaled_df[col] = 0
            logger.info(f"Feature {col}: min={min_val:.4f}, max={max_val:.4f}, mean={scaled_df[col].mean():.4f}")
        else:
            logger.warning(f"Feature {col} not found in DataFrame, skipping scaling")
    return scaled_df