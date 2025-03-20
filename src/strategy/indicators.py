# src/strategy/indicators.py
"""
This module provides a collection of technical indicators and market metrics for trading signal generation,
utilizing libraries like pandas-ta. It supports both simulated and real data sources,
with configurable flags for enabling/disabling specific indicators via src/constants.

Key Integrations:
- **src.data.data_preprocessor.preprocess_data**: Integrates computed indicators into the preprocessed DataFrame.
- **src.strategy.signal_generator.generate_signals**: Uses unscaled indicator values for trade level calculations.
- **src.strategy.signal_filter.filter_signals**: Leverages indicators for filtering signals.
- **src.utils.time_utils.calculate_days_to_next_halving**: Influences dynamic window adjustments.
- **src.constants**: Configures the SIMULATE_INDICATORS flag.

Future Considerations:
- Optimize VPVR calculation for large datasets by precomputing bins.
- Vectorize ADX and ATR calculations for performance.
- Add unit tests for edge cases.

Dependencies:
- pandas
- pandas-ta
- numpy
- src.constants
- src.utils.time_utils
"""

import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from src.constants import SIMULATE_INDICATORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure separate logger for VPVR
vpvr_logger = logging.getLogger('vpvr')
vpvr_logger.setLevel(logging.DEBUG)
vpvr_handler = logging.FileHandler('vpvr.log')
vpvr_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))
vpvr_logger.addHandler(vpvr_handler)
vpvr_logger.propagate = False

vpvr_log_counter = 0
adx_nan_warning_counter = 0

def compute_vwap(df: pd.DataFrame, min_volume: float = 1.0) -> pd.Series:
    """
    Compute Volume Weighted Average Price (VWAP) with a minimum volume threshold.
    """
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns or 'volume' not in df.columns:
            logger.warning("Empty DataFrame or missing columns passed to compute_vwap")
            return pd.Series([], index=df.index, name='vwap')

        # Filter out low-volume periods
        df_filtered = df[df['volume'] >= min_volume].copy()
        if df_filtered.empty:
            logger.warning("No data after volume filter, using original data")
            df_filtered = df.copy()

        # Compute VWAP on log-transformed prices
        df_filtered['log_close'] = np.log(df_filtered['close'])
        vwap = ta.vwap(df_filtered['high'], df_filtered['low'], df_filtered['log_close'], df_filtered['volume'])
        vwap = vwap.reindex(df.index).ffill().bfill()
        if vwap.isna().any():
            logger.warning(f"VWAP contains {vwap.isna().sum()} NaN values, filling with rolling mean")
            vwap = vwap.fillna(vwap.rolling(window=24, min_periods=1).mean())
        return vwap.rename('vwap')
    except Exception as e:
        logger.error(f"VWAP computation failed: {e}")
        default_value = np.log(df['close'].mean()) if not df['close'].empty else np.log(78877.88)
        return pd.Series([default_value] * len(df), index=df.index, name='vwap')

def compute_adx(df: pd.DataFrame, period: int = 7, min_periods: int = 3) -> pd.Series:
    """
    Compute Average Directional Index (ADX) with optimized period for hourly data.
    """
    global adx_nan_warning_counter
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            logger.warning("Empty DataFrame or missing columns passed to compute_adx")
            return pd.Series([10.0] * len(df), index=df.index, name='adx')

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=period)
        adx = adx_df[f'ADX_{period}']

        if adx.isna().all():
            if adx_nan_warning_counter % 100 == 0:
                logger.warning(f"ADX is all NaN, filling with rolling median")
            adx_nan_warning_counter += 1
            adx = adx.fillna(adx.rolling(window=24, min_periods=1).median())
        elif adx.isna().any():
            if adx_nan_warning_counter % 100 == 0:
                logger.warning(f"ADX contains {adx.isna().sum()} NaN values, filling with rolling calculation")
            adx_nan_warning_counter += 1
            for i in range(len(df)):
                if pd.isna(adx.iloc[i]):
                    lookback = min(min_periods, i + 1)
                    if lookback >= 2:
                        adx_temp = ta.adx(df['high'].iloc[:i+1], df['low'].iloc[:i+1], df['close'].iloc[:i+1], length=lookback)
                        adx.iloc[i] = adx_temp[f'ADX_{lookback}'].iloc[-1]
                    else:
                        adx.iloc[i] = adx.rolling(window=24, min_periods=1).median().iloc[i]

        return adx.fillna(10.0).rename('adx')
    except Exception as e:
        logger.error(f"ADX computation failed: {e}")
        return pd.Series([10.0] * len(df), index=df.index, name='adx')

def compute_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2, volatility_threshold: float = 0.02) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands with dynamic period and bandwidth feature.

    Args:
        prices (pd.Series): Series of closing prices.
        period (int): Period for Bollinger Bands calculation.
        std_dev (float): Standard deviation multiplier for the bands.
        volatility_threshold (float): Threshold for dynamic period adjustment.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]: Upper band, middle band, lower band, breakout signal, and bandwidth.
    """
    try:
        if prices.empty:
            logger.warning("Empty Series passed to compute_bollinger_bands")
            return (
                pd.Series([], index=prices.index, name='bb_upper'),
                pd.Series([], index=prices.index, name='bb_middle'),
                pd.Series([], index=prices.index, name='bb_lower'),
                pd.Series([], index=prices.index, name='bb_breakout'),
                pd.Series([], index=prices.index, name='bb_bandwidth')
            )

        # Dynamic period based on volatility
        price_volatility = prices.pct_change().rolling(window=24, min_periods=1).std().fillna(0.0)
        dynamic_period = period
        if price_volatility.iloc[-1] > volatility_threshold:
            dynamic_period = max(10, int(period * 0.5))
        elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
            dynamic_period = min(40, int(period * 1.5))

        bbands = ta.bbands(prices, length=dynamic_period, std=std_dev)
        expected_columns = [f'BBU_{dynamic_period}_{std_dev}', f'BBM_{dynamic_period}_{std_dev}', f'BBL_{dynamic_period}_{std_dev}']
        
        rename_dict = {col: col.rsplit('.', 1)[0] for col in bbands.columns if col.endswith(f'_{std_dev}.0')}
        if rename_dict:
            bbands = bbands.rename(columns=rename_dict)

        for expected_col in expected_columns:
            if expected_col not in bbands.columns:
                logger.warning(f"Column {expected_col} not found, filling with default values")
                default_value = prices.mean() if not prices.empty else 78877.88
                bbands[expected_col] = default_value

        bb_upper = bbands[f'BBU_{dynamic_period}_{std_dev}']
        bb_middle = bbands[f'BBM_{dynamic_period}_{std_dev}']
        bb_lower = bbands[f'BBL_{dynamic_period}_{std_dev}']
        bb_bandwidth = (bb_upper - bb_lower) / bb_middle

        default_upper = prices.mean() + 2 * prices.std() if not prices.empty else 79367.5
        default_middle = prices.mean() if not prices.empty else 78877.88
        default_lower = prices.mean() - 2 * prices.std() if not prices.empty else 78186.98
        bb_upper = bb_upper.bfill().ffill().fillna(default_upper).rename('bb_upper')
        bb_middle = bb_middle.bfill().ffill().fillna(default_middle).rename('bb_middle')
        bb_lower = bb_lower.bfill().ffill().fillna(default_lower).rename('bb_lower')
        bb_bandwidth = bb_bandwidth.bfill().ffill().fillna(0.0).rename('bb_bandwidth')

        bb_breakout = pd.Series(0, index=prices.index, name='bb_breakout')
        bb_breakout[prices > bb_upper] = 1
        bb_breakout[prices < bb_lower] = -1

        return bb_upper, bb_middle, bb_lower, bb_breakout, bb_bandwidth
    except Exception as e:
        logger.error(f"Bollinger Bands computation failed: {e}")
        default_upper = prices.mean() + 2 * prices.std() if not prices.empty else 79367.5
        default_middle = prices.mean() if not prices.empty else 78877.88
        default_lower = prices.mean() - 2 * prices.std() if not prices.empty else 78186.98
        return (
            pd.Series([default_upper] * len(prices), index=prices.index, name='bb_upper'),
            pd.Series([default_middle] * len(prices), index=prices.index, name='bb_middle'),
            pd.Series([default_lower] * len(prices), index=prices.index, name='bb_lower'),
            pd.Series([0] * len(prices), index=prices.index, name='bb_breakout'),
            pd.Series([0.0] * len(prices), index=prices.index, name='bb_bandwidth')
        )

def calculate_rsi(prices: pd.Series, window: int = 14, volatility_threshold: float = 0.015) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) with optimized volatility threshold.
    """
    try:
        if prices.empty or not np.issubdtype(prices.dtype, np.number):
            logger.warning("Invalid or empty prices passed to calculate_rsi")
            return pd.Series([50.0] * len(prices), index=prices.index, name='momentum_rsi')

        price_volatility = prices.pct_change().fillna(0.0).rolling(window=24, min_periods=1).std().fillna(0.0)
        dynamic_window = window
        if price_volatility.iloc[-1] > volatility_threshold:
            dynamic_window = max(5, int(window * 0.5))
        elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
            dynamic_window = min(30, int(window * 1.5))
        
        delta = prices.diff().fillna(0.0)
        gain = (delta.where(delta > 0, 0)).rolling(window=dynamic_window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=dynamic_window, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        rsi = rsi.where(rsi > 0, 50.0).fillna(50.0)
        return rsi.rename('momentum_rsi')
    except Exception as e:
        logger.error(f"RSI computation failed: {e}")
        return pd.Series([50.0] * len(prices), index=prices.index, name='momentum_rsi')

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, volatility_threshold: float = 0.015) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate MACD with dynamic periods.
    """
    try:
        if prices.empty or not np.issubdtype(prices.dtype, np.number):
            logger.warning("Invalid or empty prices passed to calculate_macd")
            return (
                pd.Series([0.0] * len(prices), index=prices.index, name='trend_macd'),
                pd.Series([0.0] * len(prices), index=prices.index, name='macd_signal')
            )

        price_volatility = prices.pct_change().rolling(window=24, min_periods=1).std().fillna(0.0)
        dynamic_fast = fast
        dynamic_slow = slow
        dynamic_signal = signal
        if price_volatility.iloc[-1] > volatility_threshold:
            dynamic_fast = max(6, int(fast * 0.5))
            dynamic_slow = max(12, int(slow * 0.5))
            dynamic_signal = max(5, int(signal * 0.5))
        elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
            dynamic_fast = min(24, int(fast * 1.5))
            dynamic_slow = min(52, int(slow * 1.5))
            dynamic_signal = min(18, int(signal * 1.5))
        
        exp1 = prices.ewm(span=dynamic_fast, adjust=False).mean()
        exp2 = prices.ewm(span=dynamic_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=dynamic_signal, adjust=False).mean()
        
        if macd.isna().any() or signal_line.isna().any():
            logger.warning(f"MACD contains NaNs: macd={macd.isna().sum()}, signal_line={signal_line.isna().sum()}")
        return macd.fillna(0.0).rename('trend_macd'), signal_line.fillna(0.0).rename('macd_signal')
    except Exception as e:
        logger.error(f"MACD computation failed: {e}")
        return (
            pd.Series([0.0] * len(prices), index=prices.index, name='trend_macd'),
            pd.Series([0.0] * len(prices), index=prices.index, name='macd_signal')
        )

def calculate_atr(df: pd.DataFrame, period: int = 14, volatility_threshold: float = 0.015, min_periods: int = 1) -> pd.Series:
    """
    Calculate Average True Range (ATR) with dynamic window.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
        period (int): Period for ATR calculation.
        volatility_threshold (float): Threshold for dynamic window adjustment.
        min_periods (int): Minimum periods for rolling mean.

    Returns:
        pd.Series: ATR values.
    """
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            logger.warning("Empty DataFrame or missing columns passed to calculate_atr")
            return pd.Series([], index=df.index, name='atr')
        
        high = df['high']
        low = df['low']
        close = df['close']

        price_volatility = close.pct_change().rolling(window=24, min_periods=1).std().fillna(0.0)
        dynamic_window = period
        if price_volatility.iloc[-1] > volatility_threshold:
            dynamic_window = max(5, int(period * 0.5))
        elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
            dynamic_window = min(30, int(period * 1.5))
        
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        atr = tr.rolling(window=dynamic_window, min_periods=min_periods).mean()
        if atr.isna().any():
            logger.warning(f"ATR contains {atr.isna().sum()} NaN values, filling with default")
        default_value = tr.mean() if not tr.empty else 500.0
        atr = atr.bfill().ffill().fillna(default_value)
        return atr.rename('atr')
    except Exception as e:
        logger.error(f"ATR computation failed: {e}")
        default_value = (high - low).mean() if not (high.empty or low.empty) else 500.0
        return pd.Series([default_value] * len(close), index=close.index, name='atr')

def calculate_vpvr(data: pd.DataFrame, price_col: str = 'close', volume_col: str = 'volume') -> Dict[str, pd.Series]:
    """
    Calculate Volume Profile Visible Range (VPVR) with dynamic bins.
    """
    global vpvr_log_counter
    try:
        prices = data[price_col]
        volumes = data[volume_col]
        
        price_range = prices.max() - prices.min()
        num_bins = max(50, int(price_range / 100))  # Dynamic bins
        bin_size = price_range / num_bins if price_range > 0 else 1.0
        bins = np.arange(prices.min(), prices.max() + bin_size, bin_size)
        
        hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
        
        poc_idx = np.argmax(hist)
        poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
        
        volume_threshold = np.percentile(hist, 75)
        hvn_mask = hist > volume_threshold
        lvn_mask = hist < np.percentile(hist, 25)
        
        hvn_prices = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(hvn_mask)) if hvn_mask[i]]
        lvn_prices = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(lvn_mask)) if lvn_mask[i]]
        
        hvn_upper = max(hvn_prices) if hvn_prices else poc_price
        hvn_lower = min(hvn_prices) if hvn_prices else poc_price
        lvn_upper = max(lvn_prices) if lvn_prices else poc_price
        lvn_lower = min(lvn_prices) if lvn_prices else poc_price
        
        dist_to_poc = prices - poc_price
        dist_to_hvn_upper = prices - hvn_upper
        dist_to_hvn_lower = prices - hvn_lower
        dist_to_lvn_upper = prices - lvn_upper
        dist_to_lvn_lower = prices - lvn_lower
        
        result = {
            'dist_to_poc': dist_to_poc.rename('dist_to_poc'),
            'dist_to_hvn_upper': dist_to_hvn_upper.rename('dist_to_hvn_upper'),
            'dist_to_hvn_lower': dist_to_hvn_lower.rename('dist_to_hvn_lower'),
            'dist_to_lvn_upper': dist_to_lvn_upper.rename('dist_to_lvn_upper'),
            'dist_to_lvn_lower': dist_to_lvn_lower.rename('dist_to_lvn_lower')
        }
        
        vpvr_log_counter += 1
        if vpvr_log_counter % 100 == 0:
            vpvr_logger.debug(f"VPVR calculated - POC: {poc_price:.2f}, HVN Upper: {hvn_upper:.2f}, "
                              f"HVN Lower: {hvn_lower:.2f}, LVN Upper: {lvn_upper:.2f}, "
                              f"LVN Lower: {lvn_lower:.2f}")
        return result
    except Exception as e:
        logger.error(f"Error in VPVR calculation: {e}")
        return {
            'dist_to_poc': pd.Series(0.0, index=data.index, name='dist_to_poc'),
            'dist_to_hvn_upper': pd.Series(0.0, index=data.index, name='dist_to_hvn_upper'),
            'dist_to_hvn_lower': pd.Series(0.0, index=data.index, name='dist_to_hvn_lower'),
            'dist_to_lvn_upper': pd.Series(0.0, index=data.index, name='dist_to_lvn_upper'),
            'dist_to_lvn_lower': pd.Series(0.0, index=data.index, name='dist_to_lvn_lower')
        }

def calculate_stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K, %D) with crossover signals.
    """
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            logger.warning("Empty DataFrame or missing columns passed to calculate_stochastic_oscillator")
            return (
                pd.Series([], index=df.index, name='stoch_k'),
                pd.Series([], index=df.index, name='stoch_d'),
                pd.Series([], index=df.index, name='stoch_signal')
            )

        stoch = ta.stoch(df['high'], df['low'], df['close'], k=k_period, d=d_period, smooth=smooth)
        stoch_k = stoch[f'STOCHk_{k_period}_{d_period}_{smooth}'].reindex(df.index).fillna(50.0)
        stoch_d = stoch[f'STOCHd_{k_period}_{d_period}_{smooth}'].reindex(df.index).fillna(50.0)

        stoch_signal = pd.Series(0, index=df.index, name='stoch_signal')
        
        # Compute conditions and ensure index alignment
        bullish_condition = (stoch_k > stoch_d) & (stoch_k.shift(1).reindex(df.index) <= stoch_d.shift(1).reindex(df.index))
        bearish_condition = (stoch_k < stoch_d) & (stoch_k.shift(1).reindex(df.index) >= stoch_d.shift(1).reindex(df.index))
        
        # Fill NaN values in conditions to avoid alignment issues
        bullish_condition = bullish_condition.fillna(False)
        bearish_condition = bearish_condition.fillna(False)

        stoch_signal.loc[stoch_signal.index.isin(bullish_condition.index[bullish_condition])] = 1  # Bullish crossover
        stoch_signal.loc[stoch_signal.index.isin(bearish_condition.index[bearish_condition])] = -1  # Bearish crossover

        stoch_k = stoch_k.bfill().ffill().rename('stoch_k')
        stoch_d = stoch_d.bfill().ffill().rename('stoch_d')
        stoch_signal = stoch_signal.fillna(0).rename('stoch_signal')
        return stoch_k, stoch_d, stoch_signal
    except Exception as e:
        logger.error(f"Stochastic Oscillator computation failed: {e}")
        return (
            pd.Series([50.0] * len(df), index=df.index, name='stoch_k'),
            pd.Series([50.0] * len(df), index=df.index, name='stoch_d'),
            pd.Series([0] * len(df), index=df.index, name='stoch_signal')
        )

def calculate_fibonacci_levels(df: pd.DataFrame, window: int = 24) -> Dict[str, pd.Series]:
    """
    Calculate Fibonacci retracement levels and distances to current price.
    """
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            logger.warning("Empty DataFrame or missing columns passed to calculate_fibonacci_levels")
            return {
                'dist_to_fib_236': pd.Series([], index=df.index, name='dist_to_fib_236'),
                'dist_to_fib_382': pd.Series([], index=df.index, name='dist_to_fib_382'),
                'dist_to_fib_618': pd.Series([], index=df.index, name='dist_to_fib_618')
            }

        rolling_high = df['high'].rolling(window=window, min_periods=1).max()
        rolling_low = df['low'].rolling(window=window, min_periods=1).min()
        diff = rolling_high - rolling_low

        fib_236 = rolling_low + diff * 0.236
        fib_382 = rolling_low + diff * 0.382
        fib_618 = rolling_low + diff * 0.618

        dist_to_fib_236 = df['close'] - fib_236
        dist_to_fib_382 = df['close'] - fib_382
        dist_to_fib_618 = df['close'] - fib_618

        dist_to_fib_236 = dist_to_fib_236.bfill().ffill().fillna(0.0).rename('dist_to_fib_236')
        dist_to_fib_382 = dist_to_fib_382.bfill().ffill().fillna(0.0).rename('dist_to_fib_382')
        dist_to_fib_618 = dist_to_fib_618.bfill().ffill().fillna(0.0).rename('dist_to_fib_618')

        return {
            'dist_to_fib_236': dist_to_fib_236,
            'dist_to_fib_382': dist_to_fib_382,
            'dist_to_fib_618': dist_to_fib_618
        }
    except Exception as e:
        logger.error(f"Fibonacci levels computation failed: {e}")
        return {
            'dist_to_fib_236': pd.Series([0.0] * len(df), index=df.index, name='dist_to_fib_236'),
            'dist_to_fib_382': pd.Series([0.0] * len(df), index=df.index, name='dist_to_fib_382'),
            'dist_to_fib_618': pd.Series([0.0] * len(df), index=df.index, name='dist_to_fib_618')
        }

if __name__ == "__main__":
    dummy_df = pd.DataFrame({
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100, 101, 102, 103, 104],
        'open': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range("2024-01-01", periods=5, freq="H"))
    
    logger.info(f"VWAP:\n{compute_vwap(dummy_df)}")
    logger.info(f"ADX:\n{compute_adx(dummy_df, period=7)}")
    logger.info(f"Bollinger Bands:\n{compute_bollinger_bands(dummy_df['close'])}")
    logger.info(f"RSI:\n{calculate_rsi(dummy_df['close'])}")
    logger.info(f"MACD:\n{calculate_macd(dummy_df['close'])}")
    logger.info(f"ATR:\n{calculate_atr(dummy_df)}")
    logger.info(f"VPVR:\n{calculate_vpvr(dummy_df)}")
    logger.info(f"Stochastic Oscillator:\n{calculate_stochastic_oscillator(dummy_df)}")
    logger.info(f"Fibonacci Levels:\n{calculate_fibonacci_levels(dummy_df)}")