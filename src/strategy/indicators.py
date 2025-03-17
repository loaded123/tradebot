# src/strategy/indicators.py
import pandas as pd
import pandas_ta as ta
import logging
import numpy as np
from textblob import TextBlob
from typing import Dict, Optional, Tuple
from scipy.stats import linregress
from src.constants import USE_LUXALGO_SIGNALS, USE_TRENDSPIDER_PATTERNS, USE_METASTOCK_TREND_SLOPE

# Configure main logging to console or file
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Configure separate logger for VPVR
vpvr_logger = logging.getLogger('vpvr')
vpvr_logger.setLevel(logging.DEBUG)  # Change to INFO if you don't need debug output
vpvr_handler = logging.FileHandler('vpvr.log')
vpvr_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))
vpvr_logger.addHandler(vpvr_handler)
vpvr_logger.propagate = False  # Prevent propagation to root logger

# Counter for sampling VPVR logs
vpvr_log_counter = 0

# Flag to toggle between simulated and real data (to be set in constants.py)
SIMULATE_INDICATORS = True  # Default to True; set to False when real APIs are integrated

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute Volume Weighted Average Price (VWAP) using pandas-ta with fallback."""
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns or 'volume' not in df.columns:
            logging.warning("Empty DataFrame or missing columns passed to compute_vwap")
            return pd.Series([], index=df.index)

        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        if vwap.isna().any():
            logging.warning(f"VWAP contains {vwap.isna().sum()} NaN values, filling with default")
        default_value = df['close'].mean() if not df['close'].empty else 78877.88
        return vwap.fillna(default_value)
    except Exception as e:
        logging.error(f"VWAP computation failed: {e}")
        default_value = df['close'].mean() if not df['close'].empty else 78877.88
        return pd.Series([default_value] * len(df), index=df.index)

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (ADX) using pandas-ta with fallback."""
    try:
        if df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            logging.warning("Empty DataFrame or missing columns passed to compute_adx")
            return pd.Series([10.0] * len(df), index=df.index)

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=period)
        adx = adx_df[f'ADX_{period}']
        if adx.isna().all():
            logging.warning(f"ADX is all NaN, likely insufficient data. Filling with default value of 10.0")
        elif adx.isna().any():
            logging.warning(f"ADX contains {adx.isna().sum()} NaN values, filling with default")
        return adx.fillna(10.0)  # Neutral ADX value
    except Exception as e:
        logging.error(f"ADX computation failed: {e}")
        return pd.Series([10.0] * len(df), index=df.index)

def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
    """Compute Bollinger Bands and breakout signals for high-frequency trading."""
    try:
        if df.empty or 'close' not in df.columns:
            logging.warning("Empty DataFrame or missing 'close' column passed to compute_bollinger_bands")
            return pd.DataFrame({'bb_upper': [], 'bb_middle': [], 'bb_lower': [], 'bb_breakout': []}, index=df.index)

        bbands = ta.bbands(df['close'], length=period, std=std_dev)
        # Expected columns without the '.0' suffix
        expected_columns = [f'BBU_{period}_{std_dev}', f'BBM_{period}_{std_dev}', f'BBL_{period}_{std_dev}']
        
        # Check available columns and rename if necessary
        available_columns = bbands.columns.tolist()
        logging.debug(f"Computed Bollinger Bands columns before renaming: {available_columns}")
        
        # Rename columns to remove '.0' suffix if present
        rename_dict = {}
        for col in available_columns:
            if col.endswith(f'_{std_dev}.0'):
                base_name = col.rsplit('.', 1)[0]  # Remove '.0' suffix
                rename_dict[col] = base_name
        
        if rename_dict:
            bbands = bbands.rename(columns=rename_dict)
            logging.info(f"Renamed Bollinger Bands columns: {rename_dict}")
        
        # Verify and select expected columns
        available_columns = bbands.columns.tolist()
        for expected_col in expected_columns:
            if expected_col not in available_columns:
                logging.warning(f"Column {expected_col} not found after renaming. Filling with default values.")
                default_value = df['close'].mean() if 'close' in df.columns else 78877.88
                bbands[expected_col] = default_value

        result = pd.DataFrame(index=df.index)
        result['bb_upper'] = bbands[f'BBU_{period}_{std_dev}']
        result['bb_middle'] = bbands[f'BBM_{period}_{std_dev}']
        result['bb_lower'] = bbands[f'BBL_{period}_{std_dev}']

        # Handle NaNs with forward and backward fill, then fallback
        default_upper = df['close'].mean() + 2 * df['close'].std() if not df['close'].empty else 79367.5
        default_middle = df['close'].mean() if not df['close'].empty else 78877.88
        default_lower = df['close'].mean() - 2 * df['close'].std() if not df['close'].empty else 78186.98
        result['bb_upper'] = result['bb_upper'].bfill().ffill().fillna(default_upper)
        result['bb_middle'] = result['bb_middle'].bfill().ffill().fillna(default_middle)
        result['bb_lower'] = result['bb_lower'].bfill().ffill().fillna(default_lower)

        # Generate breakout signals
        result['bb_breakout'] = 0
        result.loc[df['close'] > result['bb_upper'], 'bb_breakout'] = 1  # Bullish breakout
        result.loc[df['close'] < result['bb_lower'], 'bb_breakout'] = -1  # Bearish breakout

        return result
    except Exception as e:
        logging.error(f"Bollinger Bands computation failed: {e}")
        default_upper = df['close'].mean() + 2 * df['close'].std() if not df['close'].empty else 79367.5
        default_middle = df['close'].mean() if not df['close'].empty else 78877.88
        default_lower = df['close'].mean() - 2 * df['close'].std() if not df['close'].empty else 78186.98
        return pd.DataFrame({
            'bb_upper': [default_upper] * len(df),
            'bb_middle': [default_middle] * len(df),
            'bb_lower': [default_lower] * len(df),
            'bb_breakout': [0] * len(df)
        }, index=df.index)

def calculate_rsi(prices: pd.Series, window: int = 14, volatility_threshold: float = 0.02) -> pd.Series:
    """Calculate RSI with dynamic window based on volatility."""
    try:
        if prices.empty or not np.issubdtype(prices.dtype, np.number):
            logging.warning("Invalid or empty prices passed to calculate_rsi")
            return pd.Series([50.0] * len(prices), index=prices.index)

        price_volatility = prices.pct_change().fillna(0.0).rolling(window=24, min_periods=1).std().fillna(0.0)
        dynamic_window = window
        if price_volatility.iloc[-1] > volatility_threshold:
            dynamic_window = max(5, int(window * 0.5))
        elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
            dynamic_window = min(30, int(window * 1.5))
        
        delta = prices.diff().fillna(0.0)
        gain = (delta.where(delta > 0, 0)).rolling(window=dynamic_window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=dynamic_window, min_periods=1).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        rsi = rsi.where(rsi > 0, 50.0).fillna(50.0)
        return rsi
    except Exception as e:
        logging.error(f"RSI computation failed: {e}")
        return pd.Series([50.0] * len(prices), index=prices.index)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, volatility_threshold: float = 0.02) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD with dynamic periods based on volatility."""
    try:
        if prices.empty or not np.issubdtype(prices.dtype, np.number):
            logging.warning("Invalid or empty prices passed to calculate_macd")
            return pd.Series([0.0] * len(prices), index=prices.index), pd.Series([0.0] * len(prices), index=prices.index)

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
            logging.warning(f"MACD contains NaNs: macd={macd.isna().sum()}, signal_line={signal_line.isna().sum()}")
        return macd.fillna(0.0), signal_line.fillna(0.0)
    except Exception as e:
        logging.error(f"MACD computation failed: {e}")
        return pd.Series([0.0] * len(prices), index=prices.index), pd.Series([0.0] * len(prices), index=prices.index)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, volatility_threshold: float = 0.02, min_periods: int = 1) -> pd.Series:
    """Calculate Average True Range (ATR) with dynamic window size based on volatility and minimum periods."""
    try:
        if high.empty or low.empty or close.empty:
            logging.warning("Empty input passed to calculate_atr")
            return pd.Series([], index=close.index)
        
        if not (high.index.equals(low.index) and high.index.equals(close.index)):
            raise ValueError("Indices of high, low, and close must match")

        price_volatility = close.pct_change().rolling(window=24, min_periods=1).std().fillna(0.0)
        dynamic_window = period
        if price_volatility.iloc[-1] > volatility_threshold:
            dynamic_window = max(5, int(period * 0.5))
        elif price_volatility.iloc[-1] < volatility_threshold * 0.5:
            dynamic_window = min(30, int(period * 1.5))
        
        tr = np.maximum(high - low, np.abs(high - close.shift()), np.abs(low - close.shift()))
        atr = tr.rolling(window=dynamic_window, min_periods=min_periods).mean()
        if atr.isna().any():
            logging.warning(f"ATR contains {atr.isna().sum()} NaN values, filling with default")
        default_value = tr.mean() if not tr.empty else 500.0
        return atr.bfill().ffill().fillna(default_value)
    except Exception as e:
        logging.error(f"ATR computation failed: {e}")
        default_value = (high - low).mean() if not (high.empty or low.empty) else 500.0
        return pd.Series([default_value] * len(close), index=close.index)

def calculate_vpvr(df: pd.DataFrame, lookback: int = 500, num_bins: int = 50) -> Dict[str, float]:
    """
    Calculate Volume Profile Visible Range (VPVR) metrics: POC, HVN, LVN.
    """
    global vpvr_log_counter
    try:
        if len(df) < 2:
            logging.warning("Insufficient data for VPVR calculation")
            return {'poc': df['close'].iloc[-1] if not df.empty else 0,
                    'hvn_upper': df['close'].iloc[-1] if not df.empty else 0,
                    'hvn_lower': df['close'].iloc[-1] if not df.empty else 0,
                    'lvn_upper': df['close'].iloc[-1] if not df.empty else 0,
                    'lvn_lower': df['close'].iloc[-1] if not df.empty else 0}

        df_subset = df.tail(lookback).copy()
        if df_subset['close'].empty or df_subset['volume'].empty:
            logging.warning("Missing close or volume data in VPVR subset")
            return {'poc': df['close'].iloc[-1] if not df.empty else 0,
                    'hvn_upper': df['close'].iloc[-1] if not df.empty else 0,
                    'hvn_lower': df['close'].iloc[-1] if not df.empty else 0,
                    'lvn_upper': df['close'].iloc[-1] if not df.empty else 0,
                    'lvn_lower': df['close'].iloc[-1] if not df.empty else 0}

        price_range = df_subset['close'].max() - df_subset['close'].min()

        # Use fixed num_bins=50 to match other scripts, overriding dynamic adjustment
        adjusted_num_bins = num_bins

        # Ensure sufficient unique prices for binning
        unique_prices = df_subset['close'].dropna().nunique()
        if unique_prices < 2:
            logging.warning(f"Insufficient unique prices ({unique_prices}) for VPVR, using default values")
            return {'poc': df_subset['close'].iloc[-1],
                    'hvn_upper': df_subset['close'].iloc[-1],
                    'hvn_lower': df_subset['close'].iloc[-1],
                    'lvn_upper': df_subset['close'].iloc[-1],
                    'lvn_lower': df_subset['close'].iloc[-1]}

        if adjusted_num_bins > unique_prices - 1:
            adjusted_num_bins = max(10, unique_prices - 1)
            logging.warning(f"Adjusted num_bins to {adjusted_num_bins} due to insufficient unique prices")

        # Create bins directly from the Series with its index
        bins = pd.cut(df_subset['close'], bins=adjusted_num_bins, precision=2, include_lowest=True)
        if len(bins.cat.categories) == 0 or pd.isna(bins).all():
            logging.warning("Price bins creation failed or contains all NaNs, using default values")
            return {'poc': df_subset['close'].iloc[-1],
                    'hvn_upper': df_subset['close'].iloc[-1],
                    'hvn_lower': df_subset['close'].iloc[-1],
                    'lvn_upper': df_subset['close'].iloc[-1],
                    'lvn_lower': df_subset['close'].iloc[-1]}

        # Create a DataFrame to associate bins with volumes
        temp_df = pd.DataFrame({'price_bin': bins, 'volume': df_subset['volume']})
        volume_profile = temp_df.groupby('price_bin', observed=True)['volume'].sum()

        if volume_profile.empty:
            logging.warning("Volume profile is empty, using default values")
            return {'poc': df_subset['close'].iloc[-1],
                    'hvn_upper': df_subset['close'].iloc[-1],
                    'hvn_lower': df_subset['close'].iloc[-1],
                    'lvn_upper': df_subset['close'].iloc[-1],
                    'lvn_lower': df_subset['close'].iloc[-1]}

        poc_bin = volume_profile.idxmax()
        poc_price = poc_bin.mid if pd.notna(poc_bin) else df_subset['close'].iloc[-1]
        
        sorted_profile = volume_profile.sort_values(ascending=False)
        hvn_threshold = sorted_profile.iloc[0] * 0.7
        lvn_threshold = sorted_profile.iloc[0] * 0.3
        hvn_bins = sorted_profile[sorted_profile >= hvn_threshold].index
        lvn_bins = sorted_profile[sorted_profile <= lvn_threshold].index

        hvn_upper = hvn_bins[-1].right if len(hvn_bins) > 0 else poc_price
        hvn_lower = hvn_bins[0].left if len(hvn_bins) > 0 else poc_price
        lvn_upper = lvn_bins[-1].right if len(lvn_bins) > 0 else poc_price
        lvn_lower = lvn_bins[0].left if len(lvn_bins) > 0 else poc_price

        result = {
            'poc': float(poc_price),
            'hvn_upper': float(hvn_upper),
            'hvn_lower': float(hvn_lower),
            'lvn_upper': float(lvn_upper),
            'lvn_lower': float(lvn_lower)
        }
        # Log every 100th calculation to reduce volume
        vpvr_log_counter += 1
        if vpvr_log_counter % 100 == 0:
            vpvr_logger.debug(f"VPVR calculated - POC: {result['poc']:.2f}, HVN Upper: {result['hvn_upper']:.2f}, "
                              f"HVN Lower: {result['hvn_lower']:.2f}, LVN Upper: {result['lvn_upper']:.2f}, "
                              f"LVN Lower: {result['lvn_lower']:.2f}")
        return result

    except Exception as e:
        logging.error(f"Error in VPVR calculation: {e}")
        return {'poc': df['close'].iloc[-1] if not df.empty else 0,
                'hvn_upper': df['close'].iloc[-1] if not df.empty else 0,
                'hvn_lower': df['close'].iloc[-1] if not df.empty else 0,
                'lvn_upper': df['close'].iloc[-1] if not df.empty else 0,
                'lvn_lower': df['close'].iloc[-1] if not df.empty else 0}

def calculate_sentiment_score(symbol: str = "BTC/USD", source: str = "twitter") -> float:
    """Calculate market sentiment score from Twitter/X or news."""
    try:
        if not isinstance(symbol, str) or not isinstance(source, str):
            raise ValueError("Symbol and source must be strings")

        if not SIMULATE_INDICATORS:
            # Placeholder for real API call (to be implemented)
            logging.info(f"Fetching real sentiment data for {symbol} from {source} (API not implemented yet)")
            return 0.0  # Replace with actual API call
        else:
            logging.warning("Using simulated sentiment data. Replace with real API call.")
            response = {"tweets": [{"text": "Bitcoin is rising!"} for _ in range(10)] + [{"text": "Bitcoin is falling!"} for _ in range(5)]}
            tweets = response.get('tweets', [])
            sentiment_scores = [TextBlob(tweet['text']).sentiment.polarity for tweet in tweets]
            score = np.mean(sentiment_scores) if sentiment_scores else 0.0
            logging.info(f"Simulated sentiment score for {symbol} from {source}: {score:.2f}")
            return score
    except Exception as e:
        logging.error(f"Sentiment score computation failed: {e}")
        return 0.0

def get_onchain_metrics(symbol: str = "BTC") -> Dict[str, float]:
    """Fetch on-chain metrics (e.g., whale transactions, hash rate)."""
    try:
        if not isinstance(symbol, str):
            raise ValueError("Symbol must be a string")

        if not SIMULATE_INDICATORS:
            # Placeholder for real API call (to be implemented)
            logging.info(f"Fetching real on-chain metrics for {symbol} (API not implemented yet)")
            return {'whale_moves': 0.0, 'hash_rate': 0.0}  # Replace with actual API call
        else:
            logging.warning("Using simulated on-chain metrics. Replace with real API call.")
            response = {"whale_transactions": 5, "hash_rate": 200.0}
            metrics = response
            whale_moves = metrics.get('whale_transactions', 0)
            hash_rate = metrics.get('hash_rate', 0)
            result = {'whale_moves': float(whale_moves), 'hash_rate': float(hash_rate)}
            logging.info(f"Simulated on-chain metrics for {symbol}: {result}")
            return result
    except Exception as e:
        logging.error(f"On-chain metrics computation failed: {e}")
        return {'whale_moves': 0.0, 'hash_rate': 0.0}

def luxalgo_trend_reversal(df: pd.DataFrame, adx_threshold: float = 25.0, rsi_overbought: float = 70.0, rsi_oversold: float = 30.0) -> pd.Series:
    """Generate LuxAlgo-inspired trend and reversal signals based on ADX and RSI."""
    if not USE_LUXALGO_SIGNALS:
        logging.info("LuxAlgo trend/reversal signals disabled")
        return pd.Series(0, index=df.index)

    try:
        adx = compute_adx(df, period=14)
        rsi = calculate_rsi(df['close'], window=14)
        signals = pd.Series(0, index=df.index)

        # Trend signals: ADX > threshold indicates strong trend
        signals[(adx > adx_threshold) & (rsi > rsi_overbought)] = -1  # Potential reversal (sell)
        signals[(adx > adx_threshold) & (rsi < rsi_oversold)] = 1    # Potential reversal (buy)

        # Reversal signals: Look for RSI divergence near overbought/oversold
        price_diff = df['close'].diff()
        rsi_diff = rsi.diff()
        signals[(rsi > rsi_overbought) & (price_diff > 0) & (rsi_diff < 0)] = -1  # Bearish divergence
        signals[(rsi < rsi_oversold) & (price_diff < 0) & (rsi_diff > 0)] = 1    # Bullish divergence

        logging.info(f"Generated LuxAlgo trend/reversal signals: {signals.value_counts().to_dict()}")
        return signals
    except Exception as e:
        logging.error(f"LuxAlgo trend/reversal computation failed: {e}")
        return pd.Series(0, index=df.index)

def trendspider_pattern_recognition(df: pd.DataFrame) -> pd.Series:
    """Detect basic candlestick patterns inspired by TrendSpider (e.g., engulfing, doji)."""
    if not USE_TRENDSPIDER_PATTERNS:
        logging.info("TrendSpider pattern recognition disabled")
        return pd.Series(0, index=df.index)

    try:
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        signals = pd.Series(0, index=df.index)
        body = (df['close'] - df['open']).abs()
        total_range = df['high'] - df['low']

        # Bullish engulfing
        bullish_engulfing = (df['close'].shift(1) < df['open'].shift(1)) & \
                            (df['close'] > df['open']) & \
                            (df['close'] > df['open'].shift(1)) & \
                            (df['open'] < df['close'].shift(1))
        signals[bullish_engulfing] = 1

        # Bearish engulfing
        bearish_engulfing = (df['close'].shift(1) > df['open'].shift(1)) & \
                            (df['close'] < df['open']) & \
                            (df['close'] < df['open'].shift(1)) & \
                            (df['open'] > df['close'].shift(1))
        signals[bearish_engulfing] = -1

        # Doji (potential reversal)
        doji = (body / total_range < 0.1) & (total_range > 0)
        signals[doji & (df['close'] > df['close'].shift(1))] = 1   # Doji after downtrend
        signals[doji & (df['close'] < df['close'].shift(1))] = -1  # Doji after uptrend

        logging.info(f"Generated TrendSpider pattern signals: {signals.value_counts().to_dict()}")
        return signals
    except Exception as e:
        logging.error(f"TrendSpider pattern recognition failed: {e}")
        return pd.Series(0, index=df.index)

def metastock_trend_slope(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate trend slope using linear regression, inspired by MetaStock."""
    if not USE_METASTOCK_TREND_SLOPE:
        logging.info("MetaStock trend slope analysis disabled")
        return pd.Series(0, index=df.index)

    try:
        slopes = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            window_data = df['close'].iloc[start_idx:i+1]
            if len(window_data) < 2:
                slopes.iloc[i] = 0.0
                continue
            x = np.arange(len(window_data))
            slope, _, _, _, _ = linregress(x, window_data)
            slopes.iloc[i] = slope
        return slopes.fillna(0.0)
    except Exception as e:
        logging.error(f"MetaStock trend slope computation failed: {e}")
        return pd.Series(0, index=df.index)

if __name__ == "__main__":
    # Test with dummy data
    dummy_df = pd.DataFrame({
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100, 101, 102, 103, 104],
        'open': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range("2024-01-01", periods=5, freq="H"))
    
    logging.info(f"VWAP:\n{compute_vwap(dummy_df)}")
    logging.info(f"ADX:\n{compute_adx(dummy_df, period=14)}")
    logging.info(f"Bollinger Bands:\n{compute_bollinger_bands(dummy_df)}")
    logging.info(f"RSI:\n{calculate_rsi(dummy_df['close'])}")
    logging.info(f"MACD:\n{calculate_macd(dummy_df['close'])}")
    logging.info(f"ATR:\n{calculate_atr(dummy_df['high'], dummy_df['low'], dummy_df['close'])}")
    logging.info(f"VPVR:\n{calculate_vpvr(dummy_df, lookback=5, num_bins=50)}")
    logging.info(f"Sentiment Score: {calculate_sentiment_score()}")
    logging.info(f"On-Chain Metrics: {get_onchain_metrics()}")
    logging.info(f"LuxAlgo Trend/Reversal:\n{luxalgo_trend_reversal(dummy_df)}")
    logging.info(f"TrendSpider Patterns:\n{trendspider_pattern_recognition(dummy_df)}")
    logging.info(f"MetaStock Trend Slope:\n{metastock_trend_slope(dummy_df)}")