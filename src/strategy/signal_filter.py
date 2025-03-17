# src/strategy/signal_filter.py
import logging
import pandas as pd
import numpy as np
from typing import Dict

from src.strategy.indicators import calculate_atr
from src.constants import USE_SMRT_SCALPING

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
signals_logger = logging.getLogger('signals')

def filter_signals(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized filter for signals to enforce a minimum hold period."""
    filtered_df = signal_df.copy()
    min_hold_period = 6
    min_confidence = 0.25

    filtered_df['time_diff'] = filtered_df.index.to_series().diff().dt.total_seconds() / 3600
    filtered_df['cum_time'] = filtered_df['time_diff'].cumsum().fillna(0)
    filtered_df['dynamic_min_hold'] = np.where(filtered_df['price_volatility'] > filtered_df['price_volatility'].mean(), min_hold_period, 3)

    rsi_condition = (filtered_df['rsi'] < filtered_df['rsi_buy_threshold']) & (filtered_df['signal'] == 1) | \
                   (filtered_df['rsi'] > filtered_df['rsi_sell_threshold']) & (filtered_df['signal'] == -1)
    macd_condition = (filtered_df['macd'] > filtered_df['macd_signal']) & (filtered_df['signal'] == 1) | \
                     (filtered_df['macd'] < filtered_df['macd_signal']) & (filtered_df['signal'] == -1)
    confirming_indicators = (rsi_condition | macd_condition).astype(int) + \
                           (filtered_df['signal_confidence'] >= 0.4).astype(int)

    filtered_df['signal_valid'] = (filtered_df['cum_time'] >= filtered_df['dynamic_min_hold'].shift(1, fill_value=0)) & \
                                 (filtered_df['signal'] != 0) & \
                                 (filtered_df['signal_confidence'] >= min_confidence) & \
                                 (confirming_indicators >= 1)
    filtered_df['signal'] = np.where(filtered_df['signal_valid'], filtered_df['signal'], 0)

    filtered_df.drop(columns=['time_diff', 'cum_time', 'dynamic_min_hold', 'signal_valid'], inplace=True)
    signals_logger.info(f"Filtered signals: Total valid signals = {(filtered_df['signal'] != 0).sum()}")
    return filtered_df

def smrt_scalping_signals(df: pd.DataFrame, atr_multiplier: float = 1.0, fee_rate: float = 0.001) -> pd.Series:
    """Generate scalping signals inspired by SMRT Algo."""
    if not USE_SMRT_SCALPING:
        logging.info("SMRT Algo scalping signals disabled")
        return pd.Series(0, index=df.index)

    try:
        atr = calculate_atr(df['high'], df['low'], df['close'], period=14)
        signals = pd.Series(0, index=df.index)

        price_change = df['close'].pct_change()
        threshold = atr * atr_multiplier / df['close']

        signals[(price_change > threshold) & (df['close'] > df['close'].shift(1))] = 1
        signals[(price_change < -threshold) & (df['close'] < df['close'].shift(1))] = -1

        expected_profit = atr * atr_multiplier
        min_profit = df['close'] * fee_rate * 2
        signals[expected_profit < min_profit] = 0

        logging.info(f"Generated SMRT Algo scalping signals: {signals.value_counts().to_dict()}")
        return signals
    except Exception as e:
        logging.error(f"SMRT Algo scalping signals computation failed: {e}")
        return pd.Series(0, index=df.index)