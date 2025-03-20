# src/strategy/parameter_optimizer.py
"""
This module dynamically adapts and optimizes strategy parameters based on market conditions.
It adjusts parameters like RSI thresholds, MACD periods, and ATR multipliers to align with
detected market regimes and volatility.

Key Integrations:
- **src.strategy.market_regime.detect_market_regime**: Detects the current market regime
  (e.g., 'Bullish High Volatility') to adapt parameters.
- **src.strategy.indicators.compute_adx**: Computes the ADX indicator for trend strength,
  used in parameter optimization.
- **src.constants.USE_LUXALGO_SIGNALS**: Toggles LuxAlgo-specific parameter optimization.
- **src.strategy.backtest_visualizer_ultimate**: Provides the initial parameters and market data
  for adaptation during backtesting.

Future Considerations:
- Implement dynamic weight optimization based on backtesting results (as noted in optimize_parameters).
- Add support for more advanced optimization techniques (e.g., genetic algorithms).
- Incorporate additional market indicators (e.g., VIX, sentiment) for parameter adjustment.

Dependencies:
- pandas
- numpy
- typing.Dict
- src.strategy.market_regime
- src.strategy.indicators
- src.constants
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

from src.strategy.market_regime import detect_market_regime
from src.strategy.indicators import compute_adx
from src.constants import USE_LUXALGO_SIGNALS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def adapt_strategy_parameters(scaled_df: pd.DataFrame, initial_rsi_threshold: float = 35) -> Dict[str, float]:
    """
    Adapt strategy parameters based on market conditions and an initial RSI threshold.

    Args:
        scaled_df (pd.DataFrame): Scaled market data with features like 'close', 'price_volatility'.
        initial_rsi_threshold (float): Initial RSI threshold to base buy/sell thresholds on.

    Returns:
        Dict[str, float]: Adapted parameters including:
            - rsi_buy_threshold: RSI threshold for buy signals.
            - rsi_sell_threshold: RSI threshold for sell signals.
            - macd_fast: Fast period for MACD.
            - macd_slow: Slow period for MACD.
            - atr_multiplier: Multiplier for ATR-based stop-loss/take-profit.
            - max_risk_pct: Maximum risk percentage per trade.

    Notes:
        - Uses market regime detection to adjust parameters dynamically.
        - Adjusts RSI thresholds relative to the initial_rsi_threshold based on market conditions.
    """
    market_regime = detect_market_regime(scaled_df, window=1440)
    # Adjust RSI thresholds relative to the initial_rsi_threshold
    regime_params = {
        'Bullish Low Volatility': {
            'rsi_buy_threshold': initial_rsi_threshold,  # Use initial_rsi_threshold as base
            'rsi_sell_threshold': initial_rsi_threshold + 35,
            'macd_fast': 10,
            'macd_slow': 20,
            'atr_multiplier': 1.0,
            'max_risk_pct': 0.10
        },
        'Bullish High Volatility': {
            'rsi_buy_threshold': initial_rsi_threshold + 5,
            'rsi_sell_threshold': initial_rsi_threshold + 40,
            'macd_fast': 12,
            'macd_slow': 26,
            'atr_multiplier': 1.0,
            'max_risk_pct': 0.10
        },
        'Bearish Low Volatility': {
            'rsi_buy_threshold': initial_rsi_threshold - 5,
            'rsi_sell_threshold': initial_rsi_threshold + 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'atr_multiplier': 1.0,
            'max_risk_pct': 0.10
        },
        'Bearish High Volatility': {
            'rsi_buy_threshold': initial_rsi_threshold - 10,
            'rsi_sell_threshold': initial_rsi_threshold + 35,
            'macd_fast': 15,
            'macd_slow': 30,
            'atr_multiplier': 1.0,
            'max_risk_pct': 0.10
        },
        'Neutral': {
            'rsi_buy_threshold': initial_rsi_threshold,
            'rsi_sell_threshold': initial_rsi_threshold + 35,
            'macd_fast': 12,
            'macd_slow': 26,
            'atr_multiplier': 1.0,
            'max_risk_pct': 0.08
        },
    }
    default_params = {
        'rsi_buy_threshold': initial_rsi_threshold,
        'rsi_sell_threshold': initial_rsi_threshold + 35,
        'macd_fast': 12,
        'macd_slow': 26,
        'atr_multiplier': 1.0,
        'max_risk_pct': 0.10
    }
    params = regime_params.get(market_regime, default_params)
    logger.info(f"Adapted parameters for regime {market_regime}: {params}")
    return params

def optimize_parameters(df: pd.DataFrame, base_params: Dict[str, float]) -> Dict[str, float]:
    """
    Dynamically optimize signal parameters based on market conditions.

    Args:
        df (pd.DataFrame): Preprocessed market data with features like 'close', 'price_volatility'.
        base_params (Dict[str, float]): Base parameters to optimize, including:
            - rsi_buy_threshold
            - rsi_sell_threshold
            - macd_fast
            - macd_slow
            - atr_multiplier
            - max_risk_pct

    Returns:
        Dict[str, float]: Optimized parameters with the same keys as base_params.

    Notes:
        - Adjusts parameters based on price volatility and ADX (trend strength).
        - Only applies LuxAlgo-specific optimizations if USE_LUXALGO_SIGNALS is True.
        - Includes a placeholder for future weight optimization.
    """
    if not USE_LUXALGO_SIGNALS:
        logger.info("LuxAlgo parameter optimization disabled")
        return base_params

    try:
        price_volatility = df['close'].pct_change().rolling(window=24, min_periods=1).std().fillna(0.0)
        adx = compute_adx(df, period=14)

        optimized_params = base_params.copy()
        if price_volatility.iloc[-1] > 0.02:
            optimized_params['rsi_buy_threshold'] = max(20, base_params['rsi_buy_threshold'] - 5)
            optimized_params['rsi_sell_threshold'] = min(80, base_params['rsi_sell_threshold'] + 5)
            optimized_params['atr_multiplier'] = base_params['atr_multiplier'] * 1.2
        elif adx.iloc[-1] > 25:
            optimized_params['rsi_buy_threshold'] = min(40, base_params['rsi_buy_threshold'] + 5)
            optimized_params['rsi_sell_threshold'] = max(60, base_params['rsi_sell_threshold'] - 5)
            optimized_params['atr_multiplier'] = base_params['atr_multiplier'] * 0.8

        logger.info(f"Optimized parameters: {optimized_params}")
        # Placeholder for future weight optimization
        # TODO: Implement dynamic weight adjustment based on backtesting results
        return optimized_params
    except Exception as e:
        logger.error(f"LuxAlgo parameter optimization failed: {e}")
        return base_params