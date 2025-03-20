# src/strategy/signal_manager.py
"""
Orchestrates signal generation and filtering for the TradeBot, integrating signal generation,
filtering, and model predictions. This module acts as a coordinator between the raw preprocessed
data, the transformer model, and the signal generation and filtering logic.

Key Integrations:
- **src.strategy.signal_generator.generate_signals**: Generates raw trading signals using
  preprocessed data, scaled features, and model predictions. Passes parameters like RSI threshold,
  MACD settings, ATR multiplier, and optimized weights for signal combination.
- **src.strategy.signal_filter.filter_signals**: Applies filters (e.g., trend, volatility, confidence)
  to raw signals based on provided parameters.
- **src.models.transformer_model.TransformerPredictor**: Provides the transformer model instance
  for generating predictions, used in signal_generator.
- **src.strategy.backtest_visualizer_ultimate**: Calls this module to generate and filter signals
  during backtesting, passing optimized weights for signal combination.
- **src.constants**: Uses constants like WEIGHT_LUXALGO, WEIGHT_TRENDSPIDER, etc., as defaults
  if weights are not provided.

Future Considerations:
- Add support for dynamic parameter adjustment based on market regime changes during backtesting.
- Implement parallel processing for signal generation and filtering on large datasets.
- Consider adding a validation step to ensure weights sum to 1.0 if required by downstream logic.

Dependencies:
- asyncio
- src.strategy.signal_generator
- src.strategy.signal_filter
- src.models.transformer_model
"""

import asyncio
import logging
from src.strategy.signal_generator import generate_signals
from src.strategy.signal_filter import filter_signals
from src.models.transformer_model import TransformerPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def generate_and_filter_signals(
    scaled_df,
    preprocessed_data,
    model,
    train_columns,
    feature_scaler,
    target_scaler,
    params,
    weights=None
):
    """
    Generate and filter trading signals using preprocessed data and a transformer model.

    Args:
        scaled_df (pd.DataFrame): Scaled features for model input.
        preprocessed_data (pd.DataFrame): Unscaled preprocessed data with all features.
        model (TransformerPredictor): Trained transformer model for predictions.
        train_columns (list): List of columns used during model training.
        feature_scaler (StandardScaler): Scaler for features.
        target_scaler (StandardScaler): Scaler for target values.
        params (dict): Parameters for signal generation and filtering, including:
            - rsi_threshold (float): RSI threshold for signal generation.
            - macd_fast (int): Fast period for MACD.
            - macd_slow (int): Slow period for MACD.
            - atr_multiplier (float): Multiplier for ATR-based stop-loss/take-profit.
            - max_risk_pct (float): Maximum risk percentage per trade.
        weights (dict, optional): Weights for combining signals from different strategies.
            Expected keys: 'WEIGHT_LUXALGO', 'WEIGHT_TRENDSPIDER', 'WEIGHT_SMRT_SCALPING',
            'WEIGHT_METASTOCK', 'WEIGHT_MODEL_CONFIDENCE'. If None, defaults to constants.

    Returns:
        pd.DataFrame: Filtered signal data with columns including 'signal', 'close', etc.

    Notes:
        - Coordinates signal generation and filtering by delegating to signal_generator and signal_filter.
        - Passes optimized weights to signal_generator for weighted signal combination if provided.
        - Logs the number of signals before and after filtering for debugging.
    """
    logger.info("Generating and filtering signals")
    
    # Generate raw signals, only passing weights if provided
    kwargs = {
        "scaled_df": scaled_df,
        "preprocessed_data": preprocessed_data,
        "model": model,
        "train_columns": train_columns,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "rsi_threshold": params['rsi_threshold'],
        "macd_fast": params['macd_fast'],
        "macd_slow": params['macd_slow'],
        "atr_multiplier": params['atr_multiplier'],
        "max_risk_pct": params['max_risk_pct']
    }
    if weights is not None:
        kwargs["weights"] = weights
    
    signal_data = await generate_signals(**kwargs)
    
    # Log initial signal counts
    if not signal_data.empty:
        logger.info(f"Raw signals: Buy={(signal_data['signal'] == 1).sum()}, Sell={(signal_data['signal'] == -1).sum()}")
    
    # Filter signals
    filtered_signals = filter_signals(signal_data, params=params)
    
    # Log filtered signal counts
    if not filtered_signals.empty:
        logger.info(f"Filtered signals: Buy={(filtered_signals['signal'] == 1).sum()}, Sell={(filtered_signals['signal'] == -1).sum()}")
    
    return filtered_signals