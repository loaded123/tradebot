# src/strategy/optimizer.py
"""
This module optimizes the weights of trading indicators to maximize the Sharpe ratio using a grid search approach.
It integrates with the signal generation, filtering, and backtesting pipeline to evaluate different weight combinations.

Key Integrations:
- **src.strategy.signal_generator.generate_signals**: Generates signals for each weight combination, passing
  weights directly to influence signal combination.
- **src.strategy.signal_filter.filter_signals**: Filters the generated signals before backtesting.
- **src.models.transformer_model.load_model**: Loads the transformer model for signal generation.
- **src.strategy.backtest_engine.backtest_strategy**: Runs a backtest to compute the Sharpe ratio for each weight set.
- **src.data.data_preprocessor.scale_features**: Scales features for model input during signal generation.
- **src.constants.FEATURE_COLUMNS**: Defines the feature set used for scaling and signal generation.
- **src.strategy.backtest_visualizer_ultimate**: Calls this module to optimize weights during the backtesting pipeline.

Future Considerations:
- Implement a more efficient optimization algorithm (e.g., Bayesian optimization) to replace grid search.
- Add support for dynamic weight optimization based on market regime or recent performance.
- Parallelize the grid search to reduce computation time for large weight ranges.

Dependencies:
- asyncio
- pandas
- joblib
- itertools.product
- src.strategy.signal_generator
- src.models.transformer_model
- src.strategy.backtest_engine
- src.data.data_preprocessor
- src.strategy.signal_filter
- src.constants
"""

import asyncio
import logging
import pandas as pd
import joblib
from itertools import product
from src.strategy.signal_generator import generate_signals
from src.models.transformer_model import load_model
from src.strategy.backtest_engine import backtest_strategy
from src.data.data_preprocessor import scale_features
from src.strategy.signal_filter import filter_signals
from src.constants import FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def optimize_weights(signal_data: pd.DataFrame, preprocessed_data: pd.DataFrame, initial_capital: float, weight_ranges: dict) -> tuple:
    """
    Optimize indicator weights using grid search to maximize the Sharpe ratio.

    Args:
        signal_data (pd.DataFrame): DataFrame with initial signals (used as a template).
        preprocessed_data (pd.DataFrame): Preprocessed market data with all features.
        initial_capital (float): Initial capital for backtesting.
        weight_ranges (dict): Dictionary of weight ranges for each indicator.
            Expected keys: 'luxalgo', 'trendspider', 'smrt', 'metastock', 'model'.

    Returns:
        tuple: (best_weights, best_sharpe)
            - best_weights (tuple): Optimal weights (luxalgo, trendspider, smrt, metastock, model).
            - best_sharpe (float): Best Sharpe ratio achieved with the optimal weights.

    Notes:
        - Performs a grid search over the provided weight ranges.
        - For each weight combination, regenerates signals, filters them, and runs a backtest.
        - Tracks the best Sharpe ratio and corresponding weights.
        - Logs each tested combination and the final best result.
    """
    logger.info("Starting weight optimization via grid search")
    best_sharpe = -float('inf')
    best_weights = None

    # Prepare scaled data outside the loop to avoid redundant scaling
    scaled_df = scale_features(preprocessed_data[FEATURE_COLUMNS], feature_columns=FEATURE_COLUMNS)
    model = load_model()
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')

    # Grid search over all weight combinations
    for w_lux, w_trend, w_smrt, w_meta, w_model in product(
        weight_ranges['luxalgo'], weight_ranges['trendspider'],
        weight_ranges['smrt'], weight_ranges['metastock'], weight_ranges['model']
    ):
        # Create weights dictionary
        weights = {
            'WEIGHT_LUXALGO': w_lux,
            'WEIGHT_TRENDSPIDER': w_trend,
            'WEIGHT_SMRT_SCALPING': w_smrt,
            'WEIGHT_METASTOCK': w_meta,
            'WEIGHT_MODEL_CONFIDENCE': w_model
        }

        # Regenerate signals with the current weights
        signal_data_copy = await generate_signals(
            scaled_df=scaled_df,
            preprocessed_data=preprocessed_data,
            model=model,
            train_columns=FEATURE_COLUMNS,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            rsi_threshold=30,
            macd_fast=12,
            macd_slow=26,
            atr_multiplier=1.0,
            max_risk_pct=0.10,
            weights=weights  # Pass weights directly to generate_signals
        )
        signal_data_copy = filter_signals(signal_data_copy)

        # Run backtest to evaluate the Sharpe ratio
        results = backtest_strategy(signal_data_copy, preprocessed_data, initial_capital)
        sharpe = results['metrics']['sharpe_ratio']

        # Update best weights if the current Sharpe ratio is better
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = (w_lux, w_trend, w_smrt, w_meta, w_model)

        logger.info(f"Tested weights: LuxAlgo={w_lux:.2f}, TrendSpider={w_trend:.2f}, SMRT={w_smrt:.2f}, "
                    f"Metastock={w_meta:.2f}, Model={w_model:.2f}, Sharpe={sharpe:.2f}")

    logger.info(f"Best weights found: LuxAlgo={best_weights[0]:.2f}, TrendSpider={best_weights[1]:.2f}, "
                f"SMRT={best_weights[2]:.2f}, Metastock={best_weights[3]:.2f}, Model={best_weights[4]:.2f}, "
                f"Best Sharpe={best_sharpe:.2f}")
    return best_weights, best_sharpe