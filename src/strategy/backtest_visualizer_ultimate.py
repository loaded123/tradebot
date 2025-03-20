# src/strategy/backtest_visualizer_ultimate.py
"""
Main script for running the TradeBot backtest and visualization pipeline.

Module Map:
- Data Loading: `data_manager.load_and_preprocess_data` -> `data_fetcher` -> `data_preprocessor`
- Signal Generation: `signal_manager.generate_and_filter_signals` -> `signal_generator` -> `signal_filter`
- Model: `transformer_model.load_model`
- Backtesting: `backtest_engine.run_backtest`
- Optimization: `optimizer.optimize_weights`
- Visualization: `visualizer.plot_backtest_results`
- Risk Management: `risk_manager.manage_risk`

Key Integrations:
- **src.data.data_manager.load_and_preprocess_data**: Loads and preprocesses data, returning preprocessed_data, scaled_df, feature_scaler, and target_scaler.
- **src.strategy.signal_manager.generate_and_filter_signals**: Generates and filters signals, using optimized weights for signal combination.
- **src.models.transformer_model.load_model**: Loads the transformer model for predictions.
- **src.strategy.backtest_engine.run_backtest**: Executes the backtest to evaluate strategy performance.
- **src.strategy.optimizer.optimize_weights**: Optimizes signal weights to maximize the Sharpe ratio.
- **src.visualization.visualizer.plot_backtest_results**: Visualizes backtest results, including portfolio value and trades.
- **src.strategy.risk_manager.manage_risk**: Applies risk management, adjusting position sizes and enforcing drawdown limits.
- **src.constants.FEATURE_COLUMNS**: Defines the feature set for scaling and signal generation.

Future Considerations:
- Investigate alternative scaling methods (e.g., RobustScaler) if StandardScaler leads to issues with outliers.
- Add support for live trading by integrating with src.trading.paper_trader or real API execution.
- Parallelize weight optimization to improve performance for large weight ranges.

Dependencies:
- asyncio
- pandas
- numpy
- typing.List
- src.data.data_manager
- src.strategy.signal_manager
- src.models.transformer_model
- src.strategy.backtest_engine
- src.strategy.optimizer
- src.visualization.visualizer
- src.constants
- src.strategy.risk_manager
- winloop (for Windows event loop)

For a detailed project structure, see `docs/project_structure.md`.
"""
import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict
from itertools import product
from src.data.data_manager import load_and_preprocess_data
from src.strategy.signal_manager import generate_and_filter_signals
from src.models.transformer_model import load_model
from src.strategy.backtest_engine import run_backtest
from src.visualization.visualizer import plot_backtest_results
from src.constants import FEATURE_COLUMNS
from src.strategy.risk_manager import manage_risk

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def optimize_weights(
    scaled_df: pd.DataFrame,
    preprocessed_data: pd.DataFrame,
    model,
    train_columns: List[str],
    feature_scaler,
    target_scaler,
    params: Dict,
    current_balance: float
) -> Dict[str, float]:
    """
    Optimize weights for signal combination using grid search, ensuring weights sum to 1.

    Args:
        scaled_df (pd.DataFrame): Scaled feature data.
        preprocessed_data (pd.DataFrame): Unscaled data with indicators.
        model: Trained transformer model.
        train_columns (List[str]): List of feature columns.
        feature_scaler: Scaler for features.
        target_scaler: Scaler for target.
        params (Dict): Strategy parameters.
        current_balance (float): Initial balance for backtesting.

    Returns:
        Dict[str, float]: Best weights for signal combination.
    """
    best_sharpe = -float('inf')
    best_weights = None
    weight_combinations = []
    weight_range = np.arange(0.1, 0.6, 0.1)  # [0.1, 0.2, 0.3, 0.4, 0.5]

    # Generate all combinations where weights sum to 1
    for w1 in weight_range:
        for w2 in weight_range:
            for w3 in weight_range:
                for w4 in weight_range:
                    w5 = 1.0 - (w1 + w2 + w3 + w4)
                    if 0.1 <= w5 <= 0.5:  # Ensure all weights are within range
                        weight_combinations.append({
                            'WEIGHT_LUXALGO': w1,
                            'WEIGHT_TRENDSPIDER': w2,
                            'WEIGHT_SMRT_SCALPING': w3,
                            'WEIGHT_METASTOCK': w4,
                            'WEIGHT_MODEL_CONFIDENCE': w5
                        })

    for weights in weight_combinations:
        signal_data = await generate_and_filter_signals(
            scaled_df=scaled_df,
            preprocessed_data=preprocessed_data,
            model=model,
            train_columns=train_columns,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            params=params,
            weights=weights
        )
        if signal_data.empty or 'close' not in signal_data.columns:
            logger.warning(f"No signals generated with weights: {weights}, skipping...")
            continue

        # Apply risk management
        signal_data_with_risk, balance_after_risk = manage_risk(
            signal_data=signal_data,
            current_balance=current_balance,
            max_drawdown_pct=0.10,
            atr_multiplier=params['atr_multiplier'],
            recovery_volatility_factor=0.15,
            max_risk_pct=params['max_risk_pct'],
            min_position_size=0.002
        )

        # Run backtest
        results = run_backtest(signal_data_with_risk, preprocessed_data, balance_after_risk)
        sharpe = results['metrics']['sharpe_ratio']
        logger.info(f"Tested weights: LuxAlgo={weights['WEIGHT_LUXALGO']:.2f}, TrendSpider={weights['WEIGHT_TRENDSPIDER']:.2f}, "
                    f"SMRT={weights['WEIGHT_SMRT_SCALPING']:.2f}, Metastock={weights['WEIGHT_METASTOCK']:.2f}, "
                    f"Model={weights['WEIGHT_MODEL_CONFIDENCE']:.2f}, Sharpe={sharpe:.2f}")
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

    if best_weights is None:
        logger.warning("No valid weights found, using default weights")
        best_weights = {
            'WEIGHT_LUXALGO': 0.2,
            'WEIGHT_TRENDSPIDER': 0.2,
            'WEIGHT_SMRT_SCALPING': 0.2,
            'WEIGHT_METASTOCK': 0.2,
            'WEIGHT_MODEL_CONFIDENCE': 0.2
        }
    logger.info(f"Best weights found: {best_weights} with Sharpe Ratio: {best_sharpe:.2f}")
    return best_weights

async def main():
    """
    Main function to run the backtest and visualization pipeline with weight optimization.

    Notes:
        - Loads and preprocesses historical BTC/USD data.
        - Generates and filters trading signals using a transformer model and weighted indicator signals.
        - Optimizes weights for signal combination to maximize the Sharpe ratio.
        - Applies risk management to adjust position sizes and enforce drawdown limits.
        - Runs a backtest and visualizes the results.
    """
    logger.debug("Starting main()")
    symbol = 'BTC/USD'
    csv_path = r"C:\Users\Dennis\.vscode\tradebot\src\data\btc_usd_historical.csv"
    logger.info(f"Attempting to load data from CSV path: {csv_path}")
    logger.info(f"FEATURE_COLUMNS: {FEATURE_COLUMNS}")

    try:
        # Load and preprocess data
        preprocessed_data, scaled_df, feature_scaler, target_scaler = await load_and_preprocess_data(csv_path, symbol=symbol)
        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        logger.info(f"Preprocessed data columns: {list(preprocessed_data.columns)}")
        logger.info(f"Scaled DataFrame columns: {list(scaled_df.columns)}")

        # Load the transformer model
        model = load_model()
        logger.info("Model loaded successfully")

        # Use the market regime from preprocessed_data
        if 'market_regime' not in preprocessed_data.columns:
            raise ValueError("market_regime column not found in preprocessed_data")
        # Get the most recent market regime
        regime = preprocessed_data['market_regime'].iloc[-1]
        logger.info(f"Detected market regime: {regime}")

        # Define regime parameters based on the new categories (Low, Medium, High)
        regime_params = {
            'Low': {'rsi_threshold': 30, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 1.5, 'max_risk_pct': 0.08},
            'Medium': {'rsi_threshold': 30, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.0, 'max_risk_pct': 0.10},
            'High': {'rsi_threshold': 35, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 2.5, 'max_risk_pct': 0.12},
            'Neutral': {'rsi_threshold': 30, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.0, 'max_risk_pct': 0.08}
        }
        params = regime_params.get(regime, {'rsi_threshold': 30, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.0, 'max_risk_pct': 0.10})
        logger.info(f"Using regime parameters: {params}")

        # Initial balance
        current_balance = 17396.68

        # Optimize weights
        logger.info("Starting weight optimization")
        best_weights = await optimize_weights(
            scaled_df=scaled_df,
            preprocessed_data=preprocessed_data,
            model=model,
            train_columns=FEATURE_COLUMNS,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            params=params,
            current_balance=current_balance
        )

        # Generate final signals with best weights
        logger.info("Generating final signals with optimized weights")
        signal_data = await generate_and_filter_signals(
            scaled_df=scaled_df,
            preprocessed_data=preprocessed_data,
            model=model,
            train_columns=FEATURE_COLUMNS,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            params=params,
            weights=best_weights
        )
        if signal_data.empty or 'close' not in signal_data.columns:
            raise ValueError("No signals generated or 'close' missing from signal_data")
        logger.info(f"Signal data columns after generation: {signal_data.columns.tolist()}")
        logger.info(f"Signal data index: {signal_data.index[:5]}, last: {signal_data.index[-5:]}")
        logger.info(f"Final signals: Buy={(signal_data['signal'] == 1).sum()}, Sell={(signal_data['signal'] == -1).sum()}")

        # Apply risk management
        signal_data, current_balance = manage_risk(
            signal_data=signal_data,
            current_balance=current_balance,
            max_drawdown_pct=0.10,
            atr_multiplier=params['atr_multiplier'],
            recovery_volatility_factor=0.15,
            max_risk_pct=params['max_risk_pct'],
            min_position_size=0.002
        )
        logger.info(f"Balance after risk management: {current_balance:.2f}")

        # Run backtest
        logger.info("Starting backtest")
        results = run_backtest(signal_data, preprocessed_data, current_balance)
        logger.info(f"Backtest completed. Final portfolio value: {results['metrics']['final_value']:.2f}")

        # Visualize results
        logger.info("Generating backtest visualizations")
        plot_backtest_results(results['total'], signal_data, results['trades'], symbol)

        # Log final metrics
        metrics = results['metrics']
        logger.info(f"Final Backtest Metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
    logger.info(f"sys.path: {sys.path}")
    asyncio.run(main())