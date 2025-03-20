# src/data/data_manager.py
"""
This module manages the loading and preprocessing of market data for the trading bot,
integrating data fetching, preprocessing, and feature scaling. It ensures data consistency
across training, backtesting, and live trading by handling scalers and feature alignment.

Key Integrations:
- **src.data.data_fetcher.fetch_historical_data**: Asynchronously fetches historical data
  from a CSV file or external source based on the provided symbol and path.
- **src.data.data_preprocessor.preprocess_data**: Applies technical indicators, signals,
  and market regime detection to raw data, using FEATURE_COLUMNS from src.constants.
  - Replaces simulated indicators (e.g., luxalgo_trend_reversal) with open-source alternatives
    (e.g., SuperTrend, Ichimoku Cloud, EMA Slope).
  - Computes 'target' as the next period's log return, which is scaled here.
- **src.constants.FEATURE_COLUMNS**: Defines the feature set expected by downstream modules
  (e.g., signal_generator, transformer_model), ensuring consistency in column names.
- **src.strategy.signal_generator**: Relies on scaled features in scaled_df, including
  'macd_signal', for signal generation.
- **src.models.transformer_model**: Uses feature_scaler and target_scaler for model inference,
  requiring feature alignment with training data. The target_scaler is fitted to the actual
  min/max of log returns to improve prediction accuracy.
- **src/models/train_transformer_model.py**: Uses the preprocessed data and scalers returned
  by this module to train the Transformer model.

Future Considerations:
- Implement dynamic scaler retraining during backtesting if new features are added to FEATURE_COLUMNS.
- Add support for incremental data loading to handle large datasets efficiently.
- Consider validating scaler compatibility with preprocessed_data before transformation.

Dependencies:
- pandas
- sklearn.preprocessing.MinMaxScaler
- joblib
- numpy
- src.data.data_fetcher
- src.data.data_preprocessor
- src.constants
"""

import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import numpy as np
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data
from src.constants import FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def load_and_preprocess_data(csv_path: str, symbol: str = "BTC/USD") -> tuple:
    """
    Load and preprocess data for training or backtesting.

    Args:
        csv_path (str): Path to the CSV file containing historical data.
        symbol (str): Trading pair symbol (default: "BTC/USD").

    Returns:
        tuple: (preprocessed_data, scaled_df, feature_scaler, target_scaler)
            - preprocessed_data (pd.DataFrame): Data with all computed features, including scaled 'target'.
            - scaled_df (pd.DataFrame): Scaled features for model input, excluding 'market_regime'.
            - feature_scaler (MinMaxScaler): Fitted scaler for features.
            - target_scaler (MinMaxScaler): Fitted scaler for targets, using the actual min/max of log returns.

    Notes:
        - Loads data using fetch_historical_data and preprocesses it with preprocess_data.
        - Attempts to load pre-trained scalers; if incompatible or missing, fits new scalers
          to the current preprocessed_data and saves them.
        - Ensures all FEATURE_COLUMNS (excluding 'market_regime') are included in scaled_df,
          addressing potential mismatches with pre-trained scalers.
        - Logs column adjustments and scaling details for debugging.
        - The target_scaler is fitted to the actual range of 'target' (log returns) to avoid
          the hardcoded (-0.1, 0.1) range observed in previous runs, improving model predictions.
    """
    logger.info(f"Loading data for symbol: {symbol} from CSV: {csv_path}")
    data = await fetch_historical_data(symbol=symbol, csv_path=csv_path)

    preprocessed_data = preprocess_data(data)
    logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
    logger.info(f"Preprocessed data columns: {list(preprocessed_data.columns)}")

    # Load existing scalers
    feature_scaler_path = "feature_scaler.pkl"
    target_scaler_path = "target_scaler.pkl"
    
    feature_scaler = None
    target_scaler = None
    if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            target_scaler = joblib.load(target_scaler_path)
            logger.info("Loaded existing feature and target scalers")
        except Exception as e:
            logger.warning(f"Failed to load scalers due to: {e}. Fitting new scalers.")
    else:
        logger.warning("Scaler files not found. Fitting new scalers.")

    # Determine columns to scale (all numeric columns from FEATURE_COLUMNS present in preprocessed_data)
    numeric_cols = [col for col in FEATURE_COLUMNS if col in preprocessed_data.columns and col != 'market_regime']
    missing_cols = [col for col in FEATURE_COLUMNS if col not in preprocessed_data.columns and col != 'market_regime']
    if missing_cols:
        logger.warning(f"Missing columns from FEATURE_COLUMNS in preprocessed_data: {missing_cols}")

    if not numeric_cols:
        raise ValueError("No numeric columns to scale from FEATURE_COLUMNS")

    # Fit or transform features
    if feature_scaler is None or not all(col in getattr(feature_scaler, 'feature_names_in_', []) for col in numeric_cols):
        logger.info("Fitting new feature scaler due to incompatibility or absence")
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(preprocessed_data[numeric_cols].fillna(0))
        joblib.dump(feature_scaler, feature_scaler_path)
        logger.info(f"Saved new feature scaler to {feature_scaler_path}")

    scaled_features = feature_scaler.transform(preprocessed_data[numeric_cols].fillna(0))
    scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols, index=preprocessed_data.index)
    logger.info(f"Scaled DataFrame columns: {list(scaled_df.columns)}")

    # Fit or transform targets (using actual min/max of log returns)
    target_data = preprocessed_data[['target']].dropna()
    if target_data.empty:
        logger.warning("Target data is empty after dropping NaNs, using default scaling range [-0.1, 0.1]")
        target_scaler = MinMaxScaler(feature_range=(-0.1, 0.1))
        target_scaler.fit(np.array([[-0.1], [0.1]]))
    elif target_scaler is None or not np.allclose(target_scaler.data_min_, target_data.min().values) or not np.allclose(target_scaler.data_max_, target_data.max().values):
        logger.info("Fitting new target scaler to actual range due to incompatibility or absence")
        target_scaler = MinMaxScaler()
        target_scaler.fit(target_data)
        joblib.dump(target_scaler, target_scaler_path)
        logger.info(f"Saved new target scaler to {target_scaler_path} with min={target_data.min().values[0]:.4f}, max={target_data.max().values[0]:.4f}")
    else:
        logger.info("Using existing target scaler")

    scaled_target = target_scaler.transform(preprocessed_data[['target']].fillna(0))
    preprocessed_data['target'] = scaled_target

    return preprocessed_data, scaled_df, feature_scaler, target_scaler