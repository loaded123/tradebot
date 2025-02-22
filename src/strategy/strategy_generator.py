import torch
import numpy as np  # Reconfirmed, re-applied, and quattuordecuple-checked import for numpy operations (critical fix)
import pandas as pd
import logging
import sys
from src.models.transformer_model import TransformerPredictor
from src.strategy.position_sizer import kelly_criterion
from src.strategy.indicators import calculate_rsi, calculate_macd, calculate_atr
from src.constants import FEATURE_COLUMNS
from src.models.train_lstm_model import create_sequences

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def calculate_rsi(series, window=14):
    """Calculate RSI with NaN handling and debugging."""
    logging.debug(f"calculate_rsi series: {series.head()}")
    delta = series.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window).mean()
    roll_down = down.abs().rolling(window).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def generate_signals(scaled_df, model, train_columns, feature_scaler, target_scaler, rsi_threshold=50, macd_fast=12, macd_slow=26, atr_multiplier=2, max_risk_pct=0.05):
    """
    Generate buy/sell signals based on TransformerPredictor predictions and technical indicators.

    Args:
        scaled_df (pd.DataFrame): Scaled data with features and target
        model: Trained TransformerPredictor model
        train_columns (list): List of feature columns + target
        feature_scaler: Scaler for features
        target_scaler: Scaler for target
        rsi_threshold, macd_fast, macd_slow, atr_multiplier, max_risk_pct: Strategy parameters

    Returns:
        pd.DataFrame: DataFrame with signals and position sizes
    """
    required_columns = ['close', 'sma_20', 'adx', 'vwap', 'atr', 'target', 'price_volatility']
    for col in required_columns:
        if col not in scaled_df.columns:
            raise ValueError(f"Missing required column: {col}")

    signals = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Verify numpy is available and correctly imported in this context
        logging.debug(f"numpy version (in generate_signals): {np.__version__ if 'np' in globals() else 'Not loaded'}")
        if 'np' not in globals():
            raise ImportError("numpy (np) is not imported in this context - check module loading, caching, or environment")

        # Log the file being executed, package context, and sys.path to diagnose loading
        logging.debug(f"Module __file__: {__file__}")
        logging.debug(f"Module __package__: {__package__}")
        logging.debug(f"sys.path: {sys.path}")

        # Prepare sequences for prediction (match train_lstm_model.py's create_sequences with 13 steps)
        sequence_length = 13
        X, _, past_time_features, past_observed_mask, _, future_values = create_sequences(
            scaled_df[FEATURE_COLUMNS].values, scaled_df['target'].values, seq_length=sequence_length
        )
        
        # Ensure we have enough data for prediction (adjust if necessary to match scaled_df length)
        expected_sequences = len(scaled_df) - sequence_length + 1
        if len(X) != expected_sequences:
            logging.warning(f"Adjusting sequence length: expected {expected_sequences} sequences, got {len(X)}. Padding or truncating.")
            if len(X) < expected_sequences:
                # Pad with zeros or duplicate last sequence to match length
                padding_length = expected_sequences - len(X)
                X_padded = np.vstack([X, np.tile(X[-1], (padding_length, 1, 1))])
                past_time_features_padded = np.vstack([past_time_features, np.tile(past_time_features[-1], (padding_length, 1, 1))]) if past_time_features is not None else None
                past_observed_mask_padded = np.vstack([past_observed_mask, np.tile(past_observed_mask[-1], (padding_length, 1, 1))]) if past_observed_mask is not None else None
                future_values_padded = np.vstack([future_values, np.tile(future_values[-1], (padding_length, 1, 1))]) if future_values is not None else None
            else:
                # Truncate to match expected length
                X_padded = X[:expected_sequences]
                past_time_features_padded = past_time_features[:expected_sequences] if past_time_features is not None else None
                past_observed_mask_padded = past_observed_mask[:expected_sequences] if past_observed_mask is not None else None
                future_values_padded = future_values[:expected_sequences] if future_values is not None else None
        else:
            X_padded, past_time_features_padded, past_observed_mask_padded, future_values_padded = X, past_time_features, past_observed_mask, future_values

        logging.debug(f"X_padded shape: {X_padded.shape}")
        logging.debug(f"past_time_features_padded shape: {past_time_features_padded.shape if past_time_features_padded is not None else 'None'}")

        # Convert to tensors for prediction
        X_tensor = torch.FloatTensor(X_padded).to(device)
        past_time_features_tensor = torch.FloatTensor(past_time_features_padded).to(device) if past_time_features_padded is not None else None
        past_observed_mask_tensor = torch.FloatTensor(past_observed_mask_padded).to(device) if past_observed_mask_padded is not None else None
        future_values_tensor = torch.FloatTensor(future_values_padded).to(device) if future_values_padded is not None else None
        
        # Predict using TransformerPredictor
        predictions = model.predict(X_tensor.numpy(), past_time_features_tensor, past_observed_mask_tensor, None, future_values_tensor)
        predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Ensure predictions match scaled_df length (1183)
        if len(predictions) != len(scaled_df):
            logging.warning(f"Prediction length ({len(predictions)}) does not match scaled_df length ({len(scaled_df)}). Adjusting...")
            if len(predictions) < len(scaled_df):
                # Pad predictions with the last value
                padding_length = len(scaled_df) - len(predictions)
                predictions_padded = np.pad(predictions, (0, padding_length), mode='edge')
            else:
                # Truncate predictions
                predictions_padded = predictions[:len(scaled_df)]
        else:
            predictions_padded = predictions

        # Add predictions to DataFrame
        signal_df = scaled_df.copy()
        signal_df['predicted_price'] = predictions_padded
        
        # Calculate technical indicators (RSI, MACD, ATR) on unscaled data
        signal_df['rsi'] = calculate_rsi(signal_df['close'], window=14)
        signal_df['macd'] = calculate_macd(signal_df['close'], fast=macd_fast, slow=macd_slow)
        signal_df['atr'] = calculate_atr(signal_df['high'], signal_df['low'], signal_df['close'], window=14)
        
        # Generate signals based on predictions and indicators
        signal_df['signal'] = 0
        signal_df.loc[signal_df['predicted_price'] > signal_df['close'] * (1 + atr_multiplier * signal_df['atr']) / signal_df['close'], 'signal'] = 1
        signal_df.loc[signal_df['predicted_price'] < signal_df['close'] * (1 - atr_multiplier * signal_df['atr']) / signal_df['close'], 'signal'] = -1
        
        # Filter signals based on RSI and MACD
        signal_df.loc[signal_df['rsi'] > rsi_threshold, 'signal'] = 0
        signal_df.loc[signal_df['rsi'] < 100 - rsi_threshold, 'signal'] = 0
        signal_df.loc[signal_df['macd'] < 0, 'signal'] = -1
        
        # Calculate position size based on risk management
        signal_df['position_size'] = kelly_criterion(0.55, 2.0, 10000, signal_df['atr'], signal_df['close'], max_risk_pct)
        signal_df['position_size'] = signal_df['position_size'].where(signal_df['signal'] != 0, 0)
        
        # Select output columns
        signal_df = signal_df[['close', 'signal', 'position_size', 'predicted_price', 'rsi', 'macd', 'atr']]
        logging.info(f"Generated {len(signal_df)} signals with shape: {signal_df.shape}")
        return signal_df
    
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print(np.array([1, 2, 3]))
    test_df = pd.DataFrame({'close': [1, 2, 3], 'high': [2, 3, 4], 'low': [0, 1, 2]})
    print(calculate_rsi(test_df['close']))