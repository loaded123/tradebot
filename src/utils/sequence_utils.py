# src/utils/sequence_utils.py
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def create_sequences(features, targets, seq_length, timestamps=None):
    """
    Create sequences for time series prediction with full sequence length and proper time features.
    
    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing input features.
        targets (np.ndarray): 1D array of shape (n_samples,) containing target values.
        seq_length (int): Length of each sequence.
        timestamps (pd.Index or pd.Series, optional): Timestamps corresponding to the features (e.g., DataFrame index).
    
    Returns:
        tuple: (X, y, past_time_features, past_observed_mask, future_values, future_time_features)
        - X: Shape (n_sequences, seq_length, n_features)
        - y: Shape (n_sequences,)
        - past_time_features: Shape (n_sequences, seq_length, 5) with hour, day of week, month, day of month, quarter
        - past_observed_mask: Shape (n_sequences, seq_length, n_features)
        - future_values: Shape (n_sequences, 1, n_features)
        - future_time_features: Shape (n_sequences, 1, 5) with hour, day of week, month, day of month, quarter
    """
    # Check for NaN or inf in input features and targets
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logging.warning(f"NaN or inf detected in features before creating sequences. NaN count: {np.isnan(features).sum()}, Inf count: {np.isinf(features).sum()}")
    if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
        logging.warning(f"NaN or inf detected in targets before creating sequences. NaN count: {np.isnan(targets).sum()}, Inf count: {np.isinf(targets).sum()}")

    # Validate timestamps
    if timestamps is not None:
        if len(timestamps) != len(features):
            raise ValueError(f"Length of timestamps ({len(timestamps)}) must match length of features ({len(features)})")
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)
    else:
        logging.warning("Timestamps not provided; using default time features (zeros).")
        timestamps = pd.date_range(start="2020-01-01", periods=len(features), freq="H")

    X, y = [], []
    past_time_features, past_observed_mask = [], []
    future_values, future_time_features = [], []

    for i in range(len(features) - seq_length):
        # Feature sequence
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])

        # Past time features (hour of day, day of week, month, day of month, quarter)
        past_timestamps = timestamps[i:i + seq_length]
        past_hour = past_timestamps.hour / 23.0  # Normalize to [0, 1]
        past_dayofweek = past_timestamps.dayofweek / 6.0  # Normalize to [0, 1]
        past_month = (past_timestamps.month - 1) / 11.0  # Normalize to [0, 1]
        past_dayofmonth = (past_timestamps.day - 1) / 30.0  # Normalize to [0, 1]
        past_quarter = (past_timestamps.quarter - 1) / 3.0  # Normalize to [0, 1]
        past_features = np.stack([past_hour, past_dayofweek, past_month, past_dayofmonth, past_quarter], axis=-1)  # Shape: (seq_length, 5)
        past_time_features.append(past_features)

        # Past observed mask
        past_observed_mask.append(np.ones((seq_length, features.shape[1])))

        # Future values (placeholder, not used in current model)
        future_values.append(np.zeros((1, features.shape[1])))

        # Future time features (for the next timestep)
        future_timestamp = timestamps[i + seq_length]
        future_hour = future_timestamp.hour / 23.0
        future_dayofweek = future_timestamp.dayofweek / 6.0
        future_month = (future_timestamp.month - 1) / 11.0
        future_dayofmonth = (future_timestamp.day - 1) / 30.0
        future_quarter = (future_timestamp.quarter - 1) / 3.0
        future_features = np.array([[future_hour, future_dayofweek, future_month, future_dayofmonth, future_quarter]])  # Shape: (1, 5)
        future_time_features.append(future_features)

    return (
        np.array(X),  # Shape: (n_sequences, seq_length, n_features)
        np.array(y),  # Shape: (n_sequences,)
        np.array(past_time_features),  # Shape: (n_sequences, seq_length, 5)
        np.array(past_observed_mask),  # Shape: (n_sequences, seq_length, n_features)
        np.array(future_values),  # Shape: (n_sequences, 1, n_features)
        np.array(future_time_features)  # Shape: (n_sequences, 1, 5)
    )