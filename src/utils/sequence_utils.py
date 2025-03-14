# src/utils/sequence_utils.py
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def create_sequences(features, targets, seq_length):
    """
    Create sequences for time series prediction with full sequence length.
    
    Args:
        features (np.ndarray): 2D array of shape (n_samples, n_features) containing input features.
        targets (np.ndarray): 1D array of shape (n_samples,) containing target values.
        seq_length (int): Length of each sequence.
    
    Returns:
        tuple: (X, y, past_time_features, past_observed_mask, future_values, future_time_features)
        - X: Shape (n_sequences, seq_length, n_features)
        - y: Shape (n_sequences,)
        - past_time_features: Shape (n_sequences, seq_length, 1)
        - past_observed_mask: Shape (n_sequences, seq_length, n_features)
        - future_values: Shape (n_sequences, 1, n_features)
        - future_time_features: Shape (n_sequences, 1, 1)
    """
    X, y = [], []
    past_time_features, past_observed_mask = [], []
    future_values, future_time_features = [], []
    for i in range(len(features) - seq_length):  # Adjusted to avoid out-of-bounds
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])  # Target is the next value after the sequence
        past_time_features.append(np.zeros((seq_length, 1)))
        past_observed_mask.append(np.ones((seq_length, len(features[0]))))
        future_values.append(np.zeros((1, len(features[0]))))
        future_time_features.append(np.zeros((1, 1)))
    return (np.array(X), np.array(y), np.array(past_time_features), 
            np.array(past_observed_mask), np.array(future_values), 
            np.array(future_time_features))