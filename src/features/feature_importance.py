# src/features/feature_importance.py
"""
This module analyzes the importance of features used in the Transformer model for price prediction.
It uses permutation importance to evaluate the impact of each feature on model performance.

Key Integrations:
- **src.data.data_manager.load_and_preprocess_data**: Loads and preprocesses the historical data.
- **src.models.transformer_model.TransformerPredictor**: Loads the trained model for evaluation.
- **src.utils.sequence_utils.create_sequences**: Creates sequences for model input.
- **src.constants.FEATURE_COLUMNS**: Defines the feature set to analyze.

Future Considerations:
- Use SHAP values for more detailed feature importance analysis.
- Parallelize permutation importance computation for faster analysis.

Dependencies:
- pandas
- numpy
- torch
- sklearn
- src.data.data_manager
- src.models.transformer_model
- src.utils.sequence_utils
- src.constants
"""

import sys
import os
import asyncio
import json
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from src.data.data_manager import load_and_preprocess_data
from src.models.transformer_model import TransformerPredictor
from src.utils.sequence_utils import create_sequences
from src.constants import FEATURE_COLUMNS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('src.features.feature_importance')

def permutation_importance(model, X, y, past_time_features, device, metric_func, batch_size=64, n_repeats=10):
    """
    Compute permutation importance for each feature using batched inference.

    Args:
        model: Trained Transformer model.
        X (np.ndarray): Input features, shape (n_samples, seq_length, n_features).
        y (np.ndarray): Target values, shape (n_samples, forecast_steps).
        past_time_features (np.ndarray): Past time features, shape (n_samples, seq_length, n_time_features).
        device: Device to run the model on (CPU/GPU).
        metric_func: Function to compute the evaluation metric (e.g., mean_squared_error).
        batch_size (int): Batch size for inference.
        n_repeats (int): Number of times to permute each feature.

    Returns:
        dict: Feature importance scores.
    """
    model.eval()
    # Create DataLoader for batched inference
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y),
        torch.FloatTensor(past_time_features)
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute baseline predictions
    baseline_predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y, batch_past_time in data_loader:
            batch_X, batch_past_time = batch_X.to(device), batch_past_time.to(device)
            predictions = model(batch_X, batch_past_time).cpu().numpy()
            baseline_predictions.append(predictions)
            actuals.append(batch_y.numpy())
    baseline_predictions = np.concatenate(baseline_predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    baseline_score = metric_func(actuals, baseline_predictions)

    # Log prediction statistics to diagnose issues
    logger.info(f"Baseline predictions shape: {baseline_predictions.shape}")
    logger.info(f"Baseline predictions mean: {np.mean(baseline_predictions):.6f}, std: {np.std(baseline_predictions):.6f}")
    logger.info(f"Baseline predictions min: {np.min(baseline_predictions):.6f}, max: {np.max(baseline_predictions):.6f}")
    logger.info(f"Actuals mean: {np.mean(actuals):.6f}, std: {np.std(actuals):.6f}")
    logger.info(f"Baseline MSE: {baseline_score:.6f}")

    importance_scores = {}
    n_features = X.shape[2]

    for feature_idx in range(n_features):
        logger.info(f"Computing importance for feature {FEATURE_COLUMNS[feature_idx]} ({feature_idx+1}/{n_features})")
        scores = []
        for repeat in range(n_repeats):
            permuted_predictions = []
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, :, feature_idx])
            permuted_dataset = TensorDataset(
                torch.FloatTensor(X_permuted),
                torch.FloatTensor(y),
                torch.FloatTensor(past_time_features)
            )
            permuted_loader = DataLoader(permuted_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch_X, _, batch_past_time in permuted_loader:
                    batch_X, batch_past_time = batch_X.to(device), batch_past_time.to(device)
                    predictions = model(batch_X, batch_past_time).cpu().numpy()
                    permuted_predictions.append(predictions)
            permuted_predictions = np.concatenate(permuted_predictions, axis=0)
            score = metric_func(actuals, permuted_predictions)
            scores.append(score - baseline_score)
        importance_scores[FEATURE_COLUMNS[feature_idx]] = np.mean(scores)

    return importance_scores

def main():
    # Define the CSV path
    csv_path = r"C:\Users\Dennis\.vscode\tradebot\src\data\btc_usd_historical.csv"
    logger.info(f"Loading data from {csv_path}")

    # Load and preprocess data
    preprocessed_data, scaled_df, feature_scaler, target_scaler = asyncio.run(
        load_and_preprocess_data(csv_path, symbol='BTC/USD')
    )
    logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")

    # Create sequences
    seq_length = 24
    forecast_steps = 3
    targets = np.array([preprocessed_data['target'].shift(-i-1).values for i in range(forecast_steps)]).T
    valid_indices = ~np.any(np.isnan(targets), axis=1)
    targets = targets[valid_indices]
    features = preprocessed_data[FEATURE_COLUMNS].values[valid_indices]
    timestamps = preprocessed_data.index[valid_indices]

    X, y, past_time_features, _, _, _ = create_sequences(
        features,
        targets,
        seq_length=seq_length,
        timestamps=timestamps
    )
    logger.info(f"Sequence data shape: X={X.shape}, y={y.shape}")

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(
        input_dim=len(FEATURE_COLUMNS),
        d_model=256,
        n_heads=4,
        n_layers=4,
        dropout=0.2,
        forecast_steps=forecast_steps
    )
    # Set weights_only=True to address the security warning
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    model.to(device)
    model.eval()
    logger.info("Loaded trained model")

    # Compute permutation importance
    importance_scores = permutation_importance(
        model, X, y, past_time_features, device, mean_squared_error, batch_size=64, n_repeats=10
    )

    # Log and save results
    logger.info("Feature Importance Scores:")
    for feature, score in importance_scores.items():
        logger.info(f"{feature}: {score:.6f}")

    # Save to a file
    with open('feature_importance_scores.json', 'w') as f:
        json.dump(importance_scores, f, indent=4)
    logger.info("Saved feature importance scores to 'feature_importance_scores.json'")

if __name__ == "__main__":
    main()