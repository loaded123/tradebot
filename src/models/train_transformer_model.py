# src/models/train_transformer_model.py
"""
This module trains a Transformer model for price prediction using historical BTC/USD data.
It integrates data preprocessing, sequence creation, and model evaluation, with enhancements
for volatility-weighted loss, oversampling, and time-based cross-validation.

Key Integrations:
- **src.data.data_fetcher.fetch_historical_data**: Asynchronously fetches historical data
  from a CSV file, providing raw market data for preprocessing.
- **src.data.data_preprocessor.preprocess_data**: Preprocesses raw data with technical
  indicators (e.g., SuperTrend, Ichimoku Cloud, EMA Slope), signals (e.g., SMRT scalping),
  and features (e.g., halving cycle, volatility clustering), using FEATURE_COLUMNS from
  src.constants. The 'target' (next period's log return) is scaled to its actual min/max
  range via src.data.data_manager.load_and_preprocess_data.
- **src.data.data_manager.load_and_preprocess_data**: Loads and preprocesses data,
  returning preprocessed_data, scaled_df, feature_scaler, and target_scaler, ensuring
  consistency with the training pipeline.
- **src.models.transformer_model.TransformerPredictor**: Defines the Transformer model
  architecture, trained on scaled sequences with past time features.
- **src.utils.sequence_utils.create_sequences**: Creates input-output sequences with
  timestamps, aligning features and targets for the Transformer model.
- **src.strategy.signal_filter.smrt_scalping_signals**: Generates SMRT scalping signals
  as a feature, integrated into the preprocessed data.
- **src.constants.FEATURE_COLUMNS**: Specifies the feature set for scaling and model input,
  ensuring alignment with preprocessing and signal generation.
- **src/visualization.visualizer.plot_backtest_results**: (Indirectly) The plot_predictions
  function saves visualization outputs compatible with downstream analysis.

Future Considerations:
- Optimize memory usage for large datasets by implementing batch processing with HDF5.
- Add support for multi-task learning (e.g., predicting volatility alongside price).
- Implement Bayesian neural networks for better uncertainty estimation. This could involve using
  variational inference or Monte Carlo Dropout with a higher number of samples to capture epistemic
  uncertainty more effectively, especially for out-of-distribution data points.
- Add support for resuming training across folds by saving fold metadata.

Dependencies:
- pandas
- numpy
- torch
- torch.nn
- torch.optim
- torch.utils.data
- sklearn.preprocessing.MinMaxScaler
- matplotlib.pyplot
- joblib
- src.data.data_fetcher
- src.data.data_manager
- src.data.data_preprocessor
- src.models.transformer_model
- src.utils.sequence_utils
- src.strategy.signal_filter
- src.constants
"""

import logging
import numpy as np
import pandas as pd
import asyncio
import os
import re
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from src.data.data_manager import load_and_preprocess_data
from src.models.transformer_model import TransformerPredictor
from src.utils.sequence_utils import create_sequences
from src.constants import FEATURE_COLUMNS

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('root')

# Suppress debug messages from asyncio and PyTorch internal loggers
logging.getLogger('asyncio').setLevel(logging.INFO)
logging.getLogger('src.models.transformer_model').setLevel(logging.INFO)
logging.getLogger('torch._subclasses.fake_tensor').setLevel(logging.WARNING)
for name in logging.root.manager.loggerDict:
    if name.startswith('torch'):
        logging.getLogger(name).setLevel(logging.WARNING)
    else:
        logging.getLogger(name).setLevel(logging.INFO)

class WeightedMSELossWithDirection(nn.Module):
    def __init__(self, mse_weight=1.0, direction_weight=1.0):
        super(WeightedMSELossWithDirection, self).__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss with directional component.

        Args:
            prediction (torch.Tensor): Predicted values, shape (batch_size, forecast_steps).
            target (torch.Tensor): Target values, shape (batch_size, forecast_steps).
            volatility (torch.Tensor): Volatility weights, shape (batch_size,).

        Returns:
            torch.Tensor: Combined loss (weighted MSE + directional loss).
        """
        # MSE loss weighted by inverse volatility
        mse_loss = self.mse_loss(prediction, target)  # Shape: (batch_size, forecast_steps)
        weights = 1.0 / (volatility + 1e-6)  # Inverse weighting for volatility, shape: (batch_size,)
        weights = weights / (weights.sum() + 1e-6)  # Normalize weights

        # Expand weights to match mse_loss shape: (batch_size,) -> (batch_size, forecast_steps)
        weights = weights.unsqueeze(1).expand(-1, mse_loss.shape[1])  # Shape: (batch_size, forecast_steps)

        weighted_mse = (mse_loss * weights).mean()

        # Directional loss (predicting the sign of the change)
        pred_direction = torch.sign(prediction - target[:, 0:1])  # Use the first step for direction
        true_direction = torch.sign(target - target[:, 0:1])
        pred_direction = (pred_direction + 1) / 2  # Convert to [0, 1] for BCE
        true_direction = (true_direction + 1) / 2
        direction_loss = self.bce_loss(pred_direction, true_direction)

        total_loss = self.mse_weight * weighted_mse + self.direction_weight * direction_loss
        return total_loss

def compute_directional_accuracy(actuals, predictions):
    actual_directions = np.sign(np.diff(actuals, axis=1))
    predicted_directions = np.sign(np.diff(predictions, axis=1))
    accuracy = np.mean(actual_directions == predicted_directions)
    return accuracy

def plot_predictions(actuals, predictions, confidences, num_samples=100, forecast_steps=3, fold=None):
    plt.figure(figsize=(12, 6))
    # Plot actual prices (first step only for simplicity)
    plt.plot(actuals[:num_samples, 0], label='Actual Prices', color='blue')
    # Plot predicted prices (first step)
    plt.plot(predictions[:num_samples, 0], label='Predicted Prices (1h)', color='orange')
    plt.fill_between(range(num_samples), predictions[:num_samples, 0] - confidences[:num_samples, 0],
                     predictions[:num_samples, 0] + confidences[:num_samples, 0], color='orange', alpha=0.2, label='Confidence Interval (1h)')
    plt.xlabel('Sample')
    plt.ylabel('Price (USD)')
    fold_str = f" Fold {fold}" if fold is not None else ""
    plt.title(f'Actual vs Predicted BTC Prices (1h Forecast){fold_str}')
    plt.legend()
    filename = f'price_predictions_fold{fold}.png' if fold is not None else 'price_predictions.png'
    plt.savefig(filename)
    plt.close()

def evaluate_model(model, test_loader, device, target_scaler, preprocessed_data, train_size, val_size, seq_length=24, num_samples=100, forecast_steps=3, fold=None):
    model.eval()
    predictions = []
    actuals = []
    confidences = []

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y, batch_past_time) in enumerate(test_loader):
            batch_X, batch_y, batch_past_time = batch_X.to(device), batch_y.to(device), batch_past_time.to(device)
            # Monte Carlo Dropout for uncertainty estimation
            model.train()  # Enable dropout during inference
            batch_predictions = torch.stack([model(batch_X, batch_past_time) for _ in range(num_samples)], dim=0)
            mean_pred = batch_predictions.mean(dim=0)
            std_pred = batch_predictions.std(dim=0)
            confidence = 1.96 * std_pred  # 95% confidence interval (assuming normal distribution)
            confidence = torch.clamp(confidence, 0, 1)

            predictions.append(mean_pred.cpu().detach().numpy())
            actuals.append(batch_y.cpu().detach().numpy())
            confidences.append(confidence.cpu().detach().numpy())

    predictions = np.concatenate(predictions, axis=0)  # Shape: (n_samples, forecast_steps)
    actuals = np.concatenate(actuals, axis=0)  # Shape: (n_samples, forecast_steps)
    confidences = np.concatenate(confidences, axis=0)  # Shape: (n_samples, forecast_steps)

    # Inverse transform to log returns
    y_test_unscaled = target_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(-1, forecast_steps)
    y_pred_unscaled = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1, forecast_steps)

    # Convert log returns back to prices for MAE
    test_data = preprocessed_data.iloc[train_size + val_size:].reset_index(drop=True)
    test_data = test_data.iloc[:len(actuals) + (seq_length - 1)].tail(len(actuals))
    if len(test_data) != len(actuals):
        logger.error(f"Test data length ({len(test_data)}) does not match actuals length ({len(actuals)})")
        raise ValueError("Test data length does not match actuals length")

    last_indices = np.arange(len(actuals))
    last_close_prices = test_data['close'].iloc[last_indices].values

    if len(last_close_prices) != len(actuals):
        logger.error(f"Shape mismatch: last_close_prices ({len(last_close_prices)}) vs actuals ({len(actuals)})")
        raise ValueError("Shape mismatch between last_close_prices and actuals")

    # Compute actual and predicted prices for each forecast step
    y_test_prices = np.zeros_like(y_test_unscaled)
    y_pred_prices = np.zeros_like(y_pred_unscaled)
    for step in range(forecast_steps):
        y_test_prices[:, step] = last_close_prices * np.exp(np.sum(y_test_unscaled[:, :step+1], axis=1))
        y_pred_prices[:, step] = last_close_prices * np.exp(np.sum(y_pred_unscaled[:, :step+1], axis=1))
    confidences = confidences * (target_scaler.data_max_ - target_scaler.data_min_)  # Scale confidences to price space

    # Calculate MSE, MAE, and MAPE in price space for each step
    for step in range(forecast_steps):
        test_mse = np.mean((y_test_prices[:, step] - y_pred_prices[:, step]) ** 2)
        test_mae = np.mean(np.abs(y_test_prices[:, step] - y_pred_prices[:, step]))
        test_mape = np.mean(np.abs((y_test_prices[:, step] - y_pred_prices[:, step]) / y_test_prices[:, step])) * 100
        logger.info(f"Fold {fold} Step {step+1}: Test MSE (price space): {test_mse:.2f}, MAE (price space): {test_mae:.2f}, MAPE: {test_mape:.2f}%")

    # Compute directional accuracy
    directional_acc = compute_directional_accuracy(y_test_prices, y_pred_prices)
    logger.info(f"Fold {fold} Directional Accuracy (across all steps): {directional_acc:.2%}")

    # Plot predictions (first step only for simplicity)
    plot_predictions(y_test_prices, y_pred_prices, confidences, fold=fold)
    logger.info(f"Saved price predictions plot as 'price_predictions_fold{fold}.png'")

    return y_test_prices, y_pred_prices, confidences

def delete_checkpoints_for_fold(fold):
    """Delete all checkpoints for a given fold."""
    checkpoint_dir = '.'
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f'checkpoint_fold{fold}_'):
            os.remove(os.path.join(checkpoint_dir, file))
            logger.info(f"Deleted checkpoint: {file}")

def get_latest_checkpoint(fold):
    """
    Find the latest checkpoint file for a given fold.

    Args:
        fold (int): Fold number.

    Returns:
        tuple: (checkpoint_path, epoch) if a checkpoint exists, else (None, 0).
    """
    checkpoint_dir = '.'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f'checkpoint_fold{fold}_epoch')]
    if not checkpoint_files:
        return None, 0

    # Extract epoch numbers and find the latest
    epochs = [int(re.search(r'epoch(\d+)', f).group(1)) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_checkpoint = f'checkpoint_fold{fold}_epoch{latest_epoch}.pth'
    return latest_checkpoint, latest_epoch

def train_transformer_model(resume_training=False):
    """
    Train a Transformer model for price prediction using the full historical BTC/USD dataset.
    Updated to include new features (halving cycle, volatility clustering, normalized volume, time-based features).
    Enhanced training with weighted MSE loss, directional loss, k-fold cross-validation, and multi-step prediction.

    Args:
        resume_training (bool): If True, resume training from the latest checkpoint for each fold.
    """
    # Define the CSV path explicitly
    csv_path = r"C:\Users\Dennis\.vscode\tradebot\src\data\btc_usd_historical.csv"
    logging.info(f"Attempting to load data from CSV path: {csv_path}")

    # Verify the CSV file exists
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at {csv_path}. Please ensure the file exists.")
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Load and preprocess data using data_manager
    preprocessed_data, scaled_df, feature_scaler, target_scaler = asyncio.run(
        load_and_preprocess_data(csv_path, symbol='BTC/USD')
    )
    logging.info(f"Number of rows fetched: {len(preprocessed_data)}")
    logging.info(f"Columns: {preprocessed_data.columns.tolist()}")
    logging.info(f"Initial data shape: {preprocessed_data.shape}")
    logging.info(f"Target scaler range: min={target_scaler.data_min_[0]:.4f}, max={target_scaler.data_max_[0]:.4f}")

    # Create sequences with timestamps
    seq_length = 24
    forecast_steps = 3
    # Create multi-step targets
    targets = np.array([preprocessed_data['target'].shift(-i-1).values for i in range(forecast_steps)]).T
    # Remove rows where any target is NaN
    valid_indices = ~np.any(np.isnan(targets), axis=1)
    targets = targets[valid_indices]
    features = preprocessed_data[FEATURE_COLUMNS].values[valid_indices]
    timestamps = preprocessed_data.index[valid_indices]

    X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
        features,
        targets,
        seq_length=seq_length,
        timestamps=timestamps
    )
    logging.info(f"Sequence data shape: X={X.shape}, y={y.shape}, past_time_features={past_time_features.shape}")

    # Initialize k-fold cross-validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)  # Time-series data, so no shuffling
    fold_metrics = []

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(X)):
        logger.info(f"Starting Fold {fold+1}/{n_splits}")

        # Split into train+val and test
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        past_time_train_val, past_time_test = past_time_features[train_val_idx], past_time_features[test_idx]
        volatility_train_val = preprocessed_data['price_volatility'].values[train_val_idx]
        volatility_test = preprocessed_data['price_volatility'].values[test_idx]

        # Further split train_val into train and validation (70% train, 15% val of total data)
        train_size = int(0.7 * len(X_train_val))
        val_size = len(X_train_val) - train_size
        X_train, X_val = X_train_val[:train_size], X_train_val[train_size:]
        y_train, y_val = y_train_val[:train_size], y_train_val[train_size:]
        past_time_train, past_time_val = past_time_train_val[:train_size], past_time_train_val[train_size:]
        volatility_train = volatility_train_val[:train_size]
        volatility_val = volatility_train_val[train_size:]

        logging.info(f"Fold {fold+1}: Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        past_time_train_tensor = torch.FloatTensor(past_time_train)
        past_time_val_tensor = torch.FloatTensor(past_time_val)
        past_time_test_tensor = torch.FloatTensor(past_time_test)
        logging.info(f"Fold {fold+1}: past_time_train_tensor shape: {past_time_train_tensor.shape}")

        # Oversample volatile periods for training
        volatile_indices = np.where(volatility_train > np.quantile(volatility_train, 0.75))[0]
        oversampled_indices = np.concatenate([np.arange(len(X_train)), volatile_indices[np.random.choice(len(volatile_indices), size=len(X_train)//2, replace=True)]])
        X_train_tensor = X_train_tensor[oversampled_indices]
        y_train_tensor = y_train_tensor[oversampled_indices]
        past_time_train_tensor = past_time_train_tensor[oversampled_indices]
        # Update volatility for oversampled indices
        oversampled_volatility = volatility_train[oversampled_indices]
        logging.info(f"Fold {fold+1}: Oversampled past_time_train_tensor shape: {past_time_train_tensor.shape}")

        # DataLoader with num_workers for faster data loading
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, past_time_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, past_time_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor, past_time_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

        # Initialize model with updated architecture
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerPredictor(
            input_dim=len(FEATURE_COLUMNS),
            d_model=256,
            n_heads=4,
            n_layers=4,
            dropout=0.2,
            forecast_steps=forecast_steps
        )
        model.to(device)
        logging.info(f"Fold {fold+1}: Model initialized with input_dim={len(FEATURE_COLUMNS)}, d_model=256, n_heads=4, n_layers=4, dropout=0.2, forecast_steps={forecast_steps}")
        logging.info(f"Fold {fold+1}: Using device: {device}")

        # Define optimizer and schedulers
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=50)
        scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        criterion = WeightedMSELossWithDirection(mse_weight=1.0, direction_weight=1.0)

        # Check for existing checkpoint to resume training
        start_epoch = 0
        best_val_loss = float('inf')
        if resume_training:
            checkpoint_path, latest_epoch = get_latest_checkpoint(fold + 1)
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler_cosine.load_state_dict(checkpoint['scheduler_cosine_state_dict'])
                scheduler_plateau.load_state_dict(checkpoint['scheduler_plateau_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['val_loss']
                logger.info(f"Fold {fold+1}: Resuming training from epoch {start_epoch}, best validation loss: {best_val_loss:.6f}")
            else:
                logger.info(f"Fold {fold+1}: No checkpoint found, starting training from scratch")

        # Training loop for the fold
        patience_counter = 0
        nan_logged = False
        num_epochs = 100

        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0
            batch_idx = 0
            model.reset_logging()
            for batch_X, batch_y, batch_past_time in train_loader:
                batch_X, batch_y, batch_past_time = batch_X.to(device), batch_y.to(device), batch_past_time.to(device)
                # Compute batch indices for volatility
                start_idx = batch_idx * 64
                end_idx = min(start_idx + batch_X.size(0), len(oversampled_volatility))
                batch_volatility = torch.FloatTensor(oversampled_volatility[start_idx:end_idx]).to(device)

                if batch_idx == 0:
                    logging.info(f"Fold {fold+1} Epoch {epoch+1}, Batch volatility shape: {batch_volatility.shape}, min/max: {batch_volatility.min().item():.4f}/{batch_volatility.max().item():.4f}")
                    logging.info(f"Fold {fold+1} Epoch {epoch+1}, batch_X shape: {batch_X.shape}, batch_y shape: {batch_y.shape}, past_time_features shape: {batch_past_time.shape}")

                if (torch.isnan(batch_X).any() or torch.isinf(batch_X).any() or torch.isnan(batch_y).any() or torch.isinf(batch_y).any()) and not nan_logged:
                    logging.warning(f"Fold {fold+1} NaN or inf detected in batch_X or batch_y at Epoch {epoch+1}, Batch {batch_idx}")
                    nan_logged = True

                batch_volatility = torch.clamp(batch_volatility, min=1e-6)

                optimizer.zero_grad()
                prediction = model(batch_X, batch_past_time)

                if (torch.isnan(prediction).any() or torch.isinf(prediction).any()) and not nan_logged:
                    logging.warning(f"Fold {fold+1} NaN or inf detected in prediction at Epoch {epoch+1}, Batch {batch_idx}")
                    nan_logged = True

                loss = criterion(prediction, batch_y, batch_volatility)

                if (torch.isnan(loss) or torch.isinf(loss)) and not nan_logged:
                    logging.warning(f"Fold {fold+1} NaN or inf detected in loss at Epoch {epoch+1}, Batch {batch_idx}")
                    nan_logged = True

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                batch_idx += 1
            train_loss /= len(train_loader.dataset)
            logging.info(f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

            # Validation
            model.eval()
            val_loss = 0
            batch_idx = 0
            with torch.no_grad():
                for batch_X, batch_y, batch_past_time in val_loader:
                    batch_X, batch_y, batch_past_time = batch_X.to(device), batch_y.to(device), batch_past_time.to(device)
                    # Compute batch indices for validation volatility
                    start_idx = batch_idx * 64
                    end_idx = min(start_idx + batch_X.size(0), len(volatility_val))
                    batch_volatility = torch.FloatTensor(volatility_val[start_idx:end_idx]).to(device)
                    batch_volatility = torch.clamp(batch_volatility, min=1e-6)
                    prediction = model(batch_X, batch_past_time)
                    loss = criterion(prediction, batch_y, batch_volatility)
                    val_loss += loss.item() * batch_X.size(0)
                    batch_idx += 1
            val_loss /= len(val_loader.dataset)
            logging.info(f"Fold {fold+1} Validation Loss: {val_loss:.6f}")

            # Scheduler steps
            scheduler_cosine.step()
            scheduler_plateau.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Fold {fold+1} Learning Rate: {current_lr:.6f}")

            # Save checkpoint
            checkpoint_path = f'checkpoint_fold{fold+1}_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_cosine_state_dict': scheduler_cosine.state_dict(),
                'scheduler_plateau_state_dict': scheduler_plateau.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, checkpoint_path)
            logging.info(f"Fold {fold+1} Saved checkpoint at epoch {epoch+1}: {checkpoint_path}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')
                logging.info(f"Fold {fold+1} Saved best model at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logging.info(f"Fold {fold+1} Early stopping at epoch {epoch+1}")
                    break

        # Evaluate on test set for this fold
        y_test_prices, y_pred_prices, confidences = evaluate_model(
            model, test_loader, device, target_scaler, preprocessed_data, train_size, val_size, seq_length, forecast_steps=forecast_steps, fold=fold+1
        )

        # Collect metrics for this fold
        fold_metrics.append({
            'fold': fold+1,
            'best_val_loss': best_val_loss,
            'test_mse': np.mean((y_test_prices - y_pred_prices) ** 2),
            'test_mae': np.mean(np.abs(y_test_prices - y_pred_prices)),
            'test_mape': np.mean(np.abs((y_test_prices - y_pred_prices) / y_test_prices)) * 100,
            'directional_accuracy': compute_directional_accuracy(y_test_prices, y_pred_prices)
        })

    # Save fold metrics
    with open('fold_metrics.json', 'w') as f:
        json.dump(fold_metrics, f, indent=4)
    logging.info("Saved fold metrics to 'fold_metrics.json'")

    # Select the best fold based on validation loss
    best_fold = min(fold_metrics, key=lambda x: x['best_val_loss'])
    best_fold_idx = best_fold['fold']
    logger.info(f"Best fold: {best_fold_idx} with validation loss: {best_fold['best_val_loss']:.6f}")

    # Load the best model
    model.load_state_dict(torch.load(f'best_model_fold{best_fold_idx}.pth'))
    torch.save(model.state_dict(), 'best_model.pth')
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    logging.info("Model evaluation completed and saved as 'best_model.pth' with updated scalers")

if __name__ == "__main__":
    # Set resume_training=True to resume from the latest checkpoint
    train_transformer_model(resume_training=False)