# src/models/train_transformer_model.py
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
from sklearn.model_selection import KFold
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from src.data.data_fetcher import fetch_historical_data
from src.models.transformer_model import TransformerPredictor
from src.utils.sequence_utils import create_sequences
from src.utils.time_utils import calculate_days_to_next_halving
from src.strategy.indicators import (
    calculate_rsi, calculate_vpvr, luxalgo_trend_reversal,
    trendspider_pattern_recognition, metastock_trend_slope,
    calculate_macd, calculate_atr, compute_vwap, compute_adx,
    compute_bollinger_bands
)
from src.strategy.signal_filter import smrt_scalping_signals
from src.constants import FEATURE_COLUMNS

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('root')

# Suppress debug messages from asyncio, PyTorch internal loggers, and other loggers
logging.getLogger('asyncio').setLevel(logging.INFO)
logging.getLogger('src.models.transformer_model').setLevel(logging.INFO)
logging.getLogger('torch._subclasses.fake_tensor').setLevel(logging.WARNING)
for name in logging.root.manager.loggerDict:
    if name.startswith('torch'):
        logging.getLogger(name).setLevel(logging.WARNING)
    else:
        logging.getLogger(name).setLevel(logging.INFO)

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        squared_error = (prediction - target) ** 2
        weights = 1.0 / (volatility + 1e-6)  # Inverse weighting for volatility
        weights = weights / weights.mean()  # Normalize weights
        weighted_loss = (squared_error * weights).mean()
        return weighted_loss

def compute_directional_accuracy(actuals, predictions):
    actual_directions = np.sign(np.diff(actuals))
    predicted_directions = np.sign(np.diff(predictions))
    accuracy = np.mean(actual_directions == predicted_directions)
    return accuracy

def plot_predictions(actuals, predictions, confidences, num_samples=100):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:num_samples], label='Actual Prices', color='blue')
    plt.plot(predictions[:num_samples], label='Predicted Prices', color='orange')
    plt.fill_between(range(num_samples), predictions[:num_samples] - confidences[:num_samples],
                     predictions[:num_samples] + confidences[:num_samples], color='orange', alpha=0.2, label='Confidence Interval')
    plt.xlabel('Sample')
    plt.ylabel('Price (USD)')
    plt.title('Actual vs Predicted BTC Prices')
    plt.legend()
    plt.savefig('price_predictions.png')
    plt.close()

def evaluate_model(model, test_loader, device, target_scaler, preprocessed_data, train_size, val_size, seq_length=24, num_samples=50):
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
            confidence = 1.0 / (std_pred + 1e-6)  # Inverse of standard deviation as confidence
            confidence = torch.clamp(confidence, 0, 1)

            predictions.append(mean_pred.cpu().detach().numpy())
            actuals.append(batch_y.cpu().detach().numpy())
            confidences.append(confidence.cpu().detach().numpy())

    predictions = np.concatenate(predictions).flatten()
    actuals = np.concatenate(actuals).flatten()
    confidences = np.concatenate(confidences).flatten()

    # Inverse transform to log returns
    y_test_unscaled = target_scaler.inverse_transform(actuals.reshape(-1, 1))
    y_pred_unscaled = target_scaler.inverse_transform(predictions.reshape(-1, 1))

    # Convert log returns back to prices for MAE
    test_data = preprocessed_data.iloc[train_size + val_size:].reset_index(drop=True)
    # Adjust test_data length to match actuals (account for sequence shift)
    test_data = test_data.iloc[:len(actuals) + (seq_length - 1)].tail(len(actuals))
    if len(test_data) != len(actuals):
        logger.error(f"Test data length ({len(test_data)}) does not match actuals length ({len(actuals)})")
        raise ValueError("Test data length does not match actuals length")

    # Compute indices for the last close prices (aligned with actuals)
    last_indices = np.arange(len(actuals))
    last_close_prices = test_data['close'].iloc[last_indices].values

    if len(last_close_prices) != len(actuals):
        logger.error(f"Shape mismatch: last_close_prices ({len(last_close_prices)}) vs actuals ({len(actuals)})")
        raise ValueError("Shape mismatch between last_close_prices and actuals")

    # Compute actual and predicted prices for the next hour
    y_test_prices = last_close_prices * np.exp(y_test_unscaled.flatten())
    y_pred_prices = last_close_prices * np.exp(y_pred_unscaled.flatten())
    confidences = confidences * (target_scaler.data_max_ - target_scaler.data_min_)  # Scale confidences to price space

    # Calculate MSE, MAE, and MAPE in price space
    test_mse = np.mean((y_test_prices - y_pred_prices) ** 2)
    test_mae = np.mean(np.abs(y_test_prices - y_pred_prices))
    test_mape = np.mean(np.abs((y_test_prices - y_pred_prices) / y_test_prices)) * 100

    logger.info(f"Test MSE (price space): {test_mse:.2f}, MAE (price space): {test_mae:.2f}, MAPE: {test_mape:.2f}%")
    logger.info(f"Sample actual prices (first 5): {y_test_prices[:5].tolist()}")
    logger.info(f"Sample predicted prices (first 5): {y_pred_prices[:5].tolist()}")
    logger.info(f"Sample confidences (first 5): {confidences[:5].tolist()}")

    # Compute directional accuracy
    directional_acc = compute_directional_accuracy(y_test_prices, y_pred_prices)
    logger.info(f"Directional Accuracy: {directional_acc:.2%}")

    # Plot predictions
    plot_predictions(y_test_prices, y_pred_prices, confidences)
    logger.info("Saved price predictions plot as 'price_predictions.png'")

    return y_test_prices, y_pred_prices, confidences

def delete_checkpoints_for_fold(fold):
    """Delete all checkpoints for a given fold."""
    checkpoint_dir = '.'
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f'checkpoint_fold{fold}_'):
            os.remove(os.path.join(checkpoint_dir, file))
            logger.info(f"Deleted checkpoint: {file}")

def train_transformer_model():
    """
    Train a Transformer model for price prediction using the full historical BTC/USD dataset.
    Updated to include new features (halving cycle, volatility clustering, normalized volume, time-based features).
    Enhanced training with weighted MSE loss, increased learning rate, epochs, and patience.
    Added oversampling of volatile periods.
    Added smrt_scalping_signals as a feature.
    """
    # Define the CSV path explicitly
    csv_path = r"C:\Users\Dennis\.vscode\tradebot\src\data\btc_usd_historical.csv"
    logging.info(f"Attempting to load data from CSV path: {csv_path}")

    # Verify the CSV file exists
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at {csv_path}. Please ensure the file exists.")
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Skip deleting the best_model.pth to preserve it for evaluation
    model_path = 'best_model.pth'
    # if os.path.exists(model_path):
    #     logging.info(f"Deleting old model weights at {model_path} to retrain with full dataset")
    #     os.remove(model_path)

    # Fetch all historical data (no limit to use entire dataset)
    data = asyncio.run(fetch_historical_data(
        symbol='BTC/USD',
        timeframe='1h',
        csv_path=csv_path
    ))
    logging.info(f"Number of rows fetched: {len(data)}")
    logging.info(f"Columns: {data.columns.tolist()}")
    logging.info(f"Initial data shape: {data.shape}")

    # Remove duplicate timestamps based on index
    data = data[~data.index.duplicated(keep='first')]
    logging.info(f"Number of rows after deduplication: {len(data)}")

    # Prepare features and target (use log returns)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    # Add basic features
    data['returns'] = data['close'].pct_change().fillna(0)
    data['log_returns'] = np.log1p(data['returns']).clip(lower=-0.1, upper=0.1)
    data['price_volatility'] = data['close'].rolling(window=20).std().bfill()
    data['sma_20'] = data['close'].rolling(window=20).mean().bfill()
    data['atr'] = calculate_atr(data['high'], data['low'], data['close'], period=14).bfill()
    data['vwap'] = compute_vwap(data).bfill()
    data['adx'] = compute_adx(data, period=14).bfill()

    # Calculate RSI with a warm-up period
    warm_up_period = 50
    data['momentum_rsi'] = calculate_rsi(data['close'], window=14)
    data = data.iloc[warm_up_period:]

    # Add VPVR features (optimized)
    vpvr_lookback = 500
    vpvr_step = 100  # Compute VPVR every 100 rows to reduce computation
    vpvr_indices = np.arange(vpvr_lookback, len(data), vpvr_step)
    vpvr_results = []

    logging.info("Calculating VPVR features...")
    for i in vpvr_indices:
        window_data = data.iloc[max(0, i - vpvr_lookback):i]
        vpvr = calculate_vpvr(window_data, lookback=vpvr_lookback, num_bins=50)
        vpvr_results.append(vpvr)

    # Convert results to a DataFrame
    vpvr_df = pd.DataFrame(vpvr_results, index=data.index[vpvr_indices])

    # Interpolate VPVR metrics for all rows
    data['dist_to_poc'] = np.nan
    data['dist_to_hvn_upper'] = np.nan
    data['dist_to_hvn_lower'] = np.nan
    data['dist_to_lvn_upper'] = np.nan
    data['dist_to_lvn_lower'] = np.nan

    # Fill the computed indices
    for col in ['poc', 'hvn_upper', 'hvn_lower', 'lvn_upper', 'lvn_lower']:
        data[f'vpvr_{col}'] = vpvr_df[col]

    # Interpolate missing values
    for col in ['vpvr_poc', 'vpvr_hvn_upper', 'vpvr_hvn_lower', 'vpvr_lvn_upper', 'vpvr_lvn_lower']:
        data[col] = data[col].interpolate(method='linear').bfill().ffill()

    # Compute distances
    for i in range(len(data)):
        current_price = data['close'].iloc[i]
        data.iloc[i, data.columns.get_loc('dist_to_poc')] = (current_price - data['vpvr_poc'].iloc[i]) / data['vpvr_poc'].iloc[i] if data['vpvr_poc'].iloc[i] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_hvn_upper')] = (current_price - data['vpvr_hvn_upper'].iloc[i]) / data['vpvr_hvn_upper'].iloc[i] if data['vpvr_hvn_upper'].iloc[i] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_hvn_lower')] = (current_price - data['vpvr_hvn_lower'].iloc[i]) / data['vpvr_hvn_lower'].iloc[i] if data['vpvr_hvn_lower'].iloc[i] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_lvn_upper')] = (current_price - data['vpvr_lvn_upper'].iloc[i]) / data['vpvr_lvn_upper'].iloc[i] if data['vpvr_lvn_upper'].iloc[i] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_lvn_lower')] = (current_price - data['vpvr_lvn_lower'].iloc[i]) / data['vpvr_lvn_lower'].iloc[i] if data['vpvr_lvn_lower'].iloc[i] != 0 else 0

    # Drop temporary VPVR columns
    data = data.drop(columns=['vpvr_poc', 'vpvr_hvn_upper', 'vpvr_hvn_lower', 'vpvr_lvn_upper', 'vpvr_lvn_lower'])

    for col in ['dist_to_poc', 'dist_to_hvn_upper', 'dist_to_hvn_lower', 'dist_to_lvn_upper', 'dist_to_lvn_lower']:
        data[col] = data[col].fillna(0.0)

    # Add new features
    data['trend_macd'], _ = calculate_macd(data['close'], fast=12, slow=26)
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()

    # Compute Bollinger Bands and add to data
    bollinger_df = compute_bollinger_bands(data)
    logging.debug(f"Bollinger Bands DataFrame columns: {bollinger_df.columns.tolist()}")
    data['bb_upper'] = bollinger_df['bb_upper'].bfill()
    data['bb_middle'] = bollinger_df['bb_middle'].bfill()
    data['bb_lower'] = bollinger_df['bb_lower'].bfill()
    data['bb_breakout'] = bollinger_df['bb_breakout'].bfill()

    # Check for NaN in Bollinger Bands columns and fill with mean if necessary
    for col in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_breakout']:
        if data[col].isna().any():
            logging.warning(f"Column {col} contains NaN values after computation. Filling with mean.")
            data[col] = data[col].fillna(data[col].mean())

    data['luxalgo_signal'] = luxalgo_trend_reversal(data).fillna(0)
    data['trendspider_signal'] = trendspider_pattern_recognition(data).fillna(0)
    data['metastock_slope'] = metastock_trend_slope(data).fillna(0)

    # Add SMRT scalping signals
    data['smrt_scalping_signal'] = smrt_scalping_signals(data, atr_multiplier=1.0, fee_rate=0.001).fillna(0)
    logging.info(f"Generated SMRT scalping signals: {data['smrt_scalping_signal'].value_counts().to_dict()}")

    # Halving cycle features
    halving_dates = [
        pd.Timestamp("2012-11-28"),
        pd.Timestamp("2016-07-09"),
        pd.Timestamp("2020-05-11"),
        pd.Timestamp("2024-04-19"),
        pd.Timestamp("2028-03-15")
    ]
    data['days_to_next_halving'] = data.index.map(lambda x: calculate_days_to_next_halving(x, halving_dates)[0])
    data['days_since_last_halving'] = data.index.map(lambda x: min([(x - h).days for h in halving_dates if h <= x], default=np.nan))
    data['days_since_last_halving'] = data['days_since_last_halving'].fillna(0)

    # Volatility clustering (GARCH proxy)
    data['garch_volatility'] = data['log_returns'].rolling(window=20).std().bfill() ** 2

    # Normalized volume
    data['volume_normalized'] = data['volume'] / data['volume'].rolling(window=24).mean().bfill()

    # Time-based features
    data['hour_of_day'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['quarter'] = data.index.quarter

    # Use feature columns from constants.py
    logging.info(f"Using feature columns from constants.py: {FEATURE_COLUMNS}")

    # Check for NaN values in raw feature data and fill
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            logging.warning(f"Column {col} missing in data. Filling with zeros.")
            data[col] = 0.0
        elif data[col].isna().any():
            logging.warning(f"Column {col} contains NaN values in raw data. Filling with mean.")
            data[col] = data[col].fillna(data[col].mean())

    # Scale features
    features_scaled = feature_scaler.fit_transform(data[FEATURE_COLUMNS])
    # Explicitly clip to ensure values are within [0, 1] and remove any nan/inf
    features_scaled = np.clip(features_scaled, 0, 1)
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    logging.info(f"Features scaled: min={features_scaled.min():.2f}, max={features_scaled.max():.2f}")

    # Prepare target (log returns for the next period)
    data['target'] = data['log_returns'].shift(-1).fillna(0)
    target_scaled = target_scaler.fit_transform(data[['target']])
    # Clip target to ensure finite values
    target_scaled = np.clip(target_scaled, -1, 1)
    target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    data['target'] = target_scaled.flatten()
    logging.info(f"Target (log returns) before scaling: min={data['log_returns'].min():.2f}, max={data['log_returns'].max():.2f}")
    logging.info(f"Target after scaling: min={target_scaled.min():.2f}, max={target_scaled.max():.2f}")

    # Final DataFrame
    preprocessed_data = data.copy()
    logging.info(f"Final DataFrame shape: {preprocessed_data.shape}")
    logging.info(f"Preprocessed columns: {preprocessed_data.columns.tolist()}")

    # Check for NaN values in final preprocessed data
    if preprocessed_data[FEATURE_COLUMNS].isna().any().any():
        logging.warning(f"NaN values found in features after preprocessing: {preprocessed_data[FEATURE_COLUMNS].isna().sum()}")
    if preprocessed_data['target'].isna().any():
        logging.warning(f"NaN values found in target after preprocessing: {preprocessed_data['target'].isna().sum()}")

    # Create sequences with timestamps
    seq_length = 24
    X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
        preprocessed_data[FEATURE_COLUMNS].values,
        preprocessed_data['target'].values,
        seq_length=seq_length,
        timestamps=preprocessed_data.index
    )
    logging.info(f"Sequence data shape: X={X.shape}, y={y.shape}, past_time_features={past_time_features.shape}")
    logging.info(f"Sample past_time_features (first sequence): {past_time_features[0, :, :]}")

    # Split into train, validation, and test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    past_time_train, past_time_val, past_time_test = past_time_features[:train_size], past_time_features[train_size:train_size+val_size], past_time_features[train_size+val_size:]
    logging.info(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Convert to tensors (ensure past_time_features shape is preserved)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    past_time_train_tensor = torch.FloatTensor(past_time_train)
    past_time_val_tensor = torch.FloatTensor(past_time_val)
    past_time_test_tensor = torch.FloatTensor(past_time_test)
    logging.info(f"past_time_train_tensor shape: {past_time_train_tensor.shape}")

    # Oversample volatile periods for training
    volatility = preprocessed_data['price_volatility'].values[:train_size]
    volatile_indices = np.where(volatility > np.quantile(volatility, 0.75))[0]
    oversampled_indices = np.concatenate([np.arange(len(X_train)), volatile_indices[np.random.choice(len(volatile_indices), size=len(X_train)//2, replace=True)]])
    X_train_tensor = X_train_tensor[oversampled_indices]
    y_train_tensor = y_train_tensor[oversampled_indices]
    past_time_train_tensor = past_time_train_tensor[oversampled_indices]
    logging.info(f"Oversampled past_time_train_tensor shape: {past_time_train_tensor.shape}")

    # DataLoader with num_workers for faster data loading
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, past_time_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, past_time_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, past_time_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

    # Update model input dimension to match new FEATURE_COLUMNS length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(
        input_dim=len(FEATURE_COLUMNS),
        d_model=512,
        n_heads=8,
        n_layers=8,
        dropout=0.3
    )
    model.to(device)
    logging.info(f"Model initialized with input_dim={len(FEATURE_COLUMNS)}, d_model=512, n_heads=8, n_layers=8, dropout=0.3")
    logging.info(f"Using device: {device}")

    # Skip the training loop since we only want to evaluate
    """
    # Define optimizer and scheduler with adjusted parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)  # Keep CosineAnnealingLR for now
    criterion = WeightedMSELoss()

    # Checkpoint loading with fold tracking using metadata
    checkpoint_dir = '.'
    metadata_file = 'fold_metadata.json'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_fold')]
    completed_folds = set()

    # Initialize or load fold completion metadata
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                completed_folds = set(metadata.get('completed_folds', []))
                logger.info(f"Loaded fold metadata: completed_folds={completed_folds}")
        except Exception as e:
            logger.warning(f"Failed to load fold_metadata.json: {e}. Starting with empty completed_folds.")
            completed_folds = set()
            metadata = {'completed_folds': []}
    else:
        metadata = {'completed_folds': []}
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        logger.info("Created fold_metadata.json with empty completed_folds.")

    # Log detected checkpoints for debugging
    if checkpoint_files:
        logger.info(f"Detected checkpoints: {checkpoint_files}")
    else:
        logger.info("No checkpoints detected.")

    # Find the latest checkpoint for each fold
    latest_checkpoints = {}
    if checkpoint_files:
        for checkpoint in checkpoint_files:
            match = re.match(r'checkpoint_fold(\d+)_epoch(\d+)\.pth', checkpoint)
            if match:
                fold = int(match.group(1))
                epoch = int(match.group(2))
                if fold not in latest_checkpoints or epoch > latest_checkpoints[fold][1]:
                    latest_checkpoints[fold] = (checkpoint, epoch)

        # Determine the earliest incomplete fold to resume
        earliest_incomplete_fold = None
        earliest_incomplete_epoch = 0
        for fold in sorted(latest_checkpoints.keys()):
            if fold not in completed_folds:
                earliest_incomplete_fold = fold
                earliest_incomplete_epoch = latest_checkpoints[fold][1]
                break

        if earliest_incomplete_fold is not None:
            latest_checkpoint = latest_checkpoints[earliest_incomplete_fold][0]
            model.load_state_dict(torch.load(latest_checkpoint, weights_only=True))
            start_fold = earliest_incomplete_fold
            start_epoch = earliest_incomplete_epoch + 1
            logger.info(f"Resumed training from checkpoint: {latest_checkpoint}, fold {start_fold}, epoch {start_epoch}")
        else:
            # If all checkpointed folds are completed, start the next fold
            start_fold = max(completed_folds) + 1 if completed_folds else 1
            start_epoch = 0
            logger.info(f"All checkpointed folds completed. Starting next fold: {start_fold}")
    else:
        logger.info("No existing checkpoint found, starting training from scratch")
        start_fold = 1
        start_epoch = 0

    # Custom training loop with 5-fold cross-validation
    best_val_loss = float('inf')
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_number = fold + 1
        if fold_number < start_fold:
            continue  # Skip completed folds

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        past_time_train_fold, past_time_val_fold = past_time_features[train_idx], past_time_features[val_idx]
        X_train_tensor_fold = torch.FloatTensor(X_train_fold)
        y_train_tensor_fold = torch.FloatTensor(y_train_fold).view(-1, 1)
        X_val_tensor_fold = torch.FloatTensor(X_val_fold)
        y_val_tensor_fold = torch.FloatTensor(y_val_fold).view(-1, 1)
        past_time_train_tensor_fold = torch.FloatTensor(past_time_train_fold)
        past_time_val_tensor_fold = torch.FloatTensor(past_time_val_fold)
        logging.info(f"Fold {fold_number} past_time_train_tensor_fold shape: {past_time_train_tensor_fold.shape}")

        # Oversample volatile periods for this fold
        fold_volatility = preprocessed_data['price_volatility'].values[train_idx]
        fold_volatile_indices = np.where(fold_volatility > np.quantile(fold_volatility, 0.75))[0]
        fold_oversampled_indices = np.concatenate([np.arange(len(train_idx)), fold_volatile_indices[np.random.choice(len(fold_volatile_indices), size=len(train_idx)//2, replace=True)]])
        X_train_tensor_fold = X_train_tensor_fold[fold_oversampled_indices]
        y_train_tensor_fold = y_train_tensor_fold[fold_oversampled_indices]
        past_time_train_tensor_fold = past_time_train_tensor_fold[fold_oversampled_indices]
        logging.info(f"Fold {fold_number} oversampled past_time_train_tensor_fold shape: {past_time_train_tensor_fold.shape}")

        train_dataset_fold = TensorDataset(X_train_tensor_fold, y_train_tensor_fold, past_time_train_tensor_fold)
        val_dataset_fold = TensorDataset(X_val_tensor_fold, y_val_tensor_fold, past_time_val_tensor_fold)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=64, num_workers=4, pin_memory=True)

        patience_counter = 0
        fold_best_val_loss = float('inf')
        nan_logged = False

        for epoch in range(start_epoch if fold_number == start_fold else 0, 100):
            model.train()
            train_loss = 0
            batch_idx = 0
            model.reset_logging()  # Reset nan logging flag for the new epoch
            for batch_X, batch_y, batch_past_time in train_loader_fold:
                batch_X, batch_y, batch_past_time = batch_X.to(device), batch_y.to(device), batch_past_time.to(device)
                batch_volatility = torch.FloatTensor(preprocessed_data['price_volatility'].values[train_idx][fold_oversampled_indices][batch_idx * 64:(batch_idx + 1) * 64]).to(device)
                
                # Log batch_volatility and past_time_features only for the first batch of each epoch
                if batch_idx == 0:
                    logging.info(f"Fold {fold_number}, Epoch {epoch+1}, Batch volatility min/max: {batch_volatility.min().item():.4f}/{batch_volatility.max().item():.4f}")
                    logging.info(f"Fold {fold_number}, Epoch {epoch+1}, past_time_features shape: {batch_past_time.shape}")
                
                # Check for NaN or inf in inputs, log only once per epoch
                if (torch.isnan(batch_X).any() or torch.isinf(batch_X).any() or torch.isnan(batch_y).any() or torch.isinf(batch_y).any()) and not nan_logged:
                    logging.warning(f"NaN or inf detected in batch_X or batch_y at Fold {fold_number}, Epoch {epoch+1}, Batch {batch_idx}")
                    logging.warning(f"batch_X min/max: {batch_X.min().item():.4f}/{batch_X.max().item():.4f}")
                    logging.warning(f"batch_y min/max: {batch_y.min().item():.4f}/{batch_y.max().item():.4f}")
                    nan_logged = True
                
                # Ensure volatility is positive to avoid nan in loss
                batch_volatility = torch.clamp(batch_volatility, min=1e-6)

                optimizer.zero_grad()
                prediction = model(batch_X, batch_past_time)
                
                # Check for NaN or inf in prediction, log only once per epoch
                if (torch.isnan(prediction).any() or torch.isinf(prediction).any()) and not nan_logged:
                    logging.warning(f"NaN or inf detected in prediction at Fold {fold_number}, Epoch {epoch+1}, Batch {batch_idx}")
                    logging.warning(f"prediction min/max: {prediction.min().item():.4f}/{prediction.max().item():.4f}")
                    nan_logged = True
                
                loss = criterion(prediction, batch_y, batch_volatility)
                
                # Check for NaN or inf in loss, log only once per epoch
                if (torch.isnan(loss) or torch.isinf(loss)) and not nan_logged:
                    logging.warning(f"NaN or inf detected in loss at Fold {fold_number}, Epoch {epoch+1}, Batch {batch_idx}")
                    nan_logged = True
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                batch_idx += 1
            train_loss /= len(train_loader_fold.dataset)
            logging.info(f"Fold {fold_number}, Epoch {epoch+1}/100, Train Loss: {train_loss:.6f}")

            # Save checkpoint
            checkpoint_path = f'checkpoint_fold{fold_number}_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint for fold {fold_number} at epoch {epoch+1}")

            # Validation
            model.eval()
            val_loss = 0
            batch_idx = 0  # Reset batch_idx for validation
            with torch.no_grad():
                for batch_X, batch_y, batch_past_time in val_loader_fold:
                    batch_X, batch_y, batch_past_time = batch_X.to(device), batch_y.to(device), batch_past_time.to(device)
                    batch_volatility = torch.FloatTensor(preprocessed_data['price_volatility'].values[val_idx][batch_idx * 64:(batch_idx + 1) * 64]).to(device)
                    batch_volatility = torch.clamp(batch_volatility, min=1e-6)
                    prediction = model(batch_X, batch_past_time)
                    loss = criterion(prediction, batch_y, batch_volatility)
                    val_loss += loss.item() * batch_X.size(0)
                    batch_idx += 1
            val_loss /= len(val_loader_fold.dataset)
            logging.info(f"Fold {fold_number}, Validation Loss: {val_loss:.6f}")

            # Scheduler step and log learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Fold {fold_number}, Learning Rate: {current_lr:.6f}")

            # Early stopping per fold with fix for equal losses
            if val_loss < fold_best_val_loss:
                fold_best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold{fold_number}.pth')
                logging.info(f"Saved best model for fold {fold_number} at epoch {epoch+1}")
            elif val_loss == fold_best_val_loss:
                patience_counter = 0
                logging.info(f"Validation loss equal to best loss at Fold {fold_number}, Epoch {epoch+1}, resetting patience")
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logging.info(f"Early stopping for fold {fold_number} at epoch {epoch+1}")
                    completed_folds.add(fold_number)
                    # Save metadata
                    metadata = {'completed_folds': list(completed_folds)}
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f)
                    logger.info(f"Updated fold metadata: completed_folds={completed_folds}")
                    # Delete checkpoints for this fold
                    delete_checkpoints_for_fold(fold_number)
                    break

            if fold_best_val_loss < best_val_loss:
                best_val_loss = fold_best_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                logging.info(f"Updated best overall model at fold {fold_number} with loss {best_val_loss:.6f}")

        start_epoch = 0  # Reset for next fold
    """

    # Load best overall model for evaluation
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        logger.error(f"Best model file {model_path} not found. Please ensure the file exists.")
        raise FileNotFoundError(f"Best model file {model_path} not found.")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    logging.info("Loaded best overall model for evaluation")

    # Evaluate on test set
    y_test_prices, y_pred_prices, confidences = evaluate_model(
        model, test_loader, device, target_scaler, preprocessed_data, train_size, val_size, seq_length
    )

    # Save the model and scalers
    torch.save(model.state_dict(), 'best_model.pth')
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    logging.info("Model evaluation completed and saved as 'best_model.pth' with updated scalers")

if __name__ == "__main__":
    train_transformer_model()