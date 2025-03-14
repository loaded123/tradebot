# src/models/train_transformer_model.py
import logging
import numpy as np
import pandas as pd
import asyncio
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.data.data_fetcher import fetch_historical_data
from src.models.transformer_model import TransformerPredictor
from src.utils.sequence_utils import create_sequences
from src.strategy.indicators import (
    calculate_rsi, calculate_vpvr, luxalgo_trend_reversal,
    trendspider_pattern_recognition, metastock_trend_slope
)
from src.constants import FEATURE_COLUMNS
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def train_transformer_model():
    """
    Train a Transformer model for price prediction using historical BTC/USD data.
    Updated to include VPVR and new features (LuxAlgo, TrendSpider, MetaStock) for improved price level awareness.
    Now includes increased dropout, 5-fold cross-validation, and handles single prediction output during training.
    Confidence is computed during inference using Monte Carlo Dropout.
    """
    # Define the CSV path explicitly
    csv_path = r"C:\Users\Dennis\.vscode\tradebot\src\data\btc_usd_historical.csv"
    logging.info(f"Attempting to load data from CSV path: {csv_path}")

    # Verify the CSV file exists
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at {csv_path}. Please ensure the file exists.")
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Fetch historical data using CSV
    data = asyncio.run(fetch_historical_data(
        symbol='BTC/USD',
        timeframe='1h',
        limit=3000,
        csv_path=csv_path
    ))
    logging.info(f"Number of rows fetched: {len(data)}")
    logging.info(f"Columns: {data.columns.tolist()}")
    logging.info(f"Initial data shape: {data.shape}")

    # Filter data to 2020 onwards to reduce distribution shift
    data = data[data.index >= '2020-01-01']
    logging.info(f"Data after filtering (2020 onwards): {len(data)} rows")

    # Prepare features and target (use log returns)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))

    # Add basic features
    data['returns'] = data['close'].pct_change().fillna(0)
    data['log_returns'] = np.log1p(data['returns']).fillna(0)
    data['price_volatility'] = data['close'].rolling(window=20).std().bfill()
    data['sma_20'] = data['close'].rolling(window=20).mean().bfill()
    data['atr'] = (data['high'] - data['low']).rolling(window=14).mean().bfill()
    data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    data['adx'] = data['close'].diff().abs().rolling(window=14).mean().bfill()

    # Calculate RSI with a warm-up period
    warm_up_period = 50
    data['momentum_rsi'] = calculate_rsi(data['close'], window=14)
    data = data.iloc[warm_up_period:]

    # Add VPVR features
    vpvr_lookback = 500
    data['dist_to_poc'] = np.nan
    data['dist_to_hvn_upper'] = np.nan
    data['dist_to_hvn_lower'] = np.nan
    data['dist_to_lvn_upper'] = np.nan
    data['dist_to_lvn_lower'] = np.nan

    for i in range(vpvr_lookback, len(data)):
        window_data = data.iloc[max(0, i - vpvr_lookback):i]
        vpvr = calculate_vpvr(window_data, lookback=vpvr_lookback, num_bins=50)
        current_price = data['close'].iloc[i]
        data.iloc[i, data.columns.get_loc('dist_to_poc')] = (current_price - vpvr['poc']) / vpvr['poc'] if vpvr['poc'] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_hvn_upper')] = (current_price - vpvr['hvn_upper']) / vpvr['hvn_upper'] if vpvr['hvn_upper'] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_hvn_lower')] = (current_price - vpvr['hvn_lower']) / vpvr['hvn_lower'] if vpvr['hvn_lower'] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_lvn_upper')] = (current_price - vpvr['lvn_upper']) / vpvr['lvn_upper'] if vpvr['lvn_upper'] != 0 else 0
        data.iloc[i, data.columns.get_loc('dist_to_lvn_lower')] = (current_price - vpvr['lvn_lower']) / vpvr['lvn_lower'] if vpvr['lvn_lower'] != 0 else 0

    for col in ['dist_to_poc', 'dist_to_hvn_upper', 'dist_to_hvn_lower', 'dist_to_lvn_upper', 'dist_to_lvn_lower']:
        data[col] = data[col].fillna(0.0)

    # Add new features
    data['trend_macd'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['bollinger_middle'] = data['close'].rolling(window=20).mean().bfill()
    data['bollinger_upper'] = data['bollinger_middle'] + 2 * data['close'].rolling(window=20).std().bfill()
    data['bollinger_lower'] = data['bollinger_middle'] - 2 * data['close'].rolling(window=20).std().bfill()
    data['luxalgo_signal'] = luxalgo_trend_reversal(data).fillna(0)
    data['trendspider_signal'] = trendspider_pattern_recognition(data).fillna(0)
    data['metastock_slope'] = metastock_trend_slope(data).fillna(0)

    # Use feature columns from constants.py
    logging.info(f"Using feature columns from constants.py: {FEATURE_COLUMNS}")

    # Check for NaN values and fill
    for col in FEATURE_COLUMNS:
        if data[col].isna().any():
            logging.warning(f"Column {col} contains NaN values. Filling with mean.")
            data[col] = data[col].fillna(data[col].mean())

    # Scale features
    features_scaled = feature_scaler.fit_transform(data[FEATURE_COLUMNS])
    logging.info(f"Features scaled: min={features_scaled.min():.2f}, max={features_scaled.max():.2f}")

    # Prepare target (log returns for the next period)
    data['target'] = data['log_returns'].shift(-1).fillna(0)  # Next period's log return
    target_scaled = target_scaler.fit_transform(data[['target']])
    data['target'] = target_scaled.flatten()
    logging.info(f"Target (log returns) before scaling: min={data['log_returns'].min():.2f}, max={data['log_returns'].max():.2f}")
    logging.info(f"Target after scaling: min={target_scaled.min():.2f}, max={target_scaled.max():.2f}")

    # Final DataFrame
    preprocessed_data = data.copy()
    logging.info(f"Final DataFrame shape: {preprocessed_data.shape}")
    logging.info(f"Preprocessed columns: {preprocessed_data.columns.tolist()}")

    # Create sequences
    seq_length = 24  # Match training sequence length
    X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
        preprocessed_data[FEATURE_COLUMNS].values,
        preprocessed_data['target'].values,
        seq_length=seq_length
    )
    logging.info(f"Sequence data shape: X={X.shape}, y={y.shape}")

    # Split into train, validation, and test
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    logging.info(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Update model input dimension to match new FEATURE_COLUMNS length
    model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS), d_model=128, n_heads=8, n_layers=4, dropout=0.7)
    logging.info(f"Model initialized with input_dim={len(FEATURE_COLUMNS)}, d_model=128, n_heads=8, n_layers=4, dropout=0.7")

    # Load existing weights for fine-tuning (optional)
    try:
        model.load('best_model.pth')
        logging.info("Loaded existing model weights for fine-tuning")
    except Exception as e:
        logging.warning(f"Failed to load existing model: {e}. Training from scratch.")

    # Define optimizer and scheduler with adjusted parameters
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-6)

    # Custom training loop with 5-fold cross-validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        X_train_tensor_fold = torch.FloatTensor(X_train_fold)
        y_train_tensor_fold = torch.FloatTensor(y_train_fold)
        X_val_tensor_fold = torch.FloatTensor(X_val_fold)
        y_val_tensor_fold = torch.FloatTensor(y_val_fold)

        train_dataset_fold = TensorDataset(X_train_tensor_fold, y_train_tensor_fold)
        val_dataset_fold = TensorDataset(X_val_tensor_fold, y_val_tensor_fold)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=32)

        patience_counter = 0
        fold_best_val_loss = float('inf')

        for epoch in range(50):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader_fold:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_y = batch_y.view(-1, 1)  # Reshape target to [batch_size, 1] to match output
                optimizer.zero_grad()
                prediction = model(batch_X, training=True)
                loss = criterion(prediction, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            train_loss /= len(train_loader_fold.dataset)
            logging.info(f"Fold {fold+1}, Epoch {epoch+1}/50, Train Loss: {train_loss:.6f}")

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader_fold:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    batch_y = batch_y.view(-1, 1)
                    prediction = model(batch_X, training=True)
                    loss = criterion(prediction, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader_fold.dataset)
            logging.info(f"Fold {fold+1}, Validation Loss: {val_loss:.6f}")

            # Scheduler step and log learning rate
            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Fold {fold+1}, Learning Rate: {current_lr:.6f}")

            # Early stopping per fold
            if val_loss < fold_best_val_loss:
                fold_best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')
                logging.info(f"Saved best model for fold {fold+1} at epoch {epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logging.info(f"Early stopping for fold {fold+1} at epoch {epoch+1}")
                    break

        if fold_best_val_loss < best_val_loss:
            best_val_loss = fold_best_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Updated best overall model at fold {fold+1} with loss {best_val_loss:.6f}")

    # Load best overall model for evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    logging.info("Loaded best overall model for evaluation")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred, y_conf = model.predict(X_test_tensor.to(device))
    y_test = y_test_tensor.cpu().numpy()

    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_unscaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Convert log returns back to prices for MAE
    test_data = preprocessed_data.iloc[train_size + val_size:]
    last_indices = np.arange(len(test_data) - len(y_test), len(test_data)) - seq_length
    last_close_prices = test_data['close'].iloc[last_indices].values

    if len(last_close_prices) != len(y_test):
        logging.error(f"Shape mismatch: last_close_prices ({len(last_close_prices)}) vs y_test ({len(y_test)})")
        raise ValueError("Shape mismatch between last_close_prices and y_test")

    # Compute actual and predicted prices for the next hour
    y_test_prices = last_close_prices * np.exp(y_test_unscaled.flatten())
    y_pred_prices = last_close_prices * np.exp(y_pred_unscaled.flatten())

    # Calculate MSE and MAE in price space
    test_mse = np.mean((y_test_prices - y_pred_prices) ** 2)
    test_mae = np.mean(np.abs(y_test_prices - y_pred_prices))

    # Calculate MAPE for additional insight
    test_mape = np.mean(np.abs((y_test_prices - y_pred_prices) / y_test_prices)) * 100

    logging.info(f"Test MSE (price space): {test_mse:.2f}, MAE (price space): {test_mae:.2f}, MAPE: {test_mape:.2f}%")

    # Log sample predictions and confidences for debugging
    logging.info(f"Sample actual prices (first 5): {y_test_prices[:5].tolist()}")
    logging.info(f"Sample predicted prices (first 5): {y_pred_prices[:5].tolist()}")
    logging.info(f"Sample confidences (first 5): {y_conf[:5].flatten().tolist()}")

    # Save the model and scalers
    torch.save(model.state_dict(), 'best_model.pth')
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')
    logging.info("Model training completed and saved as 'best_model.pth' with updated scalers")

if __name__ == "__main__":
    train_transformer_model()