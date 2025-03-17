import torch
import logging
import pandas as pd
import asyncio
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from src.models.transformer_model import TransformerPredictor
from src.models.train_transformer_model import evaluate_model
from src.data.data_fetcher import fetch_historical_data
from src.utils.sequence_utils import create_sequences
from src.utils.time_utils import calculate_days_to_next_halving
import joblib
from src.constants import FEATURE_COLUMNS

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('analyze_model')

def load_and_analyze_model(model_path, csv_path):
    # Load the model with the correct architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(
        input_dim=len(FEATURE_COLUMNS),
        d_model=256,  # Match the original d_model
        n_heads=8,
        n_layers=6,   # Match the original number of layers
        dropout=0.3
    )
    # Load the state dict with weights_only=True
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path} on {device}")

    # Fetch data (using the same CSV)
    data = asyncio.run(fetch_historical_data(symbol='BTC/USD', timeframe='1h', csv_path=csv_path))
    data = data[~data.index.duplicated(keep='first')]

    # Prepare features and target (mimicking training preprocessing)
    feature_scaler = joblib.load('feature_scaler.pkl') if os.path.exists('feature_scaler.pkl') else MinMaxScaler(feature_range=(0, 1))
    target_scaler = joblib.load('target_scaler.pkl') if os.path.exists('target_scaler.pkl') else MinMaxScaler(feature_range=(-1, 1))

    # Add basic features (minimal preprocessing to match training)
    data['returns'] = data['close'].pct_change().fillna(0)
    data['log_returns'] = np.log1p(data['returns']).clip(lower=-0.1, upper=0.1)
    data['price_volatility'] = data['close'].rolling(window=20).std().bfill()

    # Add time-based features (only the 3 used during original training)
    halving_dates = [
        pd.Timestamp("2012-11-28"),
        pd.Timestamp("2016-07-09"),
        pd.Timestamp("2020-05-11"),
        pd.Timestamp("2024-04-19"),
        pd.Timestamp("2028-03-15")
    ]
    data['hour_of_day'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['days_to_next_halving'] = data.index.map(lambda x: calculate_days_to_next_halving(x, halving_dates)[0])

    # Explicitly define the time features to match the original training
    time_feature_columns = ['hour_of_day', 'day_of_week', 'days_to_next_halving']
    time_features = data[time_feature_columns].values

    # Ensure all feature columns are present
    for col in FEATURE_COLUMNS:
        if col not in data.columns:
            logger.warning(f"Column {col} missing in data. Filling with zeros.")
            data[col] = 0.0
        elif data[col].isna().any():
            logger.warning(f"Column {col} contains NaN values. Filling with mean.")
            data[col] = data[col].fillna(data[col].mean())

    # Scale features
    features_scaled = feature_scaler.fit_transform(data[FEATURE_COLUMNS])
    features_scaled = np.clip(features_scaled, 0, 1)
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1.0, neginf=0.0)

    # Prepare target (log returns for the next period)
    data['target'] = data['log_returns'].shift(-1).fillna(0)
    target_scaled = target_scaler.fit_transform(data[['target']])
    target_scaled = np.clip(target_scaled, -1, 1)
    target_scaled = np.nan_to_num(target_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    data['target'] = target_scaled.flatten()

    # Create sequences (override past_time_features to use only the 3 time features)
    seq_length = 24
    X, y, past_time_features, _, _, _ = create_sequences(
        data[FEATURE_COLUMNS].values,
        data['target'].values,
        seq_length=seq_length,
        timestamps=data.index
    )

    # Override past_time_features to use only the 3 selected time features
    # past_time_features originally has shape [num_sequences, seq_length, 5]; we need [num_sequences, seq_length, 3]
    time_features_seq = []
    for i in range(len(data) - seq_length):
        seq = time_features[i:i + seq_length]
        if len(seq) == seq_length:
            time_features_seq.append(seq)
    past_time_features = np.array(time_features_seq)
    logger.info(f"Adjusted past_time_features shape: {past_time_features.shape}")

    # Split into train, validation, and test (same as training)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    past_time_test = past_time_features[train_size + val_size:]

    # Convert to tensors
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    past_time_test_tensor = torch.FloatTensor(past_time_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor, past_time_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=4, pin_memory=True)

    # Evaluate the model
    y_test_prices, y_pred_prices, confidences = evaluate_model(
        model, test_loader, device, target_scaler, data, train_size, val_size, seq_length
    )

    # Log some model weight statistics
    for name, param in model.named_parameters():
        logger.info(f"Layer {name} - Mean: {param.data.mean().item():.4f}, Std: {param.data.std().item():.4f}")

if __name__ == "__main__":
    csv_path = r"C:\Users\Dennis\.vscode\tradebot\src\data\btc_usd_historical.csv"
    model_path = r"C:\Users\Dennis\.vscode\tradebot\best_model.pth"
    load_and_analyze_model(model_path, csv_path)