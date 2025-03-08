# src/models/model_predictor.py
import torch
import numpy as np
import pandas as pd
import logging
import asyncio
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
from src.models.train_transformer_model import create_sequences

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

from src.models.transformer_model import TransformerPredictor  # Maintained import

async def predict_live_price(model, current_data: pd.DataFrame, feature_columns: list, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler, time_steps: int = 34) -> float:
    """
    Asynchronously predict the next price for real-time trading using a trained model.
    
    Args:
        model: Trained model (TransformerPredictor)
        current_data (pd.DataFrame): Current price data with at least time_steps rows and required features
        feature_columns (list): List of feature column names
        feature_scaler (MinMaxScaler): Scaler for features
        target_scaler (MinMaxScaler): Scaler for target (price)
        time_steps (int): Number of time steps for prediction (must match training seq_length=34)
    
    Returns:
        float: Predicted next price
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Validate input
        if len(current_data) < time_steps:
            raise ValueError(f"Current data must have at least {time_steps} rows, got {len(current_data)}")
        if not all(col in current_data.columns for col in feature_columns):
            raise ValueError(f"Current data must contain all feature columns: {feature_columns}")

        # Use the most recent time_steps rows
        current_data = current_data.iloc[-time_steps:].copy()

        # Prepare features and dummy targets
        features = current_data[feature_columns].values
        features_scaled = feature_scaler.transform(features)
        dummy_targets = np.zeros(len(features))  # Dummy targets, will not be used for prediction

        # Create sequences (will generate a single sequence of length time_steps-1 for past_values)
        X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
            features_scaled.tolist(), dummy_targets.tolist(), seq_length=time_steps
        )

        # Convert to tensors
        features_tensor = torch.FloatTensor(X).to(device)  # Shape: [1, time_steps-1, len(feature_columns)]
        past_time_features_tensor = torch.FloatTensor(past_time_features).to(device)
        past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(device)
        future_values_tensor = torch.FloatTensor(future_values).to(device)
        future_time_features_tensor = torch.FloatTensor(future_time_features).to(device)

        model.eval()
        with torch.no_grad():
            prediction_scaled = model.predict(
                features_tensor.cpu().numpy(),
                past_time_features_tensor.cpu().numpy(),
                past_observed_mask_tensor.cpu().numpy(),
                future_values_tensor.cpu().numpy(),
                future_time_features_tensor.cpu().numpy()
            )

        # Inverse transform to get the predicted price
        predicted_price = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
        logging.debug(f"Predicted live price: {predicted_price:.2f}")
        return predicted_price

    except Exception as e:
        logging.error(f"Error in predict_live_price: {e}")
        return np.nan

def predict_next_movement(model, data_slice: pd.DataFrame, feature_columns: list, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler, 
                         time_steps: int = 34, threshold: float = 0.001):
    """
    Predict the next price movement direction for backtesting using a trained model.
    
    Args:
        model: Trained model (TransformerPredictor)
        data_slice (pd.DataFrame): Historical data slice with at least time_steps rows
        feature_columns (list): List of feature column names
        feature_scaler (MinMaxScaler): Scaler for features
        target_scaler (MinMaxScaler): Scaler for target (price)
        time_steps (int): Number of time steps for prediction (must match training seq_length=34)
        threshold (float): Threshold for determining movement direction
    
    Returns:
        str: 'up', 'down', or 'neutral' based on price change
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Validate input
        if len(data_slice) < time_steps:
            raise ValueError(f"Data slice must have at least {time_steps} rows, got {len(data_slice)}")
        if not all(col in data_slice.columns for col in feature_columns + ['close']):
            raise ValueError("Missing required columns in data_slice")

        # Use the most recent time_steps rows
        data_slice = data_slice.iloc[-time_steps:].copy()

        # Prepare features and dummy targets
        features = data_slice[feature_columns].values
        features_scaled = feature_scaler.transform(features)
        dummy_targets = np.zeros(len(features))  # Dummy targets, will not be used for prediction

        # Create sequences
        X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
            features_scaled.tolist(), dummy_targets.tolist(), seq_length=time_steps
        )

        # Convert to tensors
        features_tensor = torch.FloatTensor(X).to(device)  # Shape: [1, time_steps-1, len(feature_columns)]
        past_time_features_tensor = torch.FloatTensor(past_time_features).to(device)
        past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(device)
        future_values_tensor = torch.FloatTensor(future_values).to(device)
        future_time_features_tensor = torch.FloatTensor(future_time_features).to(device)

        model.eval()
        with torch.no_grad():
            prediction_scaled = model.predict(
                features_tensor.cpu().numpy(),
                past_time_features_tensor.cpu().numpy(),
                past_observed_mask_tensor.cpu().numpy(),
                future_values_tensor.cpu().numpy(),
                future_time_features_tensor.cpu().numpy()
            )

        # Inverse transform and calculate change
        predicted_price = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
        current_price = data_slice['close'].iloc[-1]
        change = (predicted_price - current_price) / current_price
        
        logging.debug(f"Predicted: {predicted_price:.2f}, Current: {current_price:.2f}, Change: {change:.4f}")
        
        # Determine direction
        if change > threshold:
            return 'up'
        elif change < -threshold:
            return 'down'
        else:
            return 'neutral'
    
    except Exception as e:
        logging.error(f"Error in predict_next_movement: {e}")
        return 'neutral'

if __name__ == "__main__":
    # Dummy test for both functions
    dummy_data = pd.DataFrame({
        'close': np.linspace(100, 110, 34),
        'open': [100] * 34,
        'high': [101] * 34,
        'low': [99] * 34,
        'volume': [1000] * 34,
        'momentum_rsi': [60] * 34,
        'trend_macd': [0.5] * 34,
        'atr': [1] * 34,
        'returns': [0.01] * 34,
        'log_returns': [0.01] * 34,
        'price_volatility': [0.02] * 34,
        'sma_20': [99.5] * 34,
        'vwap': [100.5] * 34,
        'adx': [25] * 34,
        'ema_50': [99.7] * 34,
        'bollinger_upper': [102] * 34,
        'bollinger_lower': [98] * 34,
        'bollinger_middle': [100] * 34
    })
    model = TransformerPredictor(input_dim=17, context_length=30)  # Updated for Transformer with context_length
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(dummy_data[FEATURE_COLUMNS])
    target_scaler.fit(dummy_data[['close']])
    
    # Test predict_live_price
    pred_price = asyncio.run(predict_live_price(model, dummy_data, FEATURE_COLUMNS, feature_scaler, target_scaler))
    print(f"Predicted live price: {pred_price:.2f}")
    
    # Test predict_next_movement
    pred_movement = predict_next_movement(model, dummy_data, FEATURE_COLUMNS, feature_scaler, target_scaler)
    print(f"Predicted movement: {pred_movement}")