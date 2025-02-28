# src/models/model_predictor.py

import torch
import numpy as np
import pandas as pd
import logging
import asyncio
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

from src.models.transformer_model import TransformerPredictor  # Maintained import

async def predict_live_price(model, current_data: np.ndarray, feature_columns: list, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler, time_steps: int = 1) -> float:
    """
    Asynchronously predict the next price for real-time trading using a trained model.
    
    Args:
        model: Trained model (TransformerPredictor)
        current_data (np.ndarray): Current price data [[price]] or [time_steps, features]
        feature_columns (list): List of feature column names
        feature_scaler (MinMaxScaler): Scaler for features
        target_scaler (MinMaxScaler): Scaler for target (price)
        time_steps (int): Number of time steps for prediction (default 1 for real-time)
    
    Returns:
        float: Predicted next price
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Ensure current_data is 2D (even for single time step)
        if len(current_data.shape) == 1:
            current_data = current_data.reshape(1, -1)
        if current_data.shape[1] != len(feature_columns):
            raise ValueError(f"Current data must have {len(feature_columns)} features, got {current_data.shape[1]}")

        # Scale features
        features_scaled = feature_scaler.transform(current_data)
        
        # Prepare data for the model (TransformerPredictor)
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)  # Shape: [1, 1, len(feature_columns)]
        past_time_features = torch.zeros(1, 1, 1).to(device)  # Dummy time index for real-time
        past_observed_mask = torch.ones(1, 1).to(device)  # All observed
        future_values = None  # No future values for real-time
        
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(features_tensor, past_time_features, past_observed_mask, None, future_values).cpu().numpy()  # Shape: [1, 1]

        # Inverse transform to get the predicted price
        predicted_price = target_scaler.inverse_transform(prediction_scaled)[0, 0]
        logging.debug(f"Predicted live price: {predicted_price:.2f}")
        return predicted_price

    except Exception as e:
        logging.error(f"Error in predict_live_price: {e}")
        return np.nan

def predict_next_movement(model, data_slice: pd.DataFrame, feature_columns: list, feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler, 
                         time_steps: int = 10, threshold: float = 0.001):
    """
    Predict the next price movement direction for backtesting using a trained model.
    
    Args:
        model: Trained model (TransformerPredictor)
        data_slice (pd.DataFrame): Historical data slice with time_steps rows
        feature_columns (list): List of feature column names
        feature_scaler (MinMaxScaler): Scaler for features
        target_scaler (MinMaxScaler): Scaler for target (price)
        time_steps (int): Number of time steps for prediction (default 10 for backtesting)
        threshold (float): Threshold for determining movement direction
    
    Returns:
        str: 'up', 'down', or 'neutral' based on price change
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Validate input
        if len(data_slice) != time_steps:
            raise ValueError(f"Data slice must have {time_steps} rows, got {len(data_slice)}")
        if not all(col in data_slice.columns for col in feature_columns + ['close']):
            raise ValueError("Missing required columns in data_slice")
        
        # Prepare features (17 features)
        features = data_slice[feature_columns].values  # Shape: [time_steps, 17]
        features_scaled = feature_scaler.transform(features)  # Shape: [time_steps, 17]
        
        # Prepare data for the model (TransformerPredictor)
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)  # Shape: [1, time_steps, 17]
        past_time_features = torch.zeros(1, time_steps, 1).to(device)  # Dummy time index
        past_observed_mask = torch.ones(1, time_steps).to(device)  # All observed
        future_values = None  # No future values for backtesting
        
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(features_tensor, past_time_features, past_observed_mask, None, future_values).cpu().numpy()  # Shape: [1, 1]

        # Inverse transform and calculate change
        predicted_price = target_scaler.inverse_transform(prediction_scaled)[0, 0]
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
        'close': np.linspace(100, 110, 10),
        'open': [100] * 10,
        'high': [101] * 10,
        'low': [99] * 10,
        'volume': [1000] * 10,
        'momentum_rsi': [60] * 10,
        'trend_macd': [0.5] * 10,
        'atr': [1] * 10,
        'returns': [0.01] * 10,
        'log_returns': [0.01] * 10,
        'price_volatility': [0.02] * 10,
        'sma_20': [99.5] * 10,
        'vwap': [100.5] * 10,
        'adx': [25] * 10,
        'ema_50': [99.7] * 10,
        'bollinger_upper': [102] * 10,
        'bollinger_lower': [98] * 10
    })
    model = TransformerPredictor(input_dim=17, context_length=10)  # Updated for Transformer with context_length
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(dummy_data[FEATURE_COLUMNS])
    target_scaler.fit(dummy_data[['close']])
    
    # Test predict_live_price
    current_data = np.array([[110]])  # Single price for real-time
    pred_price = asyncio.run(predict_live_price(model, current_data, FEATURE_COLUMNS, feature_scaler, target_scaler))
    print(f"Predicted live price: {pred_price:.2f}")
    
    # Test predict_next_movement
    pred_movement = predict_next_movement(model, dummy_data, FEATURE_COLUMNS, feature_scaler, target_scaler)
    print(f"Predicted movement: {pred_movement}")