# src/models/model_predictor.py

import torch
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

from src.models.transformer_model import TransformerPredictor  # Updated import

def predict_next_movement(model, data_slice, feature_columns, feature_scaler, target_scaler, 
                         time_steps=10, threshold=0.001):
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
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)  # Shape: [1, time_steps, 17]
        
        # Generate dummy past_time_features and past_observed_mask
        past_time_features = torch.zeros(1, time_steps, 1).to(device)  # Dummy time index
        past_observed_mask = torch.ones(1, time_steps).to(device)       # All observed
        
        # Predict
        model.eval()
        with torch.no_grad():
            prediction_scaled = model(features_tensor, past_time_features, past_observed_mask).cpu().numpy()  # Shape: [1, 1]
        
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
    # Dummy test
    from src.models.lstm_model import LSTMModel
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
    model = LSTMModel(input_dim=17)  # Updated input_dim
    scaler = MinMaxScaler()
    scaler.fit(dummy_data[FEATURE_COLUMNS])
    pred = predict_next_movement(model, dummy_data, FEATURE_COLUMNS, scaler, scaler)
    print(f"Prediction: {pred}")