import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data.data_preprocessor import prepare_data_for_training
from src.constants import FEATURE_COLUMNS  # Import feature columns from config

def predict_next_movement(model, data_slice, feature_columns, feature_scaler, target_scaler, time_steps=10, threshold=0.005):
    """
    Predict the next price movement direction using the LSTM model.

    :param model: The trained LSTM model
    :param data_slice: DataFrame slice for prediction
    :param feature_columns: List of columns to use for features
    :param feature_scaler: Scaler used for normalizing features
    :param target_scaler: Scaler used for normalizing target (not used here directly)
    :param time_steps: Number of time steps for the LSTM input sequence
    :param threshold: Percentage change threshold to classify as up/down
    :return: Predicted movement ('up' or 'down')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Extract features from data_slice
    features = data_slice[feature_columns].values
    
    # Reshape data for LSTM input
    features = features.reshape(1, time_steps, len(feature_columns))
    
    # Normalize features
    features_scaled = feature_scaler.transform(features.reshape(time_steps, len(feature_columns)))
    features_scaled = features_scaled.reshape(1, time_steps, len(feature_columns))
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(features_tensor)
    
    # Inverse transform the prediction to get back to original scale
    predicted_price_scaled = prediction.cpu().numpy()
    predicted_price_unscaled = target_scaler.inverse_transform(predicted_price_scaled)
    current_price = data_slice['close'].iloc[-1]  # Last known closing price
    
    # Calculate percentage change
    change = (predicted_price_unscaled.item() - current_price) / current_price
    
    # Classify movement
    if change > threshold:
        return 'up'
    elif change < -threshold:
        return 'down'
    else:
        return 'neutral'