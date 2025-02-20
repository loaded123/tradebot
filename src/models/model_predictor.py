import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from src.data.data_preprocessor import prepare_data_for_training  # Not used here
#from src.constants import FEATURE_COLUMNS  # Import feature columns from config

def predict_next_movement(model, data_slice, feature_columns, feature_scaler, target_scaler, time_steps=10, threshold=0.005):
    """Predict the next price movement direction using the LSTM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    features = data_slice[feature_columns].values
    features = features.reshape(1, time_steps, len(feature_columns))

    features_scaled = feature_scaler.transform(features.reshape(time_steps, len(feature_columns)))
    features_scaled = features_scaled.reshape(1, time_steps, len(feature_columns))

    features_tensor = torch.FloatTensor(features_scaled).to(device)

    with torch.no_grad():
        prediction = model(features_tensor)

    predicted_price_scaled = prediction.cpu().numpy()
    predicted_price_unscaled = target_scaler.inverse_transform(predicted_price_scaled)
    current_price = data_slice['close'].iloc[-1]

    change = (predicted_price_unscaled.item() - current_price) / current_price

    if change > threshold:
        return 'up'
    elif change < -threshold:
        return 'down'
    else:
        return 'neutral'