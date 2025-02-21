# src/models/train_lstm_model.py

import asyncio
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.constants import FEATURE_COLUMNS
from src.models.transformer_model import TransformerPredictor  # Ensure this imports the updated TransformerPredictor

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sequences(data_features, data_target, seq_length, num_time_features=1, observed_mask=None):
    """Create sequences for Transformer input, handling univariate future values separately, ensuring 13-step history."""
    X, y, time_features, observed_masks, future_values, future_time_features = [], [], [], [], [], []
    required_history = seq_length + max([1, 2, 3])  # 10 + 3 = 13 (unchanged, but we’ll enforce 13 steps)
    prediction_length = 1  # Predict next value
    for i in range(len(data_features) - required_history - prediction_length + 1):
        # Past features (history) - ensure exactly 13 steps
        X.append(data_features[i:(i + 13)])  # Shape [13, 17], explicitly 13 steps
        # Target (next value after history, for loss)
        y.append(data_target[i + required_history])  # Shape [1], adjusted for 13 steps
        # Past time features (dummy) - ensure exactly 13 steps
        time_seq = np.arange(13, dtype=np.float32) / 13  # Shape [13], explicitly 13 steps
        time_seq = np.expand_dims(time_seq, axis=-1)  # Shape [13, 1]
        time_features.append(time_seq)  # Shape [13, 1]
        # Past observed mask - ensure exactly 13 steps
        if observed_mask is not None:
            observed_masks.append(observed_mask[i:(i + 13)])  # Shape [13, 17], explicitly 13 steps
        # Future values (univariate target, shape [1, 1] for prediction/loss) - unchanged
        future_value = data_target[i + required_history:i + required_history + prediction_length]  # Shape [1]
        future_value_3d = np.expand_dims(future_value, axis=(1, 2))  # Shape [1, 1, 1] for univariate
        future_values.append(future_value_3d)  # Shape [1, 1, 1] per sample
        # Future time features (dummy, matching prediction_length and num_time_features) - unchanged
        future_time_seq = np.arange(required_history, required_history + prediction_length, dtype=np.float32) / (required_history + prediction_length)
        future_time_seq = np.expand_dims(future_time_seq, axis=-1)  # Shape [1, 1]
        future_time_seq_3d = np.expand_dims(future_time_seq, axis=1)  # Shape [1, 1, 1]
        future_time_features.append(future_time_seq_3d)  # Shape [1, 1, 1] per sample

    X = np.array(X)  # Shape [n_samples, 13, 17]
    y = np.array(y)  # Shape [n_samples]
    time_features = np.array(time_features)  # Shape [n_samples, 13, 1]
    if observed_mask is not None:
        observed_masks = np.array(observed_masks)  # Shape [n_samples, 13, 17]
        if len(observed_masks.shape) == 2:
            observed_masks = np.repeat(observed_masks[:, :, np.newaxis], X.shape[-1], axis=-1)
    else:
        observed_masks = np.ones_like(X, dtype=np.bool_)  # Shape [n_samples, 13, 17]
    future_values = np.array(future_values)  # Shape [n_samples, 1, 1, 1] -> Squeeze to [n_samples, 1, 1]
    future_values = np.squeeze(future_values, axis=-1)  # Shape [n_samples, 1, 1]
    future_time_features = np.array(future_time_features)  # Shape [n_samples, 1, 1, 1] -> Squeeze to [n_samples, 1, 1]
    future_time_features = np.squeeze(future_time_features, axis=-1)  # Shape [n_samples, 1, 1]

    # Log shapes for debugging
    logging.debug(f"X shape: {X.shape}, y shape: {y.shape}, time_features shape: {time_features.shape}, observed_masks shape: {observed_masks.shape if observed_masks is not None else 'None'}, future_values shape: {future_values.shape}, future_time_features shape: {future_time_features.shape}")

    return X, y, time_features, observed_masks, future_values, future_time_features

async def main():
    """Main function to orchestrate data fetching, preprocessing, and training."""
    from src.data.data_fetcher import fetch_historical_data
    from src.data.data_preprocessor import preprocess_data

    try:
        # Fetch and preprocess data
        crypto = "BTC/USD"
        df = await fetch_historical_data(crypto)
        logging.info(f"Number of rows fetched: {len(df)}")
        preprocessed_df = preprocess_data(df)
        logging.info(f"Preprocessed columns: {preprocessed_df.columns}")

        # Create sequences with longer history and future values
        sequence_length = 10  # Context length
        required_history = sequence_length + max([1, 2, 3])  # 10 + 3 = 13
        features_data = preprocessed_df[FEATURE_COLUMNS].values  # 17 features
        target_data = preprocessed_df['target'].values          # 1 target

        # Optionally create an observed mask (example: assume all data is observed)
        observed_mask = np.ones_like(features_data, dtype=np.bool_)  # Shape [n_samples, 17]

        # Create sequences
        X, y, past_time_features, past_observed_mask, future_time_features, future_values = create_sequences(
            features_data, target_data, seq_length=13
        )

        # Log shapes before splitting for debugging
        logging.debug(f"Before split - X shape: {X.shape}, y shape: {y.shape}, past_time_features shape: {past_time_features.shape}, past_observed_mask shape: {past_observed_mask.shape}, future_time_features shape: {future_time_features.shape}, future_values shape: {future_values.shape}")

        # Split data into train, validation, and test sets
        X_train, X_temp, y_train, y_temp, past_time_features_train, past_time_features_temp, past_observed_mask_train, past_observed_mask_temp, \
        future_time_features_train, future_time_features_temp, future_values_train, future_values_temp = train_test_split(
            X, y, past_time_features, past_observed_mask, future_time_features, future_values,
            train_size=0.6, test_size=0.4, shuffle=False
        )

        # Further split the temporary set into validation and test (50/50 of the 40% temp)
        X_val, X_test, y_val, y_test, past_time_features_val, past_time_features_test, past_observed_mask_val, past_observed_mask_test, \
        future_time_features_val, future_time_features_test, future_values_val, future_values_test = train_test_split(
            X_temp, y_temp, past_time_features_temp, past_observed_mask_temp, future_time_features_temp, future_values_temp,
            train_size=0.5, test_size=0.5, shuffle=False
        )

        # Log shapes after splitting for debugging
        logging.debug(f"After split - X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
        logging.debug(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}, y_test shape: {y_test.shape}")
        logging.debug(f"past_time_features_train shape: {past_time_features_train.shape if past_time_features_train is not None else 'None'}, past_time_features_val shape: {past_time_features_val.shape if past_time_features_val is not None else 'None'}, past_time_features_test shape: {past_time_features_test.shape if past_time_features_test is not None else 'None'}")
        logging.debug(f"past_observed_mask_train shape: {past_observed_mask_train.shape if past_observed_mask_train is not None else 'None'}, past_observed_mask_val shape: {past_observed_mask_val.shape if past_observed_mask_val is not None else 'None'}, past_observed_mask_test shape: {past_observed_mask_test.shape if past_observed_mask_test is not None else 'None'}")
        logging.debug(f"future_time_features_train shape: {future_time_features_train.shape if future_time_features_train is not None else 'None'}, future_time_features_val shape: {future_time_features_val.shape if future_time_features_val is not None else 'None'}, future_time_features_test shape: {future_time_features_test.shape if future_time_features_test is not None else 'None'}")
        logging.debug(f"future_values_train shape: {future_values_train.shape if future_values_train is not None else 'None'}, future_values_val shape: {future_values_val.shape if future_values_val is not None else 'None'}, future_values_test shape: {future_values_test.shape if future_values_test is not None else 'None'}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train)
        y_val_tensor = torch.FloatTensor(y_val)
        y_test_tensor = torch.FloatTensor(y_test)
        past_time_features_train_tensor = torch.FloatTensor(past_time_features_train) if past_time_features_train is not None else None
        past_time_features_val_tensor = torch.FloatTensor(past_time_features_val) if past_time_features_val is not None else None
        past_time_features_test_tensor = torch.FloatTensor(past_time_features_test) if past_time_features_test is not None else None
        past_observed_mask_train_tensor = torch.FloatTensor(past_observed_mask_train) if past_observed_mask_train is not None else None
        past_observed_mask_val_tensor = torch.FloatTensor(past_observed_mask_val) if past_observed_mask_val is not None else None
        past_observed_mask_test_tensor = torch.FloatTensor(past_observed_mask_test) if past_observed_mask_test is not None else None
        future_time_features_train_tensor = torch.FloatTensor(future_time_features_train) if future_time_features_train is not None else None
        future_time_features_val_tensor = torch.FloatTensor(future_time_features_val) if future_time_features_val is not None else None
        future_time_features_test_tensor = torch.FloatTensor(future_time_features_test) if future_time_features_test is not None else None
        future_values_train_tensor = torch.FloatTensor(future_values_train) if future_values_train is not None else None
        future_values_val_tensor = torch.FloatTensor(future_values_val) if future_values_val is not None else None
        future_values_test_tensor = torch.FloatTensor(future_values_test) if future_values_test is not None else None

        # Add logging for future_values shapes
        logging.debug(f"future_values_train_tensor shape: {future_values_train_tensor.shape if future_values_train_tensor is not None else 'None'}")
        logging.debug(f"future_values_val_tensor shape: {future_values_val_tensor.shape if future_values_val_tensor is not None else 'None'}")
        logging.debug(f"future_values_test_tensor shape: {future_values_test_tensor.shape if future_values_test_tensor is not None else 'None'}")

        # Initialize and train model
        model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS))
        model.train_model(
            X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
            past_time_features_train_tensor, past_observed_mask_train_tensor,
            past_time_features_val_tensor, past_observed_mask_val_tensor,
            future_values_train_tensor, future_time_features_train_tensor,
            future_values_val_tensor, future_time_features_val_tensor,
            epochs=200, batch_size=32, learning_rate=0.001, patience=10
        )

        # Evaluate on test set
        model.load("best_model.pth")
        predictions = model.predict(
            X_test_tensor,
            past_time_features_test_tensor,
            past_observed_mask_test_tensor,
            future_time_features_test_tensor,
            future_values_test_tensor  # Pass actual future_values for testing, not dummies
        )

        # Calculate metrics
        mse = mean_squared_error(y_test_tensor.numpy(), predictions)
        mae = mean_absolute_error(y_test_tensor.numpy(), predictions)
        logging.info(f"Test MSE: {mse}, MAE: {mae}")

        # Visualization
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test_tensor.numpy(), y=np.abs(predictions - y_test_tensor.numpy()))
        plt.xlabel('Actual Values')
        plt.ylabel('Absolute Error')
        plt.title('Error Analysis')
        plt.savefig('error_analysis.png')
        plt.close()

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    # Use winloop for Windows asyncio compatibility
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
    asyncio.run(main())