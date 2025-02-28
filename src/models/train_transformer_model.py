# src/models/train_transformer_model.py
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
from .transformer_model import TransformerPredictor
from src.data.data_fetcher import fetch_historical_data  # Absolute import
from src.data.data_preprocessor import preprocess_data, split_data  # Absolute import
from typing import List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}")

def create_sequences(data_features: Union[List, np.ndarray, pd.DataFrame], 
                    data_target: Union[List, np.ndarray, pd.Series], 
                    seq_length: int = 13, 
                    num_time_features: int = 1, 
                    observed_mask: Optional[Union[List, np.ndarray, pd.DataFrame]] = None):
    """Create sequences for Transformer input, handling univariate future values separately, ensuring 13-step history without numpy."""
    X, y, time_features, observed_masks, future_values, future_time_features = [], [], [], [], [], []
    required_history = seq_length  # Enforce exactly 13 steps for consistency
    prediction_length = 1  # Predict next value

    # Ensure data_features is a 2D numpy array or list
    if isinstance(data_features, (pd.DataFrame, pd.Series)):
        data_features = data_features.values
    if isinstance(data_features, np.ndarray):
        logging.debug(f"data_features shape before conversion: {data_features.shape}")
        if len(data_features.shape) == 1:
            data_features = data_features.reshape(-1, 1)  # Ensure 2D if 1D
        elif len(data_features.shape) != 2:
            raise ValueError(f"data_features must be 2D, got shape {data_features.shape}")
        data_features = data_features.tolist()
    elif isinstance(data_features, list):
        logging.debug(f"data_features is already list, first few rows: {data_features[:2] if data_features else 'Empty'}")
        if not data_features or not all(isinstance(row, (list, tuple)) for row in data_features):
            raise ValueError("data_features must be a 2D list [n_samples, n_features]")
        if not data_features or not data_features[0]:
            raise ValueError("data_features cannot be empty or contain empty rows")
    else:
        raise ValueError(f"Unsupported data_features type: {type(data_features)}")
    
    input_dim = len(data_features[0]) if data_features and data_features[0] else 0  # Number of features (17)
    logging.debug(f"Input dimension (n_features): {input_dim}")

    # Ensure data_target is a 1D numpy array or list
    if isinstance(data_target, (pd.DataFrame, pd.Series)):
        data_target = data_target.values
    if isinstance(data_target, np.ndarray):
        logging.debug(f"data_target shape before conversion: {data_target.shape}")
        if len(data_target.shape) == 2 and data_target.shape[1] == 1:
            data_target = data_target.flatten()
        elif len(data_target.shape) != 1:
            raise ValueError(f"data_target must be 1D, got shape {data_target.shape}")
        data_target = data_target.tolist()
    elif isinstance(data_target, list):
        logging.debug(f"data_target is already list, first few values: {data_target[:2] if data_target else 'Empty'}")
        if not data_target or not all(isinstance(x, (int, float)) for x in data_target):
            raise ValueError("data_target must be a 1D list of numbers [n_samples]")
    else:
        raise ValueError(f"Unsupported data_target type: {type(data_target)}")
    
    n_samples = len(data_features)
    if n_samples != len(data_target):
        raise ValueError(f"Length mismatch: data_features has {n_samples} samples, but data_target has {len(data_target)}")

    for i in range(n_samples - required_history - prediction_length + 1):
        # Past features (history) - ensure exactly 13 steps, [seq_length, n_features]
        sequence = data_features[i:i + seq_length]
        logging.debug(f"Sequence {i}: {sequence[:2]}... (length: {len(sequence)}, features: {len(sequence[0]) if sequence and sequence[0] else 0})")
        if not sequence or not isinstance(sequence, list) or not isinstance(sequence[0], (list, tuple)):
            raise ValueError(f"Invalid sequence at index {i}: {sequence}, type: {type(sequence)}")
        if len(sequence) != seq_length or len(sequence[0]) != input_dim:
            raise ValueError(f"Sequence {i} shape mismatch: expected [13, {input_dim}], got [len(sequence), {len(sequence[0]) if sequence and sequence[0] else 0}]")
        X.append(sequence)  # Shape [13, n_features], list of lists
        # Target (next value after history, for loss) - ensure 1D [1]
        target_value = data_target[i + required_history]
        logging.debug(f"Target {i}: {target_value} (type: {type(target_value)})")
        if not isinstance(target_value, (int, float)):
            raise ValueError(f"Invalid target value at index {i}: {target_value}, type: {type(target_value)}")
        y.append([target_value])  # Shape [1], wrapped in list for 2D
        # Past time features (dummy) - ensure exactly 13 steps
        time_seq = [float(j) / seq_length for j in range(seq_length)]  # Shape [13]
        time_features.append([[val] for val in time_seq])  # Shape [13, 1], list of lists
        # Past observed mask - ensure exactly 13 steps
        if observed_mask is not None:
            if isinstance(observed_mask, (np.ndarray, pd.DataFrame)):
                observed_mask = observed_mask.tolist()
            observed_masks.append(observed_mask[i:i + seq_length])  # Shape [13, n_features], list of lists
        else:
            observed_masks.append([[1] * input_dim for _ in range(seq_length)])  # Shape [13, n_features], list of lists
        # Future values (univariate target, shape [1, 1] for prediction/loss)
        future_value = [data_target[i + required_history]]  # Shape [1]
        future_values.append([[future_value[0]]])  # Shape [1, 1, 1], list of lists (univariate)
        # Future time features (dummy, matching prediction_length and num_time_features) - ensure 3D
        future_time_seq = [float(required_history) / (required_history + prediction_length)]  # Shape [1]
        future_time_features.append([[[val] for val in future_time_seq]])  # Shape [1, 1, 1], list of lists (3D)

    # Convert to torch tensors, ensuring correct shapes
    try:
        X_tensors = [torch.tensor(x, dtype=torch.float32) for x in X]  # List of [13, n_features] tensors
        logging.debug(f"X tensors before stack: {[x.shape for x in X_tensors[:2]]}")
        X = torch.stack(X_tensors) if X_tensors else torch.tensor([])  # Shape [n_samples, 13, n_features]
        y = torch.tensor(y, dtype=torch.float32)  # Shape [n_samples, 1]
        time_features = torch.tensor(time_features, dtype=torch.float32)  # Shape [n_samples, 13, 1]
        observed_masks = torch.tensor(observed_masks, dtype=torch.float32) if observed_masks else None  # Shape [n_samples, 13, n_features]
        future_values = torch.tensor(future_values, dtype=torch.float32)  # Shape [n_samples, 1, 1, 1]
        future_values = future_values.squeeze(-1)  # Shape [n_samples, 1, 1]
        future_time_features = torch.tensor(future_time_features, dtype=torch.float32)  # Shape [n_samples, 1, 1, 1]
        future_time_features = future_time_features.squeeze(-1)  # Shape [n_samples, 1, 1]
    except Exception as e:
        logging.error(f"Error converting sequences to tensors: {e}")
        raise

    # Log final shapes for debugging
    logging.debug(f"Final X shape: {X.shape}, y shape: {y.shape}, time_features shape: {time_features.shape}, "
                  f"observed_masks shape: {observed_masks.shape if observed_masks is not None else 'None'}, "
                  f"future_values shape: {future_values.shape}, future_time_features shape: {future_time_features.shape}")

    return X, y, time_features, observed_masks, future_values, future_time_features

def train(model: TransformerPredictor, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
          past_time_features_train: Optional[torch.Tensor] = None, past_observed_mask_train: Optional[torch.Tensor] = None,
          past_time_features_val: Optional[torch.Tensor] = None, past_observed_mask_val: Optional[torch.Tensor] = None,
          future_values_train: Optional[torch.Tensor] = None, future_time_features_train: Optional[torch.Tensor] = None,
          future_values_val: Optional[torch.Tensor] = None, future_time_features_val: Optional[torch.Tensor] = None,
          epochs: int = 200, batch_size: int = 32, learning_rate: float = 0.001, patience: int = 10) -> TransformerPredictor:
    """Train the Transformer model with early stopping and validation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    best_val_loss = float('inf')
    best_model = model.state_dict()
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i + batch_size].to(device).requires_grad_(True)
            batch_y = y_train[i:i + batch_size].to(device)

            batch_past_time = past_time_features_train[i:i + batch_size].to(device) if past_time_features_train is not None else None
            batch_past_obs = past_observed_mask_train[i:i + batch_size].to(device) if past_observed_mask_train is not None else None
            batch_future_vals = future_values_train[i:i + batch_size].to(device) if future_values_train is not None else None
            batch_future_time = future_time_features_train[i:i + batch_size].to(device) if future_time_features_train is not None else None

            optimizer.zero_grad()
            outputs = model(
                past_values=batch_x,
                past_time_features=batch_past_time,
                past_observed_mask=batch_past_obs,
                future_values=batch_future_vals,
                future_time_features=batch_future_time
            )
            logging.debug(f"Batch {i//batch_size}: outputs shape: {outputs.shape}")

            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(-1)
            elif outputs.shape[1] != 1:
                outputs = outputs[:, 0:1]

            if len(batch_y.shape) == 1:
                batch_y = batch_y.reshape(-1, 1)

            logging.debug(f"After reshape - outputs shape: {outputs.shape}, batch_y shape: {batch_y.shape}")
            if outputs.shape != batch_y.shape:
                logging.error(f"Shape mismatch: outputs {outputs.shape}, batch_y {batch_y.shape}")
                raise ValueError(f"Shape mismatch in loss calculation")

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / (len(X_train) // batch_size + 1)
        logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_outputs = model(
                past_values=X_val.to(device),
                past_time_features=past_time_features_val.to(device) if past_time_features_val is not None else None,
                past_observed_mask=past_observed_mask_val.to(device) if past_observed_mask_val is not None else None,
                future_values=future_values_val.to(device) if future_values_val is not None else None,
                future_time_features=future_time_features_val.to(device) if future_time_features_val is not None else None
            )

            if len(val_outputs.shape) == 1:
                val_outputs = val_outputs.unsqueeze(-1)
            elif val_outputs.shape[1] != 1:
                val_outputs = val_outputs[:, 0:1]

            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)

            val_loss = criterion(val_outputs, y_val.to(device)).item()
        logging.info(f"Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
            logging.info(f"Saved best model at epoch {epoch + 1}")
            torch.save(best_model, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model)
    return model

async def main():
    """Main function to orchestrate data fetching, preprocessing, scaling, and training."""
    try:
        crypto = "BTC/USD"
        df = await fetch_historical_data(crypto, exchange_name='gemini')
        logging.info(f"Number of rows fetched: {len(df)}")
        logging.info(f"Columns: {df.columns}")
        preprocessed_df = preprocess_data(df)
        logging.info(f"Preprocessed columns: {preprocessed_df.columns}")
        logging.info(f"Initial data shape: {preprocessed_df.shape}")

        sequence_length = 10
        required_history = sequence_length + max([1, 2, 3])
        if len(preprocessed_df) < required_history + 1:
            raise ValueError(f"Insufficient data for sequence length {required_history + 1}, got {len(preprocessed_df)} rows")

        # Define features (17 total)
        feature_columns = [
            'open', 'high', 'low', 'volume', 'returns', 'log_returns', 
            'price_volatility', 'sma_20', 'atr', 'vwap', 'adx', 
            'momentum_rsi', 'trend_macd', 'ema_50', 'bollinger_upper', 
            'bollinger_lower', 'bollinger_middle'
        ]
        features_data = preprocessed_df[feature_columns].values
        target_data = preprocessed_df['target'].values

        # Ensure observed mask is all 1s (fully observed)
        observed_mask = np.ones_like(features_data, dtype=np.bool_)

        # Create sequences
        X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
            features_data.tolist(), target_data.tolist(), seq_length=13
        )

        logging.debug(f"Before split - X shape: {X.shape}, y shape: {y.shape}, past_time_features shape: {past_time_features.shape}, "
                      f"past_observed_mask shape: {past_observed_mask.shape}, future_time_features shape: {future_time_features.shape}, "
                      f"future_values shape: {future_values.shape}")

        # Split data (60% train, 20% val, 20% test, no shuffling for time series)
        X_train, X_temp, y_train, y_temp, past_time_features_train, past_time_features_temp, past_observed_mask_train, past_observed_mask_temp, \
        future_time_features_train, future_time_features_temp, future_values_train, future_values_temp = train_test_split(
            X.numpy(), y.numpy(), past_time_features.numpy(), past_observed_mask.numpy(), future_time_features.numpy(), future_values.numpy(),
            train_size=0.6, test_size=0.4, shuffle=False
        )

        X_val, X_test, y_val, y_test, past_time_features_val, past_time_features_test, past_observed_mask_val, past_observed_mask_test, \
        future_time_features_val, future_time_features_test, future_values_val, future_values_test = train_test_split(
            X_temp, y_temp, past_time_features_temp, past_observed_mask_temp, future_time_features_temp, future_values_temp,
            train_size=0.5, test_size=0.5, shuffle=False
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device).requires_grad_(True)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        past_time_features_train_tensor = torch.FloatTensor(past_time_features_train).to(device) if past_time_features_train.size > 0 else None
        past_time_features_val_tensor = torch.FloatTensor(past_time_features_val).to(device) if past_time_features_val.size > 0 else None
        past_time_features_test_tensor = torch.FloatTensor(past_time_features_test).to(device) if past_time_features_test.size > 0 else None
        past_observed_mask_train_tensor = torch.FloatTensor(past_observed_mask_train).to(device) if past_observed_mask_train.size > 0 else None
        past_observed_mask_val_tensor = torch.FloatTensor(past_observed_mask_val).to(device) if past_observed_mask_val.size > 0 else None
        past_observed_mask_test_tensor = torch.FloatTensor(past_observed_mask_test).to(device) if past_observed_mask_test.size > 0 else None
        future_time_features_train_tensor = torch.FloatTensor(future_time_features_train).to(device) if future_time_features_train.size > 0 else None
        future_time_features_val_tensor = torch.FloatTensor(future_time_features_val).to(device) if future_time_features_val.size > 0 else None
        future_time_features_test_tensor = torch.FloatTensor(future_time_features_test).to(device) if future_time_features_test.size > 0 else None
        future_values_train_tensor = torch.FloatTensor(future_values_train).to(device) if future_values_train.size > 0 else None
        future_values_val_tensor = torch.FloatTensor(future_values_val).to(device) if future_values_val.size > 0 else None
        future_values_test_tensor = torch.FloatTensor(future_values_test).to(device) if future_values_test.size > 0 else None

        logging.debug(f"After split - X_train shape: {X_train_tensor.shape}, X_val shape: {X_val_tensor.shape}, X_test shape: {X_test_tensor.shape}")
        logging.debug(f"y_train shape: {y_train_tensor.shape}, y_val shape: {y_val_tensor.shape}, y_test shape: {y_test_tensor.shape}")
        logging.debug(f"past_time_features_train shape: {past_time_features_train_tensor.shape if past_time_features_train_tensor is not None else 'None'}")
        logging.debug(f"past_observed_mask_train shape: {past_observed_mask_train_tensor.shape if past_observed_mask_train_tensor is not None else 'None'}")
        logging.debug(f"future_time_features_train shape: {future_time_features_train_tensor.shape if future_time_features_train_tensor is not None else 'None'}")
        logging.debug(f"future_values_train shape: {future_values_train_tensor.shape if future_values_train_tensor is not None else 'None'}")

        # Initialize model with 17 features
        model = TransformerPredictor(input_dim=len(feature_columns), d_model=64, n_heads=4, n_layers=2, dropout=0.1).to(device)
        trained_model = train(
            model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
            past_time_features_train_tensor, past_observed_mask_train_tensor,
            past_time_features_val_tensor, past_observed_mask_val_tensor,
            future_values_train_tensor, future_time_features_train_tensor,
            future_values_val_tensor, future_time_features_val_tensor,
            epochs=200, batch_size=32, learning_rate=0.001, patience=10
        )

        # Evaluate and save predictions
        trained_model.eval()
        with torch.no_grad():
            predictions = trained_model.predict(
                X_test_tensor.cpu().numpy(),
                past_time_features_test_tensor.cpu().numpy() if past_time_features_test_tensor is not None else None,
                past_observed_mask_test_tensor.cpu().numpy() if past_observed_mask_test_tensor is not None else None,
                future_values_test_tensor.cpu().numpy() if future_values_test_tensor is not None else None,
                future_time_features_test_tensor.cpu().numpy() if future_time_features_test_tensor is not None else None
            )

        # Load scalers for inverse transform
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')

        # Inverse transform predictions and actual values
        # Ensure y_test_tensor is 2D for inverse_transform
        y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_unscaled = target_scaler.inverse_transform(predictions.reshape(-1, 1))

        mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
        mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
        logging.info(f"Test MSE (unscaled): {mse:.2f}, MAE (unscaled): {mae:.2f}")

        # Plot error analysis
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test_unscaled.flatten(), y=np.abs(predictions_unscaled.flatten() - y_test_unscaled.flatten()))
        plt.xlabel('Actual Values (USD)')
        plt.ylabel('Absolute Error (USD)')
        plt.title('Error Analysis (Unscaled BTC/USD Prices)')
        plt.savefig('error_analysis.png')
        plt.close()

        # Save the trained model and scalers for backtesting
        torch.save(trained_model.state_dict(), 'best_model.pth')
        joblib.dump(feature_scaler, 'feature_scaler.pkl')
        joblib.dump(target_scaler, 'target_scaler.pkl')

        logging.info("Model training completed and saved as 'best_model.pth'")

    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
    asyncio.run(main())