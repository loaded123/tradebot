import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import winloop
import joblib
from src.constants import FEATURE_COLUMNS
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy
import pandas
import torch
from src.models.lstm_model import LSTMModel  # Import the LSTMModel from its module

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 10  # Number of epochs to wait for improvement before stopping early
    best_model_path = "best_model.pth"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move to device
            optimizer.zero_grad()
            outputs = model(inputs)
            # Ensure shapes match: outputs is [batch,1] and labels is [batch]
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        # Early stopping logic and saving best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            try:
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Saved best model at epoch {epoch+1}")
            except Exception as save_err:
                logging.error(f"Error saving model at epoch {epoch+1}: {save_err}")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # After training, attempt to load the best model if it exists
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            logging.info("Loaded best model from file.")
        except Exception as load_err:
            logging.warning(f"Could not load best model from file: {load_err}. Returning current model state.")
    else:
        logging.warning("Best model file not found after training. Returning current model state.")

    return model

def evaluate_model(model, X_test, y_test, feature_scaler, target_scaler):
    model.eval()
    with torch.no_grad():
        assert len(X_test.shape) == 3, f"X_test should be 3D, got shape {X_test.shape}"
        n_samples, seq_length, n_features = X_test.shape
        X_test_2d = X_test.reshape(-1, n_features)  # Flatten to 2D
        logging.info(f"Shape of X_test_2d before scaling: {X_test_2d.shape}")
        try:
            X_test_scaled_2d = feature_scaler.transform(X_test_2d)
            logging.info(f"Shape of X_test_scaled_2d after scaling: {X_test_scaled_2d.shape}")
        except Exception as e:
            logging.error(f"Error scaling features: {e}. Shape of X_test_2d: {X_test_2d.shape}")
            raise
        X_test_scaled = X_test_scaled_2d.reshape(n_samples, seq_length, n_features)
        logging.info(f"Shape of X_test_scaled after reshaping: {X_test_scaled.shape}")

        assert y_test.ndim == 1, f"y_test should be 1D, got shape {y_test.shape}"
        try:
            y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))
            logging.info(f"Shape of y_test_scaled after scaling: {y_test_scaled.shape}")
        except Exception as e:
            logging.error(f"Error scaling target: {e}. Shape of y_test: {y_test.shape}")
            raise

        # Convert to tensors and move to device
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

        # Get predictions
        y_pred = model(X_test_tensor)
        logging.info(f"Shape of y_pred before inverse transform: {y_pred.shape}")

        # Inverse transform predictions and targets
        y_pred_unscaled = target_scaler.inverse_transform(y_pred.cpu().numpy())
        y_test_unscaled = target_scaler.inverse_transform(y_test_scaled)
        
        logging.info(f"Shape of y_pred_unscaled after inverse transform: {y_pred_unscaled.shape}")
        logging.info(f"Shape of y_test_unscaled after inverse transform: {y_test_unscaled.shape}")

        # Flatten the arrays to 1D for metric calculation
        y_pred_unscaled = y_pred_unscaled.flatten()
        y_test_unscaled = y_test_unscaled.flatten()

        mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
        mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

        # Error analysis
        errors = np.abs(y_pred_unscaled - y_test_unscaled)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test_unscaled, y=errors)
        plt.xlabel('Actual Values')
        plt.ylabel('Absolute Error')
        plt.title('Error Analysis')
        plt.savefig('error_analysis.png')
        plt.close()

        return mse, mae, errors

def create_sequences(data, seq_length):
    features = data[:, :-1]  # All columns except the last one (target)
    targets = data[:, -1]    # The last column is the target
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    import asyncio
    import winloop

    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())

    async def main():
        from src.data.data_fetcher import fetch_historical_data
        from src.data.data_preprocessor import preprocess_data, split_data

        crypto = "BTC/USD"
        df = await fetch_historical_data(crypto)
        logging.info(f"Number of rows fetched: {len(df)}")
        preprocessed_df = preprocess_data(df)
        sequence_length = 10
        X, y = create_sequences(preprocessed_df.values, sequence_length)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        try:
            logging.info("Starting model training...")
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            input_dim = X_train.shape[2]
            hidden_dim = 256  # Match the saved model's hidden dimension
            layer_dim = 2     # Match the saved model's number of layers
            model_path = "best_model.pth"


            if os.path.exists(model_path):
                try:
                    # Create a dummy model with the NEW architecture
                    dummy_model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim).to(device)
                    dummy_model.load_state_dict(torch.load(model_path, map_location=device))
                    # If the above line does not throw an error, then the architecture is consistent
                    logging.info(f"Loaded model from {model_path}")
                    model = dummy_model # Use the loaded model
                except RuntimeError as e:
                    logging.warning(f"Architecture mismatch, retraining: {e}")
                    if os.path.exists(model_path): # delete old mismatched model
                        os.remove(model_path)
                    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim).to(device) # create new model
                except Exception as e: # Catch other loading issues
                    logging.warning(f"Error loading model, retraining: {e}")
                    if os.path.exists(model_path): # delete old mismatched model
                        os.remove(model_path)
                    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim).to(device) # create new model
            else:
                model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim).to(device) # create new model


            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
            model = train_model(model, train_loader, val_loader, criterion, optimizer)
            logging.info("Training completed. Saving model...")
            model_save_path = os.path.join(os.getcwd(), "best_model.pth")
            logging.info(f"Model save path: {model_save_path}")
            torch.save(model.state_dict(), model_save_path)

            feature_scaler = joblib.load('feature_scaler.pkl')
            target_scaler = joblib.load('target_scaler.pkl')
            logging.info(f"Feature scaler n_features_in_: {feature_scaler.n_features_in_}")
            logging.info(f"Feature scaler feature_names_in_: {feature_scaler.feature_names_in_ if hasattr(feature_scaler, 'feature_names_in_') else 'None'}")
            logging.info(f"Target scaler n_features_in_: {target_scaler.n_features_in_}")
            logging.info(f"Target scaler feature_names_in_: {target_scaler.feature_names_in_ if hasattr(target_scaler, 'feature_names_in_') else 'None'}")
            logging.info(f"Shape of X_test before evaluation: {X_test.shape}")
            logging.info(f"Shape of y_test before evaluation: {y_test.shape}")
            mse, mae, errors = evaluate_model(model, X_test, y_test, feature_scaler, target_scaler)
            logging.info(f"Final Model MSE: {mse:.6f}, MAE: {mae:.6f}")
            logging.info(f"Sklearn version: {sklearn.__version__}")
            logging.info(f"Numpy version: {numpy.__version__}")
            logging.info(f"Pandas version: {pandas.__version__}")
            logging.info(f"Torch version: {torch.__version__}")
        except Exception as e:
            logging.error(f"Unexpected error during training or saving: {e}")

    asyncio.run(main())