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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, layer_dim=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Layer Norm for stabilization
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.layer_norm(out[:, -1, :])  # Apply layer norm on the last output
        return self.fc(out)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 10  # Number of epochs to wait for improvement before stopping early
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"Input shape: {inputs.shape}, Label shape: {labels.shape}")  # For debugging
            loss = criterion(outputs.squeeze(), labels)  # Ensure shapes match
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)  # Ensure shapes match
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            # Save model if it is the best so far
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    return model

def evaluate_model(model, X_test, y_test, feature_scaler, target_scaler):
    model.eval()
    with torch.no_grad():
        # X_test is already a 3D NumPy array [n_samples, seq_length, n_features]
        # Reshape X_test to 2D for scaling: [n_samples * seq_length, n_features]
        n_samples, seq_length, n_features = X_test.shape
        X_test_2d = X_test.reshape(-1, n_features)
        X_test_scaled_2d = feature_scaler.transform(X_test_2d)
        X_test_scaled = X_test_scaled_2d.reshape(n_samples, seq_length, n_features)

        # y_test is a 1D NumPy array [n_samples]
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1))

        # Convert to tensors and move to device
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

        # Get predictions
        y_pred = model(X_test_tensor)

        # Inverse transform predictions and targets
        y_pred_unscaled = target_scaler.inverse_transform(y_pred.cpu().numpy())
        y_test_unscaled = target_scaler.inverse_transform(y_test_scaled)

        # Calculate MSE and MAE
        mse = nn.MSELoss()(torch.FloatTensor(y_pred_unscaled), torch.FloatTensor(y_test_unscaled)).item()
        mae = nn.L1Loss()(torch.FloatTensor(y_pred_unscaled), torch.FloatTensor(y_test_unscaled)).item()
    
    return mse, mae

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

        print(f"Number of rows fetched: {len(df)}")
        preprocessed_df = preprocess_data(df)

        sequence_length = 10

        X, y = create_sequences(preprocessed_df.values, sequence_length)

        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        try:
            print("Starting model training...")
            
            # Initialize the model and move to device
            input_dim = X_train.shape[2]  # Number of features
            model = LSTMModel(input_dim=input_dim).to(device)  # Move model to device

            # Check for existing model
            model_path = "best_model.pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                print(f"Loaded model from {model_path}")
            else:
                print(f"Model file {model_path} not found, training a new model...")

            # Define the optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # Create DataLoader for training and validation
            train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
            val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

            # Train the model
            model = train_model(model, train_loader, val_loader, criterion, optimizer)

            print("Training completed. Saving model...")
            model_save_path = os.path.join(os.getcwd(), "best_model.pth")
            print(f"Model save path: {model_save_path}")
            torch.save(model.state_dict(), model_save_path)

            # Evaluate the model
            mse, mae = evaluate_model(model, X_test, y_test, joblib.load('feature_scaler.pkl'), joblib.load('target_scaler.pkl'))
            print(f"Final Model MSE: {mse:.6f}, MAE: {mae:.6f}")

        except FileNotFoundError as fnf:
            print(f"File not found error: {fnf}")
        except PermissionError as pe:
            print(f"Permission error: {pe}")
        except Exception as e:
            print(f"Unexpected error during training or saving: {e}")

    asyncio.run(main())