import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32,
                input_dim=1, hidden_dim=64, layer_dim=2, output_dim=1, learning_rate=0.001, patience=10):
    """
    Train an LSTM model with early stopping, dropout, and learning rate scheduling.

    :param X_train, y_train: Training dataset
    :param X_val, y_val: Validation dataset
    :param epochs: Maximum number of epochs
    :param patience: Number of epochs to wait for validation improvement before stopping
    :return: Best trained model
    """
    # Convert data to numpy arrays before using torch
    X_train, y_train = torch.FloatTensor(X_train.values).to(device), torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    X_val, y_val = torch.FloatTensor(X_val.values).to(device), torch.FloatTensor(y_val.values).view(-1, 1).to(device)

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim).to(device)
    criterion = nn.MSELoss()  # Use MSELoss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        # Training loop
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    return model

def generate_sample_data():
    # Generate sample data for training/testing
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.rand(100),
    }
    df = pd.DataFrame(data)
    return df

# Sample usage
if __name__ == '__main__':
    data = generate_sample_data()
    X = data[['feature1', 'feature2']]
    y = data['target']

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    print("Model trained successfully.")
