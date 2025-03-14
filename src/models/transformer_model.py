# src/models/transformer_model.py
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=22, d_model=128, n_heads=8, n_layers=4, dropout=0.7):
        """
        Initialize a custom Transformer model for time series prediction with confidence estimation via Monte Carlo Dropout.
        Increased dropout to 0.7 for better uncertainty estimation. Updated input_dim to 22 to match new FEATURE_COLUMNS.
        """
        super(TransformerPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.time_projection = nn.Linear(1, d_model)
        self.decoder = nn.Linear(d_model, 1)
        self.dropout = dropout  # Store dropout rate for inference
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, past_values: torch.Tensor, past_time_features=None, past_observed_mask=None, 
                future_values=None, future_time_features=None, training=False) -> torch.Tensor:
        """
        Forward pass for the Transformer model.
        Returns prediction only during training; during inference, predict() handles MCD.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size, seq_length, features = past_values.shape

        src = self.input_projection(past_values)
        logging.debug(f"src shape after input_projection: {src.shape}")

        if past_time_features is not None:
            time_features = self.time_projection(past_time_features)
            src = src + time_features
            logging.debug(f"time_features shape: {time_features.shape}, src shape after adding time features: {src.shape}")

        src = self.pos_encoder(src)

        if past_observed_mask is not None:
            logging.debug(f"past_observed_mask original shape: {past_observed_mask.shape}")
            past_observed_mask = past_observed_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            past_observed_mask = past_observed_mask[:, :, 0, :]
            logging.debug(f"past_observed_mask expanded shape: {past_observed_mask.shape}")
            src = src * past_observed_mask

        transformer_output = self.transformer_encoder(src)
        last_hidden = transformer_output[:, -1, :]

        prediction = self.decoder(last_hidden)
        logging.debug(f"Output shape after forward - prediction: {prediction.shape}")
        return prediction

    def train_model(self, X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
                    past_time_features=None, past_observed_mask=None, past_time_features_val=None, 
                    past_observed_mask_val=None, future_values=None, future_time_features=None,
                    future_values_val=None, future_time_features_val=None, epochs=200, batch_size=32, 
                    learning_rate=0.0005, patience=20):
        """Train the Transformer model with early stopping and validation. Increased weight decay for regularization."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-5)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        early_stopping_counter = 0

        train_dataset = TensorDataset(X, y) if past_time_features is None else TensorDataset(X, past_time_features, past_observed_mask, future_time_features, future_values, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                if past_time_features is None:
                    batch_x, batch_y = batch
                    batch_time_features, batch_mask, batch_future_time_features, batch_future_values = None, None, None, None
                else:
                    batch_x, batch_time_features, batch_mask, batch_future_time_features, batch_future_values, batch_y = batch

                optimizer.zero_grad()
                prediction = self(
                    past_values=batch_x.to(device),
                    past_time_features=batch_time_features.to(device) if batch_time_features is not None else None,
                    past_observed_mask=batch_mask.to(device) if batch_mask is not None else None,
                    future_values=batch_future_values.to(device) if batch_future_values is not None else None,
                    future_time_features=batch_future_time_features.to(device) if batch_future_time_features is not None else None,
                    training=True
                )
                if len(prediction.shape) == 1:
                    prediction = prediction.unsqueeze(-1)
                elif prediction.shape[1] != 1:
                    prediction = prediction[:, 0:1]
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.reshape(-1, 1)
                loss = criterion(prediction, batch_y.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")

            self.eval()
            with torch.no_grad():
                val_prediction = self(
                    past_values=X_val.to(device),
                    past_time_features=past_time_features_val.to(device) if past_time_features_val is not None else None,
                    past_observed_mask=past_observed_mask_val.to(device) if past_observed_mask_val is not None else None,
                    future_values=future_values_val.to(device) if future_values_val is not None else None,
                    future_time_features=future_time_features_val.to(device) if future_time_features_val is not None else None,
                    training=True
                )
                if len(val_prediction.shape) == 1:
                    val_prediction = val_prediction.unsqueeze(-1)
                elif val_prediction.shape[1] != 1:
                    val_prediction = val_prediction[:, 0:1]
                if len(y_val.shape) == 1:
                    y_val = y_val.reshape(-1, 1)
                val_loss = criterion(val_prediction, y_val.to(device)).item()
            logging.info(f"Validation Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                torch.save(self.state_dict(), "best_model.pth")
                logging.info(f"Saved best model at epoch {epoch+1}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

    def predict(self, X: np.ndarray, past_time_features=None, past_observed_mask=None, 
                future_values=None, future_time_features=None, mc_samples=100) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions and confidence scores using Monte Carlo Dropout with increased samples."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()  # Enable dropout for inference (Monte Carlo Dropout)
        X_tensor = torch.FloatTensor(X).to(device)
        if len(X_tensor.shape) != 3:
            raise ValueError(f"Expected 3D input for X, got shape {X_tensor.shape}")

        past_time_features_tensor = torch.FloatTensor(past_time_features).to(device) if past_time_features is not None else None
        past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(device) if past_observed_mask is not None else None
        future_values_tensor = torch.FloatTensor(future_values).to(device) if future_values is not None else None
        future_time_features_tensor = torch.FloatTensor(future_time_features).to(device) if future_time_features is not None else None

        # Run multiple forward passes with dropout enabled
        predictions = []
        for _ in range(mc_samples):
            with torch.no_grad():
                prediction = self(
                    past_values=X_tensor,
                    past_time_features=past_time_features_tensor,
                    past_observed_mask=past_observed_mask_tensor,
                    future_values=future_values_tensor,
                    future_time_features=future_time_features_tensor,
                    training=False
                )
                if len(prediction.shape) == 1:
                    prediction = prediction.unsqueeze(-1)
                elif prediction.shape[1] != 1:
                    prediction = prediction[:, 0:1]
                predictions.append(prediction.cpu().numpy())
        
        # Stack predictions and compute mean and variance
        predictions = np.stack(predictions, axis=0)  # Shape: [mc_samples, batch_size, 1]
        mean_prediction = np.mean(predictions, axis=0)  # Shape: [batch_size, 1]
        variance = np.var(predictions, axis=0)  # Shape: [batch_size, 1]
        
        # Convert variance to confidence (higher variance -> lower confidence)
        confidence = np.exp(-np.sqrt(variance))  # New formula for sharper confidence drop
        confidence = np.clip(confidence, 0.0, 1.0)  # Ensure confidence is in [0, 1]

        return mean_prediction, confidence

    def save(self, filepath):
        try:
            torch.save(self.state_dict(), filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load(self, filepath):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(torch.load(filepath, map_location=device))
            self.eval()
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

if __name__ == "__main__":
    model = TransformerPredictor(input_dim=22)
    sample_input = torch.randn(1, 24, 22)
    past_time_features = torch.zeros(1, 24, 1)
    past_observed_mask = torch.ones(1, 24, 22)
    future_time_features = torch.zeros(1, 1, 1)
    future_values = torch.zeros(1, 1, 22)
    prediction, confidence = model.predict(sample_input.numpy(), past_time_features.numpy(), past_observed_mask.numpy(), 
                                          future_values.numpy(), future_time_features.numpy())
    logging.debug(f"TransformerPredictor output shape - prediction: {prediction.shape}, confidence: {confidence.shape}")
    print(f"TransformerPredictor output shape - prediction: {prediction.shape}, confidence: {confidence.shape}")