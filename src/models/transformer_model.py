# src/models/transformer_model.py
import torch
import torch.nn as nn
import math
import numpy as np
import logging
from typing import Optional
from src.constants import FEATURE_COLUMNS

# Set logging configuration (reduce to INFO for training, DEBUG for development)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

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
    def __init__(self, input_dim=len(FEATURE_COLUMNS), d_model=512, n_heads=8, n_layers=8, dropout=0.3):
        """
        Initialize a custom Transformer model for time series prediction.
        Updated to d_model=512, n_layers=8, and dropout=0.3 for better performance.
        Added layer normalization and improved time feature encoding.
        Enhanced for numerical stability with input clipping and epsilon in normalization.
        """
        super(TransformerPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder layers with pre-layer normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Time feature projection (5 features: hour, day of week, month, day of month, quarter)
        self.time_projection = nn.Linear(5, d_model)
        nn.init.xavier_uniform_(self.time_projection.weight)
        nn.init.zeros_(self.time_projection.bias)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output decoder
        self.decoder = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        self.dropout = nn.Dropout(dropout)
        self.nan_logged = False

    def init_weights(self):
        # Redundant with xavier_uniform_ above, kept for compatibility
        initrange = 0.02
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.time_projection.weight.data.uniform_(-initrange, initrange)
        self.time_projection.bias.data.zero_()

    def reset_logging(self):
        """Reset the nan_logged flag at the start of each epoch."""
        self.nan_logged = False

    def forward(self, past_values: torch.Tensor, past_time_features=None, past_observed_mask=None, 
                future_values=None, future_time_features=None, training=False, batch_idx=0) -> torch.Tensor:
        """
        Forward pass for the Transformer model.
        Returns prediction only during training; inference handled by model_predictor.py with MCD.
        Enhanced with input clipping and controlled debug logging.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Clip input to prevent extreme values
        past_values = torch.clamp(past_values, min=-1e5, max=1e5)

        batch_size, seq_length, features = past_values.shape

        # Check for NaN or inf in input
        if (torch.isnan(past_values).any() or torch.isinf(past_values).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in past_values (input to forward)")
            logging.warning(f"past_values min/max: {past_values.min().item():.4f}/{past_values.max().item():.4f}")
            self.nan_logged = True

        # Project input to d_model dimension
        src = self.input_projection(past_values)
        if (torch.isnan(src).any() or torch.isinf(src).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in src after input_projection")
            logging.warning(f"src min/max: {src.min().item():.4f}/{src.max().item():.4f}")
            self.nan_logged = True

        # Add time features if provided
        if past_time_features is not None:
            expected_shape = (batch_size, seq_length, 5)
            if past_time_features.shape != expected_shape:
                logging.warning(f"Expected past_time_features shape {expected_shape}, got {past_time_features.shape}")
            time_features = self.time_projection(past_time_features)
            src = src + time_features
            if (torch.isnan(time_features).any() or torch.isinf(time_features).any()) and not self.nan_logged:
                logging.warning("NaN or inf detected in time_features")
                self.nan_logged = True

        # Add positional encoding
        src = self.pos_encoder(src)
        if (torch.isnan(src).any() or torch.isinf(src).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in src after positional encoding")
            logging.warning(f"src min/max: {src.min().item():.4f}/{src.max().item():.4f}")
            self.nan_logged = True

        # Apply mask if provided
        if past_observed_mask is not None:
            past_observed_mask = past_observed_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            past_observed_mask = past_observed_mask[:, :, 0, :]
            src = src * past_observed_mask
            if (torch.isnan(past_observed_mask).any() or torch.isinf(past_observed_mask).any()) and not self.nan_logged:
                logging.warning("NaN or inf detected in past_observed_mask")
                self.nan_logged = True

        # Transformer encoder
        transformer_output = self.transformer_encoder(src)
        if (torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in transformer_output after transformer_encoder")
            logging.warning(f"transformer_output min/max: {transformer_output.min().item():.4f}/{transformer_output.max().item():.4f}")
            self.nan_logged = True

        # Apply layer normalization
        transformer_output = self.layer_norm(transformer_output)
        if (torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in transformer_output after layer_norm")
            logging.warning(f"transformer_output min/max: {transformer_output.min().item():.4f}/{transformer_output.max().item():.4f}")
            self.nan_logged = True

        # Take the last time step
        last_hidden = transformer_output[:, -1, :]
        if (torch.isnan(last_hidden).any() or torch.isinf(last_hidden).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in last_hidden")
            logging.warning(f"last_hidden min/max: {last_hidden.min().item():.4f}/{last_hidden.max().item():.4f}")
            self.nan_logged = True

        # Apply dropout during training
        if training:
            last_hidden = self.dropout(last_hidden)
            if (torch.isnan(last_hidden).any() or torch.isinf(last_hidden).any()) and not self.nan_logged:
                logging.warning("NaN or inf detected in last_hidden after dropout")
                logging.warning(f"last_hidden min/max: {last_hidden.min().item():.4f}/{last_hidden.max().item():.4f}")
            self.nan_logged = True

        # Final prediction
        prediction = self.decoder(last_hidden)
        if (torch.isnan(prediction).any() or torch.isinf(prediction).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in prediction before return")
            logging.warning(f"prediction min/max: {prediction.min().item():.4f}/{prediction.max().item():.4f}")
            self.nan_logged = True

        # Clip prediction to prevent extreme values
        prediction = torch.clamp(prediction, min=-1e5, max=1e5)

        return prediction

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
            self.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
            self.eval()
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

if __name__ == "__main__":
    model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS))
    sample_input = torch.randn(1, 24, len(FEATURE_COLUMNS))
    past_time_features = torch.zeros(1, 24, 5)  # 5 time features
    past_observed_mask = torch.ones(1, 24, len(FEATURE_COLUMNS))
    future_time_features = torch.zeros(1, 1, 5)  # 5 time features
    future_values = torch.zeros(1, 1, len(FEATURE_COLUMNS))
    prediction = model(sample_input, past_time_features, past_observed_mask)
    logging.info(f"TransformerPredictor output shape - prediction: {prediction.shape}")
    print(f"TransformerPredictor output shape - prediction: {prediction.shape}")