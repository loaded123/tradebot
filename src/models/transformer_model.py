# src/models/transformer_model.py
"""
This module defines the TransformerPredictor model for time series prediction.

Key Integrations:
- **src.models.train_transformer_model**: Uses this model for training.
- **src.strategy.signal_generator**: Uses model predictions for signal generation.
- **src.constants.FEATURE_COLUMNS**: Defines the input dimension.

Future Considerations:
- Add a temporal encoder for time features.
- Implement volatility-weighted attention.
- Add label smoothing to the loss function.

Dependencies:
- torch
- torch.nn
- numpy
- src.constants
"""

import torch
import torch.nn as nn
import math
import numpy as np
import logging
from typing import Optional
from src.constants import FEATURE_COLUMNS

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding[:, :x.size(1), :]
        return x

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=len(FEATURE_COLUMNS), d_model=256, n_heads=4, n_layers=4, dropout=0.2, forecast_steps=3):
        """
        Initialize a Transformer model for time series prediction with reduced complexity.
        """
        super(TransformerPredictor, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.forecast_steps = forecast_steps

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        # Learnable positional encoding
        self.pos_encoder = LearnablePositionalEncoding(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Time feature projection
        self.time_projection = nn.Linear(5, d_model)
        nn.init.xavier_uniform_(self.time_projection.weight)
        nn.init.zeros_(self.time_projection.bias)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output decoder for multi-step prediction
        self.decoder = nn.Linear(d_model, forecast_steps)  # Predict multiple steps
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        self.dropout = nn.Dropout(dropout)
        self.nan_logged = False

    def reset_logging(self):
        self.nan_logged = False

    def forward(self, past_values: torch.Tensor, past_time_features=None, past_observed_mask=None, 
                future_values=None, future_time_features=None, training=False, batch_idx=0) -> torch.Tensor:
        """
        Forward pass for the Transformer model with multi-step prediction.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Clip input
        past_values = torch.clamp(past_values, min=-1e5, max=1e5)

        batch_size, seq_length, features = past_values.shape

        # Check for NaN or inf
        if (torch.isnan(past_values).any() or torch.isinf(past_values).any()) and not self.nan_logged:
            logging.warning("NaN or inf detected in past_values")
            self.nan_logged = True

        # Project input
        src = self.input_projection(past_values)

        # Add time features
        if past_time_features is not None:
            time_features = self.time_projection(past_time_features)
            src = src + time_features

        # Add positional encoding
        src = self.pos_encoder(src)

        # Apply mask
        if past_observed_mask is not None:
            past_observed_mask = past_observed_mask.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            past_observed_mask = past_observed_mask[:, :, 0, :]
            src = src * past_observed_mask

        # Transformer encoder
        transformer_output = self.transformer_encoder(src)

        # Apply layer normalization
        transformer_output = self.layer_norm(transformer_output)

        # Take the last time step
        last_hidden = transformer_output[:, -1, :]

        # Apply dropout during training
        if training:
            last_hidden = self.dropout(last_hidden)

        # Multi-step prediction
        prediction = self.decoder(last_hidden)
        prediction = torch.clamp(prediction, min=-1e5, max=1e5)

        return prediction

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        self.eval()
        logging.info(f"Model loaded from {filepath}")

def load_model() -> 'TransformerPredictor':
    input_dim = len(FEATURE_COLUMNS)
    d_model = 256
    n_heads = 4
    n_layers = 4
    dropout = 0.2
    forecast_steps = 3
    
    model = TransformerPredictor(input_dim=input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout, forecast_steps=forecast_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_path = 'best_model.pth'
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    return model

if __name__ == "__main__":
    model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS))
    sample_input = torch.randn(1, 24, len(FEATURE_COLUMNS))
    past_time_features = torch.zeros(1, 24, 5)
    past_observed_mask = torch.ones(1, 24, len(FEATURE_COLUMNS))
    prediction = model(sample_input, past_time_features, past_observed_mask)
    logging.info(f"TransformerPredictor output shape - prediction: {prediction.shape}")