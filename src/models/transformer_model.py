# src/models/transformer_model.py
import torch
import torch.nn as nn
import logging
import numpy as np
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=17, d_model=64, n_heads=4, n_layers=2, dropout=0.1, prediction_length=1):
        """
        Initialize the Transformer model for time series prediction.
        input_dim: Number of input features (default 17).
        d_model: Dimension of the transformer model (default 64).
        n_heads: Number of attention heads (default 4).
        n_layers: Number of transformer layers (default 2).
        dropout: Dropout rate (default 0.1).
        prediction_length: Length of the prediction horizon (default 1).
        """
        super(TransformerPredictor, self).__init__()
        try:
            config = TimeSeriesTransformerConfig(
                input_size=input_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                prediction_length=prediction_length,
                context_length=30,  # Reverted to 30 to match past_values length
                num_time_features=1,
                lags_sequence=[1, 2, 3]  # Lags to use for the model
            )
            self.transformer = TimeSeriesTransformerModel(config)
            self.fc = nn.Linear(d_model, 1)  # Predict a single value (univariate target, output [batch_size, 1])
            self.config = config  # Store config for reference
            logging.info(f"TimeSeriesTransformerModel initialized successfully with context_length={self.config.context_length} and lags_sequence={self.config.lags_sequence}")
            logging.debug(f"TimeSeriesTransformerConfig: {self.config}")
        except Exception as e:
            logging.error(f"Error initializing TimeSeriesTransformerModel: {e}")
            raise

    def forward(self, past_values, past_time_features=None, past_observed_mask=None, future_values=None, future_time_features=None):
        """
        Forward pass for the Transformer model.
        past_values: [batch, context_length, input_dim] (e.g., [32, 30, 17])
        past_time_features: [batch, context_length, num_time_features] (e.g., [32, 30, 1])
        past_observed_mask: [batch, context_length, input_dim] (e.g., [32, 30, 17])
        future_values: [batch, prediction_length, input_dim] (optional, for training, e.g., [32, 1, 17])
        future_time_features: [batch, prediction_length, num_time_features] (e.g., [32, 1, 1])
        """
        try:
            # Ensure inputs are PyTorch tensors
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Convert NumPy arrays to tensors if necessary
            if isinstance(past_values, np.ndarray):
                past_values = torch.FloatTensor(past_values).to(device)
            if len(past_values.shape) != 3:
                raise ValueError(f"Expected 3D input for past_values, got shape {past_values.shape}")
            batch_size, history_length, features = past_values.shape
            
            logging.debug(f"past_values shape in forward: {past_values.shape}")
            logging.debug(f"future_values shape in forward: {future_values.shape if future_values is not None else 'None'}")
            logging.debug(f"future_time_features shape in forward: {future_time_features.shape if future_time_features is not None else 'None'}")
            logging.debug(f"past_time_features shape in forward: {past_time_features.shape if past_time_features is not None else 'None'}")
            logging.debug(f"past_observed_mask shape in forward: {past_observed_mask.shape if past_observed_mask is not None else 'None'}")

            # Handle missing or incorrectly shaped past_time_features
            if past_time_features is None:
                past_time_features = torch.arange(self.config.context_length, dtype=torch.float32).repeat(batch_size, 1).unsqueeze(-1).to(device) / self.config.context_length
                logging.warning("Using dummy past_time_features.")
            elif isinstance(past_time_features, np.ndarray):
                past_time_features = torch.FloatTensor(past_time_features).to(device)
                if len(past_time_features.shape) == 2:
                    past_time_features = past_time_features.unsqueeze(-1)  # Shape [batch, context_length, 1]
                if past_time_features.shape[1] != self.config.context_length:
                    logging.warning(f"Adjusting past_time_features shape from {past_time_features.shape} to [batch, {self.config.context_length}, 1]")
                    past_time_features = past_time_features[:, :self.config.context_length, :]

            # Handle missing or incorrectly shaped past_observed_mask
            if past_observed_mask is None:
                past_observed_mask = torch.ones(batch_size, self.config.context_length, features).to(device)
                logging.warning("Using dummy past_observed_mask.")
            elif isinstance(past_observed_mask, np.ndarray):
                past_observed_mask = torch.FloatTensor(past_observed_mask).to(device)
                if len(past_observed_mask.shape) == 2:
                    past_observed_mask = past_observed_mask.unsqueeze(-1).repeat(1, 1, features)
                if past_observed_mask.shape[1] != self.config.context_length:
                    logging.warning(f"Adjusting past_observed_mask shape from {past_observed_mask.shape} to [batch, {self.config.context_length}, {features}]")
                    past_observed_mask = past_observed_mask[:, :self.config.context_length, :features]

            # Handle missing or incorrectly shaped future_values
            if future_values is None:
                future_values = torch.zeros(batch_size, self.config.prediction_length, features).to(device)
                logging.warning("Using dummy future_values with 17 features.")
            elif isinstance(future_values, np.ndarray):
                future_values = torch.FloatTensor(future_values).to(device)
            else:
                future_values = future_values.to(device) if isinstance(future_values, torch.Tensor) else torch.FloatTensor(future_values).to(device)

            # Ensure future_values is 3D [batch_size, prediction_length, features]
            if len(future_values.shape) == 1:  # 1D case (e.g., [batch_size])
                future_values = future_values.unsqueeze(0).unsqueeze(-1).repeat(1, self.config.prediction_length, features)
            elif len(future_values.shape) == 2:  # 2D case (e.g., [batch_size, prediction_length] or [batch_size, 1])
                future_values = future_values.unsqueeze(-1) if future_values.shape[1] == batch_size else future_values.unsqueeze(1)
            if future_values.shape[2] == 1:  # Univariate target, expand to 17 features
                logging.warning(f"Adjusting future_values shape from {future_values.shape} to [batch, {self.config.prediction_length}, {features}]")
                future_values_expanded = torch.zeros(batch_size, self.config.prediction_length, features).to(device)
                future_values_expanded[:, 0, 0] = future_values.squeeze(-1).squeeze(-1)
                future_values = future_values_expanded
            elif future_values.shape[2] != features:
                logging.warning(f"Adjusting future_values shape from {future_values.shape} to [batch, {self.config.prediction_length}, {features}]")
                future_values_expanded = torch.zeros(batch_size, self.config.prediction_length, features).to(device)
                future_values_expanded[:, :, 0] = future_values[:, :, 0] if future_values.shape[2] > 0 else future_values.squeeze(-1)
                future_values = future_values_expanded
            if future_values.shape != (batch_size, self.config.prediction_length, features):
                logging.warning(f"Final adjustment of future_values shape from {future_values.shape} to [batch, {self.config.prediction_length}, {features}]")
                future_values = future_values[:, :self.config.prediction_length, :features]

            # Handle missing or incorrectly shaped future_time_features
            if future_time_features is None:
                future_time_features = torch.arange(self.config.prediction_length, dtype=torch.float32).repeat(batch_size, 1).unsqueeze(-1).to(device) / self.config.prediction_length
                logging.warning("Using dummy future_time_features.")
            elif isinstance(future_time_features, np.ndarray):
                future_time_features = torch.FloatTensor(future_time_features).to(device)
                if len(future_time_features.shape) == 2:
                    future_time_features = future_time_features.unsqueeze(-1)
                if future_time_features.shape[1] != self.config.prediction_length:
                    logging.warning(f"Adjusting future_time_features shape from {future_time_features.shape} to [batch, {self.config.prediction_length}, 1]")
                    future_time_features = future_time_features[:, :self.config.prediction_length, :1]

            # Verify shapes dynamically
            logging.debug(f"Verified shapes - past_time_features: {past_time_features.shape}, past_observed_mask: {past_observed_mask.shape}, "
                          f"future_values: {future_values.shape}, future_time_features: {future_time_features.shape}")

            # Forward pass
            outputs = self.transformer(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_values=future_values,
                future_time_features=future_time_features
            )

            if hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            else:
                raise ValueError("Unexpected output format from TimeSeriesTransformerModel.forward")

            logging.debug(f"last_hidden_state shape before fc: {last_hidden_state.shape}")
            last_hidden = last_hidden_state[:, -1, :]  # Take the last hidden state for prediction
            logging.debug(f"last_hidden shape before fc: {last_hidden.shape}")
            # Ensure output is [batch_size, 1]
            output = self.fc(last_hidden)
            if output.shape[1] != 1:
                logging.error(f"Output shape after fc is unexpected: {output.shape}. Reshaping to [batch_size, 1]")
                output = output[:, 0:1]
            logging.debug(f"Output shape after fc: {output.shape}")
            if len(output.shape) == 1:
                output = output.unsqueeze(-1)
            logging.debug(f"Forward output shape after reshaping: {output.shape}")
            return output

        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            raise

    def train_model(self, X, y, X_val, y_val, past_time_features=None, past_observed_mask=None, past_time_features_val=None, past_observed_mask_val=None, future_values=None, future_time_features=None, future_values_val=None, future_time_features_val=None, epochs=100, batch_size=32, learning_rate=0.001, patience=10):
        """
        Train the Transformer model with early stopping and validation.
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Prepare training dataset
            if past_time_features is None or past_observed_mask is None or future_time_features is None:
                train_dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
            else:
                # Convert NumPy arrays to tensors
                X = torch.FloatTensor(X)
                past_time_features = torch.FloatTensor(past_time_features) if past_time_features is not None else None
                past_observed_mask = torch.FloatTensor(past_observed_mask) if past_observed_mask is not None else None
                future_time_features = torch.FloatTensor(future_time_features) if future_time_features is not None else None
                
                # Ensure future_values is 3D [batch, prediction_length, features] or univariate, use y for target
                if future_values is None:
                    future_values = y.unsqueeze(1).repeat(1, 1, X.shape[2])  # Use y as dummy future_values with all features, but target is univariate
                elif isinstance(future_values, np.ndarray):
                    future_values = torch.FloatTensor(future_values)
                    if len(future_values.shape) == 1:
                        future_values = future_values.unsqueeze(0).unsqueeze(-1).repeat(1, 1, X.shape[2])  # Shape [batch, 1, features]
                    elif len(future_values.shape) == 2:
                        future_values = future_values.unsqueeze(-1) if future_values.shape[1] == 1 else future_values.unsqueeze(1)  # Shape [batch, 1, features] or [batch, prediction_length, 1]
                    if future_values.shape[2] == 1:  # Univariate target, expand to 17 features
                        future_values_expanded = torch.zeros(X.shape[0], 1, X.shape[2]).to(device)
                        future_values_expanded[:, 0, 0] = future_values.squeeze(-1).squeeze(-1)  # Use first feature for target, rest as zeros
                        future_values = future_values_expanded
                elif len(future_values.shape) == 1:
                    future_values = future_values.unsqueeze(0).unsqueeze(-1).repeat(1, 1, X.shape[2])  # Shape [batch, 1, features]
                elif future_values.shape[2] == 1:  # PyTorch tensor with univariate target
                    future_values_expanded = torch.zeros(X.shape[0], 1, X.shape[2]).to(device)
                    future_values_expanded[:, 0, 0] = future_values.squeeze(-1).squeeze(-1)
                    future_values = future_values_expanded
                train_dataset = TensorDataset(X, past_time_features, past_observed_mask, future_time_features, future_values, torch.FloatTensor(y))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Prepare validation data
            if past_time_features_val is None or past_observed_mask_val is None or future_time_features_val is None:
                X_val_tensor, y_val_tensor = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
                future_values_val_tensor = None
            else:
                # Convert NumPy arrays to tensors
                X_val = torch.FloatTensor(X_val)
                past_time_features_val = torch.FloatTensor(past_time_features_val) if past_time_features_val is not None else None
                past_observed_mask_val = torch.FloatTensor(past_observed_mask_val) if past_observed_mask_val is not None else None
                future_time_features_val = torch.FloatTensor(future_time_features_val) if future_time_features_val is not None else None
                
                # Ensure future_values_val is 3D [batch, prediction_length, features] or univariate, use y_val for target
                if future_values_val is None:
                    future_values_val_tensor = y_val.unsqueeze(1).repeat(1, 1, X_val.shape[2])  # Use y_val as dummy
                elif isinstance(future_values_val, np.ndarray):
                    future_values_val_tensor = torch.FloatTensor(future_values_val)
                    if len(future_values_val_tensor.shape) == 1:
                        future_values_val_tensor = future_values_val_tensor.unsqueeze(0).unsqueeze(-1).repeat(1, 1, X_val.shape[2])  # Shape [batch, 1, features]
                    elif len(future_values_val_tensor.shape) == 2:
                        future_values_val_tensor = future_values_val_tensor.unsqueeze(-1) if future_values_val_tensor.shape[1] == 1 else future_values_val_tensor.unsqueeze(1)  # Shape [batch, 1, features] or [batch, prediction_length, 1]
                    if future_values_val_tensor.shape[2] == 1:  # Univariate target, expand to 17 features
                        future_values_val_expanded = torch.zeros(X_val.shape[0], 1, X_val.shape[2]).to(device)
                        future_values_val_expanded[:, 0, 0] = future_values_val_tensor.squeeze(-1).squeeze(-1)
                        future_values_val_tensor = future_values_val_expanded
                elif len(future_values_val.shape) == 1:
                    future_values_val_tensor = future_values_val.unsqueeze(0).unsqueeze(-1).repeat(1, 1, X_val.shape[2])  # Shape [batch, 1, features]
                elif future_values_val.shape[2] == 1:  # PyTorch tensor with univariate target
                    future_values_val_expanded = torch.zeros(X_val.shape[0], 1, X_val.shape[2]).to(device)
                    future_values_val_expanded[:, 0, 0] = future_values_val.squeeze(-1).squeeze(-1)
                    future_values_val_tensor = future_values_val_expanded
                else:
                    future_values_val_tensor = torch.FloatTensor(future_values_val)
                X_val_tensor, past_time_features_val_tensor, past_observed_mask_val_tensor, future_time_features_val_tensor, y_val_tensor = (
                    X_val, past_time_features_val, past_observed_mask_val, future_time_features_val, torch.FloatTensor(y_val)
                )

            best_val_loss = float('inf')
            early_stopping_counter = 0
            
            for epoch in range(epochs):
                self.train()
                total_loss = 0
                for batch in train_loader:
                    if past_time_features is None or past_observed_mask is None or future_time_features is None:
                        batch_X, batch_y = batch
                        batch_time_features, batch_mask, batch_future_time_features, batch_future_values = None, None, None, None
                    else:
                        batch_X, batch_time_features, batch_mask, batch_future_time_features, batch_future_values, batch_y = batch
                        
                        # Ensure batch_future_values is 3D [batch, prediction_length, features]
                        if batch_future_values is not None:
                            if len(batch_future_values.shape) == 1:  # 1D case (e.g., [batch_size])
                                batch_future_values = batch_future_values.unsqueeze(0).unsqueeze(-1).repeat(1, 1, batch_X.shape[2])  # Shape [batch, 1, features]
                            elif len(batch_future_values.shape) == 2:  # 2D case (e.g., [batch_size, prediction_length] or [batch_size, 1])
                                batch_future_values = batch_future_values.unsqueeze(-1) if batch_future_values.shape[1] == batch_X.shape[0] else batch_future_values.unsqueeze(1)  # Shape [batch, 1, features] or [batch, prediction_length, 1]
                            if batch_future_values.shape[2] == 1:  # Univariate
                                batch_future_values_expanded = torch.zeros(batch_X.shape[0], 1, batch_X.shape[2]).to(device)
                                batch_future_values_expanded[:, 0, 0] = batch_future_values.squeeze(-1).squeeze(-1)
                                batch_future_values = batch_future_values_expanded

                    optimizer.zero_grad()
                    output = self(
                        past_values=batch_X, 
                        past_time_features=batch_time_features, 
                        past_observed_mask=batch_mask, 
                        future_values=batch_future_values, 
                        future_time_features=batch_future_time_features
                    )
                    # Ensure output is 2D [batch_size, 1]
                    if len(output.shape) == 1:
                        output = output.unsqueeze(-1)  # Ensure 2D [batch_size, 1]
                    elif output.shape[1] != 1:
                        output = output[:, 0:1]  # Ensure only one output feature

                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.reshape(-1, 1)  # Ensure 2D [batch_size, 1]

                    logging.debug(f"Training - output shape: {output.shape}, batch_y shape: {batch_y.shape}")
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_train_loss = total_loss / len(train_loader)
                logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
                
                self.eval()
                with torch.no_grad():
                    val_output = self(
                        past_values=X_val_tensor, 
                        past_time_features=past_time_features_val_tensor, 
                        past_observed_mask=past_observed_mask_val_tensor, 
                        future_values=future_values_val_tensor, 
                        future_time_features=future_time_features_val_tensor
                    )
                    # Ensure val_output is 2D [batch_size, 1]
                    if len(val_output.shape) == 1:
                        val_output = val_output.unsqueeze(-1)  # Ensure 2D [batch_size, 1]
                    elif val_output.shape[1] != 1:
                        val_output = val_output[:, 0:1]  # Ensure only one output feature

                    if len(y_val_tensor.shape) == 1:
                        y_val_tensor = y_val_tensor.reshape(-1, 1)

                    logging.debug(f"Validation - val_output shape: {val_output.shape}, y_val_tensor shape: {y_val_tensor.shape}")
                    val_loss = criterion(val_output, y_val_tensor).item()  # Use y_val_tensor directly for consistency
                
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
        except Exception as e:
            logging.error(f"Error in train_model: {e}")
            raise

    def predict(self, X, past_time_features=None, past_observed_mask=None, future_time_features=None, future_values=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        # Ensure X is 3D [batch, context_length, input_dim]
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(device)
        else:
            X_tensor = X.to(device) if isinstance(X, torch.Tensor) else torch.FloatTensor(X).to(device)
        if len(X_tensor.shape) != 3:
            raise ValueError(f"Expected 3D input for X, got shape {X_tensor.shape}")

        # Get batch_size from X_tensor
        batch_size, _, features = X_tensor.shape

        # Handle other inputs and ensure 3D shapes
        if past_time_features is not None:
            if isinstance(past_time_features, np.ndarray):
                past_time_features = torch.FloatTensor(past_time_features).to(device)
            else:
                past_time_features = past_time_features.to(device) if isinstance(past_time_features, torch.Tensor) else torch.FloatTensor(past_time_features).to(device)
            if len(past_time_features.shape) != 3:
                past_time_features = past_time_features.unsqueeze(-1) if len(past_time_features.shape) == 2 else past_time_features
            if past_time_features.shape[2] != 1:
                logging.warning(f"Adjusting past_time_features shape from {past_time_features.shape} to [batch, {past_time_features.shape[1]}, 1]")
                past_time_features = past_time_features[:, :, :1]
        else:
            history_length = X_tensor.shape[1]
            past_time_features = torch.arange(history_length, dtype=torch.float32).repeat(batch_size, 1).unsqueeze(-1).to(device) / history_length
            logging.warning("Using dummy past_time_features.")

        if past_observed_mask is not None:
            if isinstance(past_observed_mask, np.ndarray):
                past_observed_mask = torch.FloatTensor(past_observed_mask).to(device)
            else:
                past_observed_mask = past_observed_mask.to(device) if isinstance(past_observed_mask, torch.Tensor) else torch.FloatTensor(past_observed_mask).to(device)
            if len(past_observed_mask.shape) != 3:
                past_observed_mask = past_observed_mask.unsqueeze(-1).repeat(1, 1, X_tensor.shape[2]) if len(past_observed_mask.shape) == 2 else past_observed_mask
            if past_observed_mask.shape[2] != X_tensor.shape[2]:
                logging.warning(f"Adjusting past_observed_mask shape from {past_observed_mask.shape} to [batch, {past_observed_mask.shape[1]}, {X_tensor.shape[2]}]")
                past_observed_mask = past_observed_mask[:, :, :X_tensor.shape[2]]
        else:
            history_length = X_tensor.shape[1]
            past_observed_mask = torch.ones(batch_size, history_length, X_tensor.shape[2]).to(device)
            logging.warning("Using dummy past_observed_mask.")

        if future_time_features is not None:
            if isinstance(future_time_features, np.ndarray):
                future_time_features = torch.FloatTensor(future_time_features).to(device)
            else:
                future_time_features = future_time_features.to(device) if isinstance(future_time_features, torch.Tensor) else torch.FloatTensor(future_time_features).to(device)
            if len(future_time_features.shape) != 3:
                future_time_features = future_time_features.unsqueeze(-1) if len(future_time_features.shape) == 2 else future_time_features
            if future_time_features.shape[2] != 1:
                logging.warning(f"Adjusting future_time_features shape from {future_time_features.shape} to [batch, {future_time_features.shape[1]}, 1]")
                future_time_features = future_time_features[:, :, :1]
        else:
            future_time_features = torch.arange(self.config.prediction_length, dtype=torch.float32).repeat(batch_size, 1).unsqueeze(-1).to(device) / self.config.prediction_length
            logging.warning("Using dummy future_time_features.")

        if future_values is not None:
            if isinstance(future_values, np.ndarray):
                future_values = torch.FloatTensor(future_values).to(device)
            else:
                future_values = future_values.to(device) if isinstance(future_values, torch.Tensor) else torch.FloatTensor(future_values).to(device)
            if len(future_values.shape) == 1:  # 1D case (e.g., [batch_size])
                future_values = future_values.unsqueeze(0).unsqueeze(-1).repeat(1, self.config.prediction_length, features)  # Shape [batch, 1, features]
            elif len(future_values.shape) == 2:  # 2D case (e.g., [batch_size, prediction_length] or [batch_size, 1])
                future_values = future_values.unsqueeze(-1) if future_values.shape[1] == batch_size else future_values.unsqueeze(1)  # Shape [batch, 1, features] or [batch, prediction_length, 1]
            if future_values.shape[2] == 1:  # Univariate, expand to 17 features
                logging.warning(f"Adjusting future_values shape from {future_values.shape} to [batch, {self.config.prediction_length}, {features}]")
                future_values_expanded = torch.zeros(batch_size, self.config.prediction_length, features).to(device)
                future_values_expanded[:, 0, 0] = future_values.squeeze(-1).squeeze(-1)  # Use first feature for target, rest as zeros
                future_values = future_values_expanded
            elif future_values.shape[2] != features:  # Ensure feature dimension matches input_dim
                logging.warning(f"Adjusting future_values shape from {future_values.shape} to [batch, {self.config.prediction_length}, {features}]")
                future_values_expanded = torch.zeros(batch_size, self.config.prediction_length, features).to(device)
                future_values_expanded[:, :, 0] = future_values[:, :, 0] if future_values.shape[2] > 0 else future_values.squeeze(-1)  # Copy first feature, rest as zeros
                future_values = future_values_expanded
            if future_values.shape != (batch_size, self.config.prediction_length, features):
                logging.warning(f"Final adjustment of future_values shape from {future_values.shape} to [batch, {self.config.prediction_length}, {features}]")
                future_values = future_values[:, :self.config.prediction_length, :features]
        else:
            future_values = torch.zeros(batch_size, self.config.prediction_length, features).to(device)
            logging.warning("Using dummy future_values with 17 features.")

        with torch.no_grad():
            output = self(
                past_values=X_tensor, 
                past_time_features=past_time_features, 
                past_observed_mask=past_observed_mask, 
                future_values=future_values, 
                future_time_features=future_time_features
            )
        # Ensure output is [batch_size, 1]
        if len(output.shape) == 1:
            output = output.unsqueeze(-1)  # Ensure 2D [batch_size, 1]
        elif output.shape[1] != 1:
            output = output[:, 0:1]  # Ensure only one output feature (univariate)
        logging.debug(f"Predict output shape before numpy: {output.shape}")
        logging.debug(f"Predict output type before numpy: {type(output)}")
        logging.debug(f"Predict output data: {output.tolist()[:5]}")  # Show first 5 values for debugging
        return output.cpu().numpy()  # Return 2D numpy array [batch_size, 1] for backtest compatibility

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
    # Dummy test
    model = TransformerPredictor(input_dim=17)
    sample_input = torch.randn(1, 30, 17)  # Matches context_length=30
    past_time_features = torch.zeros(1, 30, 1)
    past_observed_mask = torch.ones(1, 30, 17)
    future_time_features = torch.zeros(1, 1, 1)
    future_values = torch.zeros(1, 1, 1)  # Univariate target, will be expanded to 17 features
    output = model.predict(sample_input.numpy(), past_time_features.numpy(), past_observed_mask.numpy(), future_time_features.numpy(), future_values.numpy())
    logging.debug(f"TransformerPredictor output shape: {output.shape}")  # Should be [1, 1]
    print(f"TransformerPredictor output shape: {output.shape}")  # Should be [1, 1]