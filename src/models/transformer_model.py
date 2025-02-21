# src/models/transformer_model.py

import torch
import torch.nn as nn
import logging
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')  # Set to DEBUG for detailed logs

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=17, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        """
        Initialize a Time-Series Transformer for price prediction.

        Args:
            input_dim (int): Number of input features (17)
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            n_layers (int): Number of encoder layers
            dropout (float): Dropout rate
        """
        super(TransformerPredictor, self).__init__()
        
        try:
            # Configure the TimeSeriesTransformer with explicit context_length=10 and adjusted lags
            config = TimeSeriesTransformerConfig(
                input_size=input_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                prediction_length=1,  # Predict next value (univariate, but simulate multivariate for input)
                context_length=10,    # Explicitly match your sequence length
                num_time_features=1,  # Match the dummy time features (1 feature)
                lags_sequence=[1, 2, 3]  # Adjust lags to fit within context_length=10 (max lag = 3 to allow history of 7+1 steps)
            )
            self.transformer = TimeSeriesTransformerModel(config)
            self.fc = nn.Linear(d_model, 1)  # Output a single value (price, univariate)
            logging.info("TimeSeriesTransformerModel initialized successfully with context_length=10 and lags_sequence=[1, 2, 3]")
            logging.debug(f"TimeSeriesTransformerConfig: {self.transformer.config}")
        except Exception as e:
            logging.error(f"Error initializing TimeSeriesTransformerModel: {e}")
            raise

    def forward(self, past_values, past_time_features=None, past_observed_mask=None, future_values=None, future_time_features=None):
        """
        Forward pass through the transformer, handling multivariate future_values for input compatibility, but univariate for output.

        Args:
            past_values (torch.Tensor): Input tensor of shape [batch, history_length, features] -> [batch, 13, 17]
            past_time_features (torch.Tensor, optional): Time features, shape [batch, history_length, time_features] -> [batch, 13, 1]
            past_observed_mask (torch.Tensor, optional): Binary mask for observed values, shape [batch, history_length, features] -> [batch, 13, 17]
            future_values (torch.Tensor, optional): Future values for training (multivariate shape [batch, prediction_length, 17] with target at index 0), default None
            future_time_features (torch.Tensor, optional): Future time features, shape [batch, prediction_length, time_features] -> [batch, 1, 1]
        
        Returns:
            torch.Tensor: Predicted price, shape [batch, 1]
        """
        try:
            # Ensure input shape is [batch, history_length, features]
            if len(past_values.shape) != 3:
                raise ValueError(f"Expected 3D input for past_values, got shape {past_values.shape}")
            batch_size, history_length, features = past_values.shape
            
            logging.debug(f"past_values shape in forward: {past_values.shape}")
            logging.debug(f"future_values shape in forward: {future_values.shape if future_values is not None else 'None'}")
            logging.debug(f"future_time_features shape in forward: {future_time_features.shape if future_time_features is not None else 'None'}")
            logging.debug(f"past_time_features shape in forward: {past_time_features.shape if past_time_features is not None else 'None'}")
            logging.debug(f"past_observed_mask shape in forward: {past_observed_mask.shape if past_observed_mask is not None else 'None'}")
            
            # Default past_time_features if not provided (dummy time features, matching history_length and num_time_features=1)
            if past_time_features is None:
                past_time_features = torch.arange(history_length, dtype=torch.float32).repeat(batch_size, 1, 1).to(past_values.device) / history_length  # Shape [batch, history_length, 1]
                logging.warning("Using dummy past_time_features. Consider providing actual time features.")
            else:
                # Ensure past_time_features has shape [batch, history_length, num_time_features]
                if past_time_features.shape != (batch_size, history_length, 1):
                    logging.warning(f"Adjusting past_time_features shape from {past_time_features.shape} to [batch, history_length, 1]")
                    if past_time_features.shape == (batch_size, 1, 1, history_length):
                        past_time_features = past_time_features.squeeze(1).squeeze(1).transpose(0, 1)  # Reshape to [history_length, batch, 1] then transpose to [batch, history_length, 1]
                    elif past_time_features.shape == (batch_size, 1, history_length):
                        past_time_features = past_time_features.transpose(1, 2)  # Reshape to [batch, history_length, 1]
                    else:
                        past_time_features = past_time_features[:, :history_length, :1]  # Truncate or adjust to [batch, history_length, 1]

            # Default past_observed_mask if not provided (assume all data is observed, matching past_values shape)
            if past_observed_mask is None:
                past_observed_mask = torch.ones(batch_size, history_length, features).to(past_values.device)  # All observed, shape [batch, history_length, 17]
                logging.warning("Using dummy past_observed_mask. Consider providing actual observed mask.")
            
            # Default future_values if not provided (use actual data if available, otherwise dummy multivariate values with target at index 0)
            if future_values is None:
                future_values = torch.zeros(batch_size, self.transformer.config.prediction_length, features).to(past_values.device)  # Shape [batch, 1, 17]
                future_values[:, :, 0] = 0  # Set target feature (index 0) to zero as dummy
                logging.warning("Using dummy future_values with multivariate shape [batch, 1, 17]. Consider providing actual future values for training.")
            else:
                # Ensure future_values has shape [batch, prediction_length, features] for multivariate input, but extract univariate target
                if future_values.shape != (batch_size, self.transformer.config.prediction_length, features):
                    logging.warning(f"Adjusting future_values shape from {future_values.shape} to [batch, prediction_length, {features}]")
                    if future_values.shape == (batch_size, 1, 1, self.transformer.config.prediction_length):
                        # Expand univariate future_values to multivariate with zeros for non-target features
                        future_values_expanded = torch.zeros(batch_size, self.transformer.config.prediction_length, features).to(past_values.device)
                        future_values_expanded[:, :, 0] = future_values.squeeze(1).squeeze(1).transpose(0, 1)  # Set target feature (index 0)
                        future_values = future_values_expanded
                    elif future_values.shape == (batch_size, 1, self.transformer.config.prediction_length):
                        future_values_expanded = torch.zeros(batch_size, self.transformer.config.prediction_length, features).to(past_values.device)
                        future_values_expanded[:, :, 0] = future_values.transpose(1, 2)  # Set target feature (index 0)
                        future_values = future_values_expanded
                    else:
                        future_values = future_values[:, :self.transformer.config.prediction_length, :features]  # Truncate or adjust to [batch, prediction_length, 17]
                # Verify target feature (index 0) is valid for univariate prediction
                logging.debug(f"future_values target feature (index 0) shape: {future_values[:, :, 0].shape}")

            # Default future_time_features if not provided (dummy time features, matching prediction_length and num_time_features=1)
            if future_time_features is None:
                future_time_features = torch.arange(self.transformer.config.prediction_length, dtype=torch.float32).repeat(batch_size, 1, 1).to(past_values.device) / self.transformer.config.prediction_length  # Shape [batch, 1, 1]
                logging.warning("Using dummy future_time_features. Consider providing actual future time features.")
            else:
                # Ensure future_time_features has shape [batch, prediction_length, num_time_features]
                if future_time_features.shape != (batch_size, self.transformer.config.prediction_length, 1):
                    logging.warning(f"Adjusting future_time_features shape from {future_time_features.shape} to [batch, prediction_length, 1]")
                    if future_time_features.shape == (batch_size, 1, 1, self.transformer.config.prediction_length):
                        future_time_features = future_time_features.squeeze(1).squeeze(1).transpose(0, 1)  # Reshape to [prediction_length, batch, 1] then transpose to [batch, prediction_length, 1]
                    elif future_time_features.shape == (batch_size, 1, self.transformer.config.prediction_length):
                        future_time_features = future_time_features.transpose(1, 2)  # Reshape to [batch, prediction_length, 1]
                    else:
                        future_time_features = future_time_features[:, :self.transformer.config.prediction_length, :1]  # Truncate or adjust to [batch, prediction_length, 1]

            # Verify shapes match
            assert past_time_features.shape == (batch_size, history_length, 1), f"past_time_features shape mismatch: {past_time_features.shape} (expected [batch, history_length, 1])"
            assert past_observed_mask.shape == (batch_size, history_length, features), f"past_observed_mask shape mismatch: {past_observed_mask.shape} (expected [batch, history_length, 17])"
            assert future_values.shape == (batch_size, self.transformer.config.prediction_length, features), f"future_values shape mismatch: {future_values.shape} (expected [batch, prediction_length, 17])"
            assert future_time_features.shape == (batch_size, self.transformer.config.prediction_length, 1), f"future_time_features shape mismatch: {future_time_features.shape} (expected [batch, prediction_length, 1])"
            
            # Determine expected sequence length from config (should be 10)
            expected_context_length = self.transformer.config.context_length  # Should be 10
            logging.debug(f"Expected context length from config: {expected_context_length}")
            
            # Adjust history_length to match the expected context_length (10) if necessary, ensuring enough history for lags
            max_lag = max(self.transformer.config.lags_sequence)  # Should be 3
            required_history = expected_context_length + max_lag  # Should be 10 + 3 = 13
            if history_length < required_history:
                raise ValueError(f"History length {history_length} is less than required history {required_history} for context_length={expected_context_length} and lags={self.transformer.config.lags_sequence}")
            if history_length > required_history:
                logging.warning(f"History length {history_length} exceeds required history {required_history}. Truncating to match.")
                past_values = past_values[:, -required_history:, :]  # Use only the last 13 steps
                past_time_features = past_time_features[:, -required_history:, :]  # Use only the last 13 steps
                past_observed_mask = past_observed_mask[:, -required_history:, :]  # Use only the last 13 steps
                history_length = required_history
            
            # Re-verify shapes after adjustment
            assert past_time_features.shape == (batch_size, history_length, 1), f"Adjusted past_time_features shape mismatch: {past_time_features.shape}"
            assert past_observed_mask.shape == (batch_size, history_length, features), f"Adjusted past_observed_mask shape mismatch: {past_observed_mask.shape}"
            
            # Forward pass through the transformer, ensuring decoder produces output for multivariate input with univariate target
            try:
                outputs = self.transformer(
                    past_values,  # Positional argument
                    past_time_features,  # Positional argument
                    past_observed_mask,  # Positional argument
                    future_values=future_values,  # Keyword argument for training (multivariate shape [batch, 1, 17])
                    future_time_features=future_time_features  # Keyword argument for training (shape [batch, 1, 1])
                )
                logging.debug(f"Transformer outputs after forward: {outputs}")
                if hasattr(outputs, 'last_hidden_state'):
                    logging.debug(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")
                if hasattr(outputs, 'encoder_last_hidden_state'):
                    logging.debug(f"encoder_last_hidden_state shape: {outputs.encoder_last_hidden_state.shape}")
                if hasattr(outputs, 'params'):
                    logging.debug(f"Output params shape: {outputs.params.shape if outputs.params is not None else 'None'}")
                if hasattr(outputs, 'loc'):
                    logging.debug(f"Output loc shape: {outputs.loc.shape if outputs.loc is not None else 'None'}")
                if hasattr(outputs, 'scale'):
                    logging.debug(f"Output scale shape: {outputs.scale.shape if outputs.scale is not None else 'None'}")
            except AttributeError as e:
                logging.error(f"Error calling TimeSeriesTransformerModel.forward: {e}")
                raise ValueError("Check TimeSeriesTransformerModel implementation in transformers library (version 4.49.0) for correct forward method signature.")

            # Handle outputs flexibly, ensuring decoder output for univariate prediction (extract target feature)
            logging.debug(f"Transformer outputs type: {type(outputs)}")
            if isinstance(outputs, tuple) and len(outputs) > 0:
                if hasattr(outputs[0], 'last_hidden_state'):
                    last_hidden_state = outputs[0].last_hidden_state
                else:
                    last_hidden_state = outputs[0]  # Fallback to direct hidden states
            elif hasattr(outputs, 'last_hidden_state'):
                last_hidden_state = outputs.last_hidden_state
            else:
                raise ValueError("Unexpected output format from TimeSeriesTransformerModel.forward")

            logging.debug(f"last_hidden_state shape: {last_hidden_state.shape}")
            
            # Check if last_hidden_state has time dimension (dim=1) > 0
            if last_hidden_state.shape[1] == 0:
                logging.warning("last_hidden_state has no time steps (dim=1 = 0). Using encoder_last_hidden_state or first available state.")
                if hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
                    last_hidden = outputs.encoder_last_hidden_state[:, -1, :]  # Use last step of encoder output
                    logging.debug(f"Using encoder_last_hidden_state shape: {last_hidden.shape}")
                else:
                    raise ValueError("No valid hidden state available for prediction. Check model inputs or configuration.")
            else:
                last_hidden = last_hidden_state[:, -1, :]  # Take last hidden state [batch, d_model]
                logging.debug(f"Using last_hidden_state shape: {last_hidden.shape}")

            return self.fc(last_hidden)  # Output [batch, 1]
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            raise

    def train_model(self, X, y, X_val, y_val, past_time_features=None, past_observed_mask=None, past_time_features_val=None, past_observed_mask_val=None, future_values=None, future_time_features=None, future_values_val=None, future_time_features_val=None, epochs=100, batch_size=32, learning_rate=0.001, patience=10):
        """
        Train the transformer model with early stopping and validation, handling multivariate future_values for input compatibility but univariate for loss.

        Args:
            X (torch.Tensor): Training input features [n_samples, history_length, features] -> [n_samples, 13, 17]
            y (torch.Tensor): Training target values [n_samples] (univariate)
            X_val (torch.Tensor): Validation input features [n_val_samples, history_length, features] -> [n_val_samples, 13, 17]
            y_val (torch.Tensor): Validation target values [n_val_samples] (univariate)
            past_time_features (torch.Tensor, optional): Time features for training, shape [n_samples, history_length, time_features] -> [n_samples, 13, 1]
            past_observed_mask (torch.Tensor, optional): Observed mask for training, shape [n_samples, history_length, features] -> [n_samples, 13, 17]
            past_time_features_val (torch.Tensor, optional): Time features for validation, shape [n_val_samples, history_length, time_features] -> [n_val_samples, 13, 1]
            past_observed_mask_val (torch.Tensor, optional): Observed mask for validation, shape [n_val_samples, history_length, features] -> [n_val_samples, 13, 17]
            future_values (torch.Tensor, optional): Future values for training (multivariate shape [n_samples, 1, 17] with target at index 0), default None
            future_time_features (torch.Tensor, optional): Future time features for training, shape [n_samples, 1, 1]
            future_values_val (torch.Tensor, optional): Future values for validation (multivariate shape [n_val_samples, 1, 17] with target at index 0), default None
            future_time_features_val (torch.Tensor, optional): Future time features for validation, shape [n_val_samples, 1, 1]
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            patience (int): Patience for early stopping
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Prepare DataLoader for training data, handling multivariate future_values and time features separately
        if past_time_features is None or past_observed_mask is None or future_time_features is None:
            train_dataset = TensorDataset(X, y)
        else:
            # Ensure future_values is multivariate [n_samples, 1, 17] with target at index 0, then batch it
            if future_values is not None:
                if future_values.shape != (X.shape[0], 1, X.shape[2]):  # Should be [n_samples, 1, 17]
                    logging.warning(f"Adjusting future_values shape from {future_values.shape} to [n_samples, 1, {X.shape[2]}]")
                    if len(future_values.shape) == 2:  # [n_samples, 1]
                        future_values_tensor = torch.zeros(X.shape[0], 1, X.shape[2]).to(device)
                        future_values_tensor[:, :, 0] = torch.FloatTensor(future_values).to(device)  # Set target feature (index 0)
                        future_values = future_values_tensor
                    elif future_values.shape == (X.shape[0], 1, 1):
                        future_values_tensor = torch.zeros(X.shape[0], 1, X.shape[2]).to(device)
                        future_values_tensor[:, :, 0] = future_values.squeeze(2)  # Set target feature (index 0)
                        future_values = future_values_tensor
                    else:
                        future_values = torch.FloatTensor(future_values).to(device)[:, :1, :X.shape[2]]  # Truncate or adjust
                else:
                    future_values = torch.FloatTensor(future_values).to(device)
            else:
                future_values = None
            
            train_dataset = TensorDataset(X, past_time_features, past_observed_mask, future_time_features, future_values, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data, ensuring future_values_val is handled as a tensor
        if past_time_features_val is None or past_observed_mask_val is None or future_time_features_val is None:
            X_val_tensor, y_val_tensor = X_val, y_val
            future_values_val_tensor = None
        else:
            X_val_tensor, past_time_features_val_tensor, past_observed_mask_val_tensor, future_time_features_val_tensor, y_val_tensor = X_val, past_time_features_val, past_observed_mask_val, future_time_features_val, y_val
            # Convert future_values_val to tensor if provided, ensuring multivariate shape [n_val_samples, 1, 17]
            if future_values_val is not None:
                if future_values_val.shape != (X_val.shape[0], 1, X_val.shape[2]):  # Should be [n_val_samples, 1, 17]
                    logging.warning(f"Adjusting future_values_val shape from {future_values_val.shape} to [n_val_samples, 1, {X_val.shape[2]}]")
                    if len(future_values_val.shape) == 2:  # [n_val_samples, 1]
                        future_values_val_tensor = torch.zeros(X_val.shape[0], 1, X_val.shape[2]).to(device)
                        future_values_val_tensor[:, :, 0] = torch.FloatTensor(future_values_val).to(device)  # Set target feature (index 0)
                    elif future_values_val.shape == (X_val.shape[0], 1, 1):
                        future_values_val_tensor = torch.zeros(X_val.shape[0], 1, X_val.shape[2]).to(device)
                        future_values_val_tensor[:, :, 0] = future_values_val.squeeze(2)  # Set target feature (index 0)
                    else:
                        future_values_val_tensor = torch.FloatTensor(future_values_val).to(device)[:, :1, :X_val.shape[2]]  # Truncate or adjust
                else:
                    future_values_val_tensor = torch.FloatTensor(future_values_val).to(device)
            else:
                future_values_val_tensor = None
        
        logging.debug(f"X_val_tensor shape in train_model: {X_val_tensor.shape}, y_val_tensor shape: {y_val_tensor.shape}")
        logging.debug(f"past_time_features_val_tensor shape: {past_time_features_val_tensor.shape if past_time_features_val_tensor is not None else 'None'}")
        logging.debug(f"past_observed_mask_val_tensor shape: {past_observed_mask_val_tensor.shape if past_observed_mask_val_tensor is not None else 'None'}")
        logging.debug(f"future_time_features_val_tensor shape: {future_time_features_val_tensor.shape if future_time_features_val_tensor is not None else 'None'}")
        logging.debug(f"future_values_val_tensor shape: {future_values_val_tensor.shape if future_values_val_tensor is not None else 'None'} (multivariate for input, univariate for loss at index 0)")

        # Training loop with early stopping
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            self.train()  # Set to training mode
            total_loss = 0
            for batch in train_loader:
                if past_time_features is None or past_observed_mask is None or future_time_features is None:
                    batch_X, batch_y = batch
                    batch_time_features, batch_mask, batch_future_time_features, batch_future_values = None, None, None, None
                else:
                    batch_X, batch_time_features, batch_mask, batch_future_time_features, batch_future_values, batch_y = batch
                optimizer.zero_grad()
                logging.debug(f"batch_X shape in train_model loop: {batch_X.shape}")
                logging.debug(f"batch_time_features shape in train_model loop: {batch_time_features.shape if batch_time_features is not None else 'None'}")
                logging.debug(f"batch_mask shape in train_model loop: {batch_mask.shape if batch_mask is not None else 'None'}")
                logging.debug(f"batch_future_time_features shape in train_model loop: {batch_future_time_features.shape if batch_future_time_features is not None else 'None'}")
                logging.debug(f"batch_future_values shape in train_model loop: {batch_future_values.shape if batch_future_values is not None else 'None'}")
                if batch_y is not None:
                    logging.debug(f"batch_y shape in train_model loop: {batch_y.shape}")
                else:
                    logging.debug("batch_y is None in train_model loop")
                # Forward pass using multivariate future_values for input compatibility
                output = self(past_values=batch_X, past_time_features=batch_time_features, past_observed_mask=batch_mask, future_values=batch_future_values, future_time_features=batch_future_time_features)
                # Use future_values or batch_y for loss calculation separately, extracting univariate target (index 0)
                if batch_future_values is not None:
                    # batch_future_values is already batched by DataLoader
                    logging.debug(f"batch_future_values shape in train_model loop (batched): {batch_future_values.shape}")
                    # Use only the target feature (index 0) for univariate loss
                    batch_future_values_univariate = batch_future_values[:, :, 0]  # Shape [batch, 1]
                    loss = criterion(output.squeeze(), batch_future_values_univariate.squeeze())  # Use univariate target for loss
                elif batch_y is not None:
                    loss = criterion(output.squeeze(), batch_y)  # Fallback to target if no future_values
                else:
                    raise ValueError("Neither batch_future_values nor batch_y is available for loss calculation")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
            
            # Validation
            self.eval()
            with torch.no_grad():
                # Forward pass for validation using multivariate future_values for input compatibility
                val_outputs = self(past_values=X_val_tensor, past_time_features=past_time_features_val_tensor, past_observed_mask=past_observed_mask_val_tensor, future_values=future_values_val_tensor, future_time_features=future_time_features_val_tensor)
                # Use future_values_val_tensor for loss calculation separately (if provided), extracting univariate target (index 0)
                if future_values_val_tensor is not None:
                    val_loss = criterion(val_outputs.squeeze(), future_values_val_tensor[:, :, 0].squeeze())  # Use univariate target for loss
                elif y_val_tensor is not None:
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor)  # Fallback to target if no future_values_val
                else:
                    raise ValueError("Neither future_values_val_tensor nor y_val_tensor is available for validation loss calculation")
                val_loss = val_loss.item()
            
            logging.info(f"Validation Loss: {val_loss:.6f}")
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                try:
                    torch.save(self.state_dict(), "best_model.pth")
                    logging.info(f"Saved best model at epoch {epoch+1}")
                except Exception as e:
                    logging.error(f"Error saving model: {e}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    def predict(self, X, past_time_features=None, past_observed_mask=None, future_time_features=None, future_values=None):
        """
        Make predictions with the trained model, handling univariate prediction.

        Args:
            X (np.ndarray): Input features [n_samples, history_length, features] -> [n_samples, 13, 17]
            past_time_features (torch.Tensor, optional): Time features, shape [n_samples, history_length, time_features] -> [n_samples, 13, 1]
            past_observed_mask (torch.Tensor, optional): Observed mask, shape [n_samples, history_length, features] -> [n_samples, 13, 17]
            future_time_features (torch.Tensor, optional): Future time features, shape [n_samples, prediction_length, time_features] -> [n_samples, 1, 1]
            future_values (torch.Tensor, optional): Future values for prediction (multivariate shape [n_samples, 1, 17] with target at index 0), default None
        
        Returns:
            np.ndarray: Predicted values
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X).to(device)
        if past_time_features is not None:
            past_time_features = torch.FloatTensor(past_time_features).to(device)
        if past_observed_mask is not None:
            past_observed_mask = torch.FloatTensor(past_observed_mask).to(device)
        if future_time_features is not None:
            future_time_features = torch.FloatTensor(future_time_features).to(device)
        if future_values is not None:
            # Ensure future_values is multivariate [n_samples, 1, 17] with target at index 0
            if future_values.shape != (X_tensor.shape[0], 1, X_tensor.shape[2]):  # Should be [n_samples, 1, 17]
                logging.warning(f"Adjusting future_values shape from {future_values.shape} to [n_samples, 1, {X_tensor.shape[2]}]")
                if len(future_values.shape) == 2:  # [n_samples, 1]
                    future_values_tensor = torch.zeros(X_tensor.shape[0], 1, X_tensor.shape[2]).to(device)
                    future_values_tensor[:, :, 0] = torch.FloatTensor(future_values).to(device)  # Set target feature (index 0)
                    future_values = future_values_tensor
                elif future_values.shape == (X_tensor.shape[0], 1, 1):
                    future_values_tensor = torch.zeros(X_tensor.shape[0], 1, X_tensor.shape[2]).to(device)
                    future_values_tensor[:, :, 0] = future_values.squeeze(2)  # Set target feature (index 0)
                    future_values = future_values_tensor
                else:
                    future_values = torch.FloatTensor(future_values).to(device)[:, :1, :X_tensor.shape[2]]  # Truncate or adjust
            else:
                future_values = torch.FloatTensor(future_values).to(device)
        else:
            future_values = torch.zeros(X_tensor.shape[0], 1, X_tensor.shape[2]).to(device)  # Default dummy multivariate future_values
            future_values[:, :, 0] = 0  # Set target feature (index 0) to zero
            logging.warning("Using dummy future_values with multivariate shape [n_samples, 1, 17]. Consider providing actual future values for prediction.")
        
        logging.debug(f"X_tensor shape in predict: {X_tensor.shape}")
        logging.debug(f"past_time_features shape in predict: {past_time_features.shape if past_time_features is not None else 'None'}")
        logging.debug(f"past_observed_mask shape in predict: {past_observed_mask.shape if past_observed_mask is not None else 'None'}")
        logging.debug(f"future_time_features shape in predict: {future_time_features.shape if future_time_features is not None else 'None'}")
        logging.debug(f"future_values shape in predict: {future_values.shape if future_values is not None else 'None'} (multivariate for input, univariate for prediction at index 0)")
        
        with torch.no_grad():
            output = self(past_values=X_tensor, past_time_features=past_time_features, past_observed_mask=past_observed_mask, future_values=future_values, future_time_features=future_time_features)
        return output.squeeze().cpu().numpy()

    def save(self, filepath):
        """Save the model to a file."""
        try:
            torch.save(self.state_dict(), filepath)
            logging.info(f"Model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
    
    def load(self, filepath):
        """Load the model from a file."""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(torch.load(filepath, map_location=device))
            self.eval()
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

if __name__ == "__main__":
    import asyncio
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
    asyncio.run(main())