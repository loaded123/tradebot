# src/models/model_predictor.py
import numpy as np
import pandas as pd
import torch
import logging
import joblib
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Optional, List
from src.constants import FEATURE_COLUMNS
from src.utils.sequence_utils import create_sequences
from src.models.transformer_model import TransformerPredictor
from src.strategy.market_regime import detect_market_regime
from src.utils.time_utils import calculate_days_to_next_halving

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class ModelPredictor:
    def __init__(
        self,
        model_path: str,
        feature_scaler_path: str,
        target_scaler_path: str,
        input_dim: int = len(FEATURE_COLUMNS),  # Updated to match FEATURE_COLUMNS
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.5,
        time_steps: int = 24,
        min_buy_confidence: float = 0.75,
        min_sell_confidence: float = 0.65,
        cooldown_period: int = 3600
    ):
        """
        Initialize the ModelPredictor for live price predictions.

        Args:
            model_path (str): Path to the trained model weights.
            feature_scaler_path (str): Path to the feature scaler.
            target_scaler_path (str): Path to the target scaler.
            input_dim (int): Number of input features.
            d_model (int): Dimension of the transformer model.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
            time_steps (int): Sequence length for predictions.
            min_buy_confidence (float): Minimum confidence for buy signals.
            min_sell_confidence (float): Minimum confidence for sell signals.
            cooldown_period (int): Cooldown period between predictions (seconds).
        """
        self.model = TransformerPredictor(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        self.time_steps = time_steps
        self.min_buy_confidence = min_buy_confidence
        self.min_sell_confidence = min_sell_confidence
        self.cooldown_period = cooldown_period
        self.last_trade_time = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.halving_dates = [
            pd.Timestamp("2012-11-28"),
            pd.Timestamp("2016-07-09"),
            pd.Timestamp("2020-05-11"),
            pd.Timestamp("2024-04-19"),
            pd.Timestamp("2028-03-15")
        ]
        logging.info(f"ModelPredictor initialized on device: {self.device}")

    async def predict_live_price(self, current_data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Optional[float]]:
        """
        Predict the next price using the transformer model with Monte Carlo Dropout.

        Args:
            current_data (pd.DataFrame): Historical market data.
            feature_columns (List[str]): List of feature columns to use.

        Returns:
            Dict[str, Optional[float]]: Prediction results with price, confidence, signal, and regime.
        """
        current_time = pd.Timestamp.now()
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.cooldown_period:
            logging.debug(f"Cooldown active. Last trade at {self.last_trade_time}, waiting {self.cooldown_period} seconds.")
            return {"price": np.nan, "signal": 0, "confidence": 0.0, "regime": "Unknown"}

        try:
            if len(current_data) < self.time_steps:
                raise ValueError(f"Current data must have at least {self.time_steps} rows, got {len(current_data)}")
            if not all(col in current_data.columns for col in feature_columns):
                raise ValueError(f"Current data must contain all feature columns: {feature_columns}")

            # Prepare data
            current_data = current_data.iloc[-self.time_steps:].copy()
            features = current_data[feature_columns].values
            features_scaled = self.feature_scaler.transform(features)
            dummy_targets = np.zeros(len(features_scaled))

            # Create sequences
            X, _, past_time_features, past_observed_mask, _, _ = create_sequences(
                features_scaled, dummy_targets, seq_length=self.time_steps
            )

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            past_time_features_tensor = torch.FloatTensor(past_time_features).to(self.device) if past_time_features is not None else None
            past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(self.device) if past_observed_mask is not None else None

            # Monte Carlo Dropout for uncertainty estimation
            num_samples = 10
            predictions = []
            self.model.train()  # Enable dropout for MCD
            with torch.no_grad():
                for _ in range(num_samples):
                    pred = self.model(
                        past_values=X_tensor,
                        past_time_features=past_time_features_tensor,
                        past_observed_mask=past_observed_mask_tensor
                    )
                    predictions.append(pred.cpu().numpy())

            predictions = np.array(predictions)  # Shape: [num_samples, batch_size, 1]
            mean_pred = np.mean(predictions, axis=0)  # Shape: [batch_size, 1]
            confidence = 1.0 - np.std(predictions, axis=0) / (np.mean(np.abs(predictions), axis=0) + 1e-10)

            # Unscale prediction
            predicted_price = self.target_scaler.inverse_transform(mean_pred)[-1, 0]

            # Apply halving cycle adjustment
            current_time = current_data.index[-1]
            days_to_next, _ = calculate_days_to_next_halving(current_time, self.halving_dates)
            halving_adjustment = 1.0
            if 0 < days_to_next <= 180:  # Pre-halving bullish bias
                halving_adjustment = 1.05
            predicted_price *= halving_adjustment

            # Determine market regime and signal
            regime = detect_market_regime(current_data)
            confidence_value = float(confidence[-1, 0])
            signal = 0
            confidence_threshold = self.min_buy_confidence if 'High Volatility' not in regime else max(self.min_buy_confidence, 0.85)

            if predicted_price > current_data['close'].iloc[-1] and confidence_value >= confidence_threshold:
                signal = 1
                self.last_trade_time = current_time
            elif predicted_price < current_data['close'].iloc[-1] and confidence_value >= self.min_sell_confidence:
                signal = -1
                self.last_trade_time = current_time

            logging.debug(f"Predicted live price: {predicted_price:.2f}, Confidence: {confidence_value:.2f}, Signal: {signal}, Regime: {regime}")
            return {
                "price": float(predicted_price),
                "signal": signal,
                "confidence": confidence_value,
                "regime": regime
            }

        except Exception as e:
            logging.error(f"Error in predict_live_price: {e}")
            return {"price": np.nan, "signal": 0, "confidence": 0.0, "regime": "Unknown"}

    def predict_from_dataframe(self, data_slice: pd.DataFrame, feature_columns: List[str]) -> pd.Series:
        """
        Predict prices for a DataFrame slice.

        Args:
            data_slice (pd.DataFrame): Data slice to predict on.
            feature_columns (List[str]): List of feature columns.

        Returns:
            pd.Series: Predicted prices.
        """
        try:
            if len(data_slice) < self.time_steps:
                raise ValueError(f"Data slice must have at least {self.time_steps} rows, got {len(data_slice)}")
            if not all(col in data_slice.columns for col in feature_columns + ['close']):
                raise ValueError(f"Missing required columns in data_slice: {feature_columns + ['close']}")

            data_slice = data_slice.iloc[-self.time_steps:].copy()
            features = data_slice[feature_columns].values
            features_scaled = self.feature_scaler.transform(features)
            dummy_targets = np.zeros(len(features_scaled))

            X, _, past_time_features, past_observed_mask, _, _ = create_sequences(
                features_scaled, dummy_targets, seq_length=self.time_steps
            )

            X_tensor = torch.FloatTensor(X).to(self.device)
            past_time_features_tensor = torch.FloatTensor(past_time_features).to(self.device) if past_time_features is not None else None
            past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(self.device) if past_observed_mask is not None else None

            num_samples = 10
            predictions = []
            self.model.train()  # Enable dropout for MCD
            with torch.no_grad():
                for _ in range(num_samples):
                    pred = self.model(
                        past_values=X_tensor,
                        past_time_features=past_time_features_tensor,
                        past_observed_mask=past_observed_mask_tensor
                    )
                    predictions.append(pred.cpu().numpy())

            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            predicted_prices = self.target_scaler.inverse_transform(mean_pred).flatten()

            return pd.Series(predicted_prices, index=data_slice.index, name='predicted_price')

        except Exception as e:
            logging.error(f"Error in predict_from_dataframe: {e}")
            return pd.Series([np.nan] * len(data_slice), index=data_slice.index, name='predicted_price')

if __name__ == "__main__":
    dummy_data = pd.DataFrame({
        'close': np.linspace(100, 110, 34),
        'open': [100] * 34,
        'high': [101] * 34,
        'low': [99] * 34,
        'volume': [1000] * 34,
        'momentum_rsi': [60] * 34,
        'trend_macd': [0.5] * 34,
        'atr': [1] * 34,
        'returns': [0.01] * 34,
        'log_returns': [0.01] * 34,
        'price_volatility': [0.02] * 34,
        'sma_20': [99.5] * 34,
        'vwap': [100.5] * 34,
        'adx': [25] * 34,
        'ema_50': [99.7] * 34,
        'bollinger_upper': [102] * 34,
        'bollinger_lower': [98] * 34,
        'bollinger_middle': [100] * 34,
        'luxalgo_signal': [0] * 34,
        'trendspider_signal': [0] * 34,
        'metastock_slope': [0] * 34,
        'dist_to_poc': [0] * 34,
        'dist_to_hvn_upper': [0] * 34,
        'dist_to_hvn_lower': [0] * 34,
        'dist_to_lvn_upper': [0] * 34,
        'dist_to_lvn_lower': [0] * 34,
        'days_to_next_halving': [100] * 34,
        'days_since_last_halving': [200] * 34,
        'garch_volatility': [0.01] * 34,
        'volume_normalized': [1.0] * 34,
        'hour_of_day': [12] * 34,
        'day_of_week': [2] * 34
    })
    feature_scaler = joblib.load('feature_scaler.pkl') if joblib.load('feature_scaler.pkl') else MinMaxScaler()
    target_scaler = joblib.load('target_scaler.pkl') if joblib.load('target_scaler.pkl') else MinMaxScaler()
    feature_scaler.fit(dummy_data[FEATURE_COLUMNS])
    target_scaler.fit(dummy_data[['close']])
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

    predictor = ModelPredictor(
        model_path='best_model.pth',
        feature_scaler_path='feature_scaler.pkl',
        target_scaler_path='target_scaler.pkl'
    )
    import asyncio
    pred_result = asyncio.run(predictor.predict_live_price(dummy_data, FEATURE_COLUMNS))
    print(f"Predicted live price: {pred_result['price']:.2f}, Signal: {pred_result['signal']}, Confidence: {pred_result['confidence']:.2f}")
    pred_series = predictor.predict_from_dataframe(dummy_data, FEATURE_COLUMNS)
    print(f"Predicted series sample: {pred_series.head().tolist()}")