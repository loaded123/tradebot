# src/models/model_predictor.py
import torch
import numpy as np
import pandas as pd
import logging
import asyncio
import joblib
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
from src.models.train_transformer_model import create_sequences
from src.models.transformer_model import TransformerPredictor
from src.strategy.market_regime import detect_market_regime

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class ModelPredictor:
    def __init__(self, model_path: str, feature_scaler_path: str, target_scaler_path: str, 
                 input_dim: int = 22, d_model: int = 128, n_heads: int = 8, n_layers: int = 4, 
                 dropout: float = 0.1, time_steps: int = 24, min_buy_confidence: float = 0.80, 
                 min_sell_confidence: float = 0.70, cooldown_period: int = 3600):  # 1 hour in seconds
        self.model = TransformerPredictor(input_dim=input_dim, d_model=d_model, n_heads=n_heads, 
                                         n_layers=n_layers, dropout=dropout)
        self.model.load(filepath=model_path)
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        self.time_steps = time_steps
        self.min_buy_confidence = min_buy_confidence
        self.min_sell_confidence = min_sell_confidence
        self.cooldown_period = cooldown_period
        self.last_trade_time = None  # Track the last trade timestamp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logging.info(f"Model loaded and set to eval mode on device: {self.device}")

    async def predict_live_price(self, current_data: pd.DataFrame, feature_columns: list) -> dict:
        current_time = pd.Timestamp.now()
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() < self.cooldown_period:
            logging.debug(f"Cooldown active. Last trade at {self.last_trade_time}, waiting {self.cooldown_period} seconds.")
            return {"price": np.nan, "signal": 0, "confidence": 0.0}

        try:
            if len(current_data) < self.time_steps:
                raise ValueError(f"Current data must have at least {self.time_steps} rows, got {len(current_data)}")
            if not all(col in current_data.columns for col in feature_columns):
                raise ValueError(f"Current data must contain all feature columns: {feature_columns}")

            current_data = current_data.iloc[-self.time_steps:].copy()
            features = current_data[feature_columns].values
            features_scaled = self.feature_scaler.transform(features)
            dummy_targets = np.zeros(len(features))

            X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
                features_scaled.tolist(), dummy_targets.tolist(), seq_length=self.time_steps
            )

            X_tensor = torch.FloatTensor(X).to(self.device)
            past_time_features_tensor = torch.FloatTensor(past_time_features).to(self.device)
            past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(self.device)
            future_values_tensor = torch.FloatTensor(future_values).to(self.device)
            future_time_features_tensor = torch.FloatTensor(future_time_features).to(self.device)

            with torch.no_grad():
                prediction_scaled, confidence = self.model.predict(
                    X_tensor.cpu().numpy(),
                    past_time_features_tensor.cpu().numpy(),
                    past_observed_mask_tensor.cpu().numpy(),
                    future_values_tensor.cpu().numpy(),
                    future_time_features_tensor.cpu().numpy()
                )

            predicted_price = self.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
            regime = detect_market_regime(current_data)
            signal = 0
            if 'High Volatility' in regime:
                confidence_threshold = max(self.min_buy_confidence, 0.85)
            else:
                confidence_threshold = self.min_buy_confidence

            if confidence >= confidence_threshold:
                signal = 1
                self.last_trade_time = current_time
            elif confidence <= -self.min_sell_confidence:
                signal = -1
                self.last_trade_time = current_time

            logging.debug(f"Predicted live price: {predicted_price:.2f}, Confidence: {confidence:.2f}, Signal: {signal}, Regime: {regime}")
            return {"price": predicted_price, "signal": signal, "confidence": confidence, "regime": regime}

        except Exception as e:
            logging.error(f"Error in predict_live_price: {e}")
            return {"price": np.nan, "signal": 0, "confidence": 0.0}

    def predict_from_dataframe(self, data_slice: pd.DataFrame, feature_columns: list) -> pd.Series:
        """
        Predict prices for a DataFrame slice.
        """
        try:
            if len(data_slice) < self.time_steps:
                raise ValueError(f"Data slice must have at least {self.time_steps} rows, got {len(data_slice)}")
            if not all(col in data_slice.columns for col in feature_columns + ['close']):
                raise ValueError(f"Missing required columns in data_slice: {feature_columns + ['close']}")

            data_slice = data_slice.iloc[-self.time_steps:].copy()
            features = data_slice[feature_columns].values
            features_scaled = self.feature_scaler.transform(features)
            dummy_targets = np.zeros(len(features))

            X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
                features_scaled.tolist(), dummy_targets.tolist(), seq_length=self.time_steps
            )

            X_tensor = torch.FloatTensor(X).to(self.device)
            past_time_features_tensor = torch.FloatTensor(past_time_features).to(self.device)
            past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(self.device)
            future_values_tensor = torch.FloatTensor(future_values).to(self.device)
            future_time_features_tensor = torch.FloatTensor(future_time_features).to(self.device)

            with torch.no_grad():
                predictions_scaled, _ = self.model.predict(
                    X_tensor.cpu().numpy(),
                    past_time_features_tensor.cpu().numpy(),
                    past_observed_mask_tensor.cpu().numpy(),
                    future_values_tensor.cpu().numpy(),
                    future_time_features_tensor.cpu().numpy()
                )

            predictions = self.target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            expected_length = len(data_slice)
            if len(predictions) < expected_length:
                padding_length = expected_length - len(predictions)
                predictions = np.pad(predictions, (0, padding_length), mode='edge')
            elif len(predictions) > expected_length:
                predictions = predictions[:expected_length]

            return pd.Series(predictions, index=data_slice.index, name='predicted_price')

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
        'dist_to_lvn_lower': [0] * 34
    })
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(dummy_data[FEATURE_COLUMNS])
    target_scaler.fit(dummy_data[['close']])
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

    predictor = ModelPredictor(
        model_path='best_model.pth',
        feature_scaler_path='feature_scaler.pkl',
        target_scaler_path='target_scaler.pkl'
    )
    pred_result = asyncio.run(predictor.predict_live_price(dummy_data, FEATURE_COLUMNS))
    print(f"Predicted live price: {pred_result['price']:.2f}, Signal: {pred_result['signal']}, Confidence: {pred_result['confidence']:.2f}")
    pred_series = predictor.predict_from_dataframe(dummy_data, FEATURE_COLUMNS)
    print(f"Predicted series sample: {pred_series.head().tolist()}")