# src/strategy/signal_generator.py
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Optional
from src.models.transformer_model import TransformerPredictor
from src.strategy.indicators import calculate_rsi, calculate_macd, calculate_atr
from src.constants import FEATURE_COLUMNS
from models.train_transformer_model import create_sequences
from src.strategy.strategy_adapter_new import adapt_strategy_parameters

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s', force=True)
logging.info("Using UPDATED signal_generator.py - Feb 28, 2025 - VERSION 12")
print("signal_generator.py loaded - Feb 28, 2025 - VERSION 12")

async def generate_signals(scaled_df: pd.DataFrame, preprocessed_data: pd.DataFrame, model: TransformerPredictor, 
                          train_columns: List[str], feature_scaler, target_scaler, rsi_threshold: float = 50, 
                          macd_fast: int = 12, macd_slow: int = 26, atr_multiplier: float = 2.0, 
                          max_risk_pct: float = 0.01) -> pd.DataFrame:
    """Enhanced signal generator using model predictions, RSI, MACD, ATR, and dynamic parameters with unscaled prices."""
    print("Entering generate_signals function")
    logging.debug("Function entry - generate_signals called with preprocessed_data")

    required_columns = ['close', 'sma_20', 'adx', 'vwap', 'atr', 'target', 'price_volatility', 'high', 'low']
    print(f"Required columns: {required_columns}")
    logging.debug(f"Checking required columns: {required_columns}")
    for col in required_columns:
        if col not in scaled_df.columns:
            raise ValueError(f"Missing required column in scaled_df: {col}")
        if col not in preprocessed_data.columns:
            raise ValueError(f"Missing required column in preprocessed_data: {col}")
    print(f"Scaled_df columns: {scaled_df.columns.tolist()}")
    print(f"Preprocessed_data columns: {preprocessed_data.columns.tolist()}")
    logging.debug(f"All required columns present in scaled_df and preprocessed_data: {scaled_df.columns.tolist()} and {preprocessed_data.columns.tolist()}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model eval mode on device: {device}")
    logging.debug(f"Model set to eval mode on device: {device}")

    try:
        print("Inside try block")
        logging.debug(f"Step 1: Entering generate_signals - file: {__file__}")
        logging.debug(f"Step 2: Globals: {list(globals().keys())}")
        logging.debug(f"Step 2.5: Model device: {next(model.parameters()).device}")

        print("Calling adapt_strategy_parameters")
        params = adapt_strategy_parameters(scaled_df)
        rsi_threshold = params['rsi_threshold']
        macd_fast = params['macd_fast']
        macd_slow = params['macd_slow']
        atr_multiplier = params['atr_multiplier']
        max_risk_pct = params['max_risk_pct']
        print(f"Adapted params: {params}")
        logging.debug(f"Step 3: Adapted params: {params}")

        print("Extracting features")
        features = scaled_df[FEATURE_COLUMNS].values
        print(f"Features shape: {features.shape}")
        logging.debug(f"Step 4: Features shape: {features.shape}")
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        logging.debug(f"Step 4.5: Features reshaped: {features.shape}")

        print("Extracting targets")
        targets = scaled_df['target'].values
        print(f"Targets shape: {targets.shape}")
        logging.debug(f"Step 5: Targets shape: {targets.shape}")
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        logging.debug(f"Step 5.5: Targets reshaped: {targets.shape}")

        print("Creating sequences")
        X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
            features.tolist(), targets.flatten().tolist(), seq_length=13
        )
        print(f"X shape: {X.shape}")
        logging.debug(f"Step 6: X shape: {X.shape}")
        logging.debug(f"Step 7: y shape: {y.shape}")
        logging.debug(f"Step 8: past_time_features shape: {past_time_features.shape if past_time_features is not None else 'None'}")
        logging.debug(f"Step 9: past_observed_mask shape: {past_observed_mask.shape if past_observed_mask is not None else 'None'}")
        logging.debug(f"Step 10: future_values shape: {future_values.shape if future_values is not None else 'None'}")
        logging.debug(f"Step 11: future_time_features shape: {future_time_features.shape if future_time_features is not None else 'None'}")

        print("Converting to tensors")
        X_tensor = torch.FloatTensor(X).to(device)
        if len(X_tensor.shape) != 3:
            raise ValueError(f"X_tensor shape {X_tensor.shape} must be 3D [n_sequences, seq_length, n_features]")
        print(f"X_tensor shape: {X_tensor.shape}")
        logging.debug(f"Step 12: X_tensor shape: {X_tensor.shape}")

        y_tensor = torch.FloatTensor(y).to(device)
        if len(y_tensor.shape) != 2 or y_tensor.shape[1] != 1:
            y_tensor = y_tensor.reshape(-1, 1)
        logging.debug(f"Step 13: y_tensor shape: {y_tensor.shape}")

        past_time_features_tensor = torch.FloatTensor(past_time_features).to(device) if past_time_features is not None else None
        past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(device) if past_observed_mask is not None else None
        future_values_tensor = torch.FloatTensor(future_values).to(device) if future_values is not None else None
        future_time_features_tensor = torch.FloatTensor(future_time_features).to(device) if future_time_features is not None else None

        print("Running model.predict")
        with torch.no_grad():
            predictions = model.predict(
                X_tensor.cpu().numpy(),
                past_time_features=past_time_features_tensor.cpu().numpy() if past_time_features_tensor is not None else None,
                past_observed_mask=past_observed_mask_tensor.cpu().numpy() if past_observed_mask_tensor is not None else None,
                future_values=future_values_tensor.cpu().numpy() if future_values_tensor is not None else None,
                future_time_features=future_time_features_tensor.cpu().numpy() if future_time_features_tensor is not None else None
            )
        print(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
        print(f"Predictions sample: {predictions[:5] if hasattr(predictions, '__getitem__') else str(predictions)}")
        logging.debug(f"Step 14: Raw predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
        logging.debug(f"Step 14.5: Raw predictions sample: {predictions[:5] if hasattr(predictions, '__getitem__') else str(predictions)}")

        if not hasattr(predictions, 'shape'):
            logging.error(f"Predictions is not an array: {type(predictions)}")
            raise ValueError(f"Model.predict() returned invalid type: {type(predictions)}")
        if np.array_equal(predictions, np.arange(len(predictions))):
            logging.error("Predictions are just indices! Check model.predict() implementation.")
            raise ValueError("Predictions appear to be indices, not values.")

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.shape[1] != 1:
            predictions = predictions[:, 0:1]
        logging.debug(f"Step 15: Predictions reshaped: {predictions.shape}")

        expected_length = len(scaled_df)
        pred_length = predictions.shape[0]
        if pred_length < expected_length:
            logging.warning(f"Prediction length ({pred_length}) < scaled_df length ({expected_length}). Padding...")
            padding_length = expected_length - pred_length
            predictions = np.pad(predictions, ((0, padding_length), (0, 0)), mode='edge')
        elif pred_length > expected_length:
            predictions = predictions[:expected_length]
        logging.debug(f"Step 16: Predictions adjusted: {predictions.shape}")

        print("Unscaling predictions")
        predictions_unscaled = target_scaler.inverse_transform(predictions).flatten()
        print(f"Predictions unscaled shape: {predictions_unscaled.shape}")
        print(f"Predictions unscaled sample: {predictions_unscaled[:5]}")
        signal_df = scaled_df.copy()
        signal_df['predicted_price'] = pd.Series(predictions_unscaled, index=scaled_df.index)

        # Preserve datetime index
        signal_df.index = preprocessed_data.index

        print("Calculating indicators")
        # Use unscaled 'close', 'high', and 'low' from preprocessed_data
        unscaled_close = preprocessed_data['close'].copy()
        unscaled_high = preprocessed_data['high'].copy()
        unscaled_low = preprocessed_data['low'].copy()

        # Validate unscaled prices
        if unscaled_close.isna().any() or unscaled_high.isna().any() or unscaled_low.isna().any():
            logging.warning("Unscaled prices contain NaN values. Filling with previous valid values or default BTC price.")
            unscaled_close = unscaled_close.fillna(method='ffill').fillna(50000.0)  # Default to $50,000
            unscaled_high = unscaled_high.fillna(method='ffill').fillna(50000.0)
            unscaled_low = unscaled_low.fillna(method='ffill').fillna(50000.0)
        if (unscaled_close <= 0).any() or (unscaled_close < 10000).any():
            logging.warning("Unscaled prices appear scaled or invalid. Correcting to default BTC price.")
            unscaled_close = unscaled_close.apply(lambda x: 50000.0 if x <= 0 or x < 10000 else x)
            unscaled_high = unscaled_high.apply(lambda x: 50000.0 if x <= 0 or x < 10000 else x)
            unscaled_low = unscaled_low.apply(lambda x: 50000.0 if x <= 0 or x < 10000 else x)

        # Calculate indicators with unscaled prices and handle potential NaN values
        signal_df['rsi'] = calculate_rsi(unscaled_close, window=14)
        signal_df['macd'] = calculate_macd(unscaled_close, fast=macd_fast, slow=macd_slow)
        signal_df['atr'] = calculate_atr(unscaled_high, unscaled_low, unscaled_close, window=14)
        if signal_df['atr'].isna().any():
            logging.warning("ATR contains NaN values. Filling with forward fill or default BTC ATR.")
            signal_df['atr'] = signal_df['atr'].ffill().fillna(500.0)  # Default to typical BTC ATR (~$500)

        # Count initial signals before filtering
        initial_buy_count = 0
        initial_sell_count = 0
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0  # Default to typical BTC ATR
            if unscaled_close_val <= 0 or unscaled_close_val < 10000:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 50000.0 USD")
                unscaled_close_val = 50000.0
            if signal_df['predicted_price'].loc[idx] > unscaled_close_val * (1 + atr_multiplier * unscaled_atr_val / unscaled_close_val):
                initial_buy_count += 1
            elif signal_df['predicted_price'].loc[idx] < unscaled_close_val * (1 - atr_multiplier * unscaled_atr_val / unscaled_close_val):
                initial_sell_count += 1
        logging.info(f"Initial signals: Buy signals: {initial_buy_count}, Sell signals: {initial_sell_count}")

        # Generate initial signals with unscaled values
        signal_df['signal'] = 0
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0  # Default to typical BTC ATR
            if unscaled_close_val <= 0 or unscaled_close_val < 10000:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 50000.0 USD")
                unscaled_close_val = 50000.0
            if signal_df['predicted_price'].loc[idx] > unscaled_close_val * (1 + atr_multiplier * unscaled_atr_val / unscaled_close_val):
                signal_df.loc[idx, 'signal'] = 1  # Buy
            elif signal_df['predicted_price'].loc[idx] < unscaled_close_val * (1 - atr_multiplier * unscaled_atr_val / unscaled_close_val):
                signal_df.loc[idx, 'signal'] = -1  # Sell

        # Apply indicator-based filtering with unscaled values
        for idx in signal_df.index:
            unscaled_rsi = signal_df['rsi'].loc[idx]  # RSI is already unscaled to 0–100
            unscaled_macd = signal_df['macd'].loc[idx]  # MACD is unscaled
            if pd.isna(unscaled_rsi) or pd.isna(unscaled_macd):
                logging.warning(f"NaN detected in indicators at {idx}. Skipping filtering for this row.")
                continue
            if unscaled_rsi > rsi_threshold:  # Overbought, hold or sell
                signal_df.loc[idx, 'signal'] = 0 if signal_df.loc[idx, 'signal'] == 1 else signal_df.loc[idx, 'signal']
            elif unscaled_rsi < 100 - rsi_threshold:  # Oversold, hold or buy
                signal_df.loc[idx, 'signal'] = 0 if signal_df.loc[idx, 'signal'] == -1 else signal_df.loc[idx, 'signal']
            elif unscaled_macd < 0 and signal_df.loc[idx, 'signal'] != 1:  # Bearish MACD, sell (unless already buy)
                signal_df.loc[idx, 'signal'] = -1

        # Count signals after filtering
        final_buy_count = (signal_df['signal'] == 1).sum()
        final_sell_count = (signal_df['signal'] == -1).sum()
        logging.info(f"After filtering: Buy signals: {final_buy_count}, Sell signals: {final_sell_count}")

        # Position sizing with unscaled prices (1% of capital, capped at 0.05 BTC)
        capital = 10000  # Starting capital
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0  # Default to typical BTC ATR
            if unscaled_close_val <= 0 or unscaled_close_val < 10000:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 50000.0 USD")
                unscaled_close_val = 50000.0
            position_size = min(0.05, (capital * max_risk_pct) / unscaled_close_val)  # 1% of capital, capped at 0.05 BTC
            atr_risk = unscaled_atr_val * atr_multiplier
            volatility_adjustment = max(1, signal_df['price_volatility'].loc[idx] / scaled_df['price_volatility'].mean())
            position_size = min(position_size, max_risk_pct * capital / (unscaled_close_val + atr_risk * volatility_adjustment))
            signal_df.loc[idx, 'position_size'] = position_size
            logging.debug(f"Position size at {idx}: {position_size:.6f} BTC, Unscaled Close: {unscaled_close_val:.2f} USD, ATR: {unscaled_atr_val:.2f}, Volatility Adjustment: {volatility_adjustment}")

        signal_df['total'] = capital
        signal_df['price_volatility'] = scaled_df['price_volatility']

        signal_df = signal_df[['close', 'signal', 'position_size', 'predicted_price', 'rsi', 'macd', 'atr', 'total', 'price_volatility']]
        logging.info(f"Generated {len(signal_df)} signals with shape: {signal_df.shape}")
        logging.debug(f"Sample data - predicted_price: {signal_df['predicted_price'].head().to_list()}")
        return signal_df

    except Exception as e:
        logging.error(f"Error generating signals: {str(e)}")
        raise