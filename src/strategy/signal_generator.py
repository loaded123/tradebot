# src/strategy/signal_generator.py
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Optional
from src.models.transformer_model import TransformerPredictor
from src.strategy.indicators import calculate_atr, compute_bollinger_bands, compute_vwap, compute_adx, get_onchain_metrics
from src.constants import FEATURE_COLUMNS
from src.models.train_transformer_model import create_sequences
from src.strategy.strategy_adapter_new import adapt_strategy_parameters

# Define model input columns (features the model was trained on)
MODEL_INPUT_COLUMNS = [
    'open', 'high', 'low', 'volume', 'returns', 'log_returns', 
    'price_volatility', 'sma_20', 'atr', 'vwap', 'adx', 
    'momentum_rsi', 'trend_macd', 'ema_50', 'bollinger_upper', 
    'bollinger_lower', 'bollinger_middle'
]

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s', force=True)
logging.info("Using UPDATED signal_generator.py - Mar 08, 2025 - VERSION 40")
print("signal_generator.py loaded - Mar 08, 2025 - VERSION 40")

def calculate_historical_sentiment(preprocessed_data: pd.DataFrame, idx) -> float:
    """Estimate historical sentiment based on RSI, MACD, and price trend."""
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2 or 'momentum_rsi' not in historical_data.columns or 'trend_macd' not in historical_data.columns:
        return 0.0
    
    # Price trend (normalized change over the window)
    price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0] if len(historical_data) > 1 else 0
    
    # RSI sentiment (oversold < 30 is bullish, overbought > 70 is bearish)
    rsi = historical_data['momentum_rsi'].iloc[-1]
    rsi_sentiment = -0.5 if rsi > 70 else 0.5 if rsi < 30 else 0.0
    
    # MACD sentiment (positive MACD is bullish, negative is bearish)
    macd = historical_data['trend_macd'].iloc[-1]
    macd_sentiment = 0.3 if macd > 0 else -0.3 if macd < 0 else 0.0
    
    # Combine sentiment components
    sentiment = np.clip(price_change * 5 + rsi_sentiment + macd_sentiment, -1.0, 1.0)
    logging.debug(f"Calculated historical sentiment at {idx}: Price Change {price_change:.4f}, RSI {rsi:.2f}, MACD {macd:.2f}, Total {sentiment:.2f}")
    return sentiment

async def fetch_x_sentiment(preprocessed_data: pd.DataFrame, idx) -> float:
    """Simulate X sentiment based on recent price trend from historical data."""
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 0.0
    price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0]
    sentiment = np.clip(price_change * 10, -1, 1)  # Scale price change to sentiment range
    logging.debug(f"Fetched X sentiment at {idx} based on price change {price_change:.4f}: {sentiment:.2f}")
    return sentiment

async def get_fear_and_greed_index(preprocessed_data: pd.DataFrame, idx) -> float:
    """Simulate Fear and Greed Index based on price volatility from historical data."""
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 50.0
    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    fgi = np.clip(75 - (volatility * 1000), 0, 100)  # Scale to 0-100, higher volatility lowers FGI
    logging.debug(f"Fetched FGI at {idx} based on volatility {volatility:.4f}: {fgi:.2f}")
    return fgi

def simulate_historical_whale_moves(preprocessed_data: pd.DataFrame, idx) -> float:
    """Simulate historical whale moves based on price volatility (proxy)."""
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 0.0
    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    # Higher volatility suggests more whale activity (e.g., sell-offs or accumulation)
    whale_moves = np.clip(volatility * 50, 0, 1)  # Scale to 0-1
    logging.debug(f"Simulated whale moves at {idx} based on volatility {volatility:.4f}: {whale_moves:.2f}")
    return whale_moves

async def generate_signals(scaled_df: pd.DataFrame, preprocessed_data: pd.DataFrame, model: TransformerPredictor, 
                          train_columns: List[str], feature_scaler, target_scaler, rsi_threshold: float = 50, 
                          macd_fast: int = 12, macd_slow: int = 26, atr_multiplier: float = 4.0,  # Increased to 4.0
                          max_risk_pct: float = 0.15) -> pd.DataFrame:
    """Enhanced signal generator using model predictions, technical indicators, and historical sentiment data."""
    print("Entering generate_signals function")
    logging.debug("Function entry - generate_signals called with preprocessed_data")

    required_columns = ['close', 'sma_20', 'adx', 'vwap', 'atr', 'target', 'price_volatility', 'high', 'low', 'volume', 'momentum_rsi', 'trend_macd']
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
        max_risk_pct = 0.15  # Reduced to 15% risk per trade
        print(f"Adapted params: {params}")
        logging.debug(f"Step 3: Adapted params: {params}")

        print("Extracting features for model prediction")
        features = scaled_df[MODEL_INPUT_COLUMNS].values  # Use only model-trained features
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
            features.tolist(), targets.flatten().tolist(), seq_length=34
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

        print("Unscaling predictions and debugging scaling")
        predictions_unscaled = target_scaler.inverse_transform(predictions).flatten()
        print(f"Predictions unscaled shape: {predictions_unscaled.shape}")
        print(f"Predictions unscaled sample: {predictions_unscaled[:5]}")
        if any(p <= 0 or p < 10000 or p > 200000 for p in predictions_unscaled):
            logging.warning(f"Potential scaling error in predictions_unscaled: {predictions_unscaled[:5]}")
        signal_df = scaled_df.copy()
        signal_df['predicted_price'] = pd.Series(predictions_unscaled, index=scaled_df.index)

        # Correct model prediction bias
        prediction_errors = []
        for idx in signal_df.index:
            if idx != signal_df.index[0]:  # Skip first row to avoid NaN in diff
                actual_price = preprocessed_data['close'].loc[idx]
                predicted_price = signal_df['predicted_price'].loc[idx]
                error = predicted_price - actual_price
                prediction_errors.append(error)
        avg_error = np.mean(prediction_errors) if prediction_errors else 0
        logging.info(f"Average prediction error: {avg_error:.2f} USD")
        signal_df['predicted_price'] = signal_df['predicted_price'] - avg_error  # Adjust predictions

        # Preserve datetime index and ensure no scaling errors
        signal_df.index = preprocessed_data.index

        print("Calculating indicators with unscaled values, ensuring no scaling errors")
        # Use unscaled 'close', 'high', 'low', and 'volume' from preprocessed_data
        unscaled_close = preprocessed_data['close'].copy()
        unscaled_high = preprocessed_data['high'].copy()
        unscaled_low = preprocessed_data['low'].copy()
        unscaled_volume = preprocessed_data['volume'].copy()

        # Validate and correct unscaled prices and volume
        if unscaled_close.isna().any() or unscaled_high.isna().any() or unscaled_low.isna().any() or unscaled_volume.isna().any():
            logging.warning("Unscaled prices or volume contain NaN values. Filling with previous valid values or defaults.")
            unscaled_close = unscaled_close.fillna(method='ffill').fillna(78877.88)
            unscaled_high = unscaled_high.fillna(method='ffill').fillna(79367.5)
            unscaled_low = unscaled_low.fillna(method='ffill').fillna(78186.98)
            unscaled_volume = unscaled_volume.fillna(method='ffill').fillna(1000.0)
        if (unscaled_close <= 0).any() or (unscaled_close < 10000).any() or (unscaled_close > 200000).any():
            logging.warning("Unscaled prices appear scaled or invalid. Correcting to default BTC price range.")
            unscaled_close = unscaled_close.apply(lambda x: 78877.88 if x <= 0 or x < 10000 or x > 200000 else x)
            unscaled_high = unscaled_high.apply(lambda x: 79367.5 if x <= 0 or x < 10000 or x > 200000 else x)
            unscaled_low = unscaled_low.apply(lambda x: 78186.98 if x <= 0 or x < 10000 or x > 200000 else x)

        # Recalculate indicators if needed (should already be in preprocessed_data)
        signal_df['rsi'] = preprocessed_data['momentum_rsi'].copy()  # Use precomputed RSI
        signal_df['macd'], signal_df['macd_signal'] = preprocessed_data['trend_macd'].copy(), pd.Series(np.zeros(len(signal_df)), index=signal_df.index)  # Use precomputed MACD
        signal_df['atr'] = calculate_atr(unscaled_high, unscaled_low, unscaled_close).fillna(500.0)
        signal_df['vwap'] = compute_vwap(preprocessed_data)
        signal_df['adx'] = compute_adx(preprocessed_data)

        # Add SMA_10 and SMA_20 for trend filtering
        signal_df['sma_10'] = unscaled_close.rolling(window=10).mean().bfill()
        signal_df['sma_20'] = unscaled_close.rolling(window=20).mean().bfill()

        # Calculate Bollinger Bands
        bollinger_bands = compute_bollinger_bands(preprocessed_data)
        if isinstance(bollinger_bands, pd.DataFrame) and 'bb_breakout' in bollinger_bands.columns:
            signal_df['bb_breakout'] = bollinger_bands['bb_breakout']
        else:
            logging.warning("Bollinger Bands output missing 'bb_breakout'. Using default value of 0.")
            signal_df['bb_breakout'] = 0

        # Fetch historical sentiment data
        signal_df['sentiment_score'] = [calculate_historical_sentiment(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['x_sentiment'] = [await fetch_x_sentiment(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['fear_greed_index'] = [await get_fear_and_greed_index(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['whale_moves'] = [simulate_historical_whale_moves(preprocessed_data, idx) for idx in signal_df.index]

        # Add on-chain hash rate (static for now)
        signal_df['hash_rate'] = get_onchain_metrics(symbol="BTC")['hash_rate']

        # Signal generation with SMA crossover, ADX filter, and sentiment
        signal_df['signal'] = 0
        signal_df['signal_confidence'] = 0.0
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            unscaled_rsi = signal_df['rsi'].loc[idx]
            unscaled_macd = signal_df['macd'].loc[idx]
            sma_10 = signal_df['sma_10'].loc[idx]
            sma_20 = signal_df['sma_20'].loc[idx]
            adx = signal_df['adx'].loc[idx]
            bb_breakout = signal_df['bb_breakout'].loc[idx]
            sentiment = signal_df['sentiment_score'].loc[idx]
            whale_moves = signal_df['whale_moves'].loc[idx]
            x_sentiment = signal_df['x_sentiment'].loc[idx]
            fgi = signal_df['fear_greed_index'].loc[idx]
            
            logging.debug(f"Idx: {idx}, Sentiment: {sentiment:.2f}, X Sentiment: {x_sentiment:.2f}, Whale Moves: {whale_moves:.2f}, FGI: {fgi:.2f}")
            
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200000:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88

            # Trend direction using SMA_10/SMA_20 crossover and ADX filter
            trend = 'bullish' if sma_10 > sma_20 and adx > 20 else 'bearish' if sma_10 < sma_20 and adx > 20 else 'neutral'
            
            # Calculate confidence score with adjusted threshold
            confidence = 0.0
            price_change_threshold = unscaled_atr_val * 0.1  # Increased to 0.1
            predicted_price = signal_df['predicted_price'].loc[idx]
            if predicted_price > unscaled_close_val + price_change_threshold and sma_10 > sma_20 * 1.01:  # Added SMA confirmation
                confidence += 0.4
                if unscaled_rsi < 40:
                    confidence += 0.2
                if unscaled_macd > 0.005 and signal_df['macd_signal'].loc[idx] < unscaled_macd:
                    confidence += 0.2
                if bb_breakout == 1:
                    confidence += 0.1
                if sentiment > 0 and whale_moves > 0.2:
                    confidence += 0.2  # Increased to 0.2
                if x_sentiment > 0 and whale_moves > 0.2:
                    confidence += 0.2
                if fgi < 30:
                    confidence += 0.2
                elif fgi > 70:
                    confidence -= 0.2
                if confidence >= 0.4:
                    signal_df.loc[idx, 'signal'] = 1
            elif predicted_price < unscaled_close_val - price_change_threshold and sma_10 < sma_20 * 0.99:  # Added SMA confirmation
                confidence += 0.4
                if unscaled_rsi > 60:
                    confidence += 0.2
                if unscaled_macd < -0.005 and signal_df['macd_signal'].loc[idx] > unscaled_macd:
                    confidence += 0.2
                if bb_breakout == -1:
                    confidence += 0.1
                if sentiment < 0 and whale_moves > 0.2:
                    confidence += 0.2
                if x_sentiment < 0 and whale_moves > 0.2:
                    confidence += 0.2
                if fgi < 30:
                    confidence -= 0.2
                elif fgi > 70:
                    confidence += 0.2
                if confidence >= 0.4:
                    signal_df.loc[idx, 'signal'] = -1
            
            signal_df.loc[idx, 'signal_confidence'] = min(1.0, max(0.0, confidence))
            if signal_df.loc[idx, 'signal_confidence'] < 0.001:
                logging.warning(f"Low signal_confidence at {idx}: {signal_df.loc[idx, 'signal_confidence']}")

        # Apply indicator-based filtering
        for idx in signal_df.index:
            unscaled_rsi = signal_df['rsi'].loc[idx]
            unscaled_macd = signal_df['macd'].loc[idx]
            price_volatility = signal_df['price_volatility'].loc[idx]
            
            if price_volatility > signal_df['price_volatility'].mean():
                rsi_buy_threshold = 40
                rsi_sell_threshold = 60
            else:
                rsi_buy_threshold = 40
                rsi_sell_threshold = 60
            
            if pd.isna(unscaled_rsi) or pd.isna(unscaled_macd):
                logging.warning(f"NaN detected in indicators at {idx}. Skipping filtering.")
                continue
            if unscaled_rsi > rsi_sell_threshold:
                signal_df.loc[idx, 'signal'] = 0 if signal_df.loc[idx, 'signal'] == 1 else signal_df.loc[idx, 'signal']
            elif unscaled_rsi < rsi_buy_threshold:
                signal_df.loc[idx, 'signal'] = 0 if signal_df.loc[idx, 'signal'] == -1 else signal_df.loc[idx, 'signal']
            elif unscaled_macd < -0.005 and signal_df.loc[idx, 'signal'] != 1:
                signal_df.loc[idx, 'signal'] = -1

        # Count signals
        initial_buy_count = (signal_df['signal'] == 1).sum()
        initial_sell_count = (signal_df['signal'] == -1).sum()
        logging.info(f"Initial signals: Buy signals: {initial_buy_count}, Sell signals: {initial_sell_count}")

        signal_df = filter_signals(signal_df)
        final_buy_count = (signal_df['signal'] == 1).sum()
        final_sell_count = (signal_df['signal'] == -1).sum()
        logging.info(f"After filtering: Buy signals: {final_buy_count}, Sell signals: {final_sell_count}")

        # Position sizing with dynamic take-profit
        capital = 10000
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            price_volatility = signal_df['price_volatility'].loc[idx]
            
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200000:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88
            
            risk_factor = 0.05
            volatility_adjustment = max(0.8, min(1.2, price_volatility / (signal_df['price_volatility'].mean() * 1.2)))
            position_size = (capital * risk_factor) / (unscaled_atr_val * volatility_adjustment) / (unscaled_close_val / 100000)
            atr_risk = unscaled_atr_val * atr_multiplier
            position_size = min(position_size, max_risk_pct * capital / (unscaled_close_val + atr_risk * volatility_adjustment))
            if pd.isna(position_size) or position_size <= 0:
                logging.error(f"Invalid position_size at {idx}: {position_size:.6f} BTC, using 0.05 BTC")
                position_size = 0.05
            signal_df.loc[idx, 'position_size'] = position_size
            signal_df.loc[idx, 'take_profit'] = unscaled_close_val + (unscaled_atr_val * 4.0)  # Dynamic take-profit
            signal_df.loc[idx, 'stop_loss'] = unscaled_close_val - (unscaled_atr_val * atr_multiplier)  # Wider stop-loss
            logging.debug(f"Position size at {idx}: {position_size:.6f} BTC, Unscaled Close: {unscaled_close_val:.2f} USD, ATR: {unscaled_atr_val:.2f}, Volatility Adjustment: {volatility_adjustment:.2f}")

        signal_df['total'] = capital
        signal_df['price_volatility'] = scaled_df['price_volatility']

        signal_df = signal_df[['close', 'signal', 'position_size', 'predicted_price', 'rsi', 'macd', 'macd_signal', 'atr', 'vwap', 'adx', 'bb_breakout', 'sentiment_score', 'x_sentiment', 'fear_greed_index', 'whale_moves', 'hash_rate', 'total', 'price_volatility', 'signal_confidence', 'sma_10', 'sma_20', 'take_profit', 'stop_loss']]
        logging.info(f"Generated {len(signal_df)} signals with shape: {signal_df.shape}")
        logging.debug(f"Sample data - predicted_price: {signal_df['predicted_price'].head().to_list()}")
        logging.debug(f"Sample data - position_size: {signal_df['position_size'].head().to_list()}")

        return signal_df

    except Exception as e:
        logging.error(f"Error generating signals: {str(e)}")
        raise

def filter_signals(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Filter signals to enforce a dynamic minimum hold period (24 hours default, 12 hours in high volatility)."""
    filtered_df = signal_df.copy()
    last_trade_time = None
    for idx in filtered_df.index:
        price_volatility = filtered_df['price_volatility'].loc[idx]
        min_hold_period = 24
        if price_volatility > filtered_df['price_volatility'].mean():
            min_hold_period = 12
        if last_trade_time is None or (idx - last_trade_time).total_seconds() / 3600 >= min_hold_period:
            if filtered_df.loc[idx, 'signal'] != 0 and filtered_df.loc[idx, 'signal_confidence'] >= 0.4:
                last_trade_time = idx
            continue
        filtered_df.loc[idx, 'signal'] = 0
    return filtered_df