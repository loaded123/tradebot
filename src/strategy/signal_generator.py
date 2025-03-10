# src/strategy/signal_generator.py
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Tuple
from src.models.transformer_model import TransformerPredictor
from src.strategy.indicators import calculate_atr, compute_bollinger_bands, compute_vwap, compute_adx, get_onchain_metrics, calculate_macd, calculate_rsi
from src.constants import FEATURE_COLUMNS
from src.models.train_transformer_model import create_sequences
from src.strategy.position_sizer import calculate_position_size, kelly_criterion
from src.strategy.market_regime import detect_market_regime

# Define model input columns (features the model was trained on)
MODEL_INPUT_COLUMNS = [
    'open', 'high', 'low', 'volume', 'returns', 'log_returns', 
    'price_volatility', 'sma_20', 'atr', 'vwap', 'adx', 
    'momentum_rsi', 'trend_macd', 'ema_50', 'bollinger_upper', 
    'bollinger_lower', 'bollinger_middle'
]

# Configure main logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', force=True)
logging.info("Using UPDATED signal_generator.py - Mar 09, 2025 - VERSION 82.4 (Leverage Indicators.py)")
print("signal_generator.py loaded - Mar 09, 2025 - VERSION 82.4 (Leverage Indicators.py)")

# Configure separate loggers with summarized output
sentiment_logger = logging.getLogger('sentiment')
sentiment_handler = logging.FileHandler('sentiment.log')
sentiment_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
sentiment_logger.addHandler(sentiment_handler)
sentiment_logger.setLevel(logging.INFO)

indicators_logger = logging.getLogger('indicators')
indicators_handler = logging.FileHandler('indicators.log')
indicators_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
indicators_logger.addHandler(indicators_handler)
indicators_logger.setLevel(logging.INFO)

signals_logger = logging.getLogger('signals')
signals_handler = logging.FileHandler('signals.log')
signals_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
signals_logger.addHandler(signals_handler)
signals_logger.setLevel(logging.INFO)

def calculate_historical_sentiment(preprocessed_data: pd.DataFrame, idx) -> float:
    """Estimate historical sentiment based on RSI, MACD, and price trend."""
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2 or 'momentum_rsi' not in historical_data.columns or 'trend_macd' not in historical_data.columns:
        return 0.0
    
    price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0] if len(historical_data) > 1 else 0
    rsi = historical_data['momentum_rsi'].iloc[-1]
    rsi_sentiment = -0.5 if rsi > 70 else 0.5 if rsi < 30 else 0.0
    macd = historical_data['trend_macd'].iloc[-1]
    macd_sentiment = 0.3 if macd > 0 else -0.3 if macd < 0 else 0.0
    sentiment = np.clip(price_change * 5 + rsi_sentiment + macd_sentiment, -1.0, 1.0)
    return min(sentiment, 0.3)  # Cap sentiment influence

async def fetch_x_sentiment(preprocessed_data: pd.DataFrame, idx) -> float:
    """
    Simulate X sentiment based on price trend, RSI, volume change, and Fear & Greed Index.
    Note: Free X API tier does not support tweet search. Treated as secondary factor per user request.
    """
    logging.warning(f"Simulating X sentiment at {idx}. Free X API tier does not support tweet search. Treated as secondary factor.")
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 0.0
    
    # Price change component
    price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0]
    price_sentiment = price_change * 5

    # RSI component
    rsi = historical_data['momentum_rsi'].iloc[-1] if 'momentum_rsi' in historical_data.columns else 50
    rsi_sentiment = -0.3 if rsi > 70 else 0.3 if rsi < 30 else 0.0

    # Volume change component
    volume_change = (historical_data['volume'].iloc[-1] - historical_data['volume'].iloc[0]) / historical_data['volume'].iloc[0] if historical_data['volume'].iloc[0] != 0 else 0
    volume_sentiment = 0.2 if volume_change > 0.1 else -0.2 if volume_change < -0.1 else 0.0
    if price_change < 0:
        volume_sentiment *= -1

    # Fear & Greed Index component
    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    fgi = np.clip(75 - (volatility * 1000), 0, 100)
    fgi_sentiment = -0.2 if fgi > 70 else 0.2 if fgi < 30 else 0.0

    # Combine components
    sentiment = price_sentiment + rsi_sentiment + volume_sentiment + fgi_sentiment
    sentiment = np.clip(sentiment, -1.0, 1.0)
    logging.info(f"Simulated X sentiment at {idx}: {sentiment:.2f} (Price: {price_sentiment:.2f}, RSI: {rsi_sentiment:.2f}, Volume: {volume_sentiment:.2f}, FGI: {fgi_sentiment:.2f})")
    return sentiment

async def get_fear_and_greed_index(preprocessed_data: pd.DataFrame, idx) -> float:
    """Simulate Fear and Greed Index based on price volatility from historical data."""
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 50.0
    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    fgi = np.clip(75 - (volatility * 1000), 0, 100)
    return fgi

def simulate_historical_whale_moves(preprocessed_data: pd.DataFrame, idx) -> float:
    """Simulate historical whale moves based on price volatility (proxy)."""
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 0.0
    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    whale_moves = np.clip(volatility * 50, 0, 1)
    return whale_moves

def adapt_strategy_parameters(scaled_df: pd.DataFrame) -> dict:
    """Adapt strategy parameters based on market conditions."""
    market_regime = detect_market_regime(scaled_df)
    regime_params = {
        'Bullish Low Volatility': {'rsi_threshold': 45, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
        'Bullish High Volatility': {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
        'Bearish Low Volatility': {'rsi_threshold': 40, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
        'Bearish High Volatility': {'rsi_threshold': 35, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
        'Neutral': {'rsi_threshold': 45, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
    }
    return regime_params.get(market_regime, {'rsi_threshold': 45, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15})

def filter_signals(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Filter signals to enforce a minimum hold period, dynamically adjusted by price volatility."""
    filtered_df = signal_df.copy()
    last_trade_time = None
    min_hold_period = 4  # Increased to 4 hours to reduce overtrading
    for idx in filtered_df.index:
        current_signal = filtered_df.loc[idx, 'signal']
        price_volatility = filtered_df.loc[idx, 'price_volatility'] if 'price_volatility' in filtered_df.columns else 0.0
        dynamic_min_hold = min_hold_period if price_volatility <= filtered_df['price_volatility'].mean() else 2  # Relax in high volatility
        if (last_trade_time is None or 
            (idx - last_trade_time).total_seconds() / 3600 >= dynamic_min_hold) and current_signal != 0 and filtered_df.loc[idx, 'signal_confidence'] >= 0.05:
            last_trade_time = idx
            signals_logger.info(f"Accepted signal at {idx} with confidence {filtered_df.loc[idx, 'signal_confidence']:.2f}")
        else:
            filtered_df.loc[idx, 'signal'] = 0
            if current_signal != 0:
                signals_logger.debug(f"Signal filtered at {idx} due to min hold period ({dynamic_min_hold} hours) or low confidence ({filtered_df.loc[idx, 'signal_confidence']:.2f})")
    return filtered_df

async def generate_signals(scaled_df: pd.DataFrame, preprocessed_data: pd.DataFrame, model: TransformerPredictor, 
                          train_columns: List[str], feature_scaler, target_scaler, rsi_threshold: float = 50, 
                          macd_fast: int = 12, macd_slow: int = 26, atr_multiplier: float = 2.0, 
                          max_risk_pct: float = 0.20) -> pd.DataFrame:
    """Enhanced signal generator using model predictions, technical indicators, and historical sentiment data."""
    print("Entering generate_signals function")
    logging.debug("Function entry - generate_signals called with preprocessed_data")

    required_columns = ['close', 'sma_20', 'adx', 'vwap', 'atr', 'target', 'price_volatility', 'high', 'low', 'volume', 'momentum_rsi', 'trend_macd', 'returns']
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
        atr_multiplier = params['atr_multiplier']  # Use adapted value, default to 3.0
        max_risk_pct = params['max_risk_pct']  # Use adapted max_risk_pct
        print(f"Adapted params: {params}")
        logging.debug(f"Step 3: Adapted params: {params}")

        print("Extracting features for model prediction")
        features = scaled_df[MODEL_INPUT_COLUMNS].values
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
        if any(p <= 0 or p < 10000 or p > 200200 for p in predictions_unscaled):
            logging.warning(f"Potential scaling error in predictions_unscaled: {predictions_unscaled[:5]}")
        signal_df = scaled_df.copy()
        signal_df['predicted_price'] = pd.Series(predictions_unscaled, index=scaled_df.index)
        signal_df['raw_predicted_price'] = signal_df['predicted_price'].copy()  # Log raw predictions

        # Define unscaled variables before correction
        unscaled_close = preprocessed_data['close'].copy()
        unscaled_high = preprocessed_data['high'].copy()
        unscaled_low = preprocessed_data['low'].copy()
        unscaled_volume = preprocessed_data['volume'].copy()

        if unscaled_close.isna().any() or unscaled_high.isna().any() or unscaled_low.isna().any() or unscaled_volume.isna().any():
            logging.warning("Unscaled prices or volume contain NaN values. Filling with previous valid values or defaults.")
            unscaled_close = unscaled_close.fillna(method='ffill').fillna(78877.88)
            unscaled_high = unscaled_high.fillna(method='ffill').fillna(79367.5)
            unscaled_low = unscaled_low.fillna(method='ffill').fillna(78186.98)
            unscaled_volume = unscaled_volume.fillna(method='ffill').fillna(1000.0)
        if (unscaled_close <= 0).any() or (unscaled_close < 10000).any() or (unscaled_close > 200200).any():
            logging.warning("Unscaled prices appear scaled or invalid. Correcting to default BTC price range.")
            unscaled_close = unscaled_close.apply(lambda x: 78877.88 if x <= 0 or x < 10000 or x > 200200 else x)
            unscaled_high = unscaled_high.apply(lambda x: 79367.5 if x <= 0 or x < 10000 or x > 200200 else x)
            unscaled_low = unscaled_low.apply(lambda x: 78186.98 if x <= 0 or x < 10000 or x > 200200 else x)

        # Correct model prediction bias using a rolling 24h window with .loc
        avg_error = 0
        window = 24
        for i in range(len(signal_df)):
            idx = signal_df.index[i]
            if i < window:
                errors = [p - unscaled_close.loc[idx] for idx, p in zip(signal_df.index[:i+1], signal_df['raw_predicted_price'][:i+1]) if idx != signal_df.index[0]]
            else:
                errors = [p - unscaled_close.loc[idx] for idx, p in zip(signal_df.index[i-window+1:i+1], signal_df['raw_predicted_price'][i-window+1:i+1])]
            avg_error = np.mean(errors) if errors else avg_error
            signal_df.loc[idx, 'predicted_price'] = signal_df.loc[idx, 'raw_predicted_price'] - avg_error
        logging.info(f"Final average prediction error after rolling correction: {avg_error:.2f} USD")

        # Calculate indicators with unscaled values using indicators.py
        signal_df['rsi'] = calculate_rsi(unscaled_close)  # Replace precomputed RSI
        if signal_df['rsi'].isna().all():
            signal_df['rsi'] = 50.0  # Fallback
            logging.warning("RSI column is all NaN, using fallback value 50.0")
        macd, macd_signal = calculate_macd(unscaled_close, fast=macd_fast, slow=macd_slow)  # Replace compute_macd
        if macd.isna().all():
            macd = pd.Series(np.zeros(len(signal_df)), index=signal_df.index)
            macd_signal = pd.Series(np.zeros(len(signal_df)), index=signal_df.index)
            logging.warning("MACD column is all NaN, using fallback value 0.0")
        signal_df['macd'] = macd
        signal_df['macd_signal'] = macd_signal
        signal_df['atr'] = calculate_atr(unscaled_high, unscaled_low, unscaled_close).fillna(500.0)
        signal_df['vwap'] = compute_vwap(preprocessed_data)
        signal_df['adx'] = compute_adx(preprocessed_data).fillna(10.0)
        if signal_df['adx'].isna().all():
            signal_df['adx'] = 10.0
            logging.warning("ADX column is all NaN, using fallback value 10.0")
        signal_df['sma_10'] = unscaled_close.rolling(window=10, min_periods=1).mean().bfill()
        signal_df['sma_20'] = unscaled_close.rolling(window=20, min_periods=1).mean().bfill()
        signal_df['volume_sma_20'] = unscaled_volume.rolling(window=20, min_periods=1).mean().bfill()

        # Log indicator samples for debugging
        logging.debug(f"Sample RSI: {signal_df['rsi'].head().to_list()}")
        logging.debug(f"Sample MACD: {signal_df['macd'].head().to_list()}")
        logging.debug(f"Sample MACD Signal: {signal_df['macd_signal'].head().to_list()}")
        logging.debug(f"Sample ADX: {signal_df['adx'].head().to_list()}")
        logging.debug(f"Sample SMA_10: {signal_df['sma_10'].head().to_list()}")
        logging.debug(f"Sample SMA_20: {signal_df['sma_20'].head().to_list()}")

        bollinger_bands = compute_bollinger_bands(preprocessed_data)
        if isinstance(bollinger_bands, pd.DataFrame) and 'bb_breakout' in bollinger_bands.columns:
            signal_df['bb_breakout'] = bollinger_bands['bb_breakout']
        else:
            logging.warning("Bollinger Bands output missing 'bb_breakout'. Using default value of 0.")
            signal_df['bb_breakout'] = 0

        signal_df['sentiment_score'] = [calculate_historical_sentiment(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['x_sentiment'] = [await fetch_x_sentiment(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['fear_greed_index'] = [await get_fear_and_greed_index(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['whale_moves'] = [simulate_historical_whale_moves(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['hash_rate'] = get_onchain_metrics(symbol="BTC")['hash_rate']

        # Detect market regime
        signal_df['market_regime'] = [detect_market_regime(signal_df.loc[:idx], window=12) for idx in signal_df.index]

        summary_interval = 24
        for i in range(0, len(signal_df), summary_interval):
            end_idx = min(i + summary_interval, len(signal_df))
            window_df = signal_df.iloc[i:end_idx]
            if not window_df.empty:
                rsi_mean = window_df['rsi'].mean()
                rsi_overbought = (window_df['rsi'] > 70).sum()
                rsi_oversold = (window_df['rsi'] < 30).sum()
                macd_mean = window_df['macd'].mean()
                adx_mean = window_df['adx'].mean()
                adx_strong = (window_df['adx'] > 10).sum()
                bb_breakouts = window_df['bb_breakout'].value_counts().to_dict()
                sma_trend_bullish = (window_df['sma_10'] > window_df['sma_20']).sum()
                indicators_logger.info(
                    f"Indicator Summary [{window_df.index[0]} to {window_df.index[-1]}]: "
                    f"RSI Mean: {rsi_mean:.2f}, Overbought (>70): {rsi_overbought}, Oversold (<30): {rsi_oversold}, "
                    f"MACD Mean: {macd_mean:.2f}, ADX Mean: {adx_mean:.2f}, Strong Trend (>10): {adx_strong}, "
                    f"BB Breakouts: {bb_breakouts}, SMA Bullish (10>20): {sma_trend_bullish}/{len(window_df)}"
                )

        sentiment_summary = {
            'sentiment_score_mean': signal_df['sentiment_score'].mean(),
            'x_sentiment_mean': signal_df['x_sentiment'].mean(),
            'fear_greed_mean': signal_df['fear_greed_index'].mean(),
            'whale_moves_mean': signal_df['whale_moves'].mean(),
            'sentiment_score_positive': (signal_df['sentiment_score'] > 0).sum(),
            'x_sentiment_positive': (signal_df['x_sentiment'] > 0).sum(),
            'fear_greed_extreme_fear': (signal_df['fear_greed_index'] < 30).sum(),
            'fear_greed_extreme_greed': (signal_df['fear_greed_index'] > 70).sum(),
            'whale_moves_active': (signal_df['whale_moves'] > 0.2).sum()
        }
        sentiment_logger.info(
            f"Sentiment Summary: Sentiment Score Mean: {sentiment_summary['sentiment_score_mean']:.2f}, "
            f"X Sentiment Mean: {sentiment_summary['x_sentiment_mean']:.2f}, "
            f"Fear & Greed Mean: {sentiment_summary['fear_greed_mean']:.2f}, "
            f"Whale Moves Mean: {sentiment_summary['whale_moves_mean']:.2f}, "
            f"Positive Sentiment Score: {sentiment_summary['sentiment_score_positive']}, "
            f"Positive X Sentiment: {sentiment_summary['x_sentiment_positive']}, "
            f"Extreme Fear (<30): {sentiment_summary['fear_greed_extreme_fear']}, "
            f"Extreme Greed (>70): {sentiment_summary['fear_greed_extreme_greed']}, "
            f"Active Whale Moves (>0.2): {sentiment_summary['whale_moves_active']}"
        )

        signal_df['signal'] = 0
        signal_df['signal_confidence'] = 0.0
        mean_volatility = signal_df['price_volatility'].mean()  # Compute mean volatility for filtering
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
            predicted_price = signal_df['predicted_price'].loc[idx]
            raw_predicted_price = signal_df['raw_predicted_price'].loc[idx]
            volume = unscaled_volume.loc[idx]
            volume_sma_20 = signal_df['volume_sma_20'].loc[idx]
            market_regime = signal_df['market_regime'].loc[idx]
            macd_signal = signal_df['macd_signal'].loc[idx]
            price_volatility = signal_df['price_volatility'].loc[idx]
            
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200200:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88

            # Log indicator values for debugging
            signals_logger.debug(f"Indicator values at {idx}: ADX={adx:.2f}, SMA_10={sma_10:.2f}, SMA_20={sma_20:.2f}, RSI={unscaled_rsi:.2f}, MACD={unscaled_macd:.6f}, MACD_Signal={macd_signal:.6f}, Predicted={predicted_price:.2f}, Raw_Predicted={raw_predicted_price:.2f}, Market Regime={market_regime}")

            trend = 'bullish' if sma_10 > sma_20 and adx > 10 else 'bearish' if sma_10 < sma_20 and adx > 10 else 'neutral'
            confidence = 0.0
            price_change_threshold = unscaled_atr_val * 0.02
            sma_condition_buy = sma_10 > sma_20 * 1.0003
            sma_condition_sell = sma_10 < sma_20 * 0.9997
        
            # Skip signals during high volatility periods
            if price_volatility > 2 * mean_volatility:
                signals_logger.debug(f"Skipping signal at {idx} due to high volatility: {price_volatility:.4f} > {2 * mean_volatility:.4f}")
                continue

            # Buy conditions based on market regime with momentum boost and FGI confirmation
            if trend == 'bullish' and predicted_price > unscaled_close_val + 0.5 * price_change_threshold and sma_condition_buy and unscaled_rsi < rsi_threshold - 5 and unscaled_macd > -0.002 and volume > 0.9 * volume_sma_20 and adx > 8 and fgi < 80:
                confidence += 0.4
                if unscaled_rsi < rsi_threshold - 20:
                    confidence += 0.2
                if unscaled_macd > 0.0005 and unscaled_macd > macd_signal:  # Relaxed momentum boost
                    confidence += 0.3
                if bb_breakout == 1:
                    confidence += 0.1
                if sentiment > 0 and whale_moves > 0.2:
                    confidence += 0.2
                if x_sentiment > 0 and whale_moves > 0.2:
                    confidence += 0.2
                if fgi < 30:
                    confidence += 0.2
                elif fgi > 70:
                    confidence -= 0.2

                # Allow trades in Bullish or Neutral regimes
                if 'Bullish' in market_regime and confidence >= 0.05:
                    signal_df.loc[idx, 'signal'] = 1
                elif 'Neutral' in market_regime and confidence >= 0.4:  # Further lowered threshold
                    signal_df.loc[idx, 'signal'] = 1
            
            # Sell conditions with enhanced bearish triggers, FGI confirmation, and forced fallback
            sell_condition_trend = (trend == 'bearish' or unscaled_rsi > rsi_threshold or unscaled_macd < 0)
            sell_condition_price = (predicted_price < unscaled_close_val - 0.005 * price_change_threshold)
            sell_condition_sma_rsi_macd = (sma_condition_sell or unscaled_rsi > rsi_threshold or unscaled_macd < 0)
            sell_condition_volume_adx = (volume > volume_sma_20 and adx > 10)
            signals_logger.debug(
                f"Sell Conditions at {idx}: "
                f"Trend/Bearish={sell_condition_trend} (Trend={trend}, RSI={unscaled_rsi:.2f}, MACD={unscaled_macd:.6f}), "
                f"Price={sell_condition_price} (Predicted={predicted_price:.2f}, Close={unscaled_close_val:.2f}, Threshold={0.005 * price_change_threshold:.2f}), "
                f"SMA/RSI/MACD={sell_condition_sma_rsi_macd} (SMA_10={sma_10:.2f}, SMA_20={sma_20:.2f}), "
                f"Volume/ADX={sell_condition_volume_adx} (Volume={volume:.2f}, Volume_SMA_20={volume_sma_20:.2f}, ADX={adx:.2f})"
            )

            if sell_condition_trend and sell_condition_price and sell_condition_sma_rsi_macd and sell_condition_volume_adx:
                confidence += 0.4
                if unscaled_rsi > rsi_threshold + 10:
                    confidence += 0.2
                    signals_logger.debug(f"Sell Confidence Boost: RSI > {rsi_threshold + 10} (RSI={unscaled_rsi:.2f})")
                if unscaled_macd < -0.001 and unscaled_macd < macd_signal:  # Relaxed momentum boost
                    confidence += 0.3
                    signals_logger.debug(f"Sell Confidence Boost: MACD < -0.001 (MACD={unscaled_macd:.6f}, MACD_Signal={macd_signal:.6f})")
                if bb_breakout == -1:
                    confidence += 0.1
                    signals_logger.debug(f"Sell Confidence Boost: BB Breakout = -1")
                if sentiment < 0 and whale_moves > 0.2:
                    confidence += 0.2
                    signals_logger.debug(f"Sell Confidence Boost: Negative Sentiment and Whale Moves")
                if x_sentiment < 0 and whale_moves > 0.2:
                    confidence += 0.2
                    signals_logger.debug(f"Sell Confidence Boost: Negative X Sentiment and Whale Moves")
                if fgi > 25:
                    confidence += 0.2
                    signals_logger.debug(f"Sell Confidence Boost: FGI > 25 (FGI={fgi:.2f})")
                elif fgi < 25:
                    confidence -= 0.2
                    signals_logger.debug(f"Sell Confidence Penalty: FGI < 25 (FGI={fgi:.2f})")

                # Allow sells in Bearish or Neutral regimes, or force if RSI/MACD strongly bearish for 5 periods
                if ('Bearish' in market_regime or unscaled_rsi > rsi_threshold + 20 or unscaled_macd < -0.005) and confidence >= 0.05:
                    signal_df.loc[idx, 'signal'] = -1
                    signals_logger.info(f"Sell Signal Generated: Bearish Regime or Strong Bearish Indicators, Confidence={confidence:.2f}")
                elif 'Neutral' in market_regime and confidence >= 0.5:
                    signal_df.loc[idx, 'signal'] = -1
                    signals_logger.info(f"Sell Signal Generated: Neutral Regime, Confidence={confidence:.2f}")
                elif (unscaled_rsi > rsi_threshold + 20 or unscaled_macd < -0.005):
                    window_df = signal_df.loc[signal_df.index <= idx].tail(5)
                    if (window_df['rsi'].max() > rsi_threshold + 20 or window_df['macd'].min() < -0.005) and confidence >= 0.05:
                        signal_df.loc[idx, 'signal'] = -1
                        signals_logger.info(f"Sell Signal Generated: Forced Sell due to Strong Bearish RSI/MACD, Confidence={confidence:.2f}")
            
            signal_df.loc[idx, 'signal_confidence'] = min(1.0, max(0.0, confidence))
            if signal_df.loc[idx, 'signal_confidence'] < 0.05:
                logging.warning(f"Low signal_confidence at {idx}: {signal_df.loc[idx, 'signal_confidence']}")

        initial_buy_count = (signal_df['signal'] == 1).sum()
        initial_sell_count = (signal_df['signal'] == -1).sum()
        low_confidence_count = (signal_df['signal_confidence'] < 0.1).sum()
        signals_logger.info(
            f"Initial Signal Summary: Buy Signals: {initial_buy_count}, Sell Signals: {initial_sell_count}, "
            f"Confidence Mean: {signal_df['signal_confidence'].mean():.2f}, "
            f"Confidence Std: {signal_df['signal_confidence'].std():.2f}, "
            f"Low Confidence (<0.1): {low_confidence_count}"
        )

        for i in range(0, len(signal_df), summary_interval):
            end_idx = min(i + summary_interval, len(signal_df))
            window_df = signal_df.iloc[i:end_idx]
            if not window_df.empty:
                window_buy_count = (window_df['signal'] == 1).sum()
                window_sell_count = (window_df['signal'] == -1).sum()
                window_confidence_mean = window_df['signal_confidence'].mean()
                signals_logger.info(
                    f"Signal Summary [{window_df.index[0]} to {window_df.index[-1]}]: "
                    f"Buy Signals: {window_buy_count}, Sell Signals: {window_sell_count}, "
                    f"Confidence Mean: {window_confidence_mean:.2f}"
                )

        signal_df = filter_signals(signal_df)
        final_buy_count = (signal_df['signal'] == 1).sum()
        final_sell_count = (signal_df['signal'] == -1).sum()
        signals_logger.info(
            f"Final Signal Summary After Filtering: Buy Signals: {final_buy_count}, Sell Signals: {final_sell_count}, "
            f"Filtered Out: {(initial_buy_count + initial_sell_count) - (final_buy_count + final_sell_count)}"
        )

        capital = 10000

        # Precompute take_profit and stop_loss for signal rows to enable dynamic metrics calculation
        signal_df['take_profit'] = np.nan
        signal_df['stop_loss'] = np.nan
        signal_df['trade_outcome'] = np.nan  # Placeholder for backtest integration
        signal_indices = signal_df[signal_df['signal'] != 0].index  # Include both buy and sell signals
        for idx in signal_indices:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            price_volatility = signal_df['price_volatility'].loc[idx]
            
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200200:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88
            
            if signal_df.loc[idx, 'signal'] == 1:  # Buy signal
                signal_df.loc[idx, 'take_profit'] = unscaled_close_val + (unscaled_atr_val * 6.0)
                signal_df.loc[idx, 'stop_loss'] = unscaled_close_val - (unscaled_atr_val * 2.5)
            elif signal_df.loc[idx, 'signal'] == -1:  # Sell signal
                signal_df.loc[idx, 'take_profit'] = unscaled_close_val - (unscaled_atr_val * 6.0)
                signal_df.loc[idx, 'stop_loss'] = unscaled_close_val + (unscaled_atr_val * 2.5)
            signals_logger.debug(f"Precomputed trade levels at {idx}: Take_Profit={signal_df.loc[idx, 'take_profit']:.2f}, Stop_Loss={signal_df.loc[idx, 'stop_loss']:.2f}")

        # Dynamically calculate win rate and risk-reward ratio after trade levels are assigned
        historical_trades = signal_df[signal_df['signal'] != 0].tail(20)
        if len(historical_trades) > 0 and 'take_profit' in historical_trades.columns and 'stop_loss' in historical_trades.columns:
            trade_outcomes = historical_trades['trade_outcome'].dropna()
            if len(trade_outcomes) > 0:
                wins = sum(1 for outcome in trade_outcomes if outcome == 1)
                total_trades = len(trade_outcomes)
                win_rate = wins / total_trades if total_trades > 0 else 0.33
            else:
                win_rate = 0.33
            avg_win = (historical_trades['take_profit'].mean() - historical_trades['close'].mean()) if win_rate > 0 else 279.56
            avg_loss = (historical_trades['close'].mean() - historical_trades['stop_loss'].mean()) if win_rate < 1.0 else 117.15
            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0
        else:
            win_rate = 0.33
            risk_reward_ratio = 2.0
        logging.info(f"Dynamic Metrics: Win Rate={win_rate:.2f}, Risk/Reward Ratio={risk_reward_ratio:.2f}")

        # Now assign position sizes and ensure take_profit/stop_loss for all rows
        signal_df['position_size'] = 0.0
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200200:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88
            
            # Use Kelly Criterion for position sizing when a signal is present
            if signal_df.loc[idx, 'signal'] != 0:
                position_size = kelly_criterion(win_rate, risk_reward_ratio, capital, unscaled_atr_val, unscaled_close_val, max_risk_pct=max_risk_pct)
                # Enforce minimum position size
                position_size = max(position_size, 0.01)
                # Cap position size at 0.005 BTC to avoid excessive overrides
                position_size = min(position_size, 0.005)
                signals_logger.info(f"Trade Entry ({'Buy' if signal_df.loc[idx, 'signal'] == 1 else 'Sell'}) at {idx}: {position_size:.6f} BTC, Price: {unscaled_close_val:.2f} USD")
            else:
                position_size = calculate_position_size(capital, unscaled_atr_val, unscaled_close_val, max_risk_pct=max_risk_pct)
                position_size = max(position_size, 0.01)
                position_size = min(position_size, 0.005)
            
            signal_df.loc[idx, 'position_size'] = position_size

            # Assign take_profit and stop_loss for all rows (already done for signal rows)
            if pd.isna(signal_df.loc[idx, 'take_profit']):
                signal_df.loc[idx, 'take_profit'] = unscaled_close_val + (unscaled_atr_val * 6.0)
            if pd.isna(signal_df.loc[idx, 'stop_loss']):
                signal_df.loc[idx, 'stop_loss'] = unscaled_close_val - (unscaled_atr_val * 2.5)
            
            signals_logger.debug(f"Trade levels at {idx}: Take_Profit={signal_df.loc[idx, 'take_profit']:.2f}, Stop_Loss={signal_df.loc[idx, 'stop_loss']:.2f}, Position_Size={position_size:.6f}, Win_Rate={win_rate:.2f}, Risk_Reward={risk_reward_ratio:.2f}")

        signal_df['total'] = capital
        signal_df['price_volatility'] = scaled_df['price_volatility']

        actual_prices = unscaled_close.values
        predicted_prices = signal_df['predicted_price'].values
        if len(actual_prices) == len(predicted_prices):
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            logging.info(f"Mean Absolute Percentage Error (MAPE) of predictions: {mape:.2f}%")
            if mape > 5.0:
                logging.warning(f"High MAPE ({mape:.2f}%) detected. Consider model retraining or reduced reliance on predictions.")

        signal_df = signal_df[['close', 'signal', 'position_size', 'predicted_price', 'rsi', 'macd', 'macd_signal', 'atr', 'vwap', 'adx', 'bb_breakout', 'sentiment_score', 'x_sentiment', 'fear_greed_index', 'whale_moves', 'hash_rate', 'total', 'price_volatility', 'signal_confidence', 'sma_10', 'sma_20', 'take_profit', 'stop_loss', 'trade_outcome']]
        logging.info(f"Generated {len(signal_df)} signals with shape: {signal_df.shape}")
        logging.debug(f"Sample data - predicted_price: {signal_df['predicted_price'].head().to_list()}")
        logging.debug(f"Sample data - position_size: {signal_df['position_size'].head().to_list()}")

        return signal_df

    except Exception as e:
        logging.error(f"Error generating signals: {str(e)}")
        raise