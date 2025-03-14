# src/strategy/signal_generator.py
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Tuple, Dict
from src.models.transformer_model import TransformerPredictor
from src.strategy.indicators import (
    calculate_atr, compute_bollinger_bands, compute_vwap, compute_adx, get_onchain_metrics,
    calculate_macd, calculate_rsi, calculate_vpvr, luxalgo_trend_reversal,
    trendspider_pattern_recognition, metastock_trend_slope
)
from src.strategy.execution import hassonline_arbitrage
from src.constants import (
    FEATURE_COLUMNS, USE_LUXALGO_SIGNALS, USE_TRENDSPIDER_PATTERNS, USE_SMRT_SCALPING,
    USE_METASTOCK_TREND_SLOPE, USE_HASSONLINE_ARBITRAGE, WEIGHT_LUXALGO, WEIGHT_TRENDSPIDER,
    WEIGHT_SMRT_SCALPING, WEIGHT_METASTOCK, WEIGHT_MODEL_CONFIDENCE
)
from src.utils.sequence_utils import create_sequences
from src.strategy.position_sizer import calculate_position_size, kelly_criterion
from src.strategy.market_regime import detect_market_regime

# Configure main logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s', force=True)
logging.info("Using UPDATED signal_generator.py - Mar 14, 2025 - VERSION 83.6 (Added LuxAlgo, TrendSpider, SMRT, MetaStock, HassOnline, MCD Integration, Fixed reindex and Bollinger Bands)")
print("signal_generator.py loaded - Mar 14, 2025 - VERSION 83.6 (Added LuxAlgo, TrendSpider, SMRT, MetaStock, HassOnline, MCD Integration, Fixed reindex and Bollinger Bands)")

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
    """Simulate X sentiment based on price trend, RSI, volume change, and Fear & Greed Index."""
    logging.warning(f"Simulating X sentiment at {idx}. Free X API tier does not support tweet search. Treated as secondary factor.")
    window = 24  # 24-hour window
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 0.0
    price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0]
    price_sentiment = price_change * 5
    rsi = historical_data['momentum_rsi'].iloc[-1] if 'momentum_rsi' in historical_data.columns else 50
    rsi_sentiment = -0.3 if rsi > 70 else 0.3 if rsi < 30 else 0.0
    volume_change = (historical_data['volume'].iloc[-1] - historical_data['volume'].iloc[0]) / historical_data['volume'].iloc[0] if historical_data['volume'].iloc[0] != 0 else 0
    volume_sentiment = 0.2 if volume_change > 0.1 else -0.2 if volume_change < -0.1 else 0.0
    if price_change < 0:
        volume_sentiment *= -1
    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    fgi = np.clip(75 - (volatility * 1000), 0, 100)
    fgi_sentiment = -0.2 if fgi > 70 else 0.2 if fgi < 30 else 0.0
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
    market_regime = detect_market_regime(scaled_df, window=1440)  # Increased to 60 days
    regime_params = {
        'Bullish Low Volatility': {'rsi_buy_threshold': 35, 'rsi_sell_threshold': 70, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 1.0, 'max_risk_pct': 0.10},
        'Bullish High Volatility': {'rsi_buy_threshold': 40, 'rsi_sell_threshold': 75, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 1.0, 'max_risk_pct': 0.10},
        'Bearish Low Volatility': {'rsi_buy_threshold': 30, 'rsi_sell_threshold': 65, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 1.0, 'max_risk_pct': 0.10},
        'Bearish High Volatility': {'rsi_buy_threshold': 25, 'rsi_sell_threshold': 70, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 1.0, 'max_risk_pct': 0.10},
        'Neutral': {'rsi_buy_threshold': 35, 'rsi_sell_threshold': 70, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 1.0, 'max_risk_pct': 0.08},
    }
    return regime_params.get(market_regime, {'rsi_buy_threshold': 35, 'rsi_sell_threshold': 70, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 1.0, 'max_risk_pct': 0.10})

def filter_signals(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Filter signals to enforce a minimum hold period, dynamically adjusted by price volatility."""
    filtered_df = signal_df.copy()
    last_trade_time = None
    min_hold_period = 4  # 4 hours
    for idx in filtered_df.index:
        current_signal = filtered_df.loc[idx, 'signal']
        price_volatility = filtered_df.loc[idx, 'price_volatility'] if 'price_volatility' in filtered_df.columns else 0.0
        dynamic_min_hold = min_hold_period if price_volatility > filtered_df['price_volatility'].mean() else 2
        min_confidence = 0.15  # Reduced from 0.2
        # Relaxed criteria: Allow 0 confirming indicators if confidence is high
        rsi = filtered_df.loc[idx, 'rsi']
        macd = filtered_df.loc[idx, 'macd']
        macd_signal = filtered_df.loc[idx, 'macd_signal']
        confirming_indicators = 0
        if current_signal == 1:
            if rsi < filtered_df.loc[idx, 'rsi_buy_threshold'] and macd > macd_signal:
                confirming_indicators = 2
            elif (rsi < filtered_df.loc[idx, 'rsi_buy_threshold'] or macd > macd_signal) and filtered_df.loc[idx, 'signal_confidence'] >= 0.4:  # Reduced from 0.5
                confirming_indicators = 1
        elif current_signal == -1:
            if rsi > filtered_df.loc[idx, 'rsi_sell_threshold'] and macd < macd_signal:
                confirming_indicators = 2
            elif (rsi > filtered_df.loc[idx, 'rsi_sell_threshold'] or macd < macd_signal) and filtered_df.loc[idx, 'signal_confidence'] >= 0.4:  # Reduced from 0.5
                confirming_indicators = 1

        if (last_trade_time is None or 
            (idx - last_trade_time).total_seconds() / 3600 >= dynamic_min_hold) and \
           current_signal != 0 and \
           filtered_df.loc[idx, 'signal_confidence'] >= min_confidence and \
           (confirming_indicators >= 1 or filtered_df.loc[idx, 'signal_confidence'] >= 0.4):  # Allow 0 indicators if confidence is high
            last_trade_time = idx
            signals_logger.info(f"Accepted signal at {idx} with confidence {filtered_df.loc[idx, 'signal_confidence']:.2f}, Indicators confirmed: {confirming_indicators}")
        else:
            filtered_df.loc[idx, 'signal'] = 0
            if current_signal != 0:
                signals_logger.debug(f"Signal filtered at {idx}: Min hold ({dynamic_min_hold} hours), Confidence ({filtered_df.loc[idx, 'signal_confidence']:.2f} < {min_confidence}), Indicators ({confirming_indicators} < 1 unless confidence >= 0.4)")
    return filtered_df

def smrt_scalping_signals(df: pd.DataFrame, atr_multiplier: float = 1.0, fee_rate: float = 0.001) -> pd.Series:
    """Generate scalping signals inspired by SMRT Algo, accounting for HFT fees."""
    if not USE_SMRT_SCALPING:
        logging.info("SMRT Algo scalping signals disabled")
        return pd.Series(0, index=df.index)

    try:
        atr = calculate_atr(df['high'], df['low'], df['close'], period=14)
        signals = pd.Series(0, index=df.index)
        
        # Scalping logic: Tight entries/exits based on ATR
        price_change = df['close'].pct_change()
        threshold = atr * atr_multiplier / df['close']
        
        # Buy: Price increases by ATR threshold, sell after small profit
        signals[(price_change > threshold) & (df['close'] > df['close'].shift(1))] = 1
        signals[(price_change < -threshold) & (df['close'] < df['close'].shift(1))] = -1
        
        # Adjust for fees: Only take trades where expected profit exceeds fees
        expected_profit = atr * atr_multiplier
        min_profit = df['close'] * fee_rate * 2  # Fees for entry and exit
        signals[expected_profit < min_profit] = 0
        
        logging.info(f"Generated SMRT Algo scalping signals: {signals.value_counts().to_dict()}")
        return signals
    except Exception as e:
        logging.error(f"SMRT Algo scalping signals computation failed: {e}")
        return pd.Series(0, index=df.index)

def optimize_parameters(df: pd.DataFrame, base_params: Dict[str, float]) -> Dict[str, float]:
    """Dynamically optimize signal parameters based on market conditions (LuxAlgo-inspired)."""
    if not USE_LUXALGO_SIGNALS:
        logging.info("LuxAlgo parameter optimization disabled")
        return base_params

    try:
        price_volatility = df['close'].pct_change().rolling(window=24, min_periods=1).std().fillna(0.0)
        adx = compute_adx(df, period=14)
        
        optimized_params = base_params.copy()
        if price_volatility.iloc[-1] > 0.02:  # High volatility
            optimized_params['rsi_buy_threshold'] = max(20, base_params['rsi_buy_threshold'] - 5)
            optimized_params['rsi_sell_threshold'] = min(80, base_params['rsi_sell_threshold'] + 5)
            optimized_params['atr_multiplier'] = base_params['atr_multiplier'] * 1.2
        elif adx.iloc[-1] > 25:  # Strong trend
            optimized_params['rsi_buy_threshold'] = min(40, base_params['rsi_buy_threshold'] + 5)
            optimized_params['rsi_sell_threshold'] = max(60, base_params['rsi_sell_threshold'] - 5)
            optimized_params['atr_multiplier'] = base_params['atr_multiplier'] * 0.8
        
        logging.info(f"Optimized parameters: {optimized_params}")
        return optimized_params
    except Exception as e:
        logging.error(f"LuxAlgo parameter optimization failed: {e}")
        return base_params

async def generate_signals(scaled_df: pd.DataFrame, preprocessed_data: pd.DataFrame, model: TransformerPredictor, 
                          train_columns: List[str], feature_scaler, target_scaler, rsi_threshold: float = 35, 
                          macd_fast: int = 12, macd_slow: int = 26, atr_multiplier: float = 1.0, 
                          max_risk_pct: float = 0.10) -> pd.DataFrame:
    """Enhanced signal generator using model predictions with Monte Carlo Dropout, technical indicators, and historical sentiment data."""
    print("Entering generate_signals function")
    logging.info(f"Entering generate_signals with scaled_df shape: {scaled_df.shape}, preprocessed_data shape: {preprocessed_data.shape}")

    # Use FEATURE_COLUMNS from constants.py as the required columns
    required_columns = FEATURE_COLUMNS
    print(f"Required columns: {required_columns}")
    logging.debug(f"Checking required columns: {required_columns}")

    # Check for missing columns and dynamically calculate VPVR features if needed
    for col in required_columns:
        if col not in scaled_df.columns:
            raise ValueError(f"Missing required column in scaled_df: {col}")
        if col not in preprocessed_data.columns:
            if col.startswith('dist_to_'):
                vpvr_lookback = 500
                window_data = preprocessed_data.tail(vpvr_lookback)
                vpvr = calculate_vpvr(window_data, lookback=vpvr_lookback, num_bins=50)
                current_price = preprocessed_data['close'].iloc[-1]
                scaled_df[col] = (current_price - vpvr[col.split('_')[-1]]) / vpvr[col.split('_')[-1]]
                preprocessed_data[col] = scaled_df[col]
                logging.info(f"Added missing VPVR column {col} dynamically")
            elif col == 'luxalgo_signal':
                scaled_df[col] = luxalgo_trend_reversal(preprocessed_data)
                preprocessed_data[col] = scaled_df[col]
                logging.info(f"Added missing LuxAlgo signal column {col} dynamically")
            elif col == 'trendspider_signal':
                scaled_df[col] = trendspider_pattern_recognition(preprocessed_data)
                preprocessed_data[col] = scaled_df[col]
                logging.info(f"Added missing TrendSpider signal column {col} dynamically")
            elif col == 'metastock_slope':
                scaled_df[col] = metastock_trend_slope(preprocessed_data)
                preprocessed_data[col] = scaled_df[col]
                logging.info(f"Added missing MetaStock slope column {col} dynamically")
            else:
                raise ValueError(f"Missing required column in preprocessed_data: {col}")

    print(f"Scaled_df columns: {scaled_df.columns.tolist()}")
    print(f"Preprocessed_data columns: {preprocessed_data.columns.tolist()}")
    logging.debug(f"All required columns present in scaled_df and preprocessed_data: {scaled_df.columns.tolist()} and {preprocessed_data.columns.tolist()}")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model eval mode on device: {device}")
    logging.debug(f"Model set to eval mode on device: {device}")

    # Confirm GPU usage
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        # Verify model is on GPU
        if next(model.parameters()).is_cuda:
            print("Model is on GPU")
            logging.info("Model is on GPU")
        else:
            print("Model is NOT on GPU - moving to GPU")
            logging.warning("Model is NOT on GPU - moving to GPU")
            model.to(device)
    else:
        print("CUDA is NOT available. Using CPU")
        logging.warning("CUDA is NOT available. Using CPU")

    try:
        print("Inside try block")
        logging.debug(f"Step 1: Entering generate_signals - file: {__file__}")
        logging.debug(f"Step 2: Globals: {list(globals().keys())}")
        logging.debug(f"Step 2.5: Model device: {next(model.parameters()).device}")

        print("Calling adapt_strategy_parameters")
        params = adapt_strategy_parameters(scaled_df)
        params = optimize_parameters(preprocessed_data, params)  # LuxAlgo-inspired optimization
        rsi_buy_threshold = params['rsi_buy_threshold']
        rsi_sell_threshold = params['rsi_sell_threshold']
        macd_fast = params['macd_fast']
        macd_slow = params['macd_slow']
        atr_multiplier = params['atr_multiplier']
        max_risk_pct = params['max_risk_pct']
        print(f"Adapted and optimized params: {params}")
        logging.debug(f"Step 3: Adapted and optimized params: {params}")

        print("Extracting features for model prediction")
        features = scaled_df[required_columns].values
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
            features, targets.flatten(), seq_length=24
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

        print("Running model.predict with Monte Carlo Dropout")
        batch_size = 2000  # Optimized for RTX 4070 SUPER
        predictions = []
        confidences = []
        with torch.no_grad():
            for i in range(0, X_tensor.size(0), batch_size):
                batch_end = min(i + batch_size, X_tensor.size(0))
                batch_X = X_tensor[i:batch_end].cpu().numpy()
                batch_past_time = past_time_features_tensor[i:batch_end].cpu().numpy() if past_time_features_tensor is not None else None
                batch_mask = past_observed_mask_tensor[i:batch_end].cpu().numpy() if past_observed_mask_tensor is not None else None
                batch_future_vals = future_values_tensor[i:batch_end].cpu().numpy() if future_values_tensor is not None else None
                batch_future_time = future_time_features_tensor[i:batch_end].cpu().numpy() if future_time_features_tensor is not None else None

                batch_pred, batch_conf = model.predict(
                    batch_X, batch_past_time, batch_mask, batch_future_vals, batch_future_time
                )
                predictions.append(batch_pred)
                confidences.append(batch_conf)
                logging.info(f"Processed batch {i} to {batch_end} of {X_tensor.size(0)} sequences")
        predictions = np.concatenate(predictions)
        confidences = np.concatenate(confidences)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Confidences shape: {confidences.shape}")
        print(f"Predictions sample: {predictions[:5]}")
        print(f"Confidences sample: {confidences[:5]}")
        logging.debug(f"Step 14: Raw predictions shape: {predictions.shape}")
        logging.debug(f"Step 14.5: Raw predictions sample: {predictions[:5]}")
        logging.debug(f"Step 14.6: Raw confidences shape: {confidences.shape}")
        logging.debug(f"Step 14.7: Raw confidences sample: {confidences[:5]}")

        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.shape[1] != 1:
            predictions = predictions[:, 0:1]
        if len(confidences.shape) == 1:
            confidences = confidences.reshape(-1, 1)
        elif confidences.shape[1] != 1:
            confidences = confidences[:, 0:1]
        logging.debug(f"Step 15: Predictions reshaped: {predictions.shape}")
        logging.debug(f"Step 15.5: Confidences reshaped: {confidences.shape}")

        expected_length = len(scaled_df) - 24  # Adjust for sequence length
        pred_length = predictions.shape[0]
        if pred_length < expected_length:
            logging.warning(f"Prediction length ({pred_length}) < expected length ({expected_length}). Padding...")
            padding_length = expected_length - pred_length
            predictions = np.pad(predictions, ((0, padding_length), (0, 0)), mode='edge')
            confidences = np.pad(confidences, ((0, padding_length), (0, 0)), mode='edge')
        elif pred_length > expected_length:
            predictions = predictions[:expected_length]
            confidences = confidences[:expected_length]
        logging.debug(f"Step 16: Predictions adjusted: {predictions.shape}")
        logging.debug(f"Step 16.5: Confidences adjusted: {confidences.shape}")

        print("Unscaling predictions and debugging scaling")
        predictions_unscaled = target_scaler.inverse_transform(predictions).flatten()
        print(f"Predictions unscaled shape: {predictions_unscaled.shape}")
        print(f"Predictions unscaled sample: {predictions_unscaled[:5]}")
        if any(p <= 0 or p < 10000 or p > 200200 for p in predictions_unscaled):
            logging.warning(f"Potential scaling error in predictions_unscaled: {predictions_unscaled[:5]}")
        signal_df = scaled_df.copy()
        # Adjust index to account for sequence length
        signal_df = signal_df.iloc[24:].copy()
        signal_df['predicted_price'] = pd.Series(predictions_unscaled, index=signal_df.index)
        signal_df['raw_predicted_price'] = signal_df['predicted_price'].copy()
        signal_df['model_confidence'] = pd.Series(confidences.flatten(), index=signal_df.index)

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

        signal_df['rsi'] = calculate_rsi(unscaled_close).reindex(signal_df.index, method='ffill')
        if signal_df['rsi'].isna().all():
            signal_df['rsi'] = 50.0
            logging.warning("RSI column is all NaN, using fallback value 50.0")
        macd, macd_signal = calculate_macd(unscaled_close, fast=macd_fast, slow=macd_slow)
        if macd.isna().all():
            macd = pd.Series(np.zeros(len(signal_df)), index=signal_df.index)
            macd_signal = pd.Series(np.zeros(len(signal_df)), index=signal_df.index)
            logging.warning("MACD column is all NaN, using fallback value 0.0")
        signal_df['macd'] = macd.reindex(signal_df.index, method='ffill')
        signal_df['macd_signal'] = macd_signal.reindex(signal_df.index, method='ffill')
        signal_df['atr'] = calculate_atr(unscaled_high, unscaled_low, unscaled_close).fillna(500.0).reindex(signal_df.index, method='ffill')
        signal_df['vwap'] = compute_vwap(preprocessed_data).reindex(signal_df.index, method='ffill')
        signal_df['adx'] = compute_adx(preprocessed_data).fillna(10.0).reindex(signal_df.index, method='ffill')
        if signal_df['adx'].isna().all():
            signal_df['adx'] = 10.0
            logging.warning("ADX column is all NaN, using fallback value 10.0")
        signal_df['sma_10'] = unscaled_close.rolling(window=10, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['sma_20'] = unscaled_close.rolling(window=20, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['sma_50'] = unscaled_close.rolling(window=50, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['sma_200'] = unscaled_close.rolling(window=200, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['volume_sma_20'] = unscaled_volume.rolling(window=20, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')

        # Bollinger Bands with error handling
        bollinger_bands = compute_bollinger_bands(preprocessed_data)
        if isinstance(bollinger_bands, pd.DataFrame) and 'bb_breakout' in bollinger_bands.columns:
            signal_df['bb_breakout'] = bollinger_bands['bb_breakout'].reindex(signal_df.index, method='ffill')
        else:
            logging.warning("Bollinger Bands computation failed or missing 'bb_breakout'. Setting to 0.")
            signal_df['bb_breakout'] = 0

        signal_df['sentiment_score'] = [calculate_historical_sentiment(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['x_sentiment'] = [await fetch_x_sentiment(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['fear_greed_index'] = [await get_fear_and_greed_index(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['whale_moves'] = [simulate_historical_whale_moves(preprocessed_data, idx) for idx in signal_df.index]
        
        # Fix for hash_rate: Create a Series from the float value
        onchain_metrics = get_onchain_metrics(symbol="BTC")
        hash_rate_value = onchain_metrics['hash_rate']
        signal_df['hash_rate'] = pd.Series(hash_rate_value, index=signal_df.index)

        signal_df['market_regime'] = [detect_market_regime(preprocessed_data.loc[:idx], window=1440) for idx in signal_df.index]  # Increased to 60 days
        signal_df['rsi_buy_threshold'] = rsi_buy_threshold
        signal_df['rsi_sell_threshold'] = rsi_sell_threshold

        # Add new signals
        signal_df['luxalgo_signal'] = luxalgo_trend_reversal(preprocessed_data).reindex(signal_df.index, method='ffill')
        signal_df['trendspider_signal'] = trendspider_pattern_recognition(preprocessed_data).reindex(signal_df.index, method='ffill')
        signal_df['smrt_signal'] = smrt_scalping_signals(preprocessed_data, atr_multiplier=atr_multiplier).reindex(signal_df.index, method='ffill')
        signal_df['arbitrage_signal'] = hassonline_arbitrage(preprocessed_data, exchange_prices={}).reindex(signal_df.index, method='ffill')
        signal_df['metastock_slope'] = metastock_trend_slope(preprocessed_data).reindex(signal_df.index, method='ffill')

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
                luxalgo_signals = window_df['luxalgo_signal'].value_counts().to_dict()
                trendspider_signals = window_df['trendspider_signal'].value_counts().to_dict()
                smrt_signals = window_df['smrt_signal'].value_counts().to_dict()
                metastock_slope_mean = window_df['metastock_slope'].mean()
                indicators_logger.info(
                    f"Indicator Summary [{window_df.index[0]} to {window_df.index[-1]}]: "
                    f"RSI Mean: {rsi_mean:.2f}, Overbought (>70): {rsi_overbought}, Oversold (<30): {rsi_oversold}, "
                    f"MACD Mean: {macd_mean:.2f}, ADX Mean: {adx_mean:.2f}, Strong Trend (>10): {adx_strong}, "
                    f"BB Breakouts: {bb_breakouts}, SMA Bullish (10>20): {sma_trend_bullish}/{len(window_df)}, "
                    f"LuxAlgo Signals: {luxalgo_signals}, TrendSpider Signals: {trendspider_signals}, "
                    f"SMRT Signals: {smrt_signals}, MetaStock Slope Mean: {metastock_slope_mean:.2f}"
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
        # Calculate rolling volatility threshold (90 days)
        rolling_volatility = signal_df['price_volatility'].rolling(window=2160, min_periods=1).mean()
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            unscaled_rsi = signal_df['rsi'].loc[idx]
            unscaled_macd = signal_df['macd'].loc[idx]
            sma_10 = signal_df['sma_10'].loc[idx]
            sma_20 = signal_df['sma_20'].loc[idx]
            sma_50 = signal_df['sma_50'].loc[idx]
            sma_200 = signal_df['sma_200'].loc[idx]
            adx = signal_df['adx'].loc[idx]
            bb_breakout = signal_df['bb_breakout'].loc[idx]
            sentiment = signal_df['sentiment_score'].loc[idx]
            whale_moves = signal_df['whale_moves'].loc[idx]
            x_sentiment = signal_df['x_sentiment'].loc[idx]
            fgi = signal_df['fear_greed_index'].loc[idx]
            predicted_price = signal_df['predicted_price'].loc[idx]
            raw_predicted_price = signal_df['raw_predicted_price'].loc[idx]
            model_confidence = signal_df['model_confidence'].loc[idx]
            volume = unscaled_volume.loc[idx]
            volume_sma_20 = signal_df['volume_sma_20'].loc[idx]
            market_regime = signal_df['market_regime'].loc[idx]
            macd_signal = signal_df['macd_signal'].loc[idx]
            price_volatility = signal_df['price_volatility'].loc[idx]
            rolling_volatility_mean = rolling_volatility.loc[idx]
            rsi_buy_threshold = signal_df['rsi_buy_threshold'].loc[idx]
            rsi_sell_threshold = signal_df['rsi_sell_threshold'].loc[idx]
            luxalgo_signal = signal_df['luxalgo_signal'].loc[idx]
            trendspider_signal = signal_df['trendspider_signal'].loc[idx]
            smrt_signal = signal_df['smrt_signal'].loc[idx]
            arbitrage_signal = signal_df['arbitrage_signal'].loc[idx]
            metastock_slope = signal_df['metastock_slope'].loc[idx]

            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200200:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88

            trend = 'bullish' if sma_50 > sma_200 else 'bearish' if sma_50 < sma_200 else 'neutral'
            confidence = 0.0
            price_change_threshold = 0.0005 * unscaled_close_val  # Reduced to 0.05%

            # Modified volatility adjustment: Reduce position size
            volatility_adjustment = 1.0
            if price_volatility > 2 * rolling_volatility_mean:
                volatility_adjustment = 0.5
                signals_logger.debug(f"High volatility at {idx}: {price_volatility:.4f} > {2 * rolling_volatility_mean:.4f}, Regime: {market_regime}, Adjusting position size by 0.5x")

            # Log predicted price change for debugging
            predicted_change = predicted_price - unscaled_close_val
            signals_logger.debug(f"Predicted price change at {idx}: {predicted_change:.2f} (Predicted: {predicted_price:.2f}, Close: {unscaled_close_val:.2f}, Threshold: {price_change_threshold:.2f})")

            # Combine signals with weighted confidence, incorporating MCD confidence
            combined_confidence = (
                WEIGHT_MODEL_CONFIDENCE * model_confidence +
                WEIGHT_LUXALGO * luxalgo_signal +
                WEIGHT_TRENDSPIDER * trendspider_signal +
                WEIGHT_SMRT_SCALPING * smrt_signal +
                WEIGHT_METASTOCK * (metastock_slope / max(abs(signal_df['metastock_slope'].max()), 1e-10))
            )

            # Relaxed Buy conditions
            buy_condition_trend = market_regime in ['Bullish Low Volatility', 'Bullish High Volatility', 'Neutral'] or (sma_50 > sma_200 and signal_df['sma_50'].shift(1).loc[idx] <= signal_df['sma_200'].shift(1).loc[idx])
            buy_condition_price = predicted_price > unscaled_close_val + price_change_threshold
            buy_condition_rsi_macd = unscaled_rsi < (rsi_buy_threshold + 10)
            buy_condition_volume = volume > 0.3 * volume_sma_20  # Reduced from 0.5
            buy_conditions_met = sum([
                buy_condition_trend,
                buy_condition_price,
                buy_condition_rsi_macd,
                buy_condition_volume
            ])
            signals_logger.debug(
                f"Buy Conditions at {idx}: Conditions Met={buy_conditions_met}/4, "
                f"Year={idx.year}, "
                f"Trend/Bullish={buy_condition_trend} (Trend={trend}, SMA_50={sma_50:.2f}, SMA_200={sma_200:.2f}), "
                f"Price={buy_condition_price} (Predicted={predicted_price:.2f}, Close={unscaled_close_val:.2f}, Threshold={price_change_threshold:.2f}), "
                f"RSI/MACD={buy_condition_rsi_macd} (RSI={unscaled_rsi:.2f}), "
                f"Volume={buy_condition_volume} (Volume={volume:.2f}, Volume_SMA_20={volume_sma_20:.2f})"
            )

            if buy_conditions_met >= 3 and fgi < 80:
                confidence += 0.4
                if unscaled_rsi < rsi_buy_threshold - 15:
                    confidence += 0.15
                    signals_logger.debug(f"Buy Confidence Boost: RSI < {rsi_buy_threshold - 15} (RSI={unscaled_rsi:.2f})")
                if unscaled_macd > 0:
                    confidence += 0.2
                    signals_logger.debug(f"Buy Confidence Boost: MACD > 0 (MACD={unscaled_macd:.6f}, MACD_Signal={macd_signal:.6f})")
                if bb_breakout == 1:
                    confidence += 0.1
                    signals_logger.debug(f"Buy Confidence Boost: BB Breakout = 1")
                if sentiment > 0 and whale_moves > 0.2:
                    confidence += 0.15
                    signals_logger.debug(f"Buy Confidence Boost: Positive Sentiment and Whale Moves")
                if x_sentiment > 0 and whale_moves > 0.2:
                    confidence += 0.15
                    signals_logger.debug(f"Buy Confidence Boost: Positive X Sentiment and Whale Moves")
                if fgi < 30:
                    confidence += 0.15
                    signals_logger.debug(f"Buy Confidence Boost: FGI < 30 (FGI={fgi:.2f})")
                elif fgi > 70:
                    confidence -= 0.15
                    signals_logger.debug(f"Buy Confidence Penalty: FGI > 70 (FGI={fgi:.2f})")
                # New feature boosts
                if luxalgo_signal == 1:
                    confidence += 0.1
                    signals_logger.debug(f"Buy Confidence Boost: LuxAlgo Signal = 1")
                if trendspider_signal == 1:
                    confidence += 0.05
                    signals_logger.debug(f"Buy Confidence Boost: TrendSpider Signal = 1")
                if smrt_signal == 1:
                    confidence += 0.05
                    signals_logger.debug(f"Buy Confidence Boost: SMRT Signal = 1")
                if metastock_slope > 0:
                    confidence += 0.05
                    signals_logger.debug(f"Buy Confidence Boost: MetaStock Slope Positive")

                signal_df.loc[idx, 'signal'] = 1
                signals_logger.info(f"Buy Signal Generated: Confidence={confidence:.2f}")

            # Relaxed Sell conditions
            sell_condition_trend = market_regime in ['Bearish Low Volatility', 'Bearish High Volatility', 'Neutral'] or (sma_50 < sma_200 and signal_df['sma_50'].shift(1).loc[idx] >= signal_df['sma_200'].shift(1).loc[idx])
            sell_condition_price = predicted_price < unscaled_close_val - price_change_threshold
            sell_condition_rsi_macd = unscaled_rsi > (rsi_sell_threshold - 5)
            sell_condition_volume = volume > 0.3 * volume_sma_20  # Reduced from 0.5
            sell_conditions_met = sum([
                sell_condition_trend,
                sell_condition_price,
                sell_condition_rsi_macd,
                sell_condition_volume
            ])
            signals_logger.debug(
                f"Sell Conditions at {idx}: Conditions Met={sell_conditions_met}/4, "
                f"Year={idx.year}, "
                f"Trend/Bearish={sell_condition_trend} (Trend={trend}, RSI={unscaled_rsi:.2f}, MACD={unscaled_macd:.6f}), "
                f"Price={sell_condition_price} (Predicted={predicted_price:.2f}, Close={unscaled_close_val:.2f}, Threshold={price_change_threshold:.2f}), "
                f"RSI/MACD={sell_condition_rsi_macd} (RSI={unscaled_rsi:.2f}, MACD={unscaled_macd:.6f}, MACD_Signal={macd_signal:.6f}), "
                f"Volume={sell_condition_volume} (Volume={volume:.2f}, Volume_SMA_20={volume_sma_20:.2f})"
            )

            if sell_conditions_met >= 3:
                confidence += 0.4
                if unscaled_rsi > rsi_sell_threshold + 10:
                    confidence += 0.2
                    signals_logger.debug(f"Sell Confidence Boost: RSI > {rsi_sell_threshold + 10} (RSI={unscaled_rsi:.2f})")
                if unscaled_macd < 0 and unscaled_macd < macd_signal:
                    confidence += 0.3
                    signals_logger.debug(f"Sell Confidence Boost: MACD < 0 (MACD={unscaled_macd:.6f}, MACD_Signal={macd_signal:.6f})")
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
                # New feature boosts
                if luxalgo_signal == -1:
                    confidence += 0.1
                    signals_logger.debug(f"Sell Confidence Boost: LuxAlgo Signal = -1")
                if trendspider_signal == -1:
                    confidence += 0.05
                    signals_logger.debug(f"Sell Confidence Boost: TrendSpider Signal = -1")
                if smrt_signal == -1:
                    confidence += 0.05
                    signals_logger.debug(f"Sell Confidence Boost: SMRT Signal = -1")
                if metastock_slope < 0:
                    confidence += 0.05
                    signals_logger.debug(f"Sell Confidence Boost: MetaStock Slope Negative")

                signal_df.loc[idx, 'signal'] = -1
                signals_logger.info(f"Sell Signal Generated: Confidence={confidence:.2f}")

            # Override with arbitrage signal if present
            if USE_HASSONLINE_ARBITRAGE and arbitrage_signal != 0:
                signal_df.loc[idx, 'signal'] = arbitrage_signal
                confidence += 0.3  # Arbitrage signals are typically high-confidence
                signals_logger.info(f"Arbitrage Signal Override at {idx}: Signal={arbitrage_signal}, Confidence={confidence:.2f}")

            signal_df.loc[idx, 'signal_confidence'] = min(1.0, max(0.0, confidence + combined_confidence))
            if signal_df.loc[idx, 'signal_confidence'] < 0.15:
                logging.warning(f"Low signal_confidence at {idx}: {signal_df.loc[idx, 'signal_confidence']}")

        initial_buy_count = (signal_df['signal'] == 1).sum()
        initial_sell_count = (signal_df['signal'] == -1).sum()
        low_confidence_count = (signal_df['signal_confidence'] < 0.15).sum()
        signals_logger.info(
            f"Initial Signal Summary: Buy Signals: {initial_buy_count}, Sell Signals: {initial_sell_count}, "
            f"Confidence Mean: {signal_df['signal_confidence'].mean():.2f}, "
            f"Confidence Std: {signal_df['signal_confidence'].std():.2f}, "
            f"Low Confidence (<0.15): {low_confidence_count}"
        )

        # Log signals per year to debug distribution
        signal_df['year'] = signal_df.index.year
        signal_counts = signal_df.groupby('year')['signal'].apply(lambda x: (x != 0).sum())
        signals_logger.info(f"Signals generated per year (before filtering):\n{signal_counts}")
        signal_df.drop(columns=['year'], inplace=True)

        signal_df = filter_signals(signal_df)
        final_buy_count = (signal_df['signal'] == 1).sum()
        final_sell_count = (signal_df['signal'] == -1).sum()
        signals_logger.info(
            f"Final Signal Summary After Filtering: Buy Signals: {final_buy_count}, Sell Signals: {final_sell_count}, "
            f"Filtered Out: {(initial_buy_count + initial_sell_count) - (final_buy_count + final_sell_count)}"
        )

        # Log filtered signals per year
        signal_df['year'] = signal_df.index.year
        filtered_signal_counts = signal_df.groupby('year')['signal'].apply(lambda x: (x != 0).sum())
        signals_logger.info(f"Signals after filtering per year:\n{filtered_signal_counts}")
        signal_df.drop(columns=['year'], inplace=True)

        capital = 17396.68

        signal_df['take_profit'] = np.nan
        signal_df['stop_loss'] = np.nan
        signal_df['trailing_stop'] = np.nan
        signal_df['trade_outcome'] = np.nan
        signal_indices = signal_df[signal_df['signal'] != 0].index
        for idx in signal_indices:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200200:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88
            
            if signal_df.loc[idx, 'signal'] == 1:
                signal_df.loc[idx, 'stop_loss'] = unscaled_close_val - (unscaled_atr_val * 1.0)
                signal_df.loc[idx, 'take_profit'] = unscaled_close_val + (unscaled_atr_val * 2.0)
                signal_df.loc[idx, 'trailing_stop'] = unscaled_close_val
            elif signal_df.loc[idx, 'signal'] == -1:
                signal_df.loc[idx, 'stop_loss'] = unscaled_close_val + (unscaled_atr_val * 1.0)
                signal_df.loc[idx, 'take_profit'] = unscaled_close_val - (unscaled_atr_val * 2.0)
                signal_df.loc[idx, 'trailing_stop'] = unscaled_close_val
            signals_logger.debug(f"Precomputed trade levels at {idx}: Take_Profit={signal_df.loc[idx, 'take_profit']:.2f}, Stop_Loss={signal_df.loc[idx, 'stop_loss']:.2f}, Trailing_Stop={signal_df.loc[idx, 'trailing_stop']:.2f}")

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

        signal_df['position_size'] = 0.0
        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            price_volatility = signal_df['price_volatility'].loc[idx]
            
            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200200:
                logging.warning(f"Invalid unscaled close at {idx}: {unscaled_close_val}, using 78877.88 USD")
                unscaled_close_val = 78877.88
            
            # Apply volatility adjustment to position size
            volatility_adjustment = 1.0
            if price_volatility > 2 * rolling_volatility.loc[idx]:
                volatility_adjustment = 0.5

            if signal_df.loc[idx, 'signal'] != 0:
                position_size = kelly_criterion(win_rate, risk_reward_ratio, capital, unscaled_atr_val, unscaled_close_val, max_risk_pct=max_risk_pct)
                position_size = max(position_size, 0.01)
                position_size = min(position_size, 0.005) * volatility_adjustment
                signals_logger.info(f"Trade Entry ({'Buy' if signal_df.loc[idx, 'signal'] == 1 else 'Sell'}) at {idx}: {position_size:.6f} BTC, Price: {unscaled_close_val:.2f} USD")
            else:
                position_size = calculate_position_size(capital, unscaled_atr_val, unscaled_close_val, max_risk_pct=max_risk_pct)
                position_size = max(position_size, 0.01)
                position_size = min(position_size, 0.005) * volatility_adjustment
            
            signal_df.loc[idx, 'position_size'] = position_size

            if pd.isna(signal_df.loc[idx, 'take_profit']):
                signal_df.loc[idx, 'take_profit'] = unscaled_close_val + (unscaled_atr_val * atr_multiplier * 2.0)
            if pd.isna(signal_df.loc[idx, 'stop_loss']):
                signal_df.loc[idx, 'stop_loss'] = unscaled_close_val - (unscaled_atr_val * atr_multiplier)
            if pd.isna(signal_df.loc[idx, 'trailing_stop']):
                signal_df.loc[idx, 'trailing_stop'] = unscaled_close_val
            
            signals_logger.debug(f"Trade levels at {idx}: Take_Profit={signal_df.loc[idx, 'take_profit']:.2f}, Stop_Loss={signal_df.loc[idx, 'stop_loss']:.2f}, Trailing_Stop={signal_df.loc[idx, 'trailing_stop']:.2f}, Position_Size={position_size:.6f}, Win_Rate={win_rate:.2f}, Risk_Reward={risk_reward_ratio:.2f}")

        signal_df['total'] = capital
        signal_df['price_volatility'] = scaled_df['price_volatility'].reindex(signal_df.index, method='ffill')

        actual_prices = unscaled_close.reindex(signal_df.index).values
        predicted_prices = signal_df['predicted_price'].values
        if len(actual_prices) == len(predicted_prices):
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            logging.info(f"Mean Absolute Percentage Error (MAPE) of predictions: {mape:.2f}%")
            if mape > 5.0:
                logging.warning(f"High MAPE ({mape:.2f}%) detected. Consider model retraining or reduced reliance on predictions.")

        signal_df = signal_df[[
            'close', 'signal', 'position_size', 'predicted_price', 'rsi', 'macd', 'macd_signal', 'atr', 'vwap', 'adx',
            'bb_breakout', 'sentiment_score', 'x_sentiment', 'fear_greed_index', 'whale_moves', 'hash_rate', 'total',
            'price_volatility', 'signal_confidence', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 'take_profit', 'stop_loss',
            'trailing_stop', 'trade_outcome', 'luxalgo_signal', 'trendspider_signal', 'smrt_signal', 'arbitrage_signal',
            'metastock_slope', 'model_confidence'
        ]]
        logging.info(f"Generated {len(signal_df)} signals with shape: {signal_df.shape}")
        logging.debug(f"Sample data - predicted_price: {signal_df['predicted_price'].head().to_list()}")
        logging.debug(f"Sample data - position_size: {signal_df['position_size'].head().to_list()}")
        logging.debug(f"Sample data - model_confidence: {signal_df['model_confidence'].head().to_list()}")

        return signal_df

    except Exception as e:
        logging.error(f"Error generating signals: {str(e)}")
        raise