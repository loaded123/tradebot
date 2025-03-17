# src/strategy/signal_generator.py
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.models.transformer_model import TransformerPredictor
from src.strategy.indicators import (
    calculate_atr, compute_bollinger_bands, compute_vwap, compute_adx,
    calculate_macd, calculate_rsi, calculate_vpvr, luxalgo_trend_reversal,
    trendspider_pattern_recognition, metastock_trend_slope, get_onchain_metrics
)
from src.strategy.sentiment_analysis import (
    calculate_historical_sentiment, fetch_x_sentiment,
    get_fear_and_greed_index, simulate_historical_whale_moves
)
from src.strategy.parameter_optimizer import adapt_strategy_parameters, optimize_parameters
from src.strategy.signal_filter import filter_signals, smrt_scalping_signals
from src.strategy.position_sizer import calculate_position_size, kelly_criterion
from src.strategy.market_regime import detect_market_regime
from src.utils.sequence_utils import create_sequences
from src.utils.time_utils import calculate_days_to_next_halving
from src.constants import (
    FEATURE_COLUMNS, USE_LUXALGO_SIGNALS, USE_TRENDSPIDER_PATTERNS, USE_SMRT_SCALPING,
    USE_METASTOCK_TREND_SLOPE, USE_HASSONLINE_ARBITRAGE, WEIGHT_LUXALGO, WEIGHT_TRENDSPIDER,
    WEIGHT_SMRT_SCALPING, WEIGHT_METASTOCK, WEIGHT_MODEL_CONFIDENCE
)

# Configure main logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
main_logger = logging.getLogger('main')
main_logger.info("Using MODULARIZED signal_generator.py - Mar 17, 2025 - VERSION 87.0 (With trend-following and take-profit update)")
print("signal_generator.py loaded - Mar 17, 2025 - VERSION 87.0 (With trend-following and take-profit update)")

# Configure separate loggers
sentiment_logger = logging.getLogger('sentiment')
sentiment_handler = logging.FileHandler('sentiment.log')
sentiment_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not sentiment_logger.handlers:
    sentiment_logger.addHandler(sentiment_handler)
sentiment_logger.setLevel(logging.WARNING)

indicators_logger = logging.getLogger('indicators')
indicators_handler = logging.FileHandler('indicators.log')
indicators_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not indicators_logger.handlers:
    indicators_logger.addHandler(indicators_handler)
indicators_logger.setLevel(logging.INFO)

signals_logger = logging.getLogger('signals')
signals_handler = logging.FileHandler('signals.log')
signals_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not signals_logger.handlers:
    signals_logger.addHandler(signals_handler)
signals_logger.setLevel(logging.INFO)

async def generate_signals(
    scaled_df: pd.DataFrame,
    preprocessed_data: pd.DataFrame,
    model: TransformerPredictor,
    train_columns: List[str],
    feature_scaler,
    target_scaler,
    rsi_threshold: float = 35,
    macd_fast: int = 12,
    macd_slow: int = 26,
    atr_multiplier: float = 1.0,
    max_risk_pct: float = 0.10
) -> pd.DataFrame:
    """
    Generate trading signals using the SignalGenerator class.
    
    Args:
        scaled_df (pd.DataFrame): Scaled market data.
        preprocessed_data (pd.DataFrame): Unscaled market data.
        model (TransformerPredictor): The trained transformer model.
        train_columns (List[str]): List of feature columns used for training.
        feature_scaler: Scaler for input features.
        target_scaler: Scaler for target values.
        rsi_threshold (float): Initial RSI threshold for signal generation.
        macd_fast (int): Fast period for MACD.
        macd_slow (int): Slow period for MACD.
        atr_multiplier (float): Multiplier for ATR-based calculations.
        max_risk_pct (float): Maximum risk percentage for position sizing.
    
    Returns:
        pd.DataFrame: DataFrame with signals and associated metrics.
    """
    signal_generator = SignalGenerator(
        model=model,
        train_columns=train_columns,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        rsi_threshold=rsi_threshold,
        min_confidence=0.70,
        stop_loss_multiplier=1.5,
        take_profit_multiplier=3.0,  # Increased from 2.0 to improve risk-reward ratio
        position_size=0.1,
        halving_impact_window_days=180,
        trend_adjustment_threshold=0.85,
        capital=17396.68
    )
    # Skip first 1440 rows (60 days) to ensure sufficient data for market regime detection
    start_idx = 1440 if len(scaled_df) > 1440 else 0
    scaled_df_subset = scaled_df.iloc[start_idx:]
    preprocessed_data_subset = preprocessed_data.iloc[start_idx:]
    return await signal_generator.generate_signals(
        scaled_df=scaled_df_subset,
        preprocessed_data=preprocessed_data_subset,
        rsi_threshold=rsi_threshold,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        atr_multiplier=atr_multiplier,
        max_risk_pct=max_risk_pct
    )

class SignalGenerator:
    def __init__(
        self,
        model: TransformerPredictor,
        train_columns: List[str],
        feature_scaler,
        target_scaler,
        rsi_threshold: float = 35,
        min_confidence: float = 0.70,
        stop_loss_multiplier: float = 1.5,
        take_profit_multiplier: float = 3.0,  # Increased from 2.0
        position_size: float = 0.1,
        halving_impact_window_days: int = 180,
        trend_adjustment_threshold: float = 0.85,
        capital: float = 17396.68
    ):
        """
        Initialize the SignalGenerator with halving cycle awareness and trend-following adjustments.
        
        Args:
            model (TransformerPredictor): The transformer model for predictions.
            train_columns (List[str]): List of feature columns used for training.
            feature_scaler: Scaler for input features.
            target_scaler: Scaler for target values.
            rsi_threshold (float): Initial RSI threshold for signal generation.
            min_confidence (float): Minimum confidence threshold for signals.
            stop_loss_multiplier (float): Multiplier for stop-loss (ATR-based).
            take_profit_multiplier (float): Multiplier for take-profit (ATR-based).
            position_size (float): Default position size as a fraction of capital.
            halving_impact_window_days (int): Window around halving to adjust behavior.
            trend_adjustment_threshold (float): Confidence threshold for trend adjustments.
            capital (float): Trading capital for position sizing.
        """
        self.model = model
        self.train_columns = train_columns
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.rsi_threshold = rsi_threshold
        self.min_confidence = min_confidence
        self.stop_loss_multiplier = stop_loss_multiplier
        self.take_profit_multiplier = take_profit_multiplier
        self.position_size = position_size
        self.halving_impact_window_days = halving_impact_window_days
        self.trend_adjustment_threshold = trend_adjustment_threshold
        self.capital = capital
        self.halving_dates = [
            pd.Timestamp("2012-11-28"),
            pd.Timestamp("2016-07-09"),
            pd.Timestamp("2020-05-11"),
            pd.Timestamp("2024-04-19"),
            pd.Timestamp("2028-03-15")
        ]
        logging.info(f"SignalGenerator initialized with rsi_threshold={rsi_threshold}, take_profit_multiplier={take_profit_multiplier}")

    def calculate_halving_impact(self, current_time: pd.Timestamp) -> Tuple[float, str]:
        """
        Calculate the impact of the nearest Bitcoin halving event and determine the cycle phase.
        
        Args:
            current_time (pd.Timestamp): The current timestamp.
        
        Returns:
            Tuple[float, str]: Adjustment factor and cycle phase ("Pre-Halving", "Post-Halving", "Neutral").
        """
        days_to_next, next_halving = calculate_days_to_next_halving(current_time, self.halving_dates)
        days_since_last = None
        last_halving = None
        
        past_halvings = [h for h in self.halving_dates if h <= current_time]
        if past_halvings:
            last_halving = max(past_halvings)
            days_since_last = (current_time - last_halving).days

        adjustment_factor = 1.0
        cycle_phase = "Neutral"

        if 0 < days_to_next <= self.halving_impact_window_days:
            adjustment_factor = 1.2
            cycle_phase = "Pre-Halving"
            logging.debug(f"Pre-Halving phase detected: {days_to_next} days to next halving.")
        
        elif days_since_last is not None and 0 < days_since_last <= self.halving_impact_window_days:
            adjustment_factor = 1.1
            cycle_phase = "Post-Halving"
            logging.debug(f"Post-Halving phase detected: {days_since_last} days since last halving.")
        
        else:
            logging.debug("Neutral phase: No halving impact.")

        return adjustment_factor, cycle_phase

    async def generate_signals(
        self,
        scaled_df: pd.DataFrame,
        preprocessed_data: pd.DataFrame,
        rsi_threshold: float = 35,
        macd_fast: int = 12,
        macd_slow: int = 26,
        atr_multiplier: float = 1.0,
        max_risk_pct: float = 0.10
    ) -> pd.DataFrame:
        """
        Generate trading signals using model predictions, indicators, and sentiment data.
        
        Args:
            scaled_df (pd.DataFrame): Scaled market data.
            preprocessed_data (pd.DataFrame): Unscaled market data.
            rsi_threshold (float): Initial RSI threshold for signal generation.
            macd_fast (int): Fast period for MACD.
            macd_slow (int): Slow period for MACD.
            atr_multiplier (float): Multiplier for ATR-based calculations.
            max_risk_pct (float): Maximum risk percentage for position sizing.
        
        Returns:
            pd.DataFrame: DataFrame with signals and associated metrics.
        """
        logging.info(f"Generating signals with scaled_df shape: {scaled_df.shape}, preprocessed_data shape: {preprocessed_data.shape}")

        # Validate required columns
        required_columns = FEATURE_COLUMNS
        for col in required_columns:
            if col not in scaled_df.columns:
                raise ValueError(f"Missing required column in scaled_df: {col}")
            if col not in preprocessed_data.columns:
                if col.startswith('dist_to_'):
                    vpvr = calculate_vpvr(preprocessed_data, lookback=500, num_bins=50)
                    current_price = preprocessed_data['close'].iloc[-1]
                    scaled_df[col] = (current_price - vpvr[col.split('_')[-1]]) / vpvr[col.split('_')[-1]]
                    preprocessed_data[col] = scaled_df[col]
                elif col == 'luxalgo_signal':
                    scaled_df[col] = luxalgo_trend_reversal(preprocessed_data)
                    preprocessed_data[col] = scaled_df[col]
                elif col == 'trendspider_signal':
                    scaled_df[col] = trendspider_pattern_recognition(preprocessed_data)
                    preprocessed_data[col] = scaled_df[col]
                elif col == 'metastock_slope':
                    scaled_df[col] = metastock_trend_slope(preprocessed_data)
                    preprocessed_data[col] = scaled_df[col]
                else:
                    raise ValueError(f"Missing required column in preprocessed_data: {col}")

        # Setup model and device
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        logging.debug(f"Model moved to device: {device}")

        # Adapt and optimize parameters using rsi_threshold
        params = adapt_strategy_parameters(scaled_df, initial_rsi_threshold=rsi_threshold)
        params = optimize_parameters(preprocessed_data, params)
        rsi_buy_threshold = params['rsi_buy_threshold']
        rsi_sell_threshold = params['rsi_sell_threshold']
        macd_fast = params['macd_fast']
        macd_slow = params['macd_slow']
        atr_multiplier = params['atr_multiplier']
        max_risk_pct = params['max_risk_pct']

        # Extract features and targets
        features = scaled_df[required_columns].values
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        targets = scaled_df['target'].values
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)

        # Create sequences with timestamps
        seq_length = 24
        X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
            features, targets.flatten(), seq_length=seq_length, timestamps=scaled_df.index
        )
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        past_time_features_tensor = torch.FloatTensor(past_time_features).to(device) if past_time_features is not None else None
        past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(device) if past_observed_mask_tensor is not None else None
        future_values_tensor = torch.FloatTensor(future_values).to(device) if future_values is not None else None
        future_time_features_tensor = torch.FloatTensor(future_time_features).to(device) if future_time_features is not None else None

        # Run model predictions with Monte Carlo Dropout
        batch_size = 4000
        num_samples = 5
        predictions = []
        confidences = []
        self.model.train()  # Enable dropout for MCD
        with torch.no_grad():
            for i in range(0, X_tensor.size(0), batch_size):
                batch_end = min(i + batch_size, X_tensor.size(0))
                batch_X = X_tensor[i:batch_end]
                batch_past_time = past_time_features_tensor[i:batch_end] if past_time_features_tensor is not None else None
                batch_mask = past_observed_mask_tensor[i:batch_end] if past_observed_mask_tensor is not None else None
                batch_future_vals = future_values_tensor[i:batch_end] if future_values_tensor is not None else None
                batch_future_time = future_time_features_tensor[i:batch_end] if future_time_features_tensor is not None else None

                batch_preds = []
                for _ in range(num_samples):
                    batch_pred = self.model(
                        past_values=batch_X,
                        past_time_features=batch_past_time,
                        past_observed_mask=batch_mask,
                        future_values=batch_future_vals,
                        future_time_features=batch_future_time
                    )
                    batch_pred = batch_pred.unsqueeze(1)  # Shape: (batch_size, 1, 1)
                    batch_preds.append(batch_pred.cpu().numpy())
                mean_pred = np.mean(batch_preds, axis=0)  # Shape: (batch_size, 1, 1)
                confidence = 1.0 - np.std(batch_preds, axis=0) / (np.mean(np.abs(batch_preds), axis=0) + 1e-10)

                if batch_end == X_tensor.size(0) and mean_pred.shape[0] < batch_size:
                    padding_length = batch_size - mean_pred.shape[0]
                    padding = np.zeros((padding_length, 1, 1))
                    mean_pred = np.concatenate((mean_pred, padding), axis=0)
                    confidence = np.concatenate((confidence, padding), axis=0)

                predictions.append(mean_pred)
                confidences.append(confidence)

        predictions = np.concatenate(predictions)[:X_tensor.size(0), :, :]
        confidences = np.concatenate(confidences)[:X_tensor.size(0), :, :]

        expected_length = len(scaled_df) - seq_length
        pred_length = predictions.shape[0]
        if pred_length < expected_length:
            padding_length = expected_length - pred_length
            padding = np.zeros((padding_length, 1, 1))
            predictions = np.concatenate((predictions, padding), axis=0)
            confidences = np.concatenate((confidences, np.zeros((padding_length, 1, 1))), axis=0)
        elif pred_length > expected_length:
            predictions = predictions[:expected_length]
            confidences = confidences[:expected_length]

        predictions_unscaled = self.target_scaler.inverse_transform(predictions[:, -1, :]).flatten()
        signal_df = scaled_df.iloc[seq_length:].copy()
        signal_df['predicted_price'] = pd.Series(predictions_unscaled, index=signal_df.index, dtype=np.float64)
        signal_df['raw_predicted_price'] = signal_df['predicted_price'].copy()
        signal_df['model_confidence'] = pd.Series(confidences[:, -1, 0].flatten(), index=signal_df.index)

        signal_df['close'] = preprocessed_data['close'].reindex(signal_df.index, method='ffill')

        unscaled_close = preprocessed_data['close'].copy()
        unscaled_high = preprocessed_data['high'].copy()
        unscaled_low = preprocessed_data['low'].copy()
        unscaled_volume = preprocessed_data['volume'].copy()

        if unscaled_close.isna().any() or unscaled_high.isna().any() or unscaled_low.isna().any() or unscaled_volume.isna().any():
            unscaled_close = unscaled_close.ffill().fillna(78877.88)
            unscaled_high = unscaled_high.ffill().fillna(79367.5)
            unscaled_low = unscaled_low.ffill().fillna(78186.98)
            unscaled_volume = unscaled_volume.ffill().fillna(1000.0)

        if (unscaled_close <= 0).any() or (unscaled_close < 10000).any() or (unscaled_close > 200200).any():
            unscaled_close = unscaled_close.apply(lambda x: 78877.88 if x <= 0 or pd.isna(x) or x < 10000 or x > 200200 else x)
            unscaled_high = unscaled_high.apply(lambda x: 79367.5 if x <= 0 or pd.isna(x) or x < 10000 or x > 200200 else x)
            unscaled_low = unscaled_low.apply(lambda x: 78186.98 if x <= 0 or pd.isna(x) or x < 10000 or x > 200200 else x)

        avg_error = 0
        window = 24
        for i in range(len(signal_df)):
            idx = signal_df.index[i]
            if i < window:
                errors = [p - unscaled_close.loc[idx] for idx, p in zip(signal_df.index[:i+1], signal_df['raw_predicted_price'][:i+1]) if idx != signal_df.index[0]]
            else:
                errors = [p - unscaled_close.loc[idx] for idx, p in zip(signal_df.index[i-window+1:i+1], signal_df['raw_predicted_price'][i-window+1:i+1])]
            avg_error = np.mean(errors) if errors else avg_error
            signal_df.loc[idx, 'predicted_price'] = float(signal_df.loc[idx, 'raw_predicted_price'] - avg_error)

        # Ensure sufficient data for indicators
        min_periods = max(14, macd_slow)  # For RSI (14) and MACD slow EMA (26)
        if len(unscaled_close) < min_periods:
            logging.warning(f"Insufficient data for indicator calculation: {len(unscaled_close)} periods, need at least {min_periods}")
            signal_df['rsi'] = pd.Series(50.0, index=signal_df.index)
            signal_df['macd'] = pd.Series(0.0, index=signal_df.index)
            signal_df['macd_signal'] = pd.Series(0.0, index=signal_df.index)
        else:
            signal_df['rsi'] = calculate_rsi(unscaled_close).reindex(signal_df.index, method='ffill').fillna(50.0)
            macd, macd_signal = calculate_macd(unscaled_close, fast=macd_fast, slow=macd_slow)
            signal_df['macd'] = macd.reindex(signal_df.index, method='ffill').fillna(0.0)
            signal_df['macd_signal'] = macd_signal.reindex(signal_df.index, method='ffill').fillna(0.0)

        signal_df['atr'] = calculate_atr(unscaled_high, unscaled_low, unscaled_close).fillna(method='bfill').reindex(signal_df.index, method='ffill')
        signal_df['vwap'] = compute_vwap(preprocessed_data).reindex(signal_df.index, method='ffill')
        signal_df['adx'] = compute_adx(preprocessed_data, initial_period=5).reindex(signal_df.index, method='ffill').fillna(10.0)  # Use initial period 5 for early data
        signal_df['sma_10'] = unscaled_close.rolling(window=10, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['sma_20'] = unscaled_close.rolling(window=20, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['sma_50'] = unscaled_close.rolling(window=50, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['sma_200'] = unscaled_close.rolling(window=200, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')
        signal_df['volume_sma_20'] = unscaled_volume.rolling(window=20, min_periods=1).mean().bfill().reindex(signal_df.index, method='ffill')

        bollinger_bands = compute_bollinger_bands(preprocessed_data)
        signal_df['bb_breakout'] = bollinger_bands['bb_breakout'].reindex(signal_df.index, method='ffill') if 'bb_breakout' in bollinger_bands else 0

        signal_df['sentiment_score'] = [calculate_historical_sentiment(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['x_sentiment'] = [await fetch_x_sentiment(preprocessed_data, idx, suppress_logs=True) for idx in signal_df.index]
        signal_df['fear_greed_index'] = [await get_fear_and_greed_index(preprocessed_data, idx) for idx in signal_df.index]
        signal_df['whale_moves'] = [simulate_historical_whale_moves(preprocessed_data, idx) for idx in signal_df.index]
        onchain_metrics = get_onchain_metrics(symbol="BTC")
        signal_df['hash_rate'] = pd.Series(onchain_metrics['hash_rate'], index=signal_df.index)

        signal_df['market_regime'] = [detect_market_regime(preprocessed_data.loc[:idx], window=1440, log_interval=100) for idx in signal_df.index]
        signal_df['rsi_buy_threshold'] = rsi_buy_threshold
        signal_df['rsi_sell_threshold'] = rsi_sell_threshold

        signal_df['luxalgo_signal'] = luxalgo_trend_reversal(preprocessed_data).reindex(signal_df.index, method='ffill')
        signal_df['trendspider_signal'] = trendspider_pattern_recognition(preprocessed_data).reindex(signal_df.index, method='ffill')
        signal_df['smrt_signal'] = smrt_scalping_signals(preprocessed_data, atr_multiplier=atr_multiplier).reindex(signal_df.index, method='ffill')
        signal_df['metastock_slope'] = metastock_trend_slope(preprocessed_data).reindex(signal_df.index, method='ffill')

        current_time = signal_df.index[-1]
        halving_adjustment, cycle_phase = self.calculate_halving_impact(current_time)

        # Trend-following logic using SMA_50 and SMA_200
        trend_signal = 0
        if signal_df['sma_50'].iloc[-1] > signal_df['sma_200'].iloc[-1]:
            trend_signal = 1  # Bullish trend
        elif signal_df['sma_50'].iloc[-1] < signal_df['sma_200'].iloc[-1]:
            trend_signal = -1  # Bearish trend

        signal_df['signal'] = 0
        signal_df['signal_confidence'] = 0.0
        rolling_volatility = signal_df['price_volatility'].rolling(window=2160, min_periods=1).mean()

        # Ensure price_volatility is calculated if missing
        if 'price_volatility' not in signal_df.columns or signal_df['price_volatility'].isna().all():
            returns = unscaled_close.pct_change().fillna(0)
            signal_df['price_volatility'] = returns.rolling(window=24, min_periods=1).std().fillna(0)

        signal_df['trend'] = np.where(signal_df['sma_50'] > signal_df['sma_200'], 'bullish',
                                    np.where(signal_df['sma_50'] < signal_df['sma_200'], 'bearish', 'neutral'))
        # Add trend-following condition to buy/sell conditions
        signal_df['buy_condition_trend'] = (signal_df['market_regime'].isin(['Bullish Low Volatility', 'Bullish High Volatility', 'Neutral']) |
                                           ((signal_df['sma_50'] > signal_df['sma_200']) & (signal_df['sma_50'].shift(1) <= signal_df['sma_200'].shift(1))))
        signal_df['buy_condition_price'] = signal_df['predicted_price'] > signal_df['close'] + (0.001 * signal_df['close'])
        signal_df['buy_condition_rsi_macd'] = signal_df['rsi'] < (signal_df['rsi_buy_threshold'] + 10)
        signal_df['buy_condition_volume'] = signal_df['volume'] > 0.3 * signal_df['volume_sma_20']
        signal_df['buy_conditions_met'] = (signal_df['buy_condition_trend'].astype(int) +
                                          signal_df['buy_condition_price'].astype(int) +
                                          signal_df['buy_condition_rsi_macd'].astype(int) +
                                          signal_df['buy_condition_volume'].astype(int))

        signal_df['sell_condition_trend'] = (signal_df['market_regime'].isin(['Bearish Low Volatility', 'Bearish High Volatility', 'Neutral']) |
                                            ((signal_df['sma_50'] < signal_df['sma_200']) & (signal_df['sma_50'].shift(1) >= signal_df['sma_200'].shift(1))))
        signal_df['sell_condition_price'] = signal_df['predicted_price'] < signal_df['close'] - (0.001 * signal_df['close'])
        signal_df['sell_condition_rsi_macd'] = signal_df['rsi'] > (signal_df['rsi_sell_threshold'] - 5)
        signal_df['sell_condition_volume'] = signal_df['volume'] > 0.3 * signal_df['volume_sma_20']
        signal_df['sell_conditions_met'] = (signal_df['sell_condition_trend'].astype(int) +
                                           signal_df['sell_condition_price'].astype(int) +
                                           signal_df['sell_condition_rsi_macd'].astype(int) +
                                           signal_df['sell_condition_volume'].astype(int))

        # Confidence calculation with halving and trend adjustments
        signal_df['confidence_base'] = 0.0
        signal_df.loc[signal_df['buy_conditions_met'] >= 3, 'confidence_base'] = 0.4
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['rsi'] < signal_df['rsi_buy_threshold'] - 15), 'confidence_base'] += 0.15
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['macd'] > 0), 'confidence_base'] += 0.2
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['bb_breakout'] == 1), 'confidence_base'] += 0.1
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['sentiment_score'] > 0) & (signal_df['whale_moves'] > 0.2), 'confidence_base'] += 0.15
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['x_sentiment'] > 0) & (signal_df['whale_moves'] > 0.2), 'confidence_base'] += 0.15
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['fear_greed_index'] < 30), 'confidence_base'] += 0.15
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['fear_greed_index'] > 70), 'confidence_base'] -= 0.15
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['luxalgo_signal'] == 1), 'confidence_base'] += 0.1
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['trendspider_signal'] == 1), 'confidence_base'] += 0.05
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['smrt_signal'] == 1), 'confidence_base'] += 0.05
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['metastock_slope'] > 0), 'confidence_base'] += 0.05

        signal_df.loc[signal_df['sell_conditions_met'] >= 3, 'confidence_base'] += 0.4
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['rsi'] > signal_df['rsi_sell_threshold'] + 10), 'confidence_base'] += 0.2
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['macd'] < 0) & (signal_df['macd'] < signal_df['macd_signal']), 'confidence_base'] += 0.3
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['bb_breakout'] == -1), 'confidence_base'] += 0.1
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['sentiment_score'] < 0) & (signal_df['whale_moves'] > 0.2), 'confidence_base'] += 0.2
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['x_sentiment'] < 0) & (signal_df['whale_moves'] > 0.2), 'confidence_base'] += 0.2
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['fear_greed_index'] > 25), 'confidence_base'] += 0.2
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['fear_greed_index'] < 25), 'confidence_base'] -= 0.2
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['luxalgo_signal'] == -1), 'confidence_base'] += 0.1
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['trendspider_signal'] == -1), 'confidence_base'] += 0.05
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['smrt_signal'] == -1), 'confidence_base'] += 0.05
        signal_df.loc[(signal_df['sell_conditions_met'] >= 3) & (signal_df['metastock_slope'] < 0), 'confidence_base'] += 0.05

        # Apply halving adjustment
        signal_df['confidence_base'] = signal_df['confidence_base'] * halving_adjustment

        # Trend-following adjustment
        model_signal = 0
        if signal_df['buy_conditions_met'].iloc[-1] >= 3:
            model_signal = 1
        elif signal_df['sell_conditions_met'].iloc[-1] >= 3:
            model_signal = -1

        final_signal = 0
        adjusted_confidence = signal_df['confidence_base'].copy()
        if trend_signal != 0 and model_signal != 0:
            if trend_signal == model_signal:
                adjusted_confidence *= 1.1
                logging.debug(f"Trend and model signal alignment. Confidence boosted.")
            else:
                adjusted_confidence *= 0.9
                logging.debug(f"Trend and model signal conflict. Confidence reduced.")
                if adjusted_confidence.iloc[-1] < self.trend_adjustment_threshold:
                    final_signal = 0
                    logging.debug("Confidence below trend adjustment threshold. Signal neutralized.")
        else:
            final_signal = model_signal

        # Combine signals with weighted confidence
        signal_df['combined_confidence'] = (WEIGHT_MODEL_CONFIDENCE * signal_df['model_confidence'] +
                                          WEIGHT_LUXALGO * signal_df['luxalgo_signal'] +
                                          WEIGHT_TRENDSPIDER * signal_df['trendspider_signal'] +
                                          WEIGHT_SMRT_SCALPING * signal_df['smrt_signal'] +
                                          WEIGHT_METASTOCK * (signal_df['metastock_slope'] / (np.max(np.abs(signal_df['metastock_slope'])) + 1e-10)))
        signal_df['signal'] = final_signal  # Use trend-aligned signal
        signal_df.loc[(signal_df['buy_conditions_met'] >= 3) & (signal_df['fear_greed_index'] < 80), 'signal'] = 1
        signal_df.loc[signal_df['sell_conditions_met'] >= 3, 'signal'] = -1
        signal_df['signal_confidence'] = np.clip(adjusted_confidence + signal_df['combined_confidence'], 0.0, 1.0)
        signal_df.loc[signal_df['signal_confidence'] < self.min_confidence, 'signal'] = 0

        # Filter signals
        signal_df = filter_signals(signal_df)

        # Add volatility filter with logging
        num_signals_before = (signal_df['signal'] != 0).sum()
        signal_df.loc[signal_df['price_volatility'] > 3 * rolling_volatility, 'signal'] = 0  # Increased threshold to 3x
        num_signals_after = (signal_df['signal'] != 0).sum()
        logging.info(f"Volatility filter applied: {num_signals_before - num_signals_after} signals filtered out")

        # Adjust confidence based on trend instead of filtering signals
        signal_df.loc[(signal_df['signal'] == 1) & (signal_df['sma_50'] <= signal_df['sma_200']), 'signal_confidence'] *= 0.5
        signal_df.loc[(signal_df['signal'] == -1) & (signal_df['sma_50'] >= signal_df['sma_200']), 'signal_confidence'] *= 0.5
        signal_df.loc[signal_df['signal_confidence'] < self.min_confidence, 'signal'] = 0

        # Calculate trade levels and position sizes
        signal_df['take_profit'] = np.nan
        signal_df['stop_loss'] = np.nan
        signal_df['trailing_stop'] = np.nan
        signal_df['trade_outcome'] = np.nan
        signal_df['position_size'] = 0.0

        win_rate = 0.33
        risk_reward_ratio = 2.0
        historical_trades = signal_df[signal_df['signal'] != 0].tail(20)
        if len(historical_trades) > 0 and 'take_profit' in historical_trades.columns and 'stop_loss' in historical_trades.columns:
            trade_outcomes = historical_trades['trade_outcome'].dropna()
            if len(trade_outcomes) > 0:
                wins = sum(1 for outcome in trade_outcomes if outcome == 1)
                total_trades = len(trade_outcomes)
                win_rate = wins / total_trades if total_trades > 0 else 0.33
            avg_win = (historical_trades['take_profit'].mean() - historical_trades['close'].mean()) if win_rate > 0 else 279.56
            avg_loss = (historical_trades['close'].mean() - historical_trades['stop_loss'].mean()) if win_rate < 1.0 else 117.15
            risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 2.0

        for idx in signal_df.index:
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = signal_df['atr'].loc[idx]
            price_volatility = signal_df['price_volatility'].loc[idx]

            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = 500.0
            if unscaled_close_val <= 0 or unscaled_close_val < 10000 or unscaled_close_val > 200200:
                unscaled_close_val = 78877.88

            volatility_adjustment = 0.5 if price_volatility > 3 * rolling_volatility.loc[idx] else 1.0  # Adjusted threshold to 3x

            if signal_df.loc[idx, 'signal'] != 0:
                market_regime = signal_df.loc[idx, 'market_regime']
                stop_loss_mult = self.stop_loss_multiplier * (1.5 if 'High Volatility' in market_regime else 1.0)
                if signal_df.loc[idx, 'signal'] == 1:
                    signal_df.loc[idx, 'stop_loss'] = unscaled_close_val - (unscaled_atr_val * stop_loss_mult)
                    signal_df.loc[idx, 'take_profit'] = unscaled_close_val + (unscaled_atr_val * self.take_profit_multiplier)
                    signal_df.loc[idx, 'trailing_stop'] = unscaled_close_val
                elif signal_df.loc[idx, 'signal'] == -1:
                    signal_df.loc[idx, 'stop_loss'] = unscaled_close_val + (unscaled_atr_val * stop_loss_mult)
                    signal_df.loc[idx, 'take_profit'] = unscaled_close_val - (unscaled_atr_val * self.take_profit_multiplier)
                    signal_df.loc[idx, 'trailing_stop'] = unscaled_close_val

                confidence = signal_df.loc[idx, 'signal_confidence']
                max_position = 0.01 if confidence > 0.9 else 0.005
                position_size = kelly_criterion(win_rate, risk_reward_ratio, self.capital, unscaled_atr_val, unscaled_close_val, max_risk_pct=max_risk_pct)
                position_size = max(position_size, 0.01)
                position_size = min(position_size, max_position) * volatility_adjustment
            else:
                position_size = calculate_position_size(self.capital, unscaled_atr_val, unscaled_close_val, max_risk_pct=max_risk_pct)
                position_size = max(position_size, 0.01)
                position_size = min(position_size, 0.005) * volatility_adjustment

            signal_df.loc[idx, 'position_size'] = position_size

        # Finalize DataFrame
        signal_df['total'] = self.capital
        signal_df['cycle_phase'] = cycle_phase

        actual_prices = unscaled_close.reindex(signal_df.index).values
        predicted_prices = signal_df['predicted_price'].values
        if len(actual_prices) == len(predicted_prices):
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            logging.info(f"Mean Absolute Percentage Error (MAPE) of predictions: {mape:.2f}%")

        return signal_df[[
            'close', 'signal', 'position_size', 'predicted_price', 'rsi', 'macd', 'macd_signal', 'atr', 'vwap', 'adx',
            'bb_breakout', 'sentiment_score', 'x_sentiment', 'fear_greed_index', 'whale_moves', 'hash_rate', 'total',
            'price_volatility', 'signal_confidence', 'sma_10', 'sma_20', 'sma_50', 'sma_200', 'take_profit', 'stop_loss',
            'trailing_stop', 'trade_outcome', 'luxalgo_signal', 'trendspider_signal', 'smrt_signal', 'metastock_slope',
            'model_confidence', 'cycle_phase'
        ]]