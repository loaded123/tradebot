# src/strategy/signal_generator.py
"""
This module generates trading signals using a TransformerPredictor model, integrating preprocessed market
data and technical indicators. It includes trade level calculations (stop loss, take profit) and position
sizing based on market conditions and model confidence. Signals from multiple sources (LuxAlgo, TrendSpider,
SMRT Scalping, MetaStock, and model predictions) are combined using weighted voting.

Key Integrations:
- **src.models.transformer_model.TransformerPredictor**: The trained model generates predictions (log returns)
  which are inverse-transformed and used to derive signals and prices.
- **src.strategy.parameter_optimizer.adapt_strategy_parameters/optimize_parameters**: Adapts and optimizes
  strategy parameters (e.g., RSI thresholds) based on preprocessed data trends.
- **src.strategy.position_sizer.calculate_position_size/kelly_criterion**: Computes position sizes using
  risk management techniques, integrated into the signal generation loop.
- **src.utils.sequence_utils.create_sequences**: Prepares input sequences for the transformer model,
  handling timestamps and feature alignment.
- **src.utils.time_utils.calculate_days_to_next_halving**: Influences signal confidence via halving cycle
  adjustments, integrated through calculate_halving_impact.
- **src.data.data_preprocessor.preprocess_data**: Provides preprocessed_data with unscaled 'close' and 'atr'
  values for trade level calculations, while scaled_df is used for model input.
- **src.constants**: Uses default weights (e.g., WEIGHT_LUXALGO) if not provided via the weights parameter.

Future Considerations:
- The row-by-row trade level calculation may be a bottleneck for large datasets. Consider vectorizing
  operations or precomputing trade levels.
- The MAPE calculation assumes aligned actual and predicted prices; mismatches should trigger deeper
  sequence alignment checks.
- Add support for dynamic weight adjustment based on real-time performance or market regime.

Dependencies:
- pandas
- numpy
- torch
- src.models.transformer_model
- src.strategy.parameter_optimizer
- src.strategy.position_sizer
- src.utils.sequence_utils
- src.utils.time_utils
- src.constants
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.models.transformer_model import TransformerPredictor
from src.strategy.parameter_optimizer import adapt_strategy_parameters, optimize_parameters
from src.strategy.position_sizer import calculate_position_size, kelly_criterion
from src.utils.sequence_utils import create_sequences
from src.utils.time_utils import calculate_days_to_next_halving
from src.constants import (
    FEATURE_COLUMNS, WEIGHT_LUXALGO, WEIGHT_TRENDSPIDER, WEIGHT_SMRT_SCALPING,
    WEIGHT_METASTOCK, WEIGHT_MODEL_CONFIDENCE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
main_logger = logging.getLogger('main')
main_logger.info("Using MODULARIZED signal_generator.py - Mar 18, 2025 - VERSION 87.7 (Fixed close price fallback and prediction scaling)")
print("signal_generator.py loaded - Mar 18, 2025 - VERSION 87.7 (Fixed close price fallback and prediction scaling)")

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
    max_risk_pct: float = 0.10,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Generate trading signals using the SignalGenerator class, combining signals from multiple sources
    with weighted voting.

    Args:
        scaled_df (pd.DataFrame): Scaled market data from data_preprocessor.scale_features.
        preprocessed_data (pd.DataFrame): Unscaled market data with indicators from data_preprocessor.preprocess_data.
        model (TransformerPredictor): The trained transformer model.
        train_columns (List[str]): List of feature columns used for training.
        feature_scaler: Scaler for input features.
        target_scaler: Scaler for target values.
        rsi_threshold (float): Initial RSI threshold for signal generation.
        macd_fast (int): Fast period for MACD.
        macd_slow (int): Slow period for MACD.
        atr_multiplier (float): Multiplier for ATR-based calculations.
        max_risk_pct (float): Maximum risk percentage for position sizing.
        weights (dict, optional): Weights for combining signals from different strategies. Expected keys:
            - WEIGHT_LUXALGO
            - WEIGHT_TRENDSPIDER
            - WEIGHT_SMRT_SCALPING
            - WEIGHT_METASTOCK
            - WEIGHT_MODEL_CONFIDENCE
            If None, uses default weights from src.constants.

    Returns:
        pd.DataFrame: DataFrame with raw signals, predicted prices, and trade levels.

    Notes:
        - Skips the first 1440 rows (60 days) to ensure sufficient data for indicators.
        - Integrates with data_preprocessor.py for unscaled 'close' and 'atr' values.
        - Combines signals using weighted voting from LuxAlgo, TrendSpider, SMRT Scalping, MetaStock,
          and model predictions, with weights normalized to sum to 1.
    """
    signal_generator = SignalGenerator(
        model=model,
        train_columns=train_columns,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        rsi_threshold=rsi_threshold,
        min_confidence=0.70,
        stop_loss_multiplier=1.5,
        take_profit_multiplier=3.0,
        position_size=0.1,
        halving_impact_window_days=180,
        trend_adjustment_threshold=0.85,
        capital=17396.68
    )
    start_idx = 1440 if len(scaled_df) > 1440 else 0
    scaled_df_subset = scaled_df.iloc[start_idx:].copy()
    preprocessed_data_subset = preprocessed_data.iloc[start_idx:].copy()
    logging.info(f"scaled_df_subset columns after slicing: {list(scaled_df_subset.columns)}")
    params = {
        'rsi_threshold': rsi_threshold,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'atr_multiplier': atr_multiplier,
        'max_risk_pct': max_risk_pct
    }
    return await signal_generator.generate_signals(
        scaled_df=scaled_df_subset,
        preprocessed_data=preprocessed_data_subset,
        params=params,
        start_idx=start_idx,
        weights=weights  # Pass weights to SignalGenerator
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
        take_profit_multiplier: float = 3.0,
        position_size: float = 0.1,
        halving_impact_window_days: int = 180,
        trend_adjustment_threshold: float = 0.85,
        capital: float = 17396.68
    ):
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
            pd.Timestamp("2012-11-28").tz_localize('UTC'),
            pd.Timestamp("2016-07-09").tz_localize('UTC'),
            pd.Timestamp("2020-05-11").tz_localize('UTC'),
            pd.Timestamp("2024-04-19").tz_localize('UTC'),
            pd.Timestamp("2028-03-15").tz_localize('UTC')
        ]
        logging.info(f"SignalGenerator initialized with rsi_threshold={rsi_threshold}, take_profit_multiplier={take_profit_multiplier}")

    def calculate_halving_impact(self, current_time: pd.Timestamp) -> Tuple[float, str]:
        """
        Calculate the impact of Bitcoin halving cycles on market behavior.

        Args:
            current_time (pd.Timestamp): Current timestamp to evaluate.

        Returns:
            Tuple[float, str]: Adjustment factor and cycle phase.

        Notes:
            - Integrates with src.utils.time_utils.calculate_days_to_next_halving for halving date logic.
            - Adjusts signal confidence based on proximity to halving events.
        """
        if current_time.tz is None:
            logging.warning(f"current_time {current_time} is timezone-naive, localizing to UTC")
            current_time = current_time.tz_localize('UTC')
        elif current_time.tz != pd.Timestamp("2020-01-01").tz_localize('UTC').tz:
            logging.warning(f"current_time timezone {current_time.tz} differs from UTC, converting to UTC")
            current_time = current_time.tz_convert('UTC')
        days_to_next, next_halving = calculate_days_to_next_halving(current_time, self.halving_dates)
        days_since_last = (current_time - max([h for h in self.halving_dates if h <= current_time])).days if any(h <= current_time for h in self.halving_dates) else None
        adjustment_factor = 1.0
        cycle_phase = "Neutral"
        if 0 < days_to_next <= self.halving_impact_window_days:
            adjustment_factor = 1.2
            cycle_phase = "Pre-Halving"
        elif days_since_last is not None and 0 < days_since_last <= self.halving_impact_window_days:
            adjustment_factor = 1.1
            cycle_phase = "Post-Halving"
        return adjustment_factor, cycle_phase

    async def generate_signals(
        self,
        scaled_df: pd.DataFrame,
        preprocessed_data: pd.DataFrame,
        params: Dict[str, float] = None,
        start_idx: int = 0,
        weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions and market indicators, combining signals
        with weighted voting.

        Args:
            scaled_df (pd.DataFrame): Scaled input data for the model.
            preprocessed_data (pd.DataFrame): Unscaled data with indicators from data_preprocessor.py.
            params (Dict[str, float]): Optional parameters for strategy tuning.
            start_idx (int): Starting index used for skipping initial rows (default: 0).
            weights (dict, optional): Weights for combining signals from different strategies. Expected keys:
                - WEIGHT_LUXALGO
                - WEIGHT_TRENDSPIDER
                - WEIGHT_SMRT_SCALPING
                - WEIGHT_METASTOCK
                - WEIGHT_MODEL_CONFIDENCE
                If None, uses default weights from src.constants.

        Returns:
            pd.DataFrame: DataFrame with signals, predicted prices, and trade levels.

        Notes:
            - Uses Monte Carlo Dropout for uncertainty estimation.
            - Integrates unscaled 'close' and 'atr' from preprocessed_data for trade levels.
            - Corrects predicted prices using a moving average of errors.
            - Includes MACD and MACD signal columns for downstream filtering.
            - Combines signals using weighted voting from LuxAlgo, TrendSpider, SMRT Scalping, MetaStock,
              and model predictions, with weights normalized to sum to 1.
        """
        params = params or {}
        rsi_threshold = params.get('rsi_threshold', 35)
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        atr_multiplier = params.get('atr_multiplier', 1.0)
        max_risk_pct = params.get('max_risk_pct', 0.10)

        logging.info(f"Generating signals with scaled_df shape: {scaled_df.shape}, preprocessed_data shape: {preprocessed_data.shape}")

        required_columns = FEATURE_COLUMNS
        logging.debug(f"Required columns: {required_columns}")
        logging.debug(f"scaled_df columns before validation: {list(scaled_df.columns)}")
        for col in required_columns:
            if col not in scaled_df.columns:
                raise ValueError(f"Missing required column in scaled_df: {col}")
            if col not in preprocessed_data.columns:
                raise ValueError(f"Missing required column in preprocessed_data: {col}")
        if 'target' not in preprocessed_data.columns:
            raise ValueError("Missing 'target' column in preprocessed_data.")

        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        adapted_params = adapt_strategy_parameters(preprocessed_data, initial_rsi_threshold=rsi_threshold)
        optimized_params = optimize_parameters(preprocessed_data, adapted_params)
        rsi_buy_threshold = optimized_params['rsi_buy_threshold']
        rsi_sell_threshold = optimized_params['rsi_sell_threshold']
        macd_fast = optimized_params['macd_fast']
        macd_slow = optimized_params['macd_slow']
        atr_multiplier = optimized_params['atr_multiplier']
        max_risk_pct = optimized_params['max_risk_pct']
        params.update({
            'rsi_buy_threshold': rsi_buy_threshold,
            'rsi_sell_threshold': rsi_sell_threshold,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'atr_multiplier': atr_multiplier,
            'max_risk_pct': max_risk_pct
        })

        # Use default weights if none provided
        if weights is None:
            weights = {
                'WEIGHT_LUXALGO': WEIGHT_LUXALGO,
                'WEIGHT_TRENDSPIDER': WEIGHT_TRENDSPIDER,
                'WEIGHT_SMRT_SCALPING': WEIGHT_SMRT_SCALPING,
                'WEIGHT_METASTOCK': WEIGHT_METASTOCK,
                'WEIGHT_MODEL_CONFIDENCE': WEIGHT_MODEL_CONFIDENCE
            }
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Sum of weights must be non-zero")
        logging.info(f"Using weights: {weights}")

        features = scaled_df[required_columns].values
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        targets = preprocessed_data['target'].reindex(scaled_df.index).values
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)

        seq_length = 24
        X, y, past_time_features, past_observed_mask, future_values, future_time_features = create_sequences(
            features, targets.flatten(), seq_length=seq_length, timestamps=scaled_df.index
        )
        logging.info(f"Sequence data shapes: X={X.shape}, y={y.shape}")
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        past_time_features_tensor = torch.FloatTensor(past_time_features).to(device) if past_time_features is not None else None
        past_observed_mask_tensor = torch.FloatTensor(past_observed_mask).to(device) if past_observed_mask is not None else None
        future_values_tensor = torch.FloatTensor(future_values).to(device) if future_values is not None else None
        future_time_features_tensor = torch.FloatTensor(future_time_features).to(device) if future_time_features is not None else None

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
                    batch_pred = batch_pred.unsqueeze(1)
                    batch_preds.append(batch_pred.cpu().numpy())
                mean_pred = np.mean(batch_preds, axis=0)
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

        # Log raw predictions before inverse transformation
        logging.debug(f"Raw predictions before inverse transform (first 5): {predictions[:5, -1, :].flatten()}")

        # Inverse transform predictions with validation and log scaler range
        predictions_unscaled = self.target_scaler.inverse_transform(predictions[:, -1, :]).flatten()
        logging.debug(f"Target scaler min: {self.target_scaler.data_min_}, max: {self.target_scaler.data_max_}")
        signal_df = pd.DataFrame(index=scaled_df.index[seq_length:])
        logging.debug(f"Raw predicted log returns (first 5): {predictions_unscaled[:5]}")
        signal_df['predicted_log_return'] = pd.Series(predictions_unscaled, index=signal_df.index, dtype=np.float64)
        signal_df['raw_predicted_log_return'] = signal_df['predicted_log_return'].copy()
        signal_df['model_confidence'] = pd.Series(confidences[:, -1, 0].flatten(), index=signal_df.index)

        # Use preprocessed close directly without overriding historical values
        unscaled_close = preprocessed_data['close'].reindex(signal_df.index).copy()
        unscaled_close = unscaled_close.ffill().bfill()  # Use forward and backward fill to preserve historical prices
        signal_df['close'] = unscaled_close
        signal_df['predicted_price'] = signal_df['close'] * np.exp(signal_df['predicted_log_return'])
        signal_df['raw_predicted_price'] = signal_df['close'] * np.exp(signal_df['raw_predicted_log_return'])

        # Correct predicted prices
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

        current_time = signal_df.index[-1]
        halving_adjustment, cycle_phase = self.calculate_halving_impact(current_time)

        # Initialize signal columns
        signal_df['signal'] = 0
        signal_df['signal_confidence'] = 0.0

        # Generate model-based signals
        for idx in signal_df.index:
            predicted_return = signal_df.loc[idx, 'predicted_log_return']
            rsi = preprocessed_data['momentum_rsi'].reindex(signal_df.index).loc[idx]
            if pd.isna(rsi):
                continue
            if predicted_return > 0 and rsi < rsi_buy_threshold:
                signal_df.loc[idx, 'signal'] = 1
                signal_df.loc[idx, 'signal_confidence'] = min(0.9, abs(predicted_return) * halving_adjustment)
            elif predicted_return < 0 and rsi > rsi_sell_threshold:
                signal_df.loc[idx, 'signal'] = -1
                signal_df.loc[idx, 'signal_confidence'] = min(0.9, abs(predicted_return) * halving_adjustment)

        # Combine signals with weighted voting
        signal_df['luxalgo_signal'] = preprocessed_data['luxalgo_signal'].reindex(signal_df.index).fillna(0)
        signal_df['trendspider_signal'] = preprocessed_data['trendspider_signal'].reindex(signal_df.index).fillna(0)
        signal_df['smrt_scalping_signal'] = preprocessed_data['smrt_scalping_signal'].reindex(signal_df.index).fillna(0)
        signal_df['metastock_slope'] = preprocessed_data['metastock_slope'].reindex(signal_df.index).fillna(0)

        # Normalize weights
        normalized_weights = {k: v / sum(weights.values()) for k, v in weights.items()}
        signal_df['combined_signal'] = (
            normalized_weights['WEIGHT_LUXALGO'] * signal_df['luxalgo_signal'] +
            normalized_weights['WEIGHT_TRENDSPIDER'] * signal_df['trendspider_signal'] +
            normalized_weights['WEIGHT_SMRT_SCALPING'] * signal_df['smrt_scalping_signal'] +
            normalized_weights['WEIGHT_METASTOCK'] * signal_df['metastock_slope'] +
            normalized_weights['WEIGHT_MODEL_CONFIDENCE'] * signal_df['signal']
        )

        # Convert combined signal to discrete signals with adjusted confidence
        signal_df['signal'] = np.where(signal_df['combined_signal'] > 0.1, 1,
                                       np.where(signal_df['combined_signal'] < -0.1, -1, 0))
        signal_df['signal_confidence'] = np.abs(signal_df['combined_signal']) * signal_df['model_confidence']

        logging.info(f"Generated {len(signal_df[signal_df['signal'] != 0])} non-zero signals")

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

        log_counter = 0
        for idx in signal_df.index:
            if idx not in unscaled_close.index:
                logging.debug(f"Index {idx} not in unscaled_close, skipping.")
                continue
            unscaled_close_val = unscaled_close.loc[idx]
            unscaled_atr_val = preprocessed_data['atr'].reindex(signal_df.index).loc[idx]
            price_volatility = preprocessed_data['price_volatility'].reindex(signal_df.index).loc[idx]

            if pd.isna(unscaled_atr_val):
                unscaled_atr_val = unscaled_close_val * 0.01  # Dynamic fallback (1% of close)
                logging.debug(f"ATR fallback applied for {idx}: ATR set to {unscaled_atr_val:.2f}")

            logging.debug(f"Processing {idx}: Close={unscaled_close_val:.2f}, ATR={unscaled_atr_val:.2f}")
            rolling_volatility = preprocessed_data['price_volatility'].reindex(signal_df.index).rolling(window=2160, min_periods=1).mean()
            volatility_adjustment = 0.5 if price_volatility > 3 * rolling_volatility.loc[idx] else 1.0

            signal = signal_df.loc[idx, 'signal']
            if signal != 0:
                market_regime = preprocessed_data['market_regime'].reindex(signal_df.index).loc[idx] if 'market_regime' in preprocessed_data.columns else 'Neutral Low Volatility'
                stop_loss_mult = self.stop_loss_multiplier * (1.5 if 'High Volatility' in market_regime else 1.0)
                if signal == 1:
                    stop_loss = unscaled_close_val - (unscaled_atr_val * stop_loss_mult)
                    take_profit = unscaled_close_val + (unscaled_atr_val * self.take_profit_multiplier)
                    signal_df.loc[idx, 'stop_loss'] = max(stop_loss, 0)
                    signal_df.loc[idx, 'take_profit'] = max(take_profit, unscaled_close_val + 1)
                    signal_df.loc[idx, 'trailing_stop'] = unscaled_close_val
                    if log_counter % 100 == 0:
                        logging.debug(f"Set buy levels at {idx}: Stop_Loss={stop_loss:.2f}, Take_Profit={take_profit:.2f}")
                elif signal == -1:
                    stop_loss = unscaled_close_val + (unscaled_atr_val * stop_loss_mult)
                    take_profit = unscaled_close_val - (unscaled_atr_val * self.take_profit_multiplier)
                    signal_df.loc[idx, 'stop_loss'] = min(stop_loss, unscaled_close_val * 2)
                    signal_df.loc[idx, 'take_profit'] = max(take_profit, 0)  # Prevent negative take_profit
                    signal_df.loc[idx, 'trailing_stop'] = unscaled_close_val
                    if log_counter % 100 == 0:
                        logging.debug(f"Set sell levels at {idx}: Stop_Loss={stop_loss:.2f}, Take_Profit={take_profit:.2f}")
                log_counter += 1

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

        signal_df['total'] = self.capital
        signal_df['cycle_phase'] = cycle_phase

        # Include 'trend_macd' and 'macd_signal' for downstream filtering
        if 'trend_macd' in preprocessed_data.columns:
            signal_df['trend_macd'] = preprocessed_data['trend_macd'].reindex(signal_df.index, method='ffill')
        if 'macd_signal' in preprocessed_data.columns:
            signal_df['macd_signal'] = preprocessed_data['macd_signal'].reindex(signal_df.index, method='ffill')

        required_columns = [
            'momentum_rsi', 'atr', 'vwap', 'adx', 'bb_breakout',
            'sentiment_score', 'x_sentiment', 'fear_greed_index', 'whale_moves',
            'hash_rate', 'price_volatility', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'volume_sma_20', 'market_regime', 'luxalgo_signal', 'trendspider_signal',
            'smrt_scalping_signal', 'metastock_slope'
        ]
        for col in required_columns:
            if col in preprocessed_data.columns:
                signal_df[col] = preprocessed_data[col].reindex(signal_df.index, method='ffill')

        actual_prices = unscaled_close.values
        predicted_prices = signal_df['predicted_price'].values
        logging.debug(f"Actual prices shape: {actual_prices.shape}, Predicted prices shape: {predicted_prices.shape}")
        logging.debug(f"Sample actual prices: {actual_prices[:5]}, Sample predicted prices: {predicted_prices[:5]}")
        if len(actual_prices) == len(predicted_prices):
            mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
            logging.info(f"Mean Absolute Percentage Error (MAPE) of predictions: {mape:.2f}%")
        else:
            logging.warning(f"Mismatch in lengths: actual_prices={len(actual_prices)}, predicted_prices={len(predicted_prices)}")

        return signal_df