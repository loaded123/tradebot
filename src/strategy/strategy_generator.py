# src/strategy/strategy_generator.py

import pandas as pd
import numpy as np
import logging
from src.models.model_predictor import predict_next_movement
from src.strategy.position_sizer import kelly_criterion
from src.strategy.indicators import compute_vwap, compute_adx
from src.constants import FEATURE_COLUMNS  # Added import

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def generate_signals(preprocessed_data, model, feature_columns, feature_scaler, target_scaler, 
                     sequence_length=10, threshold=0.002, **params):
    """Generate trading signals based on model predictions and indicators."""
    required_columns = ['close', 'sma_20', 'adx', 'vwap', 'atr', 'target', 'price_volatility']
    for col in required_columns:
        if col not in preprocessed_data.columns:
            raise ValueError(f"Missing required column: {col}")

    signals = []
    current_balance = 10000
    position_value = 0

    try:
        for i in range(sequence_length, len(preprocessed_data)):
            data_slice = preprocessed_data.iloc[i - sequence_length:i]
            last_close = data_slice['close'].iloc[-1]
            atr = data_slice['atr'].iloc[-1]
            volatility = data_slice['price_volatility'].iloc[-1]
            
            # Position sizing
            win_rate = params.get('win_rate', 0.55)
            risk_reward_ratio = params.get('risk_reward_ratio', 2.0)
            max_risk_pct = params.get('max_risk_pct', 0.05)
            position_size = kelly_criterion(win_rate, risk_reward_ratio, current_balance, atr, last_close, max_risk_pct)
            
            # Prediction using only features (17)
            prediction = predict_next_movement(model, data_slice, FEATURE_COLUMNS, feature_scaler, target_scaler, 
                                              sequence_length, threshold)
            
            # Indicator signals
            sma_fast = data_slice['sma_20'].iloc[-1]
            adx = data_slice['adx'].iloc[-1]
            vwap_signal = data_slice['vwap'].iloc[-1] > last_close
            rsi = data_slice['momentum_rsi'].iloc[-1] if 'momentum_rsi' in data_slice.columns else 50
            macd = data_slice['trend_macd'].iloc[-1] if 'trend_macd' in data_slice.columns else 0
            rsi_threshold = params.get('rsi_threshold', 50)
            
            logging.debug(f"Date: {data_slice.index[-1]}, Prediction: {prediction}, SMA: {sma_fast}, "
                         f"ADX: {adx}, VWAP: {vwap_signal}, RSI: {rsi}, MACD: {macd}")
            
            if (prediction == 'up' and sma_fast > last_close * 0.99 and adx > 15 and vwap_signal and 
                rsi > max(50, rsi_threshold - 10) and macd > -0.5):
                signal = 1  # Buy
                logging.info(f"Buy signal at {data_slice.index[-1]}")
            elif (prediction == 'down' and sma_fast < last_close * 1.01 and adx > 15 and not vwap_signal and 
                  rsi < (100 - max(50, rsi_threshold - 10)) and macd < 0.5):
                signal = -1  # Sell/Short
                logging.info(f"Sell signal at {data_slice.index[-1]}")
            else:
                signal = 0
            
            # Update portfolio
            if signal == 1 and position_value == 0:  # Long entry
                position_value = current_balance * position_size
                current_balance -= position_value
            elif signal == -1 and position_value > 0:  # Long exit
                current_balance += position_value
                position_value = 0
            elif signal == -1 and position_value == 0:  # Short entry
                position_value = -current_balance * position_size
                current_balance -= position_value
            elif signal == 1 and position_value < 0:  # Short exit
                current_balance -= position_value
                position_value = 0
            
            total = current_balance + position_value
            
            signals.append({
                'date': data_slice.index[-1],
                'signal': signal,
                'close': last_close,
                'atr': atr,
                'position_size': position_size if signal != 0 else 0,
                'cash': current_balance,
                'position_value': position_value,
                'total': total,
                'price_volatility': volatility  # Add price_volatility to output
            })
        
        signal_df = pd.DataFrame(signals).set_index('date')
        logging.info(f"Generated {len(signal_df)} signals")
        return signal_df
    
    except Exception as e:
        logging.error(f"Error generating signals: {e}")
        return pd.DataFrame()