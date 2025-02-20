import pandas as pd
import numpy as np
from src.models.model_predictor import predict_next_movement
from src.strategy.position_sizer import kelly_criterion, monte_carlo_position_sizing
from src.strategy.indicators import compute_vwap, compute_adx

def generate_signals(preprocessed_data, model, feature_columns, feature_scaler, target_scaler, sequence_length=10, threshold=0.005, **params):
    """Generates trading signals."""

    required_columns = ['sma_20', 'adx', 'vwap', 'atr']
    for col in required_columns:
        if col not in preprocessed_data.columns:
            raise ValueError(f"Missing required column: {col}")

    signals = []
    current_balance = 10000
    position_value = 0

    for i in range(sequence_length, len(preprocessed_data)):
        data_slice = preprocessed_data.iloc[i - sequence_length:i]

        atr = data_slice['atr'].iloc[-1]
        last_close = data_slice['close'].iloc[-1]
        win_rate = params.get('win_rate', 0.55)
        risk_reward_ratio = params.get('risk_reward_ratio', 2.0)

        position_size = kelly_criterion(win_rate, risk_reward_ratio, current_balance, atr, last_close,
                                        max_risk_pct=params.get('max_risk_pct', 0.02))

        prediction = predict_next_movement(model, data_slice, feature_columns, feature_scaler, target_scaler,
                                          sequence_length, threshold)

        print(f"Prediction: {prediction}")  # Print the prediction

        sma_fast = data_slice['sma_20'].iloc[-1]
        adx = data_slice['adx'].iloc[-1]
        vwap_signal = data_slice['vwap'].iloc[-1] > data_slice['close'].iloc[-1]

        print(f"Index: {data_slice.index[-1]}")
        print(f"SMA Fast: {sma_fast}")
        print(f"Close: {data_slice['close'].iloc[-1]}")
        print(f"ADX: {adx}")
        print(f"VWAP Signal: {vwap_signal}")
        print(f"Params: {params}")

        if prediction == 'up' and sma_fast > data_slice['close'].iloc[-1] and adx > 20 and vwap_signal:
            print("Buy signal conditions met!")
            signal = 1
        elif prediction == 'down' and sma_fast < data_slice['close'].iloc[-1] and adx > 20 and not vwap_signal:
            print("Sell signal conditions met!")
            signal = -1
        else:
            print("No signal")
            if prediction != 'up' and prediction != 'down':
                print(f"  - Invalid prediction: {prediction}")
            if sma_fast <= data_slice['close'].iloc[-1]:
                print(f"  - SMA is not greater than close")
            if adx <= 20:
                print(f"  - ADX is not greater than 20")
            if vwap_signal == False:
                print(f"  - VWAP signal is not True")
            if vwap_signal == True:
                print(f"  - VWAP signal is not False")

            signal = 0

        if signal == 1 and position_value == 0:
            position_value = current_balance * position_size
            current_balance -= position_value
        elif signal == -1 and position_value > 0:
            current_balance += position_value
            position_value = 0
        elif signal == -1 and position_value == 0:  # Short Sell
            position_value = -current_balance * position_size
            current_balance -= position_value
        elif signal == 1 and position_value < 0:  # Close Short Sell
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
            'total': total
        })

    return pd.DataFrame(signals).set_index('date')