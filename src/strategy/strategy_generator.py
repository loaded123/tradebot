import pandas as pd
from src.models.model_predictor import predict_next_movement
from src.strategy.position_sizer import kelly_criterion, monte_carlo_position_sizing
from src.strategy.indicators import compute_vwap, compute_adx

def generate_signals(preprocessed_data, model, feature_columns, feature_scaler, target_scaler, sequence_length=10, threshold=0.005, **params):
    """
    Generate trading signals based on model predictions with optimized position sizing.

    :param preprocessed_data: Preprocessed DataFrame
    :param model: Trained model for price prediction
    :param feature_columns: List of feature column names
    :param feature_scaler: Scaler used for feature normalization
    :param target_scaler: Scaler used for target normalization
    :param sequence_length: Number of time steps used for prediction
    :param threshold: Threshold for classifying price movement
    :param params: Additional parameters for strategy adaptation
    :return: DataFrame with trading signals including portfolio value
    """
    # Ensure that necessary columns are present
    required_columns = ['sma_20', 'sma_50', 'adx', 'vwap', 'atr']
    for col in required_columns:
        if col not in preprocessed_data.columns:
            raise ValueError(f"Missing required column: {col}")

    signals = []
    current_balance = 10000
    position_value = 0

    # Compute additional indicators
    preprocessed_data['vwap'] = compute_vwap(preprocessed_data)
    preprocessed_data['adx'] = compute_adx(preprocessed_data)

    for i in range(sequence_length, len(preprocessed_data)):
        data_slice = preprocessed_data.iloc[i-sequence_length:i]
        prediction = predict_next_movement(model, data_slice, feature_columns, feature_scaler, target_scaler, sequence_length, threshold)
        
        # Trend confirmation
        sma_fast = data_slice['sma_20'].iloc[-1]
        sma_slow = data_slice['sma_50'].iloc[-1]
        adx = data_slice['adx'].iloc[-1]
        vwap_signal = data_slice['vwap'].iloc[-1] > data_slice['close'].iloc[-1]

        if prediction == 'up' and sma_fast > sma_slow and adx > 20 and vwap_signal:
            signal = 1  # Buy signal
        elif prediction == 'down' and sma_fast < sma_slow and adx > 20 and not vwap_signal:
            signal = -1  # Sell signal
        else:
            signal = 0  # Neutral, no action
        
        # Calculate position size with Kelly Criterion
        atr = data_slice['atr'].iloc[-1]
        last_close = data_slice['close'].iloc[-1]
        win_rate = 0.55  # Assumed win rate from backtesting
        risk_reward_ratio = 2  # Assumed from strategy backtests
        
        position_size = kelly_criterion(win_rate, risk_reward_ratio, current_balance, atr, last_close, max_risk_pct=params.get('max_risk_pct', 0.02))

        if signal == 1 and position_value == 0:
            position_value = current_balance * position_size
            current_balance -= position_value
        elif signal == -1 and position_value > 0:
            current_balance += position_value
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
