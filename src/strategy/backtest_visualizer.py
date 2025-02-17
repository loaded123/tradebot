import asyncio
import sys
import os

# Ensure the correct event loop policy on Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import matplotlib.pyplot as plt
import pandas as pd
import torch
import asyncio
import winloop
from src.models.train_lstm_model import LSTMModel
import joblib
from src.strategy.backtester import backtest_strategy
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data, FEATURE_COLUMNS
from src.constants import FEATURE_COLUMNS
from src.strategy.strategy_generator import generate_signals
from src.strategy.market_regime import detect_market_regime
from src.strategy.position_sizer import calculate_position_size
from src.strategy.risk_manager import manage_risk
from src.strategy.strategy_adapter import adapt_strategy_parameters

# Assuming device is defined in train_lstm_model.py or here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_backtest_results(results):
    """
    Plots cumulative returns, equity curve, and drawdowns from backtest results.

    :param results: DataFrame with 'total', 'returns', 'cumulative_returns' columns
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)

    # Equity Curve
    axes[0].plot(results.index, results['total'], label='Equity Curve', color='blue')
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title("Equity Curve")
    axes[0].legend()
    axes[0].grid()

    # Cumulative Returns
    axes[1].plot(results.index, results['cumulative_returns'], label='Cumulative Returns', color='green')
    axes[1].set_ylabel("Cumulative Return (%)")
    axes[1].set_title("Cumulative Returns")
    axes[1].legend()
    axes[1].grid()

    # Drawdown
    drawdown = (results['total'] / results['total'].cummax() - 1)
    axes[2].fill_between(results.index, drawdown, color='red', alpha=0.3, label='Drawdown')
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_title("Drawdowns")
    axes[2].legend()
    axes[2].grid()

    # Trades on Price Chart
    axes[3].plot(results.index, results['close'], label='Price', color='black')
    buy_signals = results[results['signal'] == 1]
    sell_signals = results[results['signal'] == -1]
    axes[3].plot(buy_signals.index, buy_signals['close'], '^', markersize=8, color='g', label='Buy Signal')
    axes[3].plot(sell_signals.index, sell_signals['close'], 'v', markersize=8, color='r', label='Sell Signal')
    axes[3].set_ylabel("Price")
    axes[3].set_title("Price with Buy/Sell Signals")
    axes[3].legend()
    axes[3].grid(True)

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

def load_model(model_class, model_path="best_model.pth", input_dim=12, hidden_dim=32, layer_dim=1, output_dim=1):
    # Check if the model file exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = model_class(input_dim, hidden_dim, layer_dim, output_dim).to(device)  # Send model to device
        model.load_state_dict(torch.load(model_path, map_location=device))  # Ensure it loads to the correct device
        model.eval()
    else:
        print(f"Model file {model_path} not found, training a new model...")
        model = None
    return model


def filter_signals(signal_data, min_hold_period=5):
    """
    Filter trading signals to reduce frequency, aiming to minimize transaction fees.

    :param signal_data: DataFrame with trading signals
    :param min_hold_period: Minimum number of periods to hold a position
    :return: DataFrame with filtered signals
    """
    last_signal_date = None
    last_signal = 0
    
    for i in range(len(signal_data)):
        current_signal = signal_data['signal'].iloc[i]
        
        if last_signal_date is None or (signal_data.index[i] - last_signal_date).days >= min_hold_period:
            # If enough time has passed or no previous signal, allow the signal
            signal_data.loc[signal_data.index[i], 'signal'] = current_signal
            last_signal_date = signal_data.index[i]
            last_signal = current_signal
        else:
            # Otherwise, keep the last signal
            signal_data.loc[signal_data.index[i], 'signal'] = last_signal
    
    return signal_data

async def main():
    symbol = 'BTC/USD'
    historical_data = await fetch_historical_data(symbol)

    preprocessed_data = preprocess_data(historical_data)
    
    # Debug: Print columns to ensure 'sma_20' is included before scaling
    print("Columns before scaling:", preprocessed_data.columns)
    
    # Load your model with original parameters
    model = load_model(LSTMModel)  # Pass the class name here

    if model is None:
        print("Model could not be loaded. Exiting.")
        return

    # Load the scalers
    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')

    # Ensure the feature columns match what the scaler was fitted with
    feature_columns = FEATURE_COLUMNS

    # Scale the data with the correct feature names
    data_to_scale = preprocessed_data[feature_columns]
    scaled_data = feature_scaler.transform(data_to_scale)

    # Explicitly set feature names after scaling
    scaled_df = pd.DataFrame(scaled_data, columns=feature_columns, index=preprocessed_data.index)

    # Detect market regime
    regime = detect_market_regime(preprocessed_data)
    
    # Adapt strategy parameters based on market regime
    if regime == 'Bullish Low Volatility':
        adapted_params = {'rsi_threshold': 60, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 1.5, 'max_risk_pct': 0.02}
    elif regime == 'Bullish High Volatility':
        adapted_params = {'rsi_threshold': 70, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.5, 'max_risk_pct': 0.01}
    elif regime == 'Bearish Low Volatility':
        adapted_params = {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2, 'max_risk_pct': 0.02}
    else:  # Bearish High Volatility
        adapted_params = {'rsi_threshold': 40, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3, 'max_risk_pct': 0.01}
    
    # Generate signals with adapted parameters
    signal_data = generate_signals(scaled_df, model, feature_columns, feature_scaler, target_scaler, **adapted_params)
    
    # Manage risk
    current_balance = 10000  # Example initial balance
    signal_data = manage_risk(signal_data, current_balance)
    
    # Filter signals to reduce frequency
    signal_data = filter_signals(signal_data)

    # Backtest the strategy
    results = backtest_strategy(signal_data)

    # Plot the backtest results
    plot_backtest_results(results)

asyncio.run(main())
