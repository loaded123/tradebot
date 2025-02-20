import asyncio
import sys
import os
import logging

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import matplotlib.pyplot as plt
import pandas as pd
import torch
import joblib

from src.models.lstm_model import LSTMModel
from src.strategy.backtester import backtest_strategy
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data
from src.constants import FEATURE_COLUMNS
from src.strategy.strategy_generator import generate_signals
from src.strategy.market_regime import detect_market_regime
from src.strategy.position_sizer import calculate_position_size
from src.strategy.risk_manager import manage_risk

logging.basicConfig(level=logging.INFO)

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

def load_model(model_class):
    model_path = "best_model.pth"
    input_dim = len(FEATURE_COLUMNS)
    hidden_dim = 256  # Or your hidden dimension
    layer_dim = 2     # Or your number of layers

    if os.path.exists(model_path):
        try:
            dummy_model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, layer_dim=layer_dim).to(device)
            dummy_model.load_state_dict(torch.load(model_path, map_location=device))
            logging.info(f"Loaded model from {model_path}")
            model = dummy_model
        except RuntimeError as e:
            logging.warning(f"Architecture mismatch, consider retraining: {e}")
            return None
        except Exception as e:
            logging.warning(f"Error loading model: {e}")
            return None
    else:
        logging.warning(f"Model file not found: {model_path}")
        return None

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

    model = load_model(LSTMModel)
    if model is None:
        logging.error("Model loading failed. Exiting.")
        return

    feature_scaler = joblib.load('feature_scaler.pkl')
    target_scaler = joblib.load('target_scaler.pkl')

    scaled_data = preprocessed_data[FEATURE_COLUMNS]
    scaled_data = feature_scaler.transform(scaled_data)
    scaled_df = pd.DataFrame(scaled_data, columns=FEATURE_COLUMNS, index=preprocessed_data.index)

    regime = detect_market_regime(preprocessed_data)

    regime_params = {
        'Bullish Low Volatility': {'rsi_threshold': 60, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 1.5,
                                  'max_risk_pct': 0.02},
        'Bullish High Volatility': {'rsi_threshold': 70, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.5,
                                   'max_risk_pct': 0.01},
        'Bearish Low Volatility': {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2,
                                  'max_risk_pct': 0.02},
        'Bearish High Volatility': {'rsi_threshold': 40, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3,
                                    'max_risk_pct': 0.01},
    }

    adapted_params = regime_params.get(regime)

    if adapted_params is None:
        logging.warning(f"Unknown market regime: {regime}. Using default parameters.")
        adapted_params = {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2,
                         'max_risk_pct': 0.02}

    signal_data = generate_signals(scaled_df, model, FEATURE_COLUMNS, feature_scaler, target_scaler, **adapted_params)

    current_balance = 10000
    signal_data = manage_risk(signal_data, current_balance)

    signal_data = filter_signals(signal_data)

    results = backtest_strategy(signal_data)

    plot_backtest_results(results)


if __name__ == "__main__":
    asyncio.run(main())