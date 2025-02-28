# src/strategy/backtest_visualizer_ultimate.py
import asyncio
import sys
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch
import joblib
import numpy as np
import importlib
from matplotlib.dates import DateFormatter

# Suppress Matplotlib font debugging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Force reload of signal_generator
sys.modules.pop('src.strategy.signal_generator', None)
signal_generator = importlib.import_module('src.strategy.signal_generator')
generate_signals = signal_generator.generate_signals

from src.models.transformer_model import TransformerPredictor  # Ensure using TransformerPredictor
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data
from src.constants import FEATURE_COLUMNS
from src.strategy.backtester import backtest_strategy
from src.strategy.market_regime import detect_market_regime
from src.strategy.risk_manager import manage_risk

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}")

async def plot_backtest_results(portfolio_value: pd.Series, signals: pd.DataFrame, trades: pd.DataFrame, crypto: str):
    """Plot comprehensive backtest results including equity curve, cumulative returns, drawdown, price with signals, and daily returns."""
    try:
        logging.info(f"Plotting backtest results - Signals columns: {signals.columns.tolist()}")
        if 'close' not in signals.columns:
            logging.error("Error: 'close' not in signals columns")
            raise KeyError("'close' missing from signals for plotting")

        # Convert Unix timestamps to datetime objects
        portfolio_value.index = pd.to_datetime(portfolio_value.index, unit='s', utc=True).tz_localize(None)
        signals.index = pd.to_datetime(signals.index, unit='s', utc=True).tz_localize(None)
        if not trades.empty:
            trades.index = pd.to_datetime(trades.index, unit='s', utc=True).tz_localize(None)

        plt.figure(figsize=(15, 20))  # Large figure for five subplots

        # 1. Equity Curve
        plt.subplot(5, 1, 1)
        portfolio_value.plot(label='Equity Curve')
        plt.title(f'{crypto} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.grid()
        plt.legend()

        # 2. Cumulative Returns
        plt.subplot(5, 1, 2)
        daily_returns = portfolio_value.pct_change().dropna()
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        cumulative_returns.plot(label='Cumulative Returns', color='green')
        total_return = cumulative_returns.iloc[-1] * 100
        plt.title(f'Cumulative Returns (Total: {total_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid()
        plt.legend()

        # 3. Max Drawdown
        plt.subplot(5, 1, 3)
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max * 100
        drawdown.plot(label='Drawdown', color='red')
        max_drawdown = drawdown.min()
        plt.title(f'Max Drawdown: {max_drawdown:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid()
        plt.legend()

        # 4. Price with Buy/Sell Signals and Trades
        plt.subplot(5, 1, 4)
        signals['close'].plot(label='Price', alpha=0.5)
        signals[signals['signal'] == 1]['close'].plot(marker='^', linestyle='None', color='g', label='Buy Signal')
        signals[signals['signal'] == -1]['close'].plot(marker='v', linestyle='None', color='r', label='Sell Signal')
        if not trades.empty and 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            trades['entry_price'].plot(marker='o', linestyle='None', color='b', label='Trade Entry')
            trades['exit_price'].plot(marker='o', linestyle='None', color='y', label='Trade Exit')
        else:
            logging.warning("Trades DataFrame is empty or missing 'entry_price'/'exit_price' columns; skipping trade plotting")
        
        plt.title(f'{crypto} Price with Signals and Trades')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()

        # 5. Daily Returns with Sharpe Ratio
        plt.subplot(5, 1, 5)
        daily_returns.plot(label='Daily Returns', color='purple')
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
        plt.title(f'Daily Returns (Sharpe: {sharpe_ratio:.2f})')
        plt.xlabel('Date')
        plt.ylabel('Daily Return')
        plt.grid()
        plt.legend()

        # Format x-axis for all subplots
        for ax in plt.gcf().axes:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit ticks for readability
            ax.autoscale_view()

        # Debug timestamp check
        logging.debug(f"Raw timestamp of first signal: {signals.index[0].timestamp()}")
        logging.debug(f"Expected timestamp range: 2024-12-28 to 2025-02-28 (Unix ~1,703,735,200 to ~1,706,688,800)")

        plt.tight_layout()
        
        # Sanitize the filename by replacing '/' with '-'
        sanitized_crypto = crypto.replace('/', '-')
        filename = f'{sanitized_crypto}_backtest_results.png'
        
        plt.savefig(filename)
        plt.close()
        logging.info(f"Backtest results plotted and saved as '{filename}'")

        # Note: Trading fees from Gemini (maker 0.00%–0.25%, taker 0.10%–0.40%) are not currently implemented.
        # Consider adding fees to backtest_strategy or risk_manager.py if performance improves (e.g., final value > 12,000 USD).
        # Fees would reduce final portfolio value (e.g., 9,980.12 USD to ~9,882–9,902 USD with 1,258 trades at 0.2% fee).

    except Exception as e:
        logging.error(f"Error plotting backtest results: {e}")

def load_model() -> TransformerPredictor:
    """
    Load the trained TransformerPredictor model with parameters matching training.
    """
    # Define parameters matching the training script
    input_dim = 17  # Matches len(FEATURE_COLUMNS)
    d_model = 64
    n_heads = 4
    n_layers = 2
    dropout = 0.1
    
    # Initialize the model
    model = TransformerPredictor(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )
    
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load the trained model's state dictionary
    model_path = 'best_model.pth'  # Adjust path if necessary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    # Set to evaluation mode
    model.eval()
    
    logging.info(f"Loaded model from {model_path} with input_dim={input_dim}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, dropout={dropout}")
    return model

def filter_signals(signal_data: pd.DataFrame, min_hold_period: int = 5) -> pd.DataFrame:
    # Preserve datetime index and ensure 'close' is present
    signal_data.index = pd.to_datetime(signal_data.index, unit='s', utc=True).tz_localize(None)  # Convert Unix timestamps
    if 'close' not in signal_data.columns:
        logging.warning("'close' missing from signal_data; ensuring it’s preserved")
        signal_data['close'] = signal_data['close'] if 'close' in signal_data.columns else pd.Series(index=signal_data.index)
    last_signal_idx = None
    last_signal = 0
    signal_data['filtered_signal'] = signal_data['signal']
    
    for idx in signal_data.index:
        i = signal_data.index.get_loc(idx)  # Get integer position for .iloc
        current_idx = i
        current_signal = signal_data['signal'].iloc[i]
        if (last_signal_idx is None or 
            (current_idx - last_signal_idx) >= min_hold_period):
            last_signal_idx = i
            last_signal = current_signal
        else:
            signal_data.loc[idx, 'filtered_signal'] = last_signal
    signal_data['signal'] = signal_data['filtered_signal']
    signal_data.drop(columns=['filtered_signal'], inplace=True)
    logging.info(f"Filtered signal data columns: {signal_data.columns.tolist()}")
    return signal_data

async def main():
    logging.debug("Starting main()")
    symbol = 'BTC/USD'
    try:
        historical_data = await fetch_historical_data(symbol, exchange_name='gemini')
        if historical_data.empty:
            raise ValueError("No historical data fetched.")
        logging.info(f"Historical data columns: {historical_data.columns.tolist()}")

        preprocessed_data = preprocess_data(historical_data)
        if 'close' not in preprocessed_data.columns:
            raise ValueError("'close' missing from preprocessed_data")

        model = load_model()
        if model is None:
            raise ValueError("Model loading failed.")

        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        train_columns = FEATURE_COLUMNS  # 17 features
        
        # Define indicator columns (14 features) and price columns (4 features)
        indicator_columns = [
            'volume', 'returns', 'log_returns', 'price_volatility', 'sma_20', 
            'atr', 'vwap', 'adx', 'momentum_rsi', 'trend_macd', 'ema_50', 
            'bollinger_upper', 'bollinger_lower', 'bollinger_middle'
        ]
        price_columns = ['open', 'high', 'low', 'close']

        # Scale only the indicators
        scaled_indicators = feature_scaler.transform(preprocessed_data[indicator_columns].values)
        scaled_df = pd.DataFrame(scaled_indicators, columns=indicator_columns, index=preprocessed_data.index)

        # Add the unscaled price columns back to the DataFrame, ensuring 'close' is preserved
        scaled_df[price_columns] = preprocessed_data[price_columns].copy()
        scaled_df['close'] = preprocessed_data['close'].copy()  # Explicitly ensure 'close' is included

        # Scale the target separately if needed
        scaled_df['target'] = target_scaler.transform(preprocessed_data[['target']])

        # Add additional preprocessed columns (unscaled) for consistency with signals and trades, ensuring 'close' and other prices are not lost
        scaled_df['close'] = preprocessed_data['close'].copy()  # Reaffirm 'close' presence
        scaled_df['high'] = preprocessed_data['high'].copy()
        scaled_df['low'] = preprocessed_data['low'].copy()
        scaled_df['price_volatility'] = preprocessed_data['price_volatility']
        scaled_df['sma_20'] = preprocessed_data['sma_20']
        scaled_df['adx'] = preprocessed_data['adx']
        scaled_df['vwap'] = preprocessed_data['vwap']
        scaled_df['atr'] = preprocessed_data['atr']

        logging.info(f"Scaled DataFrame columns: {scaled_df.columns.tolist()}")

        regime = detect_market_regime(preprocessed_data)
        logging.info(f"Detected market regime: {regime}")

        regime_params = {
            'Bullish Low Volatility': {'rsi_threshold': 60, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 1.5, 'max_risk_pct': 0.01},
            'Bullish High Volatility': {'rsi_threshold': 70, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.5, 'max_risk_pct': 0.01},
            'Bearish Low Volatility': {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2, 'max_risk_pct': 0.01},
            'Bearish High Volatility': {'rsi_threshold': 40, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3, 'max_risk_pct': 0.01},
        }
        params = regime_params.get(regime, {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2, 'max_risk_pct': 0.01})

        signal_data = await generate_signals(scaled_df, preprocessed_data, model, train_columns, feature_scaler, target_scaler, 
                                            rsi_threshold=params['rsi_threshold'], macd_fast=params['macd_fast'], 
                                            macd_slow=params['macd_slow'], atr_multiplier=params['atr_multiplier'], 
                                            max_risk_pct=params['max_risk_pct'])
        if signal_data.empty:
            raise ValueError("No signals generated")
        if 'close' not in signal_data.columns:
            signal_data['close'] = preprocessed_data['close'].copy()  # Ensure 'close' is added if missing
            logging.warning("'close' missing from signal_data; added from preprocessed_data")
        logging.info(f"Signal data columns before filtering: {signal_data.columns.tolist()}")
        logging.info(f"Signal data index: {signal_data.index[:5]}")  # Log first few index values for debugging

        signal_data = filter_signals(signal_data, min_hold_period=5)

        current_balance = 10000
        signal_data = manage_risk(signal_data, current_balance, atr_multiplier=params['atr_multiplier'])

        logging.info(f"Signal data columns before backtest: {signal_data.columns.tolist()}")
        logging.info(f"Signal data index before backtest: {signal_data.index[:5]}")  # Log index for debugging
        results = backtest_strategy(signal_data, preprocessed_data, initial_capital=current_balance)
        await plot_backtest_results(results['total'], signal_data, results['trades'], symbol)

    except Exception as e:
        logging.error(f"Error in backtest: {e}")
        raise

if __name__ == "__main__":
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
    logging.info(f"sys.path: {sys.path}")
    asyncio.run(main())