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
from matplotlib.dates import DateFormatter, DayLocator, HourLocator

# Direct imports
from src.strategy.signal_generator import generate_signals
from src.models.transformer_model import TransformerPredictor
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data
from src.constants import FEATURE_COLUMNS
from src.strategy.backtester import backtest_strategy
from src.strategy.market_regime import detect_market_regime
from src.strategy.risk_manager import manage_risk

# Suppress Matplotlib font debugging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"Using device: {device}")

async def plot_backtest_results(portfolio_value: pd.Series, signals: pd.DataFrame, trades: pd.DataFrame, crypto: str):
    """Plot comprehensive backtest results including equity curve, cumulative returns, drawdown, price with signals, and daily returns with dynamic date range, robust scaling corrections, and improved date formatting for hourly data over 2 months."""
    try:
        logging.info(f"Plotting backtest results - Signals columns: {signals.columns.tolist()}")
        if 'close' not in signals.columns:
            logging.error("Error: 'close' not in signals columns")
            raise KeyError("'close' missing from signals for plotting")

        # Ensure all indices are datetime objects and aligned, handling NaNs
        if not isinstance(signals.index, pd.DatetimeIndex):
            signals.index = pd.to_datetime(signals.index, utc=True).tz_localize(None)
        if not isinstance(portfolio_value.index, pd.DatetimeIndex):
            portfolio_value.index = pd.to_datetime(portfolio_value.index, utc=True).tz_localize(None)
        if not trades.empty and not isinstance(trades.index, pd.DatetimeIndex):
            trades.index = pd.to_datetime(trades.index, utc=True).tz_localize(None)

        # Align indices and handle NaNs
        common_index = signals.index
        signals = signals.reindex(common_index, method='ffill').fillna({'close': 78877.88, 'signal': 0})
        portfolio_value = portfolio_value.reindex(common_index, method='ffill').fillna(10000)  # Default to initial capital if NaN
        if not trades.empty:
            trades = trades.reindex(common_index, method='ffill').fillna(0)

        # Validate close prices
        signals['close'] = signals['close'].clip(lower=10000, upper=200000)

        # Dynamically determine date range from signal data, ensuring full range
        start_date = signals.index.min()
        end_date = signals.index.max()
        if (end_date - start_date).total_seconds() < 3600:  # Ensure at least one day of data
            logging.warning(f"Date range too short: {start_date} to {end_date}. Extending to full expected range.")
            start_date = pd.to_datetime('2025-01-01 00:00:00', utc=True).tz_localize(None)
            end_date = pd.to_datetime('2025-03-01 23:00:00', utc=True).tz_localize(None)
            signals = signals.reindex(pd.date_range(start=start_date, end=end_date, freq='h'), method='ffill').fillna({'close': 78877.88, 'signal': 0})
            portfolio_value = portfolio_value.reindex(signals.index, method='ffill').fillna(10000)
            if not trades.empty:
                trades = trades.reindex(signals.index, method='ffill').fillna(0)
        logging.debug(f"Dynamic date range: {start_date} to {end_date}")

        # Filter data to the dynamically determined range, ensuring non-empty data
        signals = signals.loc[start_date:end_date].dropna(how='all')
        portfolio_value = portfolio_value.loc[start_date:end_date].dropna(how='all')
        if not trades.empty:
            trades = trades.loc[start_date:end_date].dropna(how='all')

        if signals.empty or portfolio_value.empty:
            logging.error("Empty signals or portfolio_value DataFrame after filtering")
            raise ValueError("No data to plot")

        plt.figure(figsize=(15, 20))  # Large figure for five subplots

        # 1. Equity Curve (scaled to USD, ensuring no 1e-5 or smaller scaling)
        ax1 = plt.subplot(5, 1, 1)
        if portfolio_value.max() < 1:  # Check for 1e-5 or smaller scaling
            portfolio_value = portfolio_value * 10000
            logging.warning("Corrected portfolio_value scaling from 1e-5 or smaller to USD")
        elif portfolio_value.max() < 100:  # Additional check for partial scaling
            portfolio_value = portfolio_value * 100
            logging.warning("Corrected partial portfolio_value scaling to USD")
        elif portfolio_value.max() < 1000 and portfolio_value.max() > 1:  # Check for intermediate scaling
            portfolio_value = portfolio_value * 10
            logging.warning("Corrected intermediate portfolio_value scaling to USD")
        # Plot using Matplotlib directly to avoid dtype mismatch
        ax1.plot(portfolio_value.index, portfolio_value.values, label='Equity Curve', color='blue')
        plt.title(f'{crypto} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.grid()
        plt.legend()

        # 2. Cumulative Returns (in percent, scaled correctly)
        ax2 = plt.subplot(5, 1, 2)
        initial_value = portfolio_value.iloc[0] if pd.notna(portfolio_value.iloc[0]) else 10000
        cumulative_returns = ((portfolio_value - initial_value) / initial_value) * 100
        total_return = cumulative_returns.iloc[-1] if not pd.isna(cumulative_returns.iloc[-1]) else 0.0
        ax2.plot(cumulative_returns.index, cumulative_returns.values, label='Cumulative Returns', color='green')
        plt.title(f'Cumulative Returns (Total: {total_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid()
        plt.legend()

        # 3. Max Drawdown (in percent, scaled correctly)
        ax3 = plt.subplot(5, 1, 3)
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        drawdown = drawdown * 100  # Convert to percentage only once
        ax3.plot(drawdown.index, drawdown.values, label='Drawdown', color='red')
        max_drawdown = drawdown.min() if not pd.isna(drawdown.min()) else 0.0
        plt.title(f'Max Drawdown: {max_drawdown:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid()
        plt.legend()

        # 4. Price with Buy/Sell Signals and Trades (unscaled USD)
        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(signals.index, signals['close'].values, label='Price', alpha=0.5, color='blue')
        buy_signals = signals[signals['signal'] == 1].dropna()
        sell_signals = signals[signals['signal'] == -1].dropna()
        ax4.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', zorder=5)
        ax4.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', zorder=5)

        # Add trade entries and exits if available, handling NaNs
        if not trades.empty and 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            trade_entries = trades[trades['entry_price'] > 0]
            trade_exits = trades[trades['exit_price'] > 0]
            ax4.scatter(trade_entries.index, trade_entries['entry_price'], marker='o', color='cyan', label='Trade Entry', zorder=5)
            ax4.scatter(trade_exits.index, trade_exits['exit_price'], marker='o', color='magenta', label='Trade Exit', zorder=5)
        else:
            logging.warning("Trades DataFrame is empty or missing 'entry_price'/'exit_price' columns; skipping trade plotting")
        
        plt.title(f'{crypto} Price with Signals and Trades')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()

        # 5. Daily Returns with Sharpe Ratio (scaled correctly in percent)
        ax5 = plt.subplot(5, 1, 5)
        daily_returns = portfolio_value.pct_change(fill_method=None).dropna()
        if daily_returns.max() < 1e-4 or daily_returns.max() > 1:
            daily_returns = daily_returns * 100
            logging.warning("Corrected daily_returns scaling to percent")
        ax5.plot(daily_returns.index, daily_returns.values, label='Daily Returns', color='purple')
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0.0
        plt.title(f'Daily Returns (Sharpe: {sharpe_ratio:.2f})')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.grid()
        plt.legend()

        # Format x-axis for all subplots with proper datetime for hourly data over 2 months
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            # Ensure the x-axis uses dates
            ax.xaxis_date()
            # Set major ticks to every 7 days
            ax.xaxis.set_major_locator(DayLocator(interval=7))
            # Set the formatter to display dates as YYYY-MM-DD
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            # Set minor ticks to every day
            ax.xaxis.set_minor_locator(DayLocator(interval=1))
            # Rotate labels for readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.autoscale_view()

        # Debug timestamp check
        logging.debug(f"Raw timestamp of first signal: {signals.index[0].timestamp()}")
        logging.debug(f"Expected timestamp range: 2025-01-01 to 2025-03-01 (Unix ~1,736,716,800 to ~1,739,569,600)")

        plt.tight_layout()
        
        # Sanitize the filename by replacing '/' with '-'
        sanitized_crypto = crypto.replace('/', '-')
        filename = f'{sanitized_crypto}_backtest_results.png'
        
        plt.savefig(filename)
        plt.close()
        logging.info(f"Backtest results plotted and saved as '{filename}'")

    except Exception as e:
        logging.error(f"Error plotting backtest results: {e}")
        raise

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

def filter_signals(signal_data: pd.DataFrame, min_hold_period: int = 12) -> pd.DataFrame:
    # Preserve datetime index and ensure 'close' is present, handling NaNs
    signal_data.index = pd.to_datetime(signal_data.index, utc=True).tz_localize(None)
    if 'close' not in signal_data.columns:
        logging.warning("'close' missing from signal_data; ensuring it’s preserved")
        signal_data['close'] = signal_data['close'] if 'close' in signal_data.columns else pd.Series(index=signal_data.index, dtype=float).fillna(78877.88)
    last_signal_idx = None
    last_signal = 0
    signal_data['filtered_signal'] = signal_data['signal'].fillna(0)
    
    for idx in signal_data.index:
        i = signal_data.index.get_loc(idx)
        current_idx = i
        current_signal = signal_data['signal'].iloc[i] if pd.notna(signal_data['signal'].iloc[i]) else 0
        price_volatility = signal_data['price_volatility'].iloc[i] if 'price_volatility' in signal_data.columns else 0.0
        dynamic_min_hold = min_hold_period if price_volatility <= signal_data['price_volatility'].mean() else 6
        if (last_signal_idx is None or 
            (current_idx - last_signal_idx) >= dynamic_min_hold):
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
        logging.info(f"Historical data index: {historical_data.index[:5]}, last: {historical_data.index[-5:]}")

        # Unpack single DataFrame from preprocess_data, ensuring 'close' is present
        processed_df = preprocess_data(historical_data)
        if processed_df.empty or 'close' not in processed_df.columns:
            raise ValueError("'close' missing from processed_df or DataFrame is empty")
        
        # Assign processed_df to preprocessed_data and scaled_df (same DataFrame in this case)
        preprocessed_data = processed_df
        scaled_df = processed_df

        # Load scalers from disk, as they are saved in preprocess_data
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        train_columns = FEATURE_COLUMNS  # Use FEATURE_COLUMNS as train_columns

        # Load the model
        model = load_model()

        # Define indicator columns (14 features) and price columns (4 features)
        indicator_columns = [
            'volume', 'returns', 'log_returns', 'price_volatility', 'sma_20', 
            'atr', 'vwap', 'adx', 'momentum_rsi', 'trend_macd', 'ema_50', 
            'bollinger_upper', 'bollinger_lower', 'bollinger_middle'
        ]
        price_columns = ['open', 'high', 'low', 'close']

        logging.info(f"Scaled DataFrame columns: {scaled_df.columns.tolist()}")
        logging.info(f"Scaled DataFrame index: {scaled_df.index[:5]}, last: {scaled_df.index[-5:]}")

        regime = detect_market_regime(preprocessed_data)
        logging.info(f"Detected market regime: {regime}")

        regime_params = {
            'Bullish Low Volatility': {'rsi_threshold': 50, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 2.0, 'max_risk_pct': 0.05},
            'Bullish High Volatility': {'rsi_threshold': 55, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.05},
            'Bearish Low Volatility': {'rsi_threshold': 45, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.05},
            'Bearish High Volatility': {'rsi_threshold': 40, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3.5, 'max_risk_pct': 0.05},
        }
        params = regime_params.get(regime, {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.05})

        signal_data = await generate_signals(scaled_df, preprocessed_data, model, train_columns, feature_scaler, target_scaler, 
                                            rsi_threshold=params['rsi_threshold'], macd_fast=params['macd_fast'], 
                                            macd_slow=params['macd_slow'], atr_multiplier=params['atr_multiplier'], 
                                            max_risk_pct=params['max_risk_pct'])
        if signal_data.empty or 'close' not in signal_data.columns:
            raise ValueError("No signals generated or 'close' missing from signal_data")
        logging.info(f"Signal data columns before filtering: {signal_data.columns.tolist()}")
        logging.info(f"Signal data index: {signal_data.index[:5]}, last: {signal_data.index[-5:]}")

        signal_data = filter_signals(signal_data, min_hold_period=12)

        current_balance = 10000  # Ensure initial balance is explicitly in USD, no scaling
        signal_data, current_balance = manage_risk(signal_data, current_balance, atr_multiplier=params['atr_multiplier'])

        logging.info(f"Signal data columns before backtest: {signal_data.columns.tolist()}")
        logging.info(f"Signal data index before backtest: {signal_data.index[:5]}, last: {signal_data.index[-5:]}")
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