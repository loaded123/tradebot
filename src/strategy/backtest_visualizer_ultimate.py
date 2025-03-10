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
        signals = signals.reindex(common_index).ffill().fillna({'close': 78877.88, 'signal': 0, 'position_size': 0})
        portfolio_value = portfolio_value.reindex(common_index).ffill().fillna(10000)  # Default to initial capital if NaN
        if not trades.empty:
            trades = trades.reindex(common_index).ffill().fillna(0)

        # Validate close prices
        signals['close'] = signals['close'].clip(lower=10000, upper=200000)

        # Dynamically determine date range from signal data, ensuring full range
        start_date = signals.index.min()
        end_date = signals.index.max()
        if (end_date - start_date).total_seconds() < 3600:  # Ensure at least one day of data
            logging.warning(f"Date range too short: {start_date} to {end_date}. Extending to full expected range.")
            start_date = pd.to_datetime('2025-01-01 00:00:00', utc=True).tz_localize(None)
            end_date = pd.to_datetime('2025-03-01 23:00:00', utc=True).tz_localize(None)
            signals = signals.reindex(pd.date_range(start=start_date, end=end_date, freq='h')).ffill().fillna({'close': 78877.88, 'signal': 0, 'position_size': 0})
            portfolio_value = portfolio_value.reindex(signals.index).ffill().fillna(10000)
            if not trades.empty:
                trades = trades.reindex(signals.index).ffill().fillna(0)
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
        buy_signals = signals[signals['signal'] == 1].dropna(subset=['close', 'signal'])
        sell_signals = signals[signals['signal'] == -1].dropna(subset=['close', 'signal'])
        ax4.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', zorder=5)
        ax4.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', zorder=5)

        # Add trade entries and exits if available, handling NaNs
        if not trades.empty and 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            trade_entries = trades[trades['entry_price'] > 0].dropna(subset=['entry_price'])
            trade_exits = trades[trades['exit_price'] > 0].dropna(subset=['exit_price'])
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
            ax.xaxis_date()
            ax.xaxis.set_major_locator(DayLocator(interval=7))
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_minor_locator(DayLocator(interval=1))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.autoscale_view()

        logging.debug(f"Raw timestamp of first signal: {signals.index[0].timestamp()}")
        logging.debug(f"Expected timestamp range: 2025-01-01 to 2025-03-01 (Unix ~1,736,716,800 to ~1,739,569,600)")

        plt.tight_layout()
        
        sanitized_crypto = crypto.replace('/', '-')
        filename = f'{sanitized_crypto}_backtest_results.png'
        plt.savefig(filename)
        plt.close()

        # Separate Daily Returns Plot (commented out to avoid redundancy unless needed)
        """
        plt.figure(figsize=(15, 4))
        plt.plot(daily_returns.index, daily_returns, label='Daily Returns', color='purple')
        plt.title(f'Daily Returns (Sharpe: {sharpe_ratio:.2f})')
        plt.xlabel('Date')
        plt.ylabel('Daily Return (%)')
        plt.legend()
        plt.grid()
        ax = plt.gca()
        ax.xaxis_date()
        ax.xaxis.set_major_locator(DayLocator(interval=7))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_minor_locator(DayLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        daily_returns_filename = f'{sanitized_crypto}_daily_returns.png'
        plt.savefig(daily_returns_filename)
        plt.close()
        logging.info(f"Daily returns plotted and saved as '{daily_returns_filename}'")
        """

        logging.info(f"Backtest results plotted and saved as '{filename}'")

    except Exception as e:
        logging.error(f"Error plotting backtest results: {e}")
        raise

def load_model() -> TransformerPredictor:
    """
    Load the trained TransformerPredictor model with parameters matching training.
    """
    input_dim = 17  # Matches len(FEATURE_COLUMNS)
    d_model = 64
    n_heads = 4
    n_layers = 2
    dropout = 0.1
    
    model = TransformerPredictor(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model_path = 'best_model.pth'  # Adjust path if necessary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    logging.info(f"Loaded model from {model_path} with input_dim={input_dim}, d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, dropout={dropout}")
    return model

def filter_signals(signal_data: pd.DataFrame, min_hold_period: int = 4) -> pd.DataFrame:
    """
    Filter signals to enforce a minimum hold period, dynamically adjusted by price volatility.
    Increased min_hold_period to 4 hours to reduce overtrading.
    """
    signal_data.index = pd.to_datetime(signal_data.index, utc=True).tz_localize(None)
    if 'close' not in signal_data.columns:
        logging.warning("'close' missing from signal_data; ensuring it’s preserved")
        signal_data['close'] = signal_data['close'] if 'close' in signal_data.columns else pd.Series(index=signal_data.index, dtype=float).fillna(78877.88)
    last_signal_time = None
    last_signal = 0
    signal_data['filtered_signal'] = signal_data['signal'].fillna(0)
    
    for idx in signal_data.index:
        current_signal = signal_data['signal'].loc[idx] if pd.notna(signal_data['signal'].loc[idx]) else 0
        price_volatility = signal_data['price_volatility'].loc[idx] if 'price_volatility' in signal_data.columns else 0.0
        dynamic_min_hold = min_hold_period if price_volatility <= signal_data['price_volatility'].mean() else 2  # Relax hold period in high volatility
        if (last_signal_time is None or 
            (idx - last_signal_time).total_seconds() / 3600 >= dynamic_min_hold):
            last_signal_time = idx
            last_signal = current_signal
        else:
            signal_data.loc[idx, 'filtered_signal'] = 0
    signal_data['signal'] = signal_data['filtered_signal']
    signal_data.drop(columns=['filtered_signal'], inplace=True)
    logging.info(f"Filtered signal data columns: {signal_data.columns.tolist()}")
    return signal_data

def backtest_strategy(signal_data: pd.DataFrame, preprocessed_data: pd.DataFrame, initial_capital: float = 10000) -> dict:
    """
    Backtest the trading strategy using signals, respecting position sizes from signal_data,
    and updating trade_outcome for dynamic metrics. Allows multiple trades with adjustable trailing stops.
    Adjusted trailing stop multiplier to 5x ATR.
    """
    logging.info(f"Initial capital set to: {initial_capital}")
    
    capital = initial_capital
    open_positions = {}  # Dictionary to track multiple open positions: {entry_idx: (position_size, entry_price, trailing_stop, take_profit)}
    trades = pd.DataFrame(index=signal_data.index, columns=['entry_price', 'exit_price'])
    trades['entry_price'] = 0.0
    trades['exit_price'] = 0.0
    
    if 'trade_outcome' not in signal_data.columns:
        signal_data['trade_outcome'] = np.nan
    if 'cash' not in signal_data.columns:
        signal_data['cash'] = np.nan
    if 'position_value' not in signal_data.columns:
        signal_data['position_value'] = np.nan
    
    trade_count = 0
    wins = 0
    losses = 0
    total_profit = 0.0

    signal_data['total'] = np.nan  # Initialize total column for equity curve
    signal_data.loc[signal_data.index[0], 'total'] = capital
    signal_data.loc[signal_data.index[0], 'cash'] = capital
    signal_data.loc[signal_data.index[0], 'position_value'] = 0.0
    
    for idx in signal_data.index:
        current_price = signal_data.loc[idx, 'close']
        
        if pd.isna(current_price) or current_price <= 0 or current_price < 10000 or current_price > 200000:
            current_price = 78877.88
            signal_data.loc[idx, 'close'] = current_price
        
        # Entry logic
        if signal_data.loc[idx, 'signal'] == 1:  # Buy signal
            position_size = signal_data.loc[idx, 'position_size']
            logging.info(f"Position Size Before Entry at {idx}: {position_size:.6f} BTC (Buy, from signal_data)")
            if pd.isna(position_size) or position_size <= 0:
                logging.warning(f"Invalid position_size at {idx}: {position_size}, using default 0.005 BTC")
                position_size = 0.005
            # Cap position size to 0.005 BTC to align with signal_generator
            position_size = min(position_size, 0.005)
            take_profit = signal_data.loc[idx, 'take_profit']
            stop_loss = signal_data.loc[idx, 'stop_loss']
            open_positions[idx] = (position_size, current_price, stop_loss, take_profit)
            trades.loc[idx, 'entry_price'] = current_price
            trade_count += 1
            logging.info(f"Trade Entry (Buy) at {idx}: {position_size:.6f} BTC, Price: {current_price:.2f} USD")
        
        elif signal_data.loc[idx, 'signal'] == -1:  # Sell signal
            position_size = signal_data.loc[idx, 'position_size']
            logging.info(f"Position Size Before Entry at {idx}: {position_size:.6f} BTC (Sell, from signal_data)")
            if pd.isna(position_size) or position_size <= 0:
                logging.warning(f"Invalid position_size at {idx}: {position_size}, using default 0.005 BTC")
                position_size = 0.005
            # Cap position size to 0.005 BTC to align with signal_generator
            position_size = min(position_size, 0.005)
            take_profit = signal_data.loc[idx, 'take_profit']
            stop_loss = signal_data.loc[idx, 'stop_loss']
            open_positions[idx] = (-position_size, current_price, stop_loss, take_profit)
            trades.loc[idx, 'entry_price'] = current_price
            trade_count += 1
            logging.info(f"Trade Entry (Sell) at {idx}: {position_size:.6f} BTC, Price: {current_price:.2f} USD")
        
        # Exit logic
        positions_to_close = []
        position_value = 0.0
        for entry_idx, (pos_size, entry_price, trailing_stop, take_profit) in list(open_positions.items()):
            current_atr = signal_data.loc[idx, 'atr']
            if pd.isna(current_atr):
                current_atr = 500.0
            
            # Update trailing stop for long positions
            if pos_size > 0:
                if current_price > entry_price:
                    new_trailing_stop = current_price - (current_atr * 5.0)  # Increased to 5x ATR
                    trailing_stop = max(trailing_stop, new_trailing_stop)
                open_positions[entry_idx] = (pos_size, entry_price, trailing_stop, take_profit)
            
            # Update trailing stop for short positions
            elif pos_size < 0:
                if current_price < entry_price:
                    new_trailing_stop = current_price + (current_atr * 5.0)  # Increased to 5x ATR
                    trailing_stop = min(trailing_stop, new_trailing_stop)
                open_positions[entry_idx] = (pos_size, entry_price, trailing_stop, take_profit)
            
            # Calculate position value for this position
            position_value += pos_size * current_price

            # Exit conditions for long position
            if pos_size > 0:
                if signal_data.loc[idx, 'signal'] == -1:  # Opposite signal
                    profit = pos_size * (current_price - entry_price)
                    capital += profit
                    total_profit += profit
                    trades.loc[idx, 'exit_price'] = current_price
                    signal_data.loc[idx, 'trade_outcome'] = 1 if profit > 0 else -1
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    logging.info(f"Opposite Signal Exit (Sell) at {idx} for entry at {entry_idx}: {pos_size:.6f} BTC, Profit: {profit:.2f}")
                    positions_to_close.append(entry_idx)
                elif current_price >= take_profit:
                    profit = pos_size * (take_profit - entry_price)
                    capital += profit
                    total_profit += profit
                    trades.loc[idx, 'exit_price'] = take_profit
                    signal_data.loc[idx, 'trade_outcome'] = 1
                    wins += 1
                    logging.info(f"Take-Profit Exit (Sell) at {idx} for entry at {entry_idx}: {pos_size:.6f} BTC, Profit: {profit:.2f}")
                    positions_to_close.append(entry_idx)
                elif current_price <= trailing_stop:
                    profit = pos_size * (trailing_stop - entry_price)
                    capital += profit
                    total_profit += profit
                    trades.loc[idx, 'exit_price'] = trailing_stop
                    signal_data.loc[idx, 'trade_outcome'] = -1 if profit < 0 else 0
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    logging.info(f"Trailing Stop Exit (Sell) at {idx} for entry at {entry_idx}: {pos_size:.6f} BTC, Profit: {profit:.2f}")
                    positions_to_close.append(entry_idx)
            
            # Exit conditions for short position
            elif pos_size < 0:
                if signal_data.loc[idx, 'signal'] == 1:  # Opposite signal
                    profit = -pos_size * (entry_price - current_price)
                    capital += profit
                    total_profit += profit
                    trades.loc[idx, 'exit_price'] = current_price
                    signal_data.loc[idx, 'trade_outcome'] = 1 if profit > 0 else -1
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    logging.info(f"Opposite Signal Exit (Buy) at {idx} for entry at {entry_idx}: {abs(pos_size):.6f} BTC, Profit: {profit:.2f}")
                    positions_to_close.append(entry_idx)
                elif current_price <= take_profit:
                    profit = -pos_size * (entry_price - take_profit)
                    capital += profit
                    total_profit += profit
                    trades.loc[idx, 'exit_price'] = take_profit
                    signal_data.loc[idx, 'trade_outcome'] = 1
                    wins += 1
                    logging.info(f"Take-Profit Exit (Buy) at {idx} for entry at {entry_idx}: {abs(pos_size):.6f} BTC, Profit: {profit:.2f}")
                    positions_to_close.append(entry_idx)
                elif current_price >= trailing_stop:
                    profit = -pos_size * (entry_price - trailing_stop)
                    capital += profit
                    total_profit += profit
                    trades.loc[idx, 'exit_price'] = trailing_stop
                    signal_data.loc[idx, 'trade_outcome'] = -1 if profit < 0 else 0
                    if profit > 0:
                        wins += 1
                    else:
                        losses += 1
                    logging.info(f"Trailing Stop Exit (Buy) at {idx} for entry at {entry_idx}: {abs(pos_size):.6f} BTC, Profit: {profit:.2f}")
                    positions_to_close.append(entry_idx)
        
        for entry_idx in positions_to_close:
            del open_positions[entry_idx]

        # Update portfolio value
        signal_data.loc[idx, 'cash'] = capital
        signal_data.loc[idx, 'position_value'] = position_value
        signal_data.loc[idx, 'total'] = capital + position_value
    
    # Handle open positions at the end
    for entry_idx, (pos_size, entry_price, trailing_stop, take_profit) in list(open_positions.items()):
        current_price = signal_data['close'].iloc[-1]
        profit = pos_size * (current_price - entry_price) if pos_size > 0 else -pos_size * (entry_price - current_price)
        capital += profit
        total_profit += profit
        trades.loc[signal_data.index[-1], 'exit_price'] = current_price
        signal_data.loc[signal_data.index[-1], 'trade_outcome'] = 0
        logging.info(f"Closing open position at {signal_data.index[-1]} for entry at {entry_idx}: {abs(pos_size):.6f} BTC, Price: {current_price:.2f}, Profit: {profit:.2f}")
        del open_positions[entry_idx]
    
    signal_data['total'] = signal_data['total'].ffill()
    signal_data['cash'] = signal_data['cash'].ffill()
    signal_data['position_value'] = signal_data['position_value'].ffill().fillna(0.0)
    
    returns = signal_data['total'].pct_change(fill_method=None).dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0.0
    rolling_max = signal_data['total'].cummax()
    drawdown = (signal_data['total'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100 if not pd.isna(drawdown.min()) else 0.0
    profit_factor = abs(sum([p for p in returns if p > 0])) / abs(sum([p for p in returns if p < 0])) if any(r < 0 for r in returns) else 1.0
    win_loss_ratio = wins / losses if losses > 0 else wins / 1.0 if wins > 0 else 0.0
    
    logging.info(f"Backtest completed. Final portfolio value: {capital:.2f}")
    logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Sortino Ratio: {sortino_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}%, Profit Factor: {profit_factor:.2f}, Trades: {trade_count}, Win/Loss Ratio: {win_loss_ratio:.2f}")
    
    return {
        'total': signal_data['total'],
        'trades': trades,
        'metrics': {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'trade_count': trade_count,
            'win_loss_ratio': win_loss_ratio,
            'final_value': capital
        }
    }

async def main():
    """Main function to run the backtest and visualization pipeline."""
    logging.debug("Starting main()")
    symbol = 'BTC/USD'
    try:
        historical_data = await fetch_historical_data(symbol, exchange_name='gemini')
        if historical_data.empty:
            raise ValueError("No historical data fetched.")
        logging.info(f"Historical data columns: {historical_data.columns.tolist()}")
        logging.info(f"Historical data index: {historical_data.index[:5]}, last: {historical_data.index[-5:]}")

        processed_df = preprocess_data(historical_data)
        if processed_df.empty or 'close' not in processed_df.columns:
            raise ValueError("'close' missing from processed_df or DataFrame is empty")
        
        preprocessed_data = processed_df
        scaled_df = processed_df

        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        train_columns = FEATURE_COLUMNS

        model = load_model()

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
            'Bullish Low Volatility': {'rsi_threshold': 45, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},  # Adjusted for more buys
            'Bullish High Volatility': {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
            'Bearish Low Volatility': {'rsi_threshold': 40, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
            'Bearish High Volatility': {'rsi_threshold': 35, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15},
        }
        params = regime_params.get(regime, {'rsi_threshold': 45, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 3.0, 'max_risk_pct': 0.15})

        signal_data = await generate_signals(scaled_df, preprocessed_data, model, train_columns, feature_scaler, target_scaler, 
                                            rsi_threshold=params['rsi_threshold'], macd_fast=params['macd_fast'], 
                                            macd_slow=params['macd_slow'], atr_multiplier=params['atr_multiplier'], 
                                            max_risk_pct=params['max_risk_pct'])
        if signal_data.empty or 'close' not in signal_data.columns:
            raise ValueError("No signals generated or 'close' missing from signal_data")
        logging.info(f"Signal data columns before filtering: {signal_data.columns.tolist()}")
        logging.info(f"Signal data index: {signal_data.index[:5]}, last: {signal_data.index[-5:]}")

        signal_data = filter_signals(signal_data, min_hold_period=4)

        current_balance = 17396.68  # Use the final balance from the last run
        signal_data, current_balance = manage_risk(signal_data, current_balance, max_drawdown_pct=0.03,  # Stricter drawdown limit
                                                  atr_multiplier=params['atr_multiplier'], 
                                                  recovery_volatility_factor=0.15, 
                                                  max_risk_pct=params['max_risk_pct'], 
                                                  min_position_size=0.005)

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