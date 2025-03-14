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
from matplotlib.dates import DateFormatter, DayLocator
import json
from datetime import datetime

# Direct imports
from src.strategy.signal_generator import generate_signals
from src.models.transformer_model import TransformerPredictor
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data, scale_features
from src.constants import FEATURE_COLUMNS
from src.strategy.market_regime import detect_market_regime
from src.strategy.risk_manager import manage_risk

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")

async def plot_backtest_results(portfolio_value: pd.Series, signals: pd.DataFrame, trades: pd.DataFrame, crypto: str):
    """Plot comprehensive backtest results including equity curve, cumulative returns, drawdown, price with signals, and daily returns."""
    try:
        logger.info(f"Plotting backtest results - Signals columns: {signals.columns.tolist()}")
        if 'close' not in signals.columns:
            logger.error("Error: 'close' not in signals columns")
            raise KeyError("'close' missing from signals for plotting")

        signals.index = pd.to_datetime(signals.index, utc=True).tz_localize(None)
        portfolio_value.index = pd.to_datetime(portfolio_value.index, utc=True).tz_localize(None)
        if not trades.empty:
            trades.index = pd.to_datetime(trades.index, utc=True).tz_localize(None)

        common_index = signals.index
        signals = signals.reindex(common_index).ffill().fillna({'close': 78877.88, 'signal': 0, 'position_size': 0})
        portfolio_value = portfolio_value.reindex(common_index).ffill().fillna(10000)
        if not trades.empty:
            trades = trades.reindex(common_index).ffill().fillna(0)

        signals['close'] = signals['close'].clip(lower=10000, upper=200000)

        # Set extended date range
        start_date = pd.to_datetime('2020-01-01 00:00:00', utc=True).tz_localize(None)
        end_date = pd.to_datetime('2025-03-12 23:00:00', utc=True).tz_localize(None)
        if (end_date - start_date).total_seconds() < 3600:
            logger.warning(f"Date range too short: {start_date} to {end_date}. Using extended range.")
        signals = signals.reindex(pd.date_range(start=start_date, end=end_date, freq='h')).ffill().fillna({'close': 78877.88, 'signal': 0, 'position_size': 0})
        portfolio_value = portfolio_value.reindex(signals.index).ffill().fillna(10000)
        if not trades.empty:
            trades = trades.reindex(signals.index).ffill().fillna(0)

        signals = signals.loc[start_date:end_date].dropna(how='all')
        portfolio_value = portfolio_value.loc[start_date:end_date].dropna(how='all')
        if not trades.empty:
            trades = trades.loc[start_date:end_date].dropna(how='all')

        if signals.empty or portfolio_value.empty:
            logger.error("Empty signals or portfolio_value DataFrame after filtering")
            raise ValueError("No data to plot")

        # Calculate MAPE correctly
        if 'predicted_price' in signals.columns:
            valid_mask = (~signals['close'].isna()) & (~signals['predicted_price'].isna())
            mape = np.mean(np.abs((signals.loc[valid_mask, 'close'] - signals.loc[valid_mask, 'predicted_price']) / signals.loc[valid_mask, 'close'])) * 100
            logger.info(f"Mean Absolute Percentage Error (MAPE) of predictions: {mape:.2f}%")
        else:
            logger.warning("Cannot calculate MAPE: 'predicted_price' not in signals")

        # Downsample data for plotting to avoid MAXTICKS exceeded
        signals_ds = signals.resample('4h').mean().dropna()
        portfolio_value_ds = portfolio_value.resample('4h').mean().dropna()
        if not trades.empty:
            trades_ds = trades.resample('4h').mean().dropna()

        plt.figure(figsize=(15, 25))

        # 1. Equity Curve
        ax1 = plt.subplot(5, 1, 1)
        ax1.plot(portfolio_value_ds.index, portfolio_value_ds.values, label='Equity Curve', color='blue')
        plt.title(f'{crypto} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.grid()
        plt.legend()

        # 2. Cumulative Returns
        ax2 = plt.subplot(5, 1, 2)
        initial_value = portfolio_value_ds.iloc[0] if pd.notna(portfolio_value_ds.iloc[0]) else 10000
        cumulative_returns = ((portfolio_value_ds - initial_value) / initial_value) * 100
        total_return = cumulative_returns.iloc[-1] if not pd.isna(cumulative_returns.iloc[-1]) else 0.0
        ax2.plot(cumulative_returns.index, cumulative_returns.values, label='Cumulative Returns', color='green')
        plt.title(f'Cumulative Returns (Total: {total_return:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.grid()
        plt.legend()

        # 3. Max Drawdown
        ax3 = plt.subplot(5, 1, 3)
        rolling_max = portfolio_value_ds.cummax()
        drawdown = (portfolio_value_ds - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min() if not pd.isna(drawdown.min()) else 0.0
        ax3.plot(drawdown.index, drawdown.values, label='Drawdown', color='red')
        plt.title(f'Max Drawdown: {max_drawdown:.2f}%')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid()
        plt.legend()

        # 4. Price with Buy/Sell Signals and Trades
        ax4 = plt.subplot(5, 1, 4)
        ax4.plot(signals_ds.index, signals_ds['close'].values, label='Price', alpha=0.5, color='blue')
        buy_signals = signals[signals['signal'] == 1].dropna(subset=['close', 'signal'])
        sell_signals = signals[signals['signal'] == -1].dropna(subset=['close', 'signal'])
        trades_profit = trades['profit'].dropna()
        top_trades = trades_profit.nlargest(5).index.union(trades_profit.nsmallest(5).index)
        trade_entries = trades.loc[top_trades].dropna(subset=['entry_price'])
        trade_exits = trades.loc[top_trades].dropna(subset=['exit_price'])
        ax4.scatter(buy_signals.index, signals.loc[buy_signals.index, 'close'], marker='^', color='green', label='Buy Signal', zorder=5, alpha=0.5)
        ax4.scatter(sell_signals.index, signals.loc[sell_signals.index, 'close'], marker='v', color='red', label='Sell Signal', zorder=5, alpha=0.5)
        ax4.scatter(trade_entries.index, trade_entries['entry_price'], marker='o', color='cyan', label='Trade Entry', zorder=5, alpha=0.5)
        ax4.scatter(trade_exits.index, trade_exits['exit_price'], marker='o', color='magenta', label='Trade Exit', zorder=5, alpha=0.5)

        # Add new signal markers
        if 'luxalgo_signal' in signals.columns:
            luxalgo_buys = signals[signals['luxalgo_signal'] == 1].dropna(subset=['close', 'luxalgo_signal'])
            luxalgo_sells = signals[signals['luxalgo_signal'] == -1].dropna(subset=['close', 'luxalgo_signal'])
            ax4.scatter(luxalgo_buys.index, signals.loc[luxalgo_buys.index, 'close'], marker='s', color='lime', label='LuxAlgo Buy', zorder=5, alpha=0.3)
            ax4.scatter(luxalgo_sells.index, signals.loc[luxalgo_sells.index, 'close'], marker='s', color='pink', label='LuxAlgo Sell', zorder=5, alpha=0.3)

        if 'trendspider_signal' in signals.columns:
            trendspider_buys = signals[signals['trendspider_signal'] == 1].dropna(subset=['close', 'trendspider_signal'])
            trendspider_sells = signals[signals['trendspider_signal'] == -1].dropna(subset=['close', 'trendspider_signal'])
            ax4.scatter(trendspider_buys.index, signals.loc[trendspider_buys.index, 'close'], marker='D', color='cyan', label='TrendSpider Buy', zorder=5, alpha=0.3)
            ax4.scatter(trendspider_sells.index, signals.loc[trendspider_sells.index, 'close'], marker='D', color='orange', label='TrendSpider Sell', zorder=5, alpha=0.3)

        if 'smrt_signal' in signals.columns:
            smrt_buys = signals[signals['smrt_signal'] == 1].dropna(subset=['close', 'smrt_signal'])
            smrt_sells = signals[signals['smrt_signal'] == -1].dropna(subset=['close', 'smrt_signal'])
            ax4.scatter(smrt_buys.index, signals.loc[smrt_buys.index, 'close'], marker='*', color='purple', label='SMRT Buy', zorder=5, alpha=0.3)
            ax4.scatter(smrt_sells.index, signals.loc[smrt_sells.index, 'close'], marker='*', color='brown', label='SMRT Sell', zorder=5, alpha=0.3)

        if 'arbitrage_signal' in signals.columns:
            arbitrage_buys = signals[signals['arbitrage_signal'] == 1].dropna(subset=['close', 'arbitrage_signal'])
            arbitrage_sells = signals[signals['arbitrage_signal'] == -1].dropna(subset=['close', 'arbitrage_signal'])
            ax4.scatter(arbitrage_buys.index, signals.loc[arbitrage_buys.index, 'close'], marker='P', color='teal', label='Arbitrage Buy', zorder=5, alpha=0.3)
            ax4.scatter(arbitrage_sells.index, signals.loc[arbitrage_sells.index, 'close'], marker='P', color='gold', label='Arbitrage Sell', zorder=5, alpha=0.3)

        for idx in top_trades:
            if pd.notna(trades.loc[idx, 'entry_price']) and pd.notna(trades.loc[idx, 'exit_price']):
                entry_price = trades.loc[idx, 'entry_price']
                exit_price = trades.loc[idx, 'exit_price']
                profit = (exit_price - entry_price) * signals.loc[idx, 'position_size'] if signals.loc[idx, 'signal'] == 1 else (entry_price - exit_price) * abs(signals.loc[idx, 'position_size'])
                color = 'green' if profit > 0 else 'red'
                ax4.text(idx, exit_price, f'P/L: {profit:.2f}', color=color, fontsize=8, rotation=45)

        plt.title(f'{crypto} Price with Signals and Trades')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid()

        # 5. Cumulative P/L
        ax5 = plt.subplot(5, 1, 5)
        initial_value = portfolio_value_ds.iloc[0]
        cumulative_pl = (portfolio_value_ds - initial_value).cumsum()
        ax5.plot(cumulative_pl.index, cumulative_pl.values, label='Cumulative P/L', color='orange')
        plt.title('Cumulative Profit/Loss')
        plt.xlabel('Date')
        plt.ylabel('Profit/Loss (USD)')
        plt.grid()
        plt.legend()

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.xaxis_date()
            ax.xaxis.set_major_locator(DayLocator(interval=30))
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_minor_locator(DayLocator(interval=7))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.autoscale_view()

        plt.tight_layout()
        sanitized_crypto = crypto.replace('/', '-')
        filename = f'{sanitized_crypto}_backtest_results.png'
        plt.savefig(filename, dpi=300)
        plt.close()
        logger.info(f"Backtest results plotted and saved as '{filename}'")

    except Exception as e:
        logger.error(f"Error plotting backtest results: {e}")
        raise

def load_model() -> TransformerPredictor:
    """Load the trained TransformerPredictor model."""
    input_dim = len(FEATURE_COLUMNS)
    d_model = 128
    n_heads = 8
    n_layers = 4
    dropout = 0.7
    
    model = TransformerPredictor(input_dim=input_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_path = 'best_model.pth'
    state_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info(f"Loaded model from {model_path} with partial state dict matching")
    return model

def filter_signals(signal_data: pd.DataFrame, min_hold_period: int = 6) -> pd.DataFrame:
    """Filter signals to enforce a minimum hold period, using confidence."""
    signal_data.index = pd.to_datetime(signal_data.index, utc=True).tz_localize(None)
    last_signal_time = None
    signal_data['filtered_signal'] = signal_data['signal'].fillna(0)
    for idx in signal_data.index:
        current_signal = signal_data.loc[idx, 'signal']
        confidence = signal_data.loc[idx, 'signal_confidence'] if 'signal_confidence' in signal_data.columns else 0.0
        price_volatility = signal_data.loc[idx, 'price_volatility'] if 'price_volatility' in signal_data.columns else 0.0
        dynamic_min_hold = min_hold_period if price_volatility <= signal_data['price_volatility'].mean() else 2
        min_confidence = 0.15
        
        if (last_signal_time is None or (idx - last_signal_time).total_seconds() / 3600 >= dynamic_min_hold) and \
           current_signal != 0 and confidence >= min_confidence:
            last_signal_time = idx
            logger.debug(f"Accepted signal at {idx} with confidence {confidence:.2f}")
        else:
            signal_data.loc[idx, 'filtered_signal'] = 0
            if current_signal != 0:
                logger.debug(f"Filtered signal at {idx}: Min hold ({dynamic_min_hold} hours), Confidence ({confidence:.2f} < {min_confidence})")

    signal_data['signal'] = signal_data['filtered_signal']
    signal_data.drop(columns=['filtered_signal'], inplace=True)
    logger.info(f"Filtered signal data columns: {signal_data.columns.tolist()}")
    return signal_data

def backtest_strategy(signal_data: pd.DataFrame, preprocessed_data: pd.DataFrame, initial_capital: float = 10000) -> dict:
    """Backtest the trading strategy with precomputed trade levels and confidence-based filtering."""
    logger.info(f"Initial capital set to: {initial_capital}")
    capital = initial_capital
    open_positions = {}
    trades_df = pd.DataFrame(columns=['timestamp', 'symbol', 'trade_type', 'size', 'entry_price', 'exit_price', 'profit'])
    trades = pd.DataFrame(index=signal_data.index, columns=['entry_price', 'exit_price', 'profit'])
    trades[['entry_price', 'exit_price', 'profit']] = 0.0

    if 'trade_outcome' not in signal_data.columns:
        signal_data['trade_outcome'] = np.nan
    if 'cash' not in signal_data.columns:
        signal_data['cash'] = np.nan
    if 'position_value' not in signal_data.columns:
        signal_data['position_value'] = np.nan
    if 'total' not in signal_data.columns:
        signal_data['total'] = np.nan
    if 'portfolio_value' not in signal_data.columns:
        signal_data['portfolio_value'] = np.nan

    signal_data.loc[signal_data.index[0], 'cash'] = capital
    signal_data.loc[signal_data.index[0], 'total'] = capital
    signal_data.loc[signal_data.index[0], 'portfolio_value'] = capital

    trade_count = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    trade_history = []

    for idx in signal_data.index:
        current_price = signal_data.loc[idx, 'close']
        current_atr = signal_data.loc[idx, 'atr'] if 'atr' in signal_data.columns else 500.0
        if pd.isna(current_price) or current_price <= 0 or current_price < 10000 or current_price > 200000:
            current_price = 78877.88
            signal_data.loc[idx, 'close'] = current_price
        if pd.isna(current_atr):
            current_atr = 500.0

        position_size = signal_data.loc[idx, 'position_size']
        take_profit = signal_data.loc[idx, 'take_profit']
        stop_loss = signal_data.loc[idx, 'stop_loss']
        trailing_stop = signal_data.loc[idx, 'trailing_stop'] if 'trailing_stop' in signal_data.columns else np.nan
        confidence = signal_data.loc[idx, 'signal_confidence'] if 'signal_confidence' in signal_data.columns else 0.0

        if signal_data.loc[idx, 'signal'] == 1 and position_size > 0 and pd.notna(take_profit) and pd.notna(stop_loss) and confidence >= 0.15:
            open_positions[idx] = (position_size, current_price, stop_loss, take_profit, trailing_stop, False)
            trades.loc[idx, 'entry_price'] = current_price
            trade_history.append({
                'timestamp': idx.isoformat(),
                'symbol': 'BTC/USD',
                'trade_type': 'Buy',
                'size': position_size,
                'entry_price': current_price,
                'exit_price': None,
                'profit': None
            })
            trade_count += 1
            logger.info(f"Trade Entry (Buy) at {idx}: {position_size:.6f} BTC, Price: {current_price:.2f} USD, Confidence: {confidence:.2f}")
        elif signal_data.loc[idx, 'signal'] == -1 and position_size > 0 and pd.notna(take_profit) and pd.notna(stop_loss) and confidence >= 0.15:
            open_positions[idx] = (-position_size, current_price, stop_loss, take_profit, trailing_stop, False)
            trades.loc[idx, 'entry_price'] = current_price
            trade_history.append({
                'timestamp': idx.isoformat(),
                'symbol': 'BTC/USD',
                'trade_type': 'Sell',
                'size': position_size,
                'entry_price': current_price,
                'exit_price': None,
                'profit': None
            })
            trade_count += 1
            logger.info(f"Trade Entry (Sell) at {idx}: {position_size:.6f} BTC, Price: {current_price:.2f} USD, Confidence: {confidence:.2f}")

        position_value = 0.0
        for pos in open_positions.values():
            pos_size = pos[0]
            position_value += pos_size * current_price
        signal_data.loc[idx, 'cash'] = capital
        signal_data.loc[idx, 'position_value'] = position_value
        signal_data.loc[idx, 'total'] = capital + position_value
        signal_data.loc[idx, 'portfolio_value'] = capital + position_value

        positions_to_close = []
        for entry_idx, (pos_size, entry_price, fixed_stop, take_profit, trailing_stop, trailing_active) in list(open_positions.items()):
            if pd.isna(trailing_stop):
                trailing_stop = fixed_stop

            if pos_size > 0:
                if current_price >= entry_price + current_atr and pd.notna(trailing_stop):
                    trailing_active = True
                if trailing_active and pd.notna(trailing_stop):
                    trailing_stop = max(trailing_stop, current_price - current_atr)
                effective_stop = fixed_stop if not trailing_active else min(fixed_stop, trailing_stop)
                exit_condition = current_price >= take_profit or current_price <= effective_stop
                exit_reason = "Take-Profit" if current_price >= take_profit else "Stop-Loss/Trailing Stop"
            else:
                if current_price <= entry_price - current_atr and pd.notna(trailing_stop):
                    trailing_active = True
                if trailing_active and pd.notna(trailing_stop):
                    trailing_stop = min(trailing_stop, current_price + current_atr)
                effective_stop = fixed_stop if not trailing_active else max(fixed_stop, trailing_stop)
                exit_condition = current_price <= take_profit or current_price >= effective_stop
                exit_reason = "Take-Profit" if current_price <= take_profit else "Stop-Loss/Trailing Stop"

            if exit_condition:
                if pos_size > 0:
                    profit = pos_size * (current_price - entry_price)
                else:
                    profit = -pos_size * (entry_price - current_price)
                capital += profit
                trades.loc[idx, 'exit_price'] = current_price
                trades.loc[idx, 'profit'] = profit
                signal_data.loc[idx, 'trade_outcome'] = 1 if profit > 0 else -1 if profit < 0 else 0
                if profit > 0:
                    wins += 1
                elif profit < 0:
                    losses += 1
                total_profit += profit
                logger.info(f"{exit_reason} Exit ({'Sell' if pos_size > 0 else 'Buy'}) at {idx} for entry at {entry_idx}: Profit: {profit:.2f}")
                positions_to_close.append(entry_idx)
                # Update trade history
                for trade in trade_history:
                    if trade['timestamp'] == entry_idx.isoformat() and trade['exit_price'] is None:
                        trade['exit_price'] = current_price
                        trade['profit'] = profit
                        break

            open_positions[entry_idx] = (pos_size, entry_price, fixed_stop, take_profit, trailing_stop, trailing_active)
            signal_data.loc[idx, 'trailing_stop'] = trailing_stop if entry_idx in open_positions else np.nan

        for entry_idx in positions_to_close:
            del open_positions[entry_idx]

    for entry_idx, (pos_size, entry_price, fixed_stop, take_profit, trailing_stop, trailing_active) in list(open_positions.items()):
        current_price = signal_data['close'].iloc[-1]
        if pos_size > 0:
            profit = pos_size * (current_price - entry_price)
        else:
            profit = -pos_size * (entry_price - current_price)
        capital += profit
        trades.loc[signal_data.index[-1], 'exit_price'] = current_price
        trades.loc[signal_data.index[-1], 'profit'] = profit
        signal_data.loc[signal_data.index[-1], 'trade_outcome'] = 0
        total_profit += profit
        signal_data.loc[signal_data.index[-1], 'portfolio_value'] = capital
        logger.info(f"Closing open position at {signal_data.index[-1]} for entry at {entry_idx}: Profit: {profit:.2f}")
        # Update trade history
        for trade in trade_history:
            if trade['timestamp'] == entry_idx.isoformat() and trade['exit_price'] is None:
                trade['exit_price'] = current_price
                trade['profit'] = profit
                break
        del open_positions[entry_idx]

    signal_data['total'] = signal_data['total'].ffill()
    signal_data['cash'] = signal_data['cash'].ffill()
    signal_data['position_value'] = signal_data['position_value'].ffill().fillna(0.0)
    signal_data['portfolio_value'] = signal_data['portfolio_value'].ffill()

    returns = signal_data['total'].pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24) if returns.std() != 0 else 0.0
    downside_returns = returns[returns < 0]
    sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252 * 24) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0.0
    rolling_max = signal_data['total'].cummax()
    drawdown = (signal_data['total'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100 if not pd.isna(drawdown.min()) else 0.0
    profit_factor = abs(sum([p for p in trades['profit'] if p > 0])) / abs(sum([p for p in trades['profit'] if p < 0])) if any(p < 0 for p in trades['profit']) else float('inf')
    win_loss_ratio = wins / losses if losses > 0 else wins / 1.0 if wins > 0 else 0.0

    logger.info(f"Backtest completed. Final portfolio value: {capital:.2f}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Sortino Ratio: {sortino_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}%, "
                f"Profit Factor: {profit_factor:.2f}, Trades: {trade_count}, Win/Loss Ratio: {win_loss_ratio:.2f}")

    # Save results for dashboard
    performance = {
        'pnl': capital - initial_capital,
        'num_trades': trade_count,
        'win_rate': win_loss_ratio / (win_loss_ratio + 1) if win_loss_ratio > 0 else 0.0,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'last_update': datetime.now().isoformat(),
        'pnl_timestamps': signal_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'pnl_values': (signal_data['total'] - initial_capital).tolist()
    }
    with open("backtest_results.json", "w") as f:
        json.dump(performance, f)
    logger.info("Backtest results saved to backtest_results.json")

    # Save trade data
    with open("trade_data.json", "w") as f:
        json.dump({'trades': trade_history}, f)
    logger.info("Trade data saved to trade_data.json")

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
    logger.debug("Starting main()")
    symbol = 'BTC/USD'
    csv_path = r"C:\Users\Dennis\.vscode\tradebot\src\data\btc_usd_historical.csv"
    logger.info(f"Attempting to load data from CSV path: {csv_path}")

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}. Please ensure the file exists.")
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    try:
        historical_data = await fetch_historical_data(symbol, csv_path=csv_path)
        if historical_data.empty:
            raise ValueError("No historical data fetched.")
        logger.info(f"Historical data columns: {historical_data.columns.tolist()}")
        logger.info(f"Historical data index: {historical_data.index[:5]}, last: {historical_data.index[-5:]}")
        logger.info(f"Data range: {historical_data.index.min()} to {historical_data.index.max()}")

        preprocessed_data = preprocess_data(historical_data)
        if preprocessed_data.empty or 'close' not in preprocessed_data.columns:
            raise ValueError("'close' missing from preprocessed_df or DataFrame is empty")
        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        logger.info(f"Preprocessed columns: {preprocessed_data.columns.tolist()}")

        # Add and scale the target column
        preprocessed_data['target'] = preprocessed_data['log_returns']
        scaled_df = scale_features(preprocessed_data, FEATURE_COLUMNS + ['target'])
        logger.info(f"Scaled data shape: {scaled_df.shape}")
        logger.info(f"Scaled columns: {scaled_df.columns.tolist()}")

        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        train_columns = FEATURE_COLUMNS

        model = load_model()

        regime = detect_market_regime(preprocessed_data)
        logger.info(f"Detected market regime: {regime}")

        regime_params = {
            'Bullish Low Volatility': {'rsi_threshold': 30, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 2.0, 'max_risk_pct': 0.10},
            'Bullish High Volatility': {'rsi_threshold': 35, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.0, 'max_risk_pct': 0.10},
            'Bearish Low Volatility': {'rsi_threshold': 25, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.0, 'max_risk_pct': 0.10},
            'Bearish High Volatility': {'rsi_threshold': 20, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 2.0, 'max_risk_pct': 0.10},
            'Neutral': {'rsi_threshold': 30, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.0, 'max_risk_pct': 0.08}
        }
        params = regime_params.get(regime, {'rsi_threshold': 30, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.0, 'max_risk_pct': 0.10})

        logger.info("Entering generate_signals function")
        signal_data = await generate_signals(
            scaled_df,
            preprocessed_data,
            model,
            train_columns,
            feature_scaler,
            target_scaler,
            rsi_threshold=params['rsi_threshold'],
            macd_fast=params['macd_fast'],
            macd_slow=params['macd_slow'],
            atr_multiplier=params['atr_multiplier'],
            max_risk_pct=params['max_risk_pct']
        )
        if signal_data.empty or 'close' not in signal_data.columns:
            raise ValueError("No signals generated or 'close' missing from signal_data")
        logger.info(f"Signal data columns before filtering: {signal_data.columns.tolist()}")
        logger.info(f"Signal data index: {signal_data.index[:5]}, last: {signal_data.index[-5:]}")

        signal_data = filter_signals(signal_data, min_hold_period=6)

        current_balance = 17396.68
        signal_data, current_balance = manage_risk(signal_data, current_balance, max_drawdown_pct=0.10,
                                                  atr_multiplier=params['atr_multiplier'],
                                                  recovery_volatility_factor=0.15,
                                                  max_risk_pct=params['max_risk_pct'],
                                                  min_position_size=0.002)

        logger.info(f"Signal data columns before backtest: {signal_data.columns.tolist()}")
        logger.info(f"Signal data index before backtest: {signal_data.index[:5]}, last: {signal_data.index[-5:]}")
        results = backtest_strategy(signal_data, preprocessed_data, initial_capital=current_balance)
        await plot_backtest_results(results['total'], signal_data, results['trades'], symbol)

    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise

if __name__ == "__main__":
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
    logger.info(f"sys.path: {sys.path}")
    asyncio.run(main())