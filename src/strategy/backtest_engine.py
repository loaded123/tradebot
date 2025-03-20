# src/strategy/backtest_engine.py
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from src.strategy.risk_manager import manage_risk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

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

        # Use precomputed take_profit and stop_loss from signal_generator.py
        if signal_data.loc[idx, 'signal'] == 1 and position_size > 0 and pd.notna(take_profit) and pd.notna(stop_loss) and confidence >= 0.1:  # Changed from 0.25 to 0.1
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
        elif signal_data.loc[idx, 'signal'] == -1 and position_size > 0 and pd.notna(take_profit) and pd.notna(stop_loss) and confidence >= 0.1:  # Changed from 0.25 to 0.1
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

def run_backtest(signal_data, preprocessed_data, initial_capital):
    signal_data, current_balance = manage_risk(
        signal_data=signal_data,
        current_balance=initial_capital,
        max_drawdown_pct=0.10,
        atr_multiplier=1.0,
        recovery_volatility_factor=0.15,
        max_risk_pct=0.10,
        min_position_size=0.002
    )
    return backtest_strategy(signal_data, preprocessed_data, initial_capital)