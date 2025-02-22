# src/strategy/backtester.py

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def backtest_strategy(signal_data, initial_capital=10000, transaction_cost=0.001, slippage=0.0005):
    """
    Backtest the trading strategy based on signal data.

    Args:
        signal_data (pd.DataFrame): DataFrame with 'close', 'signal', 'position_size' columns
        initial_capital (float): Starting capital
        transaction_cost (float): Cost per trade (e.g., 0.1% = 0.001)
        slippage (float): Slippage per trade (e.g., 0.05% = 0.0005)
    
    Returns:
        pd.DataFrame: Backtest results with performance metrics
    """
    try:
        if 'signal' not in signal_data.columns:
            raise ValueError("Signal column missing in signal_data")

        portfolio = {'total': initial_capital, 'position': 0, 'cash': initial_capital}
        results = []
        
        for i, row in signal_data.iterrows():
            date = i
            close_price = row['close']
            signal = row['signal']
            position_size = row.get('position_size', 0)  # Default to 0 if not provided
            
            # Adjust price for slippage
            effective_price = close_price * (1 + slippage if signal == 1 else 1 - slippage if signal == -1 else 1)
            
            if signal == 1 and portfolio['position'] == 0:  # Buy
                units = (portfolio['cash'] * position_size) / effective_price
                cost = units * effective_price * (1 + transaction_cost)
                if cost <= portfolio['cash']:
                    portfolio['position'] = units
                    portfolio['cash'] -= cost
                else:
                    logging.warning(f"Insufficient funds on {date}: {cost} > {portfolio['cash']}")
            elif signal == -1 and portfolio['position'] > 0:  # Sell
                proceeds = portfolio['position'] * effective_price * (1 - transaction_cost)
                portfolio['cash'] += proceeds
                portfolio['position'] = 0
            elif signal == -1 and portfolio['position'] == 0:  # Short
                units = (portfolio['cash'] * position_size) / effective_price
                cost = units * effective_price * (1 + transaction_cost)
                if cost <= portfolio['cash']:
                    portfolio['position'] = -units
                    portfolio['cash'] -= cost
            elif signal == 1 and portfolio['position'] < 0:  # Cover short
                proceeds = -portfolio['position'] * effective_price * (1 - transaction_cost)
                portfolio['cash'] += proceeds
                portfolio['position'] = 0
            
            portfolio['total'] = portfolio['cash'] + (portfolio['position'] * close_price)
            
            # Calculate returns
            if not results:
                returns = 0
                cumulative_returns = 0
            else:
                previous_total = results[-1]['total']
                returns = (portfolio['total'] - previous_total) / previous_total if previous_total != 0 else 0
                cumulative_returns = (portfolio['total'] - initial_capital) / initial_capital
            
            results.append({
                'date': date,
                'close': close_price,
                'signal': signal,
                'cash': portfolio['cash'],
                'position': portfolio['position'],
                'position_size': position_size,
                'total': portfolio['total'],
                'returns': returns,
                'cumulative_returns': cumulative_returns * 100  # In percentage
            })
        
        df_results = pd.DataFrame(results).set_index('date')
    
        # Additional metrics
        daily_returns = df_results['returns'].dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
        max_drawdown = (df_results['total'] / df_results['total'].cummax() - 1).min() * 100  # In percentage
        trades = len(df_results[df_results['signal'] != 0])
        wins = len(df_results[(df_results['signal'] == 1) & (df_results['returns'] > 0)]) + len(df_results[(df_results['signal'] == -1) & (df_results['returns'] < 0)])
        losses = trades - wins
        win_loss_ratio = wins / losses if losses > 0 else float('inf')
        
        logging.info(f"Backtest completed. Final portfolio value: {df_results['total'].iloc[-1]:.2f}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}%, Trades: {trades}, Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        return df_results
    
    except Exception as e:
        logging.error(f"Error in backtest: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Dummy test
    dummy_data = pd.DataFrame({
        'close': [100, 101, 102, 101, 100],
        'signal': [0, 1, 0, -1, 0],
        'position_size': [0, 0.5, 0, 0.5, 0]
    }, index=pd.date_range("2024-01-01", periods=5, freq="H"))
    results = backtest_strategy(dummy_data)
    print(results)