# src/strategy/backtester.py

import pandas as pd
import logging
import math  # For sqrt

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def backtest_strategy(signal_data, preprocessed_data: pd.DataFrame, initial_capital=10000, transaction_cost=0.001, slippage=0.0005):
    """
    Backtest the trading strategy based on signal data using unscaled prices.

    Args:
        signal_data (pd.DataFrame): DataFrame with 'close', 'signal', 'position_size' columns
        preprocessed_data (pd.DataFrame): DataFrame with unscaled historical data
        initial_capital (float): Starting capital
        transaction_cost (float): Cost per trade (e.g., 0.1% = 0.001)
        slippage (float): Slippage per trade (e.g., 0.05% = 0.0005)
    
    Returns:
        dict: Dictionary containing 'total' (portfolio value series) and 'trades' (DataFrame with entry/exit prices)
    """
    try:
        if 'signal' not in signal_data.columns:
            raise ValueError("Signal column missing in signal_data")

        # Preserve datetime index
        signal_data.index = signal_data.index.to_pydatetime() if hasattr(signal_data.index, 'to_pydatetime') else signal_data.index
        preprocessed_data.index = preprocessed_data.index.to_pydatetime() if hasattr(preprocessed_data.index, 'to_pydatetime') else preprocessed_data.index

        portfolio = {'total': initial_capital, 'position': 0, 'cash': initial_capital}
        results = []
        # Initialize trades DataFrame with the same index as signal_data
        trades = pd.DataFrame(index=signal_data.index, columns=['entry_price', 'exit_price', 'size'])
        
        for idx in signal_data.index:
            i = signal_data.index.get_loc(idx)  # Get integer position for .iloc
            # Use preprocessed_data for unscaled close price, falling back to signal_data if necessary
            close_price = preprocessed_data['close'].loc[idx] if idx in preprocessed_data.index else signal_data['close'].iloc[i]
            signal = signal_data['signal'].iloc[i]
            position_size = signal_data['position_size'].iloc[i]  # In BTC, unscaled
            
            # Validate unscaled close price using preprocessed_data as primary source
            if close_price <= 0 or pd.isna(close_price) or close_price < 10000:
                logging.warning(f"Invalid or scaled close price at {idx}: {close_price}, using 50000.0 USD")
                close_price = 50000.0  # Default to typical BTC price
            
            # Adjust price for slippage
            effective_price = close_price * (1 + slippage if signal == 1 else 1 - slippage if signal == -1 else 1)
            
            # Calculate maximum affordable units based on available cash
            if effective_price * (1 + transaction_cost) <= 0 or pd.isna(effective_price):
                logging.warning(f"Invalid effective price at {idx}: {effective_price}, using 50000.0 USD")
                effective_price = 50000.0  # Default to typical BTC price
            max_units = portfolio['cash'] / (effective_price * (1 + transaction_cost))
            
            if signal == 1 and portfolio['position'] == 0:  # Buy to open long
                units = min(position_size, max_units)  # Use position_size but cap at affordable amount
                if units > 0:
                    cost = units * effective_price * (1 + transaction_cost)
                    if cost > portfolio['cash']:
                        logging.warning(f"Insufficient funds on {idx}: Cost {cost:.2f} exceeds cash {portfolio['cash']:.2f}. Scaling down.")
                        units = portfolio['cash'] / (effective_price * (1 + transaction_cost))
                        cost = units * effective_price * (1 + transaction_cost)
                    portfolio['position'] = units
                    portfolio['cash'] -= cost
                    # Record entry price and size for long trade
                    trades.loc[idx, 'entry_price'] = effective_price
                    trades.loc[idx, 'size'] = units
                    logging.info(f"Trade Entry (Buy) at {idx}: {units:.6f} BTC at {effective_price:.2f} USD")
                else:
                    logging.warning(f"Insufficient funds on {idx}: Cannot afford {position_size:.6f} BTC at {effective_price:.2f} USD")
            elif signal == -1 and portfolio['position'] > 0:  # Sell to close long
                proceeds = portfolio['position'] * effective_price * (1 - transaction_cost)
                portfolio['cash'] += proceeds
                # Record exit price for long trade
                trades.loc[idx, 'exit_price'] = effective_price
                logging.info(f"Trade Exit (Sell) at {idx}: {portfolio['position']:.6f} BTC at {effective_price:.2f} USD")
                portfolio['position'] = 0
            elif signal == -1 and portfolio['position'] == 0:  # Sell to open short
                units = min(position_size, max_units)  # Use position_size but cap at affordable amount
                if units > 0:
                    cost = units * effective_price * (1 + transaction_cost)
                    if cost > portfolio['cash']:
                        logging.warning(f"Insufficient funds on {idx}: Cost {cost:.2f} exceeds cash {portfolio['cash']:.2f}. Scaling down.")
                        units = portfolio['cash'] / (effective_price * (1 + transaction_cost))
                        cost = units * effective_price * (1 + transaction_cost)
                    portfolio['position'] = -units
                    portfolio['cash'] -= cost
                    # Record entry price and size for short trade
                    trades.loc[idx, 'entry_price'] = effective_price
                    trades.loc[idx, 'size'] = units
                    logging.info(f"Trade Entry (Short) at {idx}: {units:.6f} BTC at {effective_price:.2f} USD")
                else:
                    logging.warning(f"Insufficient funds on {idx}: Cannot afford {position_size:.6f} BTC at {effective_price:.2f} USD")
            elif signal == 1 and portfolio['position'] < 0:  # Buy to close short
                proceeds = -portfolio['position'] * effective_price * (1 - transaction_cost)
                portfolio['cash'] += proceeds
                # Record exit price for short trade
                trades.loc[idx, 'exit_price'] = effective_price
                logging.info(f"Trade Exit (Cover) at {idx}: {-portfolio['position']:.6f} BTC at {effective_price:.2f} USD")
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
                'date': idx,
                'close': close_price,  # Store unscaled for clarity
                'signal': signal,
                'cash': portfolio['cash'],
                'position': portfolio['position'],
                'position_size': position_size,
                'total': portfolio['total'],
                'returns': returns,
                'cumulative_returns': cumulative_returns * 100  # In percentage
            })
        
        df_results = pd.DataFrame(results).set_index('date')
    
        # Additional metrics (without numpy)
        daily_returns = df_results['returns'].dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252) if daily_returns.std() != 0 else 0
        max_drawdown = (df_results['total'] / df_results['total'].cummax() - 1).min() * 100  # In percentage
        trades_count = len(df_results[df_results['signal'] != 0])
        wins = len(df_results[(df_results['signal'] == 1) & (df_results['returns'] > 0)]) + len(df_results[(df_results['signal'] == -1) & (df_results['returns'] < 0)])
        losses = trades_count - wins
        win_loss_ratio = wins / losses if losses > 0 else float('inf')
        
        logging.info(f"Backtest completed. Final portfolio value: {df_results['total'].iloc[-1]:.2f}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}%, Trades: {trades_count}, Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        # Return dictionary with portfolio value and trades
        return {
            'total': df_results['total'],
            'trades': trades
        }
    
    except Exception as e:
        logging.error(f"Error in backtest: {e}")
        return {'total': pd.Series(), 'trades': pd.DataFrame()}

if __name__ == "__main__":
    # Dummy test with realistic BTC prices
    dummy_data = pd.DataFrame({
        'close': [50000, 51000, 52000, 51000, 50000],  # Unscaled BTC prices (~$50,000)
        'signal': [0, 1, 0, -1, 0],
        'position_size': [0, 0.002, 0, 0.002, 0]  # Position size of 0.002 BTC (~$100 per trade)
    }, index=pd.date_range("2025-02-28", periods=5, freq="H"))
    preprocessed_data = dummy_data.copy()  # Mock preprocessed data with unscaled prices
    results = backtest_strategy(dummy_data, preprocessed_data)
    print("Portfolio Value:\n", results['total'])
    print("Trades:\n", results['trades'])