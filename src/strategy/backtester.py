import pandas as pd

def backtest_strategy(signal_data, initial_capital=100000):
    """
    Backtest the trading strategy based on signal data.

    :param signal_data: DataFrame with trading signals
    :param initial_capital: Starting capital for backtesting
    :return: DataFrame with backtest results
    """
    portfolio = {'total': initial_capital, 'position': 0, 'cash': initial_capital}
    results = []

    for i, row in signal_data.iterrows():
        date = i
        close_price = row['close']
        signal = row['signal']
        position_size = row.get('position_size', 1)  # Use 1 if position_size not available
        
        if signal == 1:  # Buy signal
            if portfolio['position'] == 0:
                # Calculate the number of units to buy based on position size
                portfolio['position'] = (portfolio['cash'] * position_size) / close_price
                portfolio['cash'] = portfolio['cash'] * (1 - position_size)
        elif signal == -1:  # Sell signal
            if portfolio['position'] > 0:
                portfolio['cash'] += portfolio['position'] * close_price
                portfolio['position'] = 0
        
        portfolio['total'] = portfolio['cash'] + (portfolio['position'] * close_price)
        
        # Calculate returns and cumulative returns
        if i == signal_data.index[0]:
            previous_total = initial_capital
            returns = 0
            cumulative_returns = 0
        else:
            previous_total = results[-1]['total']
            returns = (portfolio['total'] - previous_total) / previous_total
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
            'cumulative_returns': cumulative_returns
        })

    return pd.DataFrame(results).set_index('date')