# src/strategy/backtester.py
import pandas as pd
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
pd.set_option('future.no_silent_downcasting', True)

def backtest_strategy(signal_data, preprocessed_data: pd.DataFrame, initial_capital=10000, transaction_cost=0.001, slippage=0.001, stop_loss=0.05, take_profit=0.15):
    """
    Backtest the trading strategy with adjusted risk parameters.

    Args:
        signal_data (pd.DataFrame): DataFrame with 'close', 'signal', 'position_size' columns
        preprocessed_data (pd.DataFrame): DataFrame with unscaled historical data
        initial_capital (float): Starting capital
        transaction_cost (float): Base cost per trade (e.g., 0.1% = 0.001)
        slippage (float): Base slippage per trade (e.g., 0.1% = 0.001)
        stop_loss (float): Stop-loss percentage (5%)
        take_profit (float): Take-profit percentage (15%)

    Returns:
        dict: Dictionary containing 'total' (portfolio value series) and 'trades' (DataFrame with entry/exit prices)
    """
    try:
        if 'signal' not in signal_data.columns or 'position_size' not in signal_data.columns:
            raise ValueError("Signal or position_size column missing")

        signal_data.index = pd.to_datetime(signal_data.index, utc=True).tz_localize(None)
        preprocessed_data.index = pd.to_datetime(preprocessed_data.index, utc=True).tz_localize(None)

        common_index = signal_data.index.union(preprocessed_data.index)
        signal_data = signal_data.reindex(common_index, method='ffill').fillna(0)
        preprocessed_data = preprocessed_data.reindex(common_index, method='ffill').fillna({'close': 78877.88, 'high': 79367.5, 'low': 78186.98, 'volume': 1000.0, 'atr': 500.0})
        if len(preprocessed_data) < len(signal_data):
            logging.warning(f"Preprocessed data shorter than signal data. Extending.")
            preprocessed_data = preprocessed_data.reindex(signal_data.index, method='ffill').fillna({'close': 78877.88, 'high': 79367.5, 'low': 78186.98, 'volume': 1000.0, 'atr': 500.0})

        portfolio = {'total': initial_capital, 'position': 0.0, 'cash': initial_capital}
        logging.info(f"Initial capital set to: {initial_capital:.2f}")

        results = []
        entry_price = None
        last_trade_time = None
        min_holding_period = pd.Timedelta(hours=24)

        trades = pd.DataFrame(index=signal_data.index, columns=['entry_price', 'exit_price', 'size', 'profit']).fillna(0.0)
        
        for idx in signal_data.index:
            close_price = preprocessed_data.loc[idx, 'close']
            signal = signal_data.loc[idx, 'signal']
            position_size = signal_data.loc[idx, 'position_size']
            atr = signal_data.loc[idx, 'atr'] if 'atr' in signal_data.columns and not pd.isna(signal_data.loc[idx, 'atr']) else preprocessed_data.loc[idx, 'atr'] if 'atr' in preprocessed_data.columns else 500.0
            confidence = signal_data.loc[idx, 'signal_confidence'] if 'signal_confidence' in signal_data.columns else 0.5
            
            if pd.isna(close_price) or close_price <= 0 or close_price < 10000 or close_price > 200000:
                logging.warning(f"Invalid close price at {idx}: {close_price}, using 78877.88 USD")
                close_price = 78877.88
            if pd.isna(atr) or atr <= 0:
                logging.warning(f"Invalid ATR at {idx}: {atr}, using 500.0 USD")
                atr = 500.0
            if pd.isna(position_size) or position_size <= 0:
                logging.error(f"Invalid position_size at {idx}: {position_size:.6f}, using 0.05 BTC")
                position_size = 0.05

            transaction_cost_dynamic = transaction_cost + (0.001 * (position_size / 0.05))
            slippage_dynamic = slippage * (atr / signal_data['atr'].mean()) if 'atr' in signal_data.columns and not pd.isna(signal_data['atr'].mean()) else slippage
            effective_price = close_price * (1 + slippage_dynamic if signal == 1 else 1 - slippage_dynamic if signal == -1 else 1)
            units = position_size * min(1.5, confidence * 3) if confidence > 0.5 else position_size

            if confidence < 0.5:
                signal = 0
            if signal != 0 and last_trade_time and (idx - last_trade_time) < min_holding_period:
                signal = 0
            elif signal != 0 and signal_data.loc[idx, 'price_volatility'] > signal_data['price_volatility'].mean():
                min_holding_period = pd.Timedelta(hours=12)

            dynamic_stop_loss = max(stop_loss, atr / close_price * 2)
            dynamic_take_profit = take_profit

            if pd.notna(portfolio['position']) and portfolio['position'] != 0 and pd.notna(entry_price):
                if portfolio['position'] > 0:
                    if close_price <= entry_price * (1 - dynamic_stop_loss):
                        proceeds = portfolio['position'] * close_price * (1 - transaction_cost_dynamic)
                        portfolio['cash'] += proceeds
                        trade_profit = proceeds - (portfolio['position'] * entry_price * (1 + transaction_cost_dynamic))
                        trades.loc[idx, 'exit_price'] = float(close_price) if pd.notna(close_price) else 0.0
                        trades.loc[idx, 'profit'] = trade_profit
                        portfolio['position'] = 0.0
                        logging.info(f"Stop-Loss Exit (Sell) at {idx}: {portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f}")
                    elif close_price >= entry_price * (1 + dynamic_take_profit):
                        proceeds = portfolio['position'] * close_price * (1 - transaction_cost_dynamic)
                        portfolio['cash'] += proceeds
                        trade_profit = proceeds - (portfolio['position'] * entry_price * (1 + transaction_cost_dynamic))
                        trades.loc[idx, 'exit_price'] = float(close_price) if pd.notna(close_price) else 0.0
                        trades.loc[idx, 'profit'] = trade_profit
                        portfolio['position'] = 0.0
                        logging.info(f"Take-Profit Exit (Sell) at {idx}: {portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f}")
                    if pd.notna(atr) and atr > 6.0 * signal_data['atr'].mean():
                        proceeds = portfolio['position'] * close_price * (1 - transaction_cost_dynamic)
                        portfolio['cash'] += proceeds
                        trade_profit = proceeds - (portfolio['position'] * entry_price * (1 + transaction_cost_dynamic))
                        trades.loc[idx, 'exit_price'] = float(close_price) if pd.notna(close_price) else 0.0
                        trades.loc[idx, 'profit'] = trade_profit
                        portfolio['position'] = 0.0
                        logging.info(f"Volatility Exit at {idx}: {portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f}")
                elif portfolio['position'] < 0:
                    if close_price >= entry_price * (1 + dynamic_stop_loss):
                        proceeds = -portfolio['position'] * close_price * (1 - transaction_cost_dynamic)
                        portfolio['cash'] += proceeds
                        trade_profit = (portfolio['position'] * entry_price * (1 - transaction_cost_dynamic)) - proceeds
                        trades.loc[idx, 'exit_price'] = float(close_price) if pd.notna(close_price) else 0.0
                        trades.loc[idx, 'profit'] = trade_profit
                        portfolio['position'] = 0.0
                        logging.info(f"Stop-Loss Exit (Cover) at {idx}: {-portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f}")
                    elif close_price <= entry_price * (1 - dynamic_take_profit):
                        proceeds = -portfolio['position'] * close_price * (1 - transaction_cost_dynamic)
                        portfolio['cash'] += proceeds
                        trade_profit = (portfolio['position'] * entry_price * (1 - transaction_cost_dynamic)) - proceeds
                        trades.loc[idx, 'exit_price'] = float(close_price) if pd.notna(close_price) else 0.0
                        trades.loc[idx, 'profit'] = trade_profit
                        portfolio['position'] = 0.0
                        logging.info(f"Take-Profit Exit (Cover) at {idx}: {-portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f}")
                    if pd.notna(atr) and atr > 6.0 * signal_data['atr'].mean():
                        proceeds = -portfolio['position'] * close_price * (1 - transaction_cost_dynamic)
                        portfolio['cash'] += proceeds
                        trade_profit = (portfolio['position'] * entry_price * (1 - transaction_cost_dynamic)) - proceeds
                        trades.loc[idx, 'exit_price'] = float(close_price) if pd.notna(close_price) else 0.0
                        trades.loc[idx, 'profit'] = trade_profit
                        portfolio['position'] = 0.0
                        logging.info(f"Volatility Exit at {idx}: {-portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f}")

            if signal == 1 and (pd.isna(portfolio['position']) or portfolio['position'] == 0):
                cost = units * effective_price * (1 + transaction_cost_dynamic)
                if pd.isna(cost) or cost > portfolio['cash']:
                    logging.warning(f"Insufficient funds at {idx}: Cost {cost:.2f} exceeds cash {portfolio['cash']:.2f}, using 0.05 BTC")
                    units = 0.05
                    cost = units * effective_price * (1 + transaction_cost_dynamic)
                portfolio['position'] = units
                portfolio['cash'] -= cost
                trades.loc[idx, 'entry_price'] = float(effective_price) if pd.notna(effective_price) else 0.0
                trades.loc[idx, 'size'] = units
                entry_price = float(effective_price) if pd.notna(effective_price) else 0.0
                last_trade_time = idx
                logging.info(f"Trade Entry (Buy) at {idx}: {units:.6f} BTC, Price: {effective_price:.2f} USD")
            elif signal == -1 and pd.notna(portfolio['position']) and portfolio['position'] > 0:
                if pd.isna(effective_price):
                    logging.error(f"NaN effective_price at {idx}")
                    continue
                proceeds = portfolio['position'] * effective_price * (1 - transaction_cost_dynamic)
                portfolio['cash'] += proceeds
                trade_profit = proceeds - (portfolio['position'] * entry_price * (1 + transaction_cost_dynamic))
                trades.loc[idx, 'exit_price'] = float(effective_price) if pd.notna(effective_price) else 0.0
                trades.loc[idx, 'profit'] = trade_profit
                portfolio['position'] = 0.0
                entry_price = None
                last_trade_time = idx
                logging.info(f"Trade Exit (Sell) at {idx}: {portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f} USD")
            elif signal == -1 and (pd.isna(portfolio['position']) or portfolio['position'] == 0):
                cost = units * effective_price * (1 + transaction_cost_dynamic)
                if pd.isna(cost) or cost > portfolio['cash']:
                    logging.warning(f"Insufficient funds at {idx}: Cost {cost:.2f} exceeds cash {portfolio['cash']:.2f}, using 0.05 BTC")
                    units = 0.05
                    cost = units * effective_price * (1 + transaction_cost_dynamic)
                portfolio['position'] = -units
                portfolio['cash'] -= cost
                trades.loc[idx, 'entry_price'] = float(effective_price) if pd.notna(effective_price) else 0.0
                trades.loc[idx, 'size'] = units
                entry_price = float(effective_price) if pd.notna(effective_price) else 0.0
                last_trade_time = idx
                logging.info(f"Trade Entry (Short) at {idx}: {units:.6f} BTC, Price: {effective_price:.2f} USD")
            elif signal == 1 and pd.notna(portfolio['position']) and portfolio['position'] < 0:
                if pd.isna(effective_price):
                    logging.error(f"NaN effective_price at {idx}")
                    continue
                proceeds = -portfolio['position'] * effective_price * (1 - transaction_cost_dynamic)
                portfolio['cash'] += proceeds
                trade_profit = (portfolio['position'] * entry_price * (1 - transaction_cost_dynamic)) - proceeds
                trades.loc[idx, 'exit_price'] = float(effective_price) if pd.notna(effective_price) else 0.0
                trades.loc[idx, 'profit'] = trade_profit
                portfolio['position'] = 0.0
                entry_price = None
                last_trade_time = idx
                logging.info(f"Trade Exit (Cover) at {idx}: {-portfolio['position']:.6f} BTC, Profit: {trade_profit:.2f} USD")

            portfolio['total'] = portfolio['cash'] + (portfolio['position'] * close_price) if pd.notna(portfolio['cash']) and pd.notna(portfolio['position']) and pd.notna(close_price) else initial_capital

            if not results:
                returns = 0.0
                cumulative_returns = 0.0
            else:
                previous_total = results[-1]['total'] if pd.notna(results[-1]['total']) else initial_capital
                returns = (portfolio['total'] - previous_total) / previous_total if previous_total != 0 else 0.0
                cumulative_returns = (portfolio['total'] - initial_capital) / initial_capital if pd.notna(portfolio['total']) else 0.0
            
            results.append({
                'date': idx,
                'close': float(close_price) if pd.notna(close_price) else 78877.88,
                'signal': float(signal) if pd.notna(signal) else 0.0,
                'cash': float(portfolio['cash']) if pd.notna(portfolio['cash']) else 0.0,
                'position': float(portfolio['position']) if pd.notna(portfolio['position']) else 0.0,
                'position_size': float(position_size) if pd.notna(position_size) else 0.05,
                'total': float(portfolio['total']) if pd.notna(portfolio['total']) else initial_capital,
                'returns': returns,
                'cumulative_returns': cumulative_returns * 100
            })
        
        df_results = pd.DataFrame(results).set_index('date')
        daily_returns = df_results['returns'].dropna()
        if daily_returns.max() < 1e-4 or daily_returns.max() > 1:
            daily_returns = daily_returns * 100
            logging.warning("Corrected daily_returns scaling to percent")
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252) if daily_returns.std() != 0 else 0.0
        max_drawdown = (df_results['total'].dropna() / df_results['total'].dropna().cummax() - 1).min() * 100 if not df_results['total'].dropna().empty else 0.0

        downside_returns = daily_returns[daily_returns < 0].dropna() * 100
        sortino_ratio = (daily_returns.mean() * 100 / downside_returns.std()) * math.sqrt(252) if not downside_returns.empty and downside_returns.std() != 0 else 0.0

        trade_profits = trades['profit'].dropna()
        gross_profit = trade_profits[trade_profits > 0].sum()
        gross_loss = abs(trade_profits[trade_profits < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0
        wins = len(trade_profits[trade_profits > 0])
        losses = len(trade_profits[trade_profits < 0])
        win_loss_ratio = wins / losses if losses > 0 and not pd.isna(wins) and not pd.isna(losses) else float('inf') if wins > 0 else 0.0
        
        trades_count = len(trades[(trades['entry_price'] != 0) | (trades['exit_price'] != 0)])
        
        logging.info(f"Backtest completed. Final portfolio value: {df_results['total'].iloc[-1] if pd.notna(df_results['total'].iloc[-1]) else initial_capital:.2f}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}, Sortino Ratio: {sortino_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}%, "
                     f"Profit Factor: {profit_factor:.2f}, Trades: {trades_count}, Win/Loss Ratio: {win_loss_ratio:.2f}")
        
        return {'total': df_results['total'].fillna(initial_capital), 'trades': trades.fillna(0.0)}
    
    except Exception as e:
        logging.error(f"Error in backtest: {e}")
        return {'total': pd.Series([initial_capital], index=[signal_data.index[0]] if not signal_data.empty else pd.DatetimeIndex(['2025-01-01 00:00:00'])), 'trades': pd.DataFrame()}

def walk_forward_backtest(signal_data, preprocessed_data, train_period=30, test_period=10):
    """
    Perform walk-forward analysis to test strategy robustness across rolling windows.

    Args:
        signal_data (pd.DataFrame): DataFrame with trading signals
        preprocessed_data (pd.DataFrame): DataFrame with historical price data
        train_period (int): Number of periods for training (in hours)
        test_period (int): Number of periods for testing (in hours)
    
    Returns:
        pd.DataFrame: Results of walk-forward tests
    """
    results = []
    step_size = test_period
    for start in range(0, len(signal_data) - train_period - test_period + 1, step_size):
        train_end = start + train_period
        test_end = train_end + test_period
        train_data = signal_data.iloc[start:train_end].dropna(how='all')
        test_data = signal_data.iloc[train_end:test_end].dropna(how='all')
        train_preprocessed = preprocessed_data.iloc[start:train_end].dropna(how='all')
        test_preprocessed = preprocessed_data.iloc[train_end:test_end].dropna(how='all')
        if not test_data.empty and not test_preprocessed.empty:
            result = backtest_strategy(test_data, test_preprocessed)
            results.append({
                'start': test_data.index[0],
                'end': test_data.index[-1],
                'final_value': result['total'].iloc[-1] if pd.notna(result['total'].iloc[-1]) else 10000,
                'sharpe': (result['total'].pct_change().dropna().mean() / result['total'].pct_change().dropna().std()) * math.sqrt(252) if result['total'].pct_change().dropna().std() != 0 else 0.0
            })
            logging.debug(f"Walk-forward test: Start {test_data.index[0]}, End {test_data.index[-1]}, Final Value {result['total'].iloc[-1]:.2f}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    dummy_data = pd.DataFrame({
        'close': [78000] * 50 + [108000] * 50,
        'signal': [0] * 25 + [1] + [0] * 23 + [0] * 25 + [-1] + [0] * 23,
        'position_size': [0] * 25 + [0.05] + [0] * 23 + [0] * 25 + [0.05] + [0] * 23,
        'price_volatility': [0.01] * 100,
        'signal_confidence': [0.5] * 100,
        'total': [10000.0] * 100,
        'atr': [500.0] * 100,
        'sma_10': [78000] * 50 + [108000] * 50,
        'sma_20': [78000] * 50 + [108000] * 50,
        'adx': [30.0] * 100
    }, index=pd.date_range("2025-01-01", periods=100, freq="H"))
    preprocessed_data = pd.DataFrame({
        'close': [78000] * 50 + [108000] * 50,
        'high': [78100] * 50 + [108100] * 50,
        'low': [77900] * 50 + [107900] * 50
    }, index=pd.date_range("2025-01-01", periods=100, freq="H"))
    results = backtest_strategy(dummy_data, preprocessed_data)
    print("Portfolio Value:\n", results['total'])
    print("Trades:\n", results['trades'])
    wf_results = walk_forward_backtest(dummy_data, preprocessed_data)
    print("Walk-Forward Results:\n", wf_results)