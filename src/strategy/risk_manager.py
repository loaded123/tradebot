import pandas as pd

def manage_risk(df, current_balance, max_drawdown_pct=0.07, atr_multiplier=2.5, recovery_volatility_factor=0.05):
    """Manage risk with dynamic position sizing and stop-loss."""

    equity_curve = df['total']
    max_equity = equity_curve.cummax()
    drawdown = (equity_curve - max_equity) / max_equity

    last_drawdown_date = None
    position_value = 0  # Track current position value

    for i in range(len(df)):
        atr = df['atr'].iloc[i]
        last_close = df['close'].iloc[i]
        signal = df['signal'].iloc[i]

        # 1. Drawdown Management:
        if drawdown.iloc[i] < -max_drawdown_pct:
            reduction_factor = 1 + (drawdown.iloc[i] / max_drawdown_pct)
            df.loc[df.index[i], 'position_size'] *= max(reduction_factor, 0.5)
            last_drawdown_date = df.index[i]

        # 2. Volatility-Based Recovery:
        elif last_drawdown_date:
            volatility = df['price_volatility'].iloc[i]
            recovery_factor = 1 + (recovery_volatility_factor * volatility)
            df.loc[df.index[i], 'position_size'] = min(df.loc[df.index[i], 'position_size'] * recovery_factor, current_balance / last_close)

        # 3. Normal Position Increase (if no drawdown):
        elif signal != 0:  # Only increase position if there is a signal
            df.loc[df.index[i], 'position_size'] = min(df.loc[df.index[i], 'position_size'] * 1.05, current_balance / last_close)

        # 4. Stop-Loss Implementation:
        stop_loss = last_close - (atr * atr_multiplier)

        if position_value > 0 and last_close <= stop_loss:  # Check if position is open and stop-loss is hit
            print(f"Stop-loss triggered at {df.index[i]}")  # Print statement for debugging
            df.loc[df.index[i], 'signal'] = -1  # Force a sell signal
            position_value = 0  # Reset position value
            df.loc[df.index[i], 'position_size'] = 0  # Reset position size
            current_balance += position_value
        elif position_value < 0 and last_close >= stop_loss:  # Check if short position is open and stop-loss is hit
            print(f"Stop-loss triggered at {df.index[i]}")  # Print statement for debugging
            df.loc[df.index[i], 'signal'] = 1  # Force a buy signal
            position_value = 0  # Reset position value
            df.loc[df.index[i], 'position_size'] = 0  # Reset position size
            current_balance -= position_value

        # Update position value
        if signal == 1 and position_value == 0:
            position_value = df.loc[df.index[i], 'position_size'] * last_close
            current_balance -= position_value
        elif signal == -1 and position_value > 0:
            current_balance += position_value
            position_value = 0
        elif signal == -1 and position_value == 0:
            position_value = -df.loc[df.index[i], 'position_size'] * last_close
            current_balance -= position_value
        elif signal == 1 and position_value < 0:
            current_balance -= position_value
            position_value = 0

        df.loc[df.index[i], 'cash'] = current_balance
        df.loc[df.index[i], 'position_value'] = position_value
        df.loc[df.index[i], 'total'] = current_balance + position_value
        df.loc[df.index[i], 'stop_loss'] = stop_loss  # Keep stop loss value
    return df