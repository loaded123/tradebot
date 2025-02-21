# src/strategy/risk_manager.py

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def manage_risk(df, current_balance, max_drawdown_pct=0.1, atr_multiplier=1.5, recovery_volatility_factor=0.05):
    """
    Manage risk with dynamic position sizing and stop-loss.

    Args:
        df (pd.DataFrame): DataFrame with 'total', 'atr', 'close', 'signal', 'position_size', 'price_volatility'
        current_balance (float): Initial cash balance
        max_drawdown_pct (float): Maximum allowable drawdown percentage
        atr_multiplier (float): Multiplier for ATR-based stop-loss
        recovery_volatility_factor (float): Factor for position size recovery
    
    Returns:
        pd.DataFrame: Updated DataFrame with risk-managed columns
    """
    try:
        # Validate inputs
        required_cols = ['total', 'atr', 'close', 'signal', 'position_size', 'price_volatility']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")
        
        df = df.copy()
        equity_curve = df['total']
        max_equity = equity_curve.cummax()
        drawdown = (equity_curve - max_equity) / max_equity
        position_value = 0  # Track current position value in USD
        last_drawdown_date = None
        
        for i, row in df.iterrows():
            date = df.index[i]
            atr = row['atr']
            last_close = row['close']
            signal = row['signal']
            volatility = row['price_volatility']
            
            # Dynamic ATR multiplier adjustment based on volatility
            dynamic_atr_mult = atr_multiplier * (1 + volatility / df['price_volatility'].mean()) if volatility > 0 else atr_multiplier
            
            # Stop-loss levels
            stop_loss_long = last_close - (atr * dynamic_atr_mult)  # For long positions
            stop_loss_short = last_close + (atr * dynamic_atr_mult)  # For short positions
            
            # 1. Drawdown Management
            if drawdown.iloc[i] < -max_drawdown_pct:
                reduction_factor = max(0.7, 1 + (drawdown.iloc[i] / max_drawdown_pct))
                df.loc[date, 'position_size'] *= max(reduction_factor, 0.5)  # Reduce size, min 50%
                last_drawdown_date = date
                logging.info(f"Drawdown exceeded at {date}: {drawdown.iloc[i]:.2%}, reduced position size")
            
            # 2. Volatility-Based Recovery
            elif last_drawdown_date is not None:
                recovery_factor = 1 + (recovery_volatility_factor * volatility)
                max_units = current_balance / last_close if last_close > 0 else 0
                df.loc[date, 'position_size'] = min(df.loc[date, 'position_size'] * recovery_factor, max_units)
                if (date - last_drawdown_date).total_seconds() > 24 * 3600:  # Reset after 24 hours
                    last_drawdown_date = None
            
            # 3. Normal Position Increase
            elif signal != 0:
                max_units = current_balance / last_close if last_close > 0 else 0
                df.loc[date, 'position_size'] = min(df.loc[date, 'position_size'] * 1.05, max_units)
            
            # 4. Stop-Loss Implementation
            if position_value > 0 and last_close <= stop_loss_long:  # Long stop-loss
                logging.info(f"Stop-loss (long) triggered at {date}: Close {last_close:.2f} <= Stop {stop_loss_long:.2f}")
                df.loc[date, 'signal'] = -1  # Force sell
                current_balance += position_value
                position_value = 0
                df.loc[date, 'position_size'] = 0
            elif position_value < 0 and last_close >= stop_loss_short:  # Short stop-loss
                logging.info(f"Stop-loss (short) triggered at {date}: Close {last_close:.2f} >= Stop {stop_loss_short:.2f}")
                df.loc[date, 'signal'] = 1  # Force buy to cover
                current_balance -= position_value  # Negative position_value is a liability
                position_value = 0
                df.loc[date, 'position_size'] = 0
            
            # Update position based on signal
            if signal == 1 and position_value == 0:  # Enter long
                position_value = df.loc[date, 'position_size'] * last_close
                current_balance -= position_value
            elif signal == -1 and position_value > 0:  # Exit long
                current_balance += position_value
                position_value = 0
            elif signal == -1 and position_value == 0:  # Enter short
                position_value = -df.loc[date, 'position_size'] * last_close
                current_balance -= position_value  # Borrowed value reduces cash
            elif signal == 1 and position_value < 0:  # Exit short
                current_balance -= position_value  # Repay borrowed value
                position_value = 0
            
            # Update DataFrame
            df.loc[date, 'cash'] = current_balance
            df.loc[date, 'position_value'] = position_value
            df.loc[date, 'total'] = current_balance + position_value
            df.loc[date, 'stop_loss'] = stop_loss_long if position_value >= 0 else stop_loss_short
        
        logging.info(f"Risk management completed. Final balance: {current_balance:.2f}")
        return df
    
    except Exception as e:
        logging.error(f"Error in risk management: {e}")
        return df

if __name__ == "__main__":
    # Dummy test
    dummy_df = pd.DataFrame({
        'total': [10000, 9500, 9200, 9800, 10000],
        'atr': [1, 1, 1, 1, 1],
        'close': [100, 95, 92, 98, 100],
        'signal': [1, 0, 0, -1, 0],
        'position_size': [0.5, 0.5, 0.5, 0.5, 0.5],
        'price_volatility': [0.02, 0.03, 0.04, 0.03, 0.02]
    }, index=pd.date_range("2024-01-01", periods=5, freq="H"))
    result = manage_risk(dummy_df, 10000)
    print(result)