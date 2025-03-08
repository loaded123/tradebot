# src/strategy/risk_manager.py

import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def manage_risk(df, current_balance, max_drawdown_pct=0.05, atr_multiplier=3.0, recovery_volatility_factor=0.15):
    """
    Manage risk with dynamic position sizing and stop-loss, ensuring unscaled prices and optimizing for higher gains and lower drawdown.

    Args:
        df (pd.DataFrame): DataFrame with 'total', 'atr', 'close', 'signal', 'position_size', 'price_volatility'
        current_balance (float): Initial cash balance in USD
        max_drawdown_pct (float): Maximum allowable drawdown percentage (maintained at 5%)
        atr_multiplier (float): Multiplier for ATR-based stop-loss (increased for wider stops)
        recovery_volatility_factor (float): Factor for position size recovery (increased for larger positions)
    
    Returns:
        tuple: Updated DataFrame and final current_balance
    """
    try:
        # Validate inputs
        required_cols = ['total', 'atr', 'close', 'signal', 'position_size', 'price_volatility']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = df.copy()
        # Ensure numeric columns are float64 to fix dtype warnings
        for col in required_cols:
            df[col] = df[col].astype('float64')

        equity_curve = df['total']
        max_equity = equity_curve.cummax()
        drawdown = (equity_curve - max_equity) / max_equity if max_equity.max() != 0 else pd.Series(0, index=df.index, dtype='float64')
        position_value = 0.0  # Track current position value in USD, explicitly float
        last_drawdown_date = None

        # Calculate mean volatility once to avoid repeated computation
        mean_volatility = df['price_volatility'].mean()

        # Use iloc for iteration over rows
        for i in range(len(df)):
            date = df.index[i]  # Get date from index
            atr = df.iloc[i]['atr']
            last_close = df.iloc[i]['close']
            signal = df.iloc[i]['signal']
            volatility = df.iloc[i]['price_volatility']
            position_size = df.iloc[i]['position_size']

            # Skip or handle the first 13 rows where ATR is NaN due to 14-period window
            if pd.isna(atr) and i < 13:
                logging.warning(f"Invalid ATR at {date}: nan, using 500.0 USD for initial period")
                atr = 500.0
            elif pd.isna(atr) or atr <= 0:
                logging.warning(f"Invalid ATR at {date}: {atr}, using 500.0 USD")
                atr = 500.0

            # Validate unscaled values, ensuring no scaling errors
            if pd.isna(last_close) or last_close <= 0 or not (10000 <= last_close <= 200000):
                logging.warning(f"Invalid unscaled close at {date}: {last_close}, using 78877.88 USD")
                last_close = 78877.88
            if pd.isna(position_size) or position_size <= 0:
                logging.error(f"Invalid position_size at {date}: {position_size:.6f} BTC, critical scaling issue detected. Using 0.05 BTC.")
                position_size = 0.05

            # Dynamic ATR multiplier adjustment based on volatility
            dynamic_atr_mult = atr_multiplier * (1 + (volatility / (mean_volatility * 1.5))) if mean_volatility > 0 and volatility > 0 else atr_multiplier

            # Adjusted stop-loss and take-profit levels (much wider for higher gains, lower drawdown)
            stop_loss_long = last_close - (atr * dynamic_atr_mult * 2.5)  # Wider long stop-loss
            stop_loss_short = last_close + (atr * dynamic_atr_mult * 2.5)  # Wider short stop-loss
            take_profit_long = last_close + (atr * dynamic_atr_mult * 4.0)  # Wider long take-profit
            take_profit_short = last_close - (atr * dynamic_atr_mult * 4.0)  # Wider short take-profit

            # 1. Drawdown Management (conservative but allowing larger positions)
            if drawdown.iloc[i] < -max_drawdown_pct:
                reduction_factor = max(0.6, 1 + (drawdown.iloc[i] / max_drawdown_pct))  # Less aggressive reduction
                df.loc[date, 'position_size'] = max(reduction_factor * position_size, 0.05)  # Maintain 0.05 BTC minimum
                last_drawdown_date = date
                logging.info(f"Drawdown exceeded at {date}: {drawdown.iloc[i]:.2%}, reduced position size to {df.loc[date, 'position_size']:.6f} BTC")

            # 2. Volatility-Based Recovery (more aggressive for larger positions)
            elif last_drawdown_date is not None:
                recovery_factor = 1 + (recovery_volatility_factor * volatility * 2)  # More aggressive recovery
                max_units = current_balance / last_close if last_close > 0 else 0.05  # Ensure minimum 0.05 BTC
                df.loc[date, 'position_size'] = min(position_size * recovery_factor, max_units)
                if (pd.to_datetime(date) - pd.to_datetime(last_drawdown_date)).total_seconds() > 48 * 3600:  # Reset after 48 hours
                    last_drawdown_date = None

            # 3. Normal Position Increase (more aggressive)
            elif signal != 0:
                max_units = current_balance / last_close if last_close > 0 else 0.05  # Ensure minimum 0.05 BTC
                df.loc[date, 'position_size'] = min(position_size * 1.3, max_units)  # More aggressive increase

            # 4. Stop-Loss and Take-Profit Implementation, ensuring unscaled values
            if position_value > 0 and pd.notna(last_close):  # Long position
                if pd.notna(last_close) and last_close <= stop_loss_long:
                    logging.info(f"Stop-loss (long) triggered at {date}: Close {last_close:.2f} <= Stop {stop_loss_long:.2f}")
                    df.loc[date, 'signal'] = -1  # Force sell
                    current_balance += position_value
                    position_value = 0.0
                    df.loc[date, 'position_size'] = 0.0
                elif pd.notna(last_close) and last_close >= take_profit_long:
                    logging.info(f"Take-profit (long) triggered at {date}: Close {last_close:.2f} >= Take-Profit {take_profit_long:.2f}")
                    df.loc[date, 'signal'] = -1  # Force sell
                    current_balance += position_value
                    position_value = 0.0
                    df.loc[date, 'position_size'] = 0.0
            elif position_value < 0 and pd.notna(last_close):  # Short position
                if pd.notna(last_close) and last_close >= stop_loss_short:
                    logging.info(f"Stop-loss (short) triggered at {date}: Close {last_close:.2f} >= Stop {stop_loss_short:.2f}")
                    df.loc[date, 'signal'] = 1  # Force buy to cover
                    current_balance -= position_value  # Negative position_value is a liability
                    position_value = 0.0
                    df.loc[date, 'position_size'] = 0.0
                elif pd.notna(last_close) and last_close <= take_profit_short:
                    logging.info(f"Take-profit (short) triggered at {date}: Close {last_close:.2f} <= Take-Profit {take_profit_short:.2f}")
                    df.loc[date, 'signal'] = 1  # Force buy to cover
                    current_balance -= position_value  # Repay borrowed value
                    position_value = 0.0
                    df.loc[date, 'position_size'] = 0.0

            # Update position based on signal, ensuring unscaled values
            if signal == 1 and position_value == 0:  # Enter long
                position_value = position_size * last_close
                current_balance -= position_value
            elif signal == -1 and position_value > 0:  # Exit long
                current_balance += position_value
                position_value = 0.0
            elif signal == -1 and position_value == 0:  # Enter short
                position_value = -position_size * last_close
                current_balance -= position_value  # Borrowed value reduces cash
            elif signal == 1 and position_value < 0:  # Exit short
                current_balance -= position_value  # Repay borrowed value
                position_value = 0.0

            # Update DataFrame, ensuring unscaled and float64 types
            df.loc[date, 'cash'] = float(current_balance)
            df.loc[date, 'position_value'] = float(position_value)
            df.loc[date, 'total'] = float(current_balance + position_value)
            df.loc[date, 'stop_loss'] = float(stop_loss_long if position_value >= 0 else stop_loss_short)
            df.loc[date, 'take_profit'] = float(take_profit_long if position_value >= 0 else take_profit_short)

        logging.info(f"Risk management completed. Final balance: {current_balance:.2f}")
        return df, current_balance  # Return both DataFrame and final balance

    except Exception as e:
        logging.error(f"Error in risk management: {e}")
        return df, current_balance  # Return original df and current_balance in case of error

if __name__ == "__main__":
    # Dummy test with realistic BTC prices and position sizes
    dummy_df = pd.DataFrame({
        'total': [10000.0] * 5,
        'atr': [500.0] * 5,  # Realistic BTC ATR (~$500)
        'close': [80000.0, 81000.0, 82000.0, 81000.0, 80000.0],  # Unscaled BTC prices (~$80,000)
        'signal': [1, 0, 0, -1, 0],
        'position_size': [0.05, 0.05, 0.05, 0.05, 0.05],  # Realistic BTC position size (~$4,000/trade)
        'price_volatility': [0.02, 0.03, 0.04, 0.03, 0.02]
    }, index=pd.date_range("2025-01-01", periods=5, freq="H"))
    result_df, final_balance = manage_risk(dummy_df, 10000.0)
    print(f"Result DataFrame:\n{result_df}")
    print(f"Final Balance: {final_balance:.2f}")