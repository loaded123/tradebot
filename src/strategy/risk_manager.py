# src/strategy/risk_manager.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def manage_risk(df, current_balance, max_drawdown_pct=0.05, atr_multiplier=3.0, recovery_volatility_factor=0.15, 
                max_risk_pct=0.20, min_position_size=0.01):
    """
    Manage risk with dynamic position sizing and stop-loss, ensuring unscaled prices and optimizing for higher gains 
    and lower drawdown. Respects signal_generator's position_size unless risk constraints are exceeded.

    Args:
        df (pd.DataFrame): DataFrame with 'total', 'atr', 'close', 'signal', 'position_size', 'price_volatility'
        current_balance (float): Initial cash balance in USD
        max_drawdown_pct (float): Maximum allowable drawdown percentage (default 5%)
        atr_multiplier (float): Multiplier for ATR-based stop-loss (default 3.0)
        recovery_volatility_factor (float): Factor for position size recovery (default 0.15)
        max_risk_pct (float): Maximum risk percentage per trade (default 20%)
        min_position_size (float): Minimum position size in BTC (default 0.01)

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
        position_value = 0.0  # Track current position value in USD
        last_drawdown_date = None

        # Calculate mean volatility once to avoid repeated computation
        mean_volatility = df['price_volatility'].mean()

        # Use iloc for iteration over rows
        for i in range(len(df)):
            date = df.index[i]
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

            # Validate unscaled values
            if pd.isna(last_close) or last_close <= 0 or not (10000 <= last_close <= 200000):
                logging.warning(f"Invalid unscaled close at {date}: {last_close}, using 78877.88 USD")
                last_close = 78877.88
            if pd.isna(position_size) or position_size <= 0:
                logging.warning(f"Invalid position_size at {date}: {position_size:.6f} BTC, using default {min_position_size} BTC")
                position_size = min_position_size

            # Dynamic ATR multiplier adjustment based on volatility
            dynamic_atr_mult = atr_multiplier * (1 + (volatility / (mean_volatility * 1.5))) if mean_volatility > 0 and volatility > 0 else atr_multiplier

            # Adjusted stop-loss and take-profit levels (aligned with signal_generator)
            stop_loss_long = last_close - (atr * min(dynamic_atr_mult, 2.0))
            stop_loss_short = last_close + (atr * min(dynamic_atr_mult, 2.0))
            take_profit_long = last_close + (atr * min(dynamic_atr_mult, 5.5 if volatility <= mean_volatility else 4.5))
            take_profit_short = last_close - (atr * min(dynamic_atr_mult, 5.5 if volatility <= mean_volatility else 4.5))

            # Calculate risk per trade based on stop-loss distance
            if signal != 0:
                if signal == 1:  # Long position
                    stop_distance = last_close - stop_loss_long
                else:  # Short position
                    stop_distance = stop_loss_short - last_close
                risk_per_unit = stop_distance
                max_risk = current_balance * max_risk_pct
                max_units = max_risk / risk_per_unit if risk_per_unit > 0 else min_position_size

                # Debug logging for insight
                logging.debug(f"Risk calc at {date}: max_risk={max_risk:.2f}, risk_per_unit={risk_per_unit:.2f}, max_units={max_units:.6f}, position_size={position_size:.6f}")
                current_risk = position_size * risk_per_unit
                logging.debug(f"Current risk at {date}: current_risk={current_risk:.2f}")

                # Respect signal_generator's position_size unless it exceeds max_risk
                if current_risk > max_risk:
                    adjusted_position_size = max(min(max_units, position_size), min_position_size)
                    logging.warning(f"Position size adjusted at {date} from {position_size:.6f} to {adjusted_position_size:.6f} BTC to meet risk limit")
                    df.loc[date, 'position_size'] = adjusted_position_size
                else:
                    df.loc[date, 'position_size'] = position_size  # No override
                    logging.debug(f"Position size preserved at {date}: {df.loc[date, 'position_size']:.6f} BTC")

            # 1. Drawdown Management (conservative but allowing larger positions)
            if drawdown.iloc[i] < -max_drawdown_pct:
                reduction_factor = max(0.6, 1 + (drawdown.iloc[i] / max_drawdown_pct))
                df.loc[date, 'position_size'] = max(reduction_factor * df.loc[date, 'position_size'], min_position_size)
                last_drawdown_date = date
                logging.info(f"Drawdown exceeded at {date}: {drawdown.iloc[i]:.2%}, reduced position size to {df.loc[date, 'position_size']:.6f} BTC")

            # 2. Volatility-Based Recovery (more aggressive for larger positions)
            elif last_drawdown_date is not None:
                recovery_factor = 1 + (recovery_volatility_factor * volatility * 2)
                max_units = current_balance / last_close if last_close > 0 else min_position_size
                df.loc[date, 'position_size'] = min(df.loc[date, 'position_size'] * recovery_factor, max_units)
                if (pd.to_datetime(date) - pd.to_datetime(last_drawdown_date)).total_seconds() > 24 * 3600:
                    last_drawdown_date = None

            # 3. Normal Position Increase (more aggressive)
            elif signal != 0:
                max_units = current_balance / last_close if last_close > 0 else min_position_size
                df.loc[date, 'position_size'] = min(df.loc[date, 'position_size'] * 1.3, max_units)

            # Update position based on signal, ensuring unscaled values
            if signal == 1 and position_value == 0:
                position_value = df.loc[date, 'position_size'] * last_close
                current_balance -= position_value
            elif signal == -1 and position_value > 0:
                current_balance += position_value
                position_value = 0.0
            elif signal == -1 and position_value == 0:
                position_value = -df.loc[date, 'position_size'] * last_close
                current_balance -= position_value
            elif signal == 1 and position_value < 0:
                current_balance -= position_value
                position_value = 0.0

            # Update DataFrame, ensuring unscaled and float64 types
            df.loc[date, 'cash'] = float(current_balance)
            df.loc[date, 'position_value'] = float(position_value)
            df.loc[date, 'total'] = float(current_balance + position_value)
            df.loc[date, 'stop_loss'] = float(stop_loss_long if position_value >= 0 else stop_loss_short)
            df.loc[date, 'take_profit'] = float(take_profit_long if position_value >= 0 else take_profit_short)

        # Ensure no negative balance
        current_balance = max(current_balance, 0.0)
        logging.info(f"Risk management completed. Final balance: {current_balance:.2f}")
        return df, current_balance

    except Exception as e:
        logging.error(f"Error in risk management: {e}")
        return df, current_balance

if __name__ == "__main__":
    # Dummy test with realistic BTC prices and position sizes
    dummy_df = pd.DataFrame({
        'total': [10000.0] * 5,
        'atr': [500.0] * 5,
        'close': [80000.0, 81000.0, 82000.0, 81000.0, 80000.0],
        'signal': [1, 0, 0, -1, 0],
        'position_size': [0.005, 0.005, 0.005, 0.005, 0.005],
        'price_volatility': [0.02, 0.03, 0.04, 0.03, 0.02]
    }, index=pd.date_range("2025-01-01", periods=5, freq="H"))
    result_df, final_balance = manage_risk(dummy_df, 10000.0, min_position_size=0.01)
    print(f"Result DataFrame:\n{result_df}")
    print(f"Final Balance: {final_balance:.2f}")