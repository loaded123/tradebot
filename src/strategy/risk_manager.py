# src/strategy/risk_manager.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def manage_risk(signal_data: pd.DataFrame, current_balance: float, max_drawdown_pct: float = 0.10,
                atr_multiplier: float = 2.0, recovery_volatility_factor: float = 0.15,
                max_risk_pct: float = 0.10, min_position_size: float = 0.002) -> tuple:
    """
    Apply risk management to the signal data, including drawdown limits, dynamic position sizing,
    and tighter stop-losses.

    Args:
        signal_data: DataFrame containing trading signals and market data.
        current_balance: Current portfolio balance.
        max_drawdown_pct: Maximum allowable drawdown percentage before pausing trades.
        atr_multiplier: Multiplier for ATR to set stop-loss and take-profit levels.
        recovery_volatility_factor: Volatility factor for resuming trading after a drawdown.
        max_risk_pct: Maximum risk percentage per trade.
        min_position_size: Minimum position size to ensure trade viability.

    Returns:
        tuple: Updated signal_data and current_balance.
    """
    try:
        signal_data = signal_data.copy()
        initial_balance = current_balance
        peak_balance = initial_balance
        trading_paused = False

        # Ensure necessary columns exist
        required_cols = ['close', 'atr', 'price_volatility', 'signal', 'position_size', 'take_profit', 'stop_loss']
        for col in required_cols:
            if col not in signal_data.columns:
                raise ValueError(f"Missing required column: {col}")

        signal_data['portfolio_value'] = initial_balance  # Initialize portfolio value
        mean_volatility = signal_data['price_volatility'].mean()

        for idx in signal_data.index:
            # Update portfolio value (this will be updated in backtest_strategy)
            # For now, assume portfolio_value is computed later in backtest
            portfolio_value = signal_data.loc[idx, 'portfolio_value']

            # Update peak balance and check drawdown
            peak_balance = max(peak_balance, portfolio_value)
            current_drawdown = (peak_balance - portfolio_value) / peak_balance

            # Pause trading if drawdown exceeds the limit
            if current_drawdown > max_drawdown_pct:
                trading_paused = True
                logging.warning(f"Trading paused at {idx}: Drawdown {current_drawdown:.2%} exceeds max {max_drawdown_pct:.2%}")
            # Resume trading if volatility decreases and drawdown recovers
            elif trading_paused and current_drawdown < (max_drawdown_pct / 2) and \
                 signal_data.loc[idx, 'price_volatility'] < mean_volatility * (1 - recovery_volatility_factor):
                trading_paused = False
                logging.info(f"Trading resumed at {idx}: Drawdown {current_drawdown:.2%}, Volatility acceptable")

            # Skip trading if paused
            if trading_paused:
                signal_data.loc[idx, 'signal'] = 0
                signal_data.loc[idx, 'position_size'] = 0
                continue

            # Dynamic position sizing based on volatility and ATR
            atr = signal_data.loc[idx, 'atr']
            price_volatility = signal_data.loc[idx, 'price_volatility']
            position_size = signal_data.loc[idx, 'position_size']

            # Reduce position size in high volatility
            if price_volatility > mean_volatility * 1.5:
                position_size *= 0.5
                logging.debug(f"Reduced position size at {idx} due to high volatility: {position_size:.6f}")
            # Cap position size based on risk
            risk_amount = current_balance * max_risk_pct
            stop_loss_distance = atr * atr_multiplier
            position_size = min(position_size, risk_amount / (stop_loss_distance * signal_data.loc[idx, 'close']))
            position_size = max(position_size, min_position_size)

            signal_data.loc[idx, 'position_size'] = position_size

            # Tighter stop-losses: Adjust based on ATR and market conditions
            if signal_data.loc[idx, 'signal'] == 1:  # Buy
                stop_loss = signal_data.loc[idx, 'close'] - (atr * 1.0)  # Tighter stop-loss (1x ATR)
                take_profit = signal_data.loc[idx, 'close'] + (atr * 3.0)  # 1:3 risk/reward ratio
            elif signal_data.loc[idx, 'signal'] == -1:  # Sell
                stop_loss = signal_data.loc[idx, 'close'] + (atr * 1.0)
                take_profit = signal_data.loc[idx, 'close'] - (atr * 3.0)
            else:
                stop_loss = signal_data.loc[idx, 'stop_loss']
                take_profit = signal_data.loc[idx, 'take_profit']

            signal_data.loc[idx, 'stop_loss'] = stop_loss
            signal_data.loc[idx, 'take_profit'] = take_profit
            logging.debug(f"Adjusted levels at {idx}: Stop_Loss={stop_loss:.2f}, Take_Profit={take_profit:.2f}")

        return signal_data, current_balance

    except Exception as e:
        logging.error(f"Error in risk management: {e}")
        raise