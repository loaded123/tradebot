# src/strategy/position_sizer.py
"""
This module provides functions for calculating position sizes based on risk management strategies,
including ATR-based sizing, Kelly Criterion, and Monte Carlo simulations. It ensures positions are
aligned with risk limits and market conditions.

Key Integrations:
- **src.strategy.signal_generator**: Uses position sizes in signal generation, aligning with ATR-based
  stop-loss/take-profit calculations.
- **src.strategy.risk_manager**: Applies position sizes during risk management, ensuring consistency
  with ATR multipliers and risk percentages.
- **src.strategy.backtest_engine**: Uses position sizes during backtesting to calculate trade outcomes.

Future Considerations:
- Add support for dynamic risk adjustments based on portfolio performance or market regime.
- Incorporate slippage and transaction costs into position sizing calculations.
- Implement more advanced sizing strategies (e.g., volatility-adjusted Kelly Criterion).

Dependencies:
- numpy
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_position_size(current_balance: float, atr: float, last_close: float, max_risk_pct: float = 0.20) -> float:
    """
    Calculate position size based on ATR, aligned with risk management settings.

    Args:
        current_balance (float): Current portfolio balance.
        atr (float): Average True Range value for the current period.
        last_close (float): Last closing price of the asset.
        max_risk_pct (float): Maximum risk percentage per trade (default: 0.20).

    Returns:
        float: Number of units to trade, capped at 0.005 BTC to avoid excessive overrides.

    Notes:
        - Uses a 2x ATR stop-loss distance to calculate position size.
        - Caps position size to ensure trade viability and avoid excessive risk.
    """
    try:
        if any(x <= 0 for x in [current_balance, atr, last_close]):
            raise ValueError("Inputs must be positive")
        
        risk_amount = current_balance * max_risk_pct
        stop_loss_distance = atr * 2  # Aligned with 2 ATR stop-loss in signal_generator and risk_manager
        position_size = risk_amount / stop_loss_distance
        units = position_size / last_close
        units = min(units, 0.005)  # Cap position size at 0.005 BTC to avoid excessive overrides
        return units if units > 0 else 0
    
    except Exception as e:
        logger.error(f"Error in position size calculation: {e}")
        return 0

def kelly_criterion(win_rate: float, risk_reward_ratio: float, current_balance: float, atr: float, last_close: float, max_risk_pct: float = 0.20) -> float:
    """
    Calculate optimal position size using the Kelly Criterion, aligned with risk management settings.

    Args:
        win_rate (float): Historical win rate of the strategy (between 0 and 1).
        risk_reward_ratio (float): Risk-reward ratio of the strategy.
        current_balance (float): Current portfolio balance.
        atr (float): Average True Range value (used for consistency with other methods).
        last_close (float): Last closing price of the asset.
        max_risk_pct (float): Maximum risk percentage per trade (default: 0.20).

    Returns:
        float: Number of units to trade, capped at 0.005 BTC to avoid excessive overrides.

    Notes:
        - Applies the Kelly Criterion formula to determine the optimal fraction of capital to risk.
        - Caps the position size to avoid excessive risk.
    """
    try:
        if not (0 <= win_rate <= 1) or risk_reward_ratio <= 0 or current_balance <= 0:
            raise ValueError("Invalid inputs for Kelly Criterion")
        
        kelly_fraction = (win_rate * risk_reward_ratio - (1 - win_rate)) / risk_reward_ratio
        if kelly_fraction <= 0:
            return 0
        
        position_size = current_balance * kelly_fraction
        max_position_value = current_balance * max_risk_pct
        position_size = min(position_size, max_position_value)
        units = position_size / last_close
        units = min(units, 0.005)  # Cap position size at 0.005 BTC to avoid excessive overrides
        return units if units > 0 else 0
    
    except Exception as e:
        logger.error(f"Error in Kelly Criterion: {e}")
        return 0

def monte_carlo_position_sizing(account_balance: float, risk_per_trade: float, win_rate: float, avg_win: float, avg_loss: float, num_simulations: int = 10000) -> float:
    """
    Perform Monte Carlo simulation to determine position sizing based on simulated trade outcomes.

    Args:
        account_balance (float): Current portfolio balance.
        risk_per_trade (float): Risk percentage per trade.
        win_rate (float): Historical win rate of the strategy (between 0 and 1).
        avg_win (float): Average profit per winning trade.
        avg_loss (float): Average loss per losing trade.
        num_simulations (int): Number of Monte Carlo simulations (default: 10000).

    Returns:
        float: Optimal fraction of capital to risk, capped at risk_per_trade.

    Notes:
        - Simulates 100 trades per simulation to estimate portfolio growth.
        - Returns the average growth factor, capped at the specified risk_per_trade.
    """
    try:
        if any(x <= 0 for x in [account_balance, risk_per_trade, avg_win, avg_loss]) or not (0 <= win_rate <= 1):
            raise ValueError("Invalid inputs for Monte Carlo")
        
        risk_amount = account_balance * risk_per_trade
        results = []
        
        for _ in range(num_simulations):
            balance = account_balance
            for _ in range(100):  # 100 trades per simulation
                if np.random.rand() < win_rate:
                    balance += avg_win
                else:
                    balance -= avg_loss
                balance = max(balance, 0)
            results.append(balance)
        
        optimal_fraction = np.mean(results) / account_balance
        return min(optimal_fraction, risk_per_trade)  # Cap at risk_per_trade
    
    except Exception as e:
        logger.error(f"Error in Monte Carlo sizing: {e}")
        return 0

if __name__ == "__main__":
    # Dummy test
    print(kelly_criterion(0.55, 2.0, 10000, 1, 100, max_risk_pct=0.20))
    print(monte_carlo_position_sizing(10000, 0.20, 0.55, 100, 50))