# src/strategy/position_sizer.py

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def calculate_position_size(current_balance, atr, last_close, max_risk_pct=0.1):
    """Calculate position size based on ATR."""
    try:
        if any(x <= 0 for x in [current_balance, atr, last_close]):
            raise ValueError("Inputs must be positive")
        
        risk_amount = current_balance * max_risk_pct
        stop_loss_distance = atr * 2
        position_size = risk_amount / stop_loss_distance
        units = position_size / last_close
        return units if units > 0 else 0
    
    except Exception as e:
        logging.error(f"Error in position size calculation: {e}")
        return 0

def kelly_criterion(win_rate, risk_reward_ratio, current_balance, atr, last_close, max_risk_pct=0.05):
    """Calculate optimal position size using Kelly Criterion."""
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
        return units if units > 0 else 0
    
    except Exception as e:
        logging.error(f"Error in Kelly Criterion: {e}")
        return 0

def monte_carlo_position_sizing(account_balance, risk_per_trade, win_rate, avg_win, avg_loss, num_simulations=10000):
    """Monte Carlo simulation for position sizing."""
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
        logging.error(f"Error in Monte Carlo sizing: {e}")
        return 0

if __name__ == "__main__":
    # Dummy test
    print(kelly_criterion(0.55, 2.0, 10000, 1, 100, max_risk_pct=0.1))  # Updated default max_risk_pct
    print(monte_carlo_position_sizing(10000, 0.1, 0.55, 100, 50))  # Updated risk_per_trade for consistency