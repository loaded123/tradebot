import numpy as np

def calculate_position_size(current_balance, atr, last_close, max_risk_pct=0.02):
    """Calculate position size based on ATR and current account balance."""
    risk_amount = current_balance * max_risk_pct
    stop_loss_distance = atr * 2  # Example: 2x ATR for stop loss
    position_size = risk_amount / stop_loss_distance
    units = position_size / last_close
    return units

def kelly_criterion(win_rate, risk_reward_ratio, current_balance, atr, last_close, max_risk_pct=0.02):
    """Calculates the optimal position size using the Kelly Criterion."""

    kelly_fraction = (win_rate * risk_reward_ratio - (1 - win_rate)) / risk_reward_ratio
    if kelly_fraction <= 0:
        return 0  # No trade

    position_size = current_balance * kelly_fraction

    # Apply maximum risk percentage constraint
    max_position_value = current_balance * max_risk_pct  # Constrain position size
    position_size = min(position_size, max_position_value)  # Constrain position size

    units = position_size / last_close  # Calculate the number of units to trade
    return units

def monte_carlo_position_sizing(account_balance, risk_per_trade, win_rate, avg_win, avg_loss, num_simulations=10000):
    """
    Monte Carlo simulation for position sizing optimization.
    - account_balance: The current account balance.
    - risk_per_trade: The risk per trade (e.g., percentage of account balance).
    - win_rate: The probability of a winning trade.
    - avg_win: The average profit from a winning trade.
    - avg_loss: The average loss from a losing trade.
    - num_simulations: The number of Monte Carlo simulations to run.
    """
    # Risk per trade in monetary value
    risk_amount = account_balance * risk_per_trade

    # Arrays to store results of each simulation
    results = []

    for _ in range(num_simulations):
        # Simulate a sequence of trades
        simulated_account_balance = account_balance
        for _ in range(100):  # Number of trades in each simulation
            if np.random.rand() < win_rate:
                # Win trade
                simulated_account_balance += avg_win
            else:
                # Loss trade
                simulated_account_balance -= avg_loss
            # Ensure account balance doesn't drop below zero
            simulated_account_balance = max(simulated_account_balance, 0)
        
        results.append(simulated_account_balance)

    # Calculate optimal position size based on simulation results
    optimal_position_size = np.mean(results) / account_balance
    return optimal_position_size