import numpy as np

def calculate_position_size(current_balance, atr, last_close, max_risk_pct=0.02):
    """
    Calculate position size based on ATR and current account balance.

    :param current_balance: Current balance of the trading account
    :param atr: Average True Range from the last known value
    :param last_close: Last closing price
    :param max_risk_pct: Maximum percentage of account balance to risk per trade
    :return: Number of units to trade
    """
    risk_amount = current_balance * max_risk_pct
    stop_loss_distance = atr * 2  # Example: 2x ATR for stop loss
    position_size = risk_amount / stop_loss_distance
    units = position_size / last_close
    return units

def kelly_criterion(win_rate, reward_risk_ratio):
    """
    Calculate the optimal position size using the Kelly Criterion.

    :param win_rate: Probability of a successful trade (0 to 1)
    :param reward_risk_ratio: Expected reward-to-risk ratio
    :return: Optimal fraction of capital to risk per trade
    """
    if reward_risk_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0  # Avoid invalid calculations
    kelly_fraction = win_rate - ((1 - win_rate) / reward_risk_ratio)
    return max(0, min(kelly_fraction, 1))  # Keep it between 0 and 1

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