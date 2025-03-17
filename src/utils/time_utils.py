# src/utils/time_utils.py
import pandas as pd
from typing import List, Tuple

def calculate_days_to_next_halving(current_time: pd.Timestamp, halving_dates: List[pd.Timestamp]) -> Tuple[int, pd.Timestamp]:
    """
    Calculate the number of days to the next Bitcoin halving event.
    
    Args:
        current_time (pd.Timestamp): The current timestamp.
        halving_dates (List[pd.Timestamp]): List of halving dates.
    
    Returns:
        Tuple[int, pd.Timestamp]: Days to the next halving and the date of the next halving.
    """
    future_halvings = [h for h in halving_dates if h > current_time]
    if not future_halvings:
        return 9999, halving_dates[-1]  # Far future if no upcoming halving
    
    next_halving = min(future_halvings)
    days_to_next = (next_halving - current_time).days
    return max(days_to_next, 0), next_halving