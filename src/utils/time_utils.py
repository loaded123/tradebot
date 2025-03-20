# src/utils/time_utils.py
import pandas as pd
from typing import List, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Define Bitcoin halving dates (approximate dates for demonstration; adjust as needed)
HALVING_DATES = [
    pd.Timestamp("2012-11-28", tz='UTC'),  # First halving
    pd.Timestamp("2016-07-09", tz='UTC'),  # Second halving
    pd.Timestamp("2020-05-11", tz='UTC'),  # Third halving
    pd.Timestamp("2024-04-20", tz='UTC'),  # Fourth halving
    pd.Timestamp("2028-03-15", tz='UTC'),  # Fifth halving (estimated)
]

def calculate_days_to_next_halving(current_time: pd.Timestamp, halving_dates: List[pd.Timestamp] = HALVING_DATES) -> Tuple[int, pd.Timestamp]:
    """
    Calculate the number of days to the next Bitcoin halving event.
    
    Args:
        current_time (pd.Timestamp): The current timestamp.
        halving_dates (List[pd.Timestamp]): List of halving dates (default: HALVING_DATES).
    
    Returns:
        Tuple[int, pd.Timestamp]: Days to the next halving and the date of the next halving.
    """
    # Ensure current_time is timezone-aware (UTC)
    if current_time.tz is None:
        logger.warning(f"current_time {current_time} is timezone-naive, localizing to UTC")
        current_time = current_time.tz_localize('UTC')
    elif current_time.tz != pd.Timestamp("2020-01-01").tz_localize('UTC').tz:
        logger.warning(f"current_time timezone {current_time.tz} differs from UTC, converting to UTC")
        current_time = current_time.tz_convert('UTC')

    # Ensure all halving dates are timezone-aware (UTC)
    halving_dates = [
        h.tz_localize('UTC') if h.tz is None else h.tz_convert('UTC') for h in halving_dates
    ]

    future_halvings = [h for h in halving_dates if h > current_time]
    if not future_halvings:
        return 9999, halving_dates[-1]  # Far future if no upcoming halving
    
    next_halving = min(future_halvings)
    days_to_next = (next_halving - current_time).days
    return max(days_to_next, 0), next_halving

def calculate_days_since_last_halving(current_time: pd.Timestamp, halving_dates: List[pd.Timestamp] = HALVING_DATES) -> int:
    """
    Calculate the number of days since the last Bitcoin halving event.
    
    Args:
        current_time (pd.Timestamp): The current timestamp.
        halving_dates (List[pd.Timestamp]): List of halving dates (default: HALVING_DATES).
    
    Returns:
        int: Days since the last halving. Returns 0 if no previous halving exists.
    """
    # Ensure current_time is timezone-aware (UTC)
    if current_time.tz is None:
        logger.warning(f"current_time {current_time} is timezone-naive, localizing to UTC")
        current_time = current_time.tz_localize('UTC')
    elif current_time.tz != pd.Timestamp("2020-01-01").tz_localize('UTC').tz:
        logger.warning(f"current_time timezone {current_time.tz} differs from UTC, converting to UTC")
        current_time = current_time.tz_convert('UTC')

    # Ensure all halving dates are timezone-aware (UTC)
    halving_dates = [
        h.tz_localize('UTC') if h.tz is None else h.tz_convert('UTC') for h in halving_dates
    ]

    past_halvings = [h for h in halving_dates if h <= current_time]
    if not past_halvings:
        logger.warning(f"No past halving events found for timestamp {current_time}. Returning 0.")
        return 0  # No previous halving

    last_halving = max(past_halvings)
    days_since_last = (current_time - last_halving).days
    return max(days_since_last, 0)