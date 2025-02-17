# src/utils/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Gemini API credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_SECRET = os.getenv("GEMINI_API_SECRET")

# Ensure they are loaded correctly
if not GEMINI_API_KEY or not GEMINI_API_SECRET:
    raise ValueError("❌ Missing GEMINI API credentials. Check your .env file.")

class Config:
    # Trading pair and other configurations
    TRADING_PAIR = 'BTC/USD'  # Example
    TIME_FRAME = '1m'  # 1-minute timeframe
    HISTORICAL_DATA_POINTS = 1000  # Number of historical data points to fetch
    POLLING_INTERVAL = 60  # Time in seconds between fetches
    ERROR_DELAY = 10  # Delay on error

    # Simulation mode toggle (set to True for testing)
    SIMULATION_MODE = True  # Set to True for testing, False for live trading
    
    # API Keys (for live trading)
    API_KEY = GEMINI_API_KEY
    API_SECRET = GEMINI_API_SECRET

    # Strategy and model parameters
    STRATEGY_NAME = 'simple_strategy'

    # Set other configuration values as needed (e.g., risk per trade, stop-loss settings, etc.)
    RISK_PER_TRADE = 0.02  # Example: 2% risk per trade
    INITIAL_CAPITAL = 10000  # Example initial capital for backtesting or real-time trading
