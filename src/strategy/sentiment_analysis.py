# src/strategy/sentiment_analysis.py
"""
This module simulates sentiment analysis for the trading bot, estimating market sentiment based on
price trends, technical indicators, and volatility. It provides simulated X sentiment, historical sentiment,
Fear and Greed Index, and whale move indicators.

Key Integrations:
- **src.data.data_preprocessor**: Uses preprocessed data with features like 'close', 'momentum_rsi',
  'trend_macd', and 'volume' to compute sentiment scores.
- **src.strategy.signal_generator**: Can integrate sentiment scores to adjust signal confidence or weights.
- **src.strategy.backtest_visualizer_ultimate**: Provides preprocessed data for sentiment analysis during backtesting.

Future Considerations:
- Replace simulated X sentiment with actual API data when available (e.g., using X API for tweet sentiment).
- Incorporate more advanced sentiment analysis (e.g., NLP on news articles or social media).
- Add support for real-time sentiment updates in live trading scenarios.

Dependencies:
- pandas
- numpy
- typing.Dict
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def fetch_x_sentiment(preprocessed_data: pd.DataFrame, idx: pd.Timestamp, suppress_logs: bool = False) -> float:
    """
    Simulate X sentiment based on price trend, RSI, volume change, and Fear & Greed Index.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed market data with columns like 'close', 'momentum_rsi', 'volume'.
        idx (pd.Timestamp): Timestamp for sentiment calculation.
        suppress_logs (bool): If True, suppress logging of sentiment simulation details (default: False).

    Returns:
        float: Simulated sentiment score between -1.0 and 1.0.

    Notes:
        - Combines price change, RSI, volume change, and a simulated Fear & Greed Index to estimate sentiment.
        - Clips the final sentiment score to ensure it stays within the valid range.
    """
    if not suppress_logs:
        logger.warning(f"Simulating X sentiment at {idx}. Free X API tier does not support tweet search.")

    window = 24
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 0.0

    # Price change sentiment
    price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0]
    price_sentiment = price_change * 5

    # RSI sentiment
    rsi = historical_data['momentum_rsi'].iloc[-1] if 'momentum_rsi' in historical_data.columns else 50.0
    if pd.isna(rsi):
        rsi = 50.0
    rsi_sentiment = -0.3 if rsi > 70 else 0.3 if rsi < 30 else 0.0

    # Volume sentiment
    volume_change = (historical_data['volume'].iloc[-1] - historical_data['volume'].iloc[0]) / historical_data['volume'].iloc[0] if historical_data['volume'].iloc[0] != 0 else 0
    volume_sentiment = 0.2 if volume_change > 0.1 else -0.2 if volume_change < -0.1 else 0.0
    if price_change < 0:
        volume_sentiment *= -1

    # Fear and Greed Index sentiment
    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    if pd.isna(volatility):
        volatility = 0.0
    fgi = np.clip(75 - (volatility * 1000), 0, 100)
    fgi_sentiment = -0.2 if fgi > 70 else 0.2 if fgi < 30 else 0.0

    # Combine sentiments
    sentiment = price_sentiment + rsi_sentiment + volume_sentiment + fgi_sentiment
    sentiment = np.clip(sentiment, -1.0, 1.0)

    if not suppress_logs:
        logger.info(f"Simulated X sentiment at {idx}: {sentiment:.2f}")

    return sentiment

def calculate_historical_sentiment(preprocessed_data: pd.DataFrame, idx: pd.Timestamp) -> float:
    """
    Estimate historical sentiment based on RSI, MACD, and price trend.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed market data with columns like 'close', 'momentum_rsi', 'trend_macd'.
        idx (pd.Timestamp): Timestamp for sentiment calculation.

    Returns:
        float: Sentiment score between -1.0 and 1.0.

    Notes:
        - Combines price change, RSI, and MACD to estimate historical market sentiment.
        - Clips the final sentiment score to ensure it stays within the valid range.
    """
    window = 24
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2 or 'momentum_rsi' not in historical_data.columns or 'trend_macd' not in historical_data.columns:
        return 0.0

    price_change = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[0]) / historical_data['close'].iloc[0] if len(historical_data) > 1 else 0
    rsi = historical_data['momentum_rsi'].iloc[-1]
    if pd.isna(rsi):
        rsi = 50.0
    rsi_sentiment = -0.5 if rsi > 70 else 0.5 if rsi < 30 else 0.0

    macd = historical_data['trend_macd'].iloc[-1]
    if pd.isna(macd):
        macd = 0.0
    macd_sentiment = 0.3 if macd > 0 else -0.3 if macd < 0 else 0.0

    sentiment = np.clip(price_change * 5 + rsi_sentiment + macd_sentiment, -1.0, 1.0)
    return min(sentiment, 0.3)

async def get_fear_and_greed_index(preprocessed_data: pd.DataFrame, idx: pd.Timestamp) -> float:
    """
    Simulate Fear and Greed Index based on price volatility.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed market data with 'close' column.
        idx (pd.Timestamp): Timestamp for calculation.

    Returns:
        float: Simulated Fear and Greed Index (0 to 100).

    Notes:
        - Estimates the Fear and Greed Index using price volatility over a 24-period window.
        - Clips the result to ensure it stays within the valid range.
    """
    window = 24
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 50.0

    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    if pd.isna(volatility):
        volatility = 0.0
    fgi = np.clip(75 - (volatility * 1000), 0, 100)
    return fgi

def simulate_historical_whale_moves(preprocessed_data: pd.DataFrame, idx: pd.Timestamp) -> float:
    """
    Simulate historical whale moves based on price volatility.

    Args:
        preprocessed_data (pd.DataFrame): Preprocessed market data with 'close' column.
        idx (pd.Timestamp): Timestamp for calculation.

    Returns:
        float: Simulated whale moves score (0 to 1).

    Notes:
        - Estimates whale activity using price volatility over a 24-period window.
        - Clips the result to ensure it stays within the valid range.
    """
    window = 24
    historical_data = preprocessed_data.loc[:idx].tail(window) if len(preprocessed_data.loc[:idx]) >= window else preprocessed_data.loc[:idx]
    if len(historical_data) < 2:
        return 0.0

    volatility = historical_data['close'].pct_change().std() * np.sqrt(24)
    if pd.isna(volatility):
        volatility = 0.0
    whale_moves = np.clip(volatility * 50, 0, 1)
    return whale_moves