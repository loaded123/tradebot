# src/data/gemini_websocket.py

import asyncio
import torch
import sys
import os
import json
import logging
import signal
from typing import Optional
import ccxt.async_support as ccxt
from dotenv import load_dotenv
import websocket  # Explicitly import websocket for debugging
import numpy as np
import pandas as pd

# Add `src/` directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.transformer_model import TransformerPredictor
from src.data.data_preprocessor import preprocess_data, feature_columns
from src.data.data_fetcher import fetch_historical_data
from src.strategy.strategy_generator import generate_signals  # Use for signal generation
from src.strategy.position_sizer import kelly_criterion
from src.api.gemini import create_gemini_exchange
from src.constants import FEATURE_COLUMNS  # Import feature columns for consistency
from src.utils.logger import logger

# Setup logging for better visibility into WebSocket events
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# WebSocket URL for Gemini
WS_URL = "wss://api.gemini.com/v1/marketdata/BTCUSD"

# Global variables for WebSocket and model
ws = None
model = None
context_buffer = None  # Buffer for maintaining historical context

async def load_model():
    """Asynchronously load the trained TransformerPredictor model."""
    # Path to model in the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_path = os.path.join(root_dir, 'best_model.pth')
    
    # Add debugging to verify the path
    logging.debug(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Please ensure the file exists at the root directory.")
        raise FileNotFoundError(f"No such file or directory: '{model_path}'")
    
    model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS), d_model=64, n_heads=4, n_layers=2, dropout=0.1)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        logging.info(f"TransformerPredictor loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
    
    return model

async def initialize_context_buffer(symbol: str = 'BTC/USD'):
    """Initialize the context buffer with historical data for context_length=10."""
    global context_buffer
    historical_data = await fetch_historical_data(symbol)
    preprocessed_data = preprocess_data(historical_data)
    context_buffer = preprocessed_data[FEATURE_COLUMNS].tail(10).values  # Shape [10, 17]
    if len(context_buffer) < 10:
        context_buffer = np.pad(context_buffer, ((10 - len(context_buffer), 0), (0, 0)), mode='constant')  # Pad with zeros
    logging.debug(f"Initialized context_buffer shape: {context_buffer.shape}")

async def calculate_atr(exchange: ccxt.Exchange, symbol: str, window: int = 14) -> float:
    """
    Asynchronously calculate Average True Range (ATR) from historical data.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USD')
        window: ATR window period
    
    Returns:
        float: ATR value
    """
    historical_data = await fetch_historical_data(symbol)
    preprocessed_data = preprocess_data(historical_data)
    high = preprocessed_data['high'].tail(window).values
    low = preprocessed_data['low'].tail(window).values
    close = preprocessed_data['close'].tail(window).values
    tr = np.max([high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))], axis=0)[1:]
    atr = np.mean(tr)
    logging.debug(f"Calculated ATR for {symbol}: {atr}")
    return atr

async def calculate_trade_amount(exchange: ccxt.Exchange, symbol: str = 'BTC/USD') -> float:
    """
    Asynchronously calculate the amount to trade based on current balance and risk management.
    
    Args:
        exchange: CCXT exchange instance
        symbol: Trading pair (e.g., 'BTC/USD')
    
    Returns:
        float: Amount to trade (in BTC)
    """
    try:
        logging.debug(f"Fetching balance for {symbol}")
        balance = await exchange.fetch_balance()
        btc_balance = balance['free'].get('BTC', 0.0)
        usd_balance = balance['free'].get('USD', 0.0)
        
        # Fetch current price for USD to BTC conversion
        logging.debug(f"Fetching ticker for {symbol}")
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Calculate ATR for risk management
        atr = await calculate_atr(exchange, symbol)
        
        # Use Kelly Criterion for position sizing
        position_size = kelly_criterion(0.5, 2.0, 10000, atr, current_price, 0.05)  # 5% max risk
        
        logging.debug(f"BTC balance: {btc_balance}, USD balance: {usd_balance}, Current price: {current_price}, Position size: {position_size}")
        if btc_balance > position_size:
            return position_size
        elif usd_balance > (position_size * current_price):
            return min(position_size, usd_balance / current_price)  # Convert USD to BTC
        return 0  # If neither condition is met, return 0
    except Exception as e:
        logging.error(f"Error calculating trade amount: {e}")
        return 0

async def set_stop_loss(exchange: ccxt.Exchange, symbol: str, amount: float, entry_price: float, stop_loss_price: float):
    """Asynchronously place a stop-loss order"""
    try:
        logging.debug(f"Setting stop-loss for {symbol} at {stop_loss_price}")
        # Gemini supports stop-limit orders; adjust accordingly
        if not exchange.has['createStopLimitOrder']:
            logging.warning("Gemini does not support stop-limit orders directly; using limit order instead.")
            limit_order = await exchange.create_limit_sell_order(symbol, amount, stop_loss_price)
            logging.info(f"Stop-loss limit order set at {stop_loss_price}: {limit_order}")
        else:
            stop_loss_order = await exchange.create_stop_limit_order(symbol, 'sell', amount, stop_loss_price, stop_loss_price)
            logging.info(f"Stop-loss order set at {stop_loss_price}: {stop_loss_order}")
    except Exception as e:
        logging.error(f"Failed to set stop-loss: {e}")

async def set_take_profit(exchange: ccxt.Exchange, symbol: str, amount: float, entry_price: float, take_profit_price: float):
    """Asynchronously place a take-profit order"""
    try:
        logging.debug(f"Setting take-profit for {symbol} at {take_profit_price}")
        # Gemini supports limit orders for take-profit; adjust accordingly
        take_profit_order = await exchange.create_limit_sell_order(symbol, amount, take_profit_price)
        logging.info(f"Take-profit limit order set at {take_profit_price}: {take_profit_order}")
    except Exception as e:
        logging.error(f"Failed to set take-profit: {e}")

async def on_message(ws, message):
    """Asynchronously handle incoming WebSocket messages."""
    global model, context_buffer
    try:
        logging.debug(f"Received WebSocket message: {message}")
        data = json.loads(message)
        
        if 'events' in data:
            for event in data['events']:
                if event['type'] == 'trade':
                    logging.info(f"Trade event received: {event}")
                    
                    # Extract the price and timestamp for prediction
                    price = float(event['price'])
                    timestamp = pd.to_datetime(event['timestamp'], unit='ms')

                    # Prepare data for prediction using context_buffer
                    latest_data = pd.DataFrame({
                        'open': [price], 'high': [price], 'low': [price], 'close': [price], 'volume': [0]
                    }, index=[timestamp])
                    preprocessed_data = preprocess_data(latest_data)

                    # Update context_buffer with the latest data
                    context_buffer = np.roll(context_buffer, -1, axis=0)  # Shift buffer
                    context_buffer[-1] = preprocessed_data[FEATURE_COLUMNS].values[0]  # Add latest data

                    # Prepare data for prediction (shape [1, 10, 17] to match Transformer input)
                    X = context_buffer.reshape(1, 10, len(FEATURE_COLUMNS))  # Shape [1, 10, 17]
                    X_tensor = torch.FloatTensor(X)

                    # Predict using Transformer model
                    device = torch.device('cpu')  # Use CPU for WebSocket (simpler for real-time)
                    with torch.no_grad():
                        past_time_features = torch.FloatTensor(np.zeros((1, 10, 1)))  # Dummy past time features [1, 10, 1]
                        past_observed_mask = torch.ones((1, 10, len(FEATURE_COLUMNS)))  # All observed [1, 10, 17]
                        future_values = torch.FloatTensor(np.zeros((1, 1, 1)))  # Dummy future values [1, 1, 1]
                        future_time_features = torch.FloatTensor(np.zeros((1, 1, 1)))  # Dummy future time features [1, 1, 1]

                        predictions = model.predict(
                            X_tensor.to(device),
                            past_time_features=past_time_features.to(device),
                            past_observed_mask=past_observed_mask.to(device),
                            future_values=future_values.to(device),
                            future_time_features=future_time_features.to(device)
                        )

                    # Extract predicted price (assuming output is [1, 1])
                    predicted_price = predictions.cpu().numpy().flatten()[0]
                    current_price = price

                    # Generate signals using the Transformer prediction (simplified logic)
                    signal = 1 if predicted_price > current_price * 1.001 else -1 if predicted_price < current_price * 0.999 else 0  # 0.1% threshold

                    if signal != 0:
                        exchange = await create_gemini_exchange()
                        amount = await calculate_trade_amount(exchange, 'BTC/USD')
                        if amount > 0:
                            if signal == 1:  # Buy signal
                                logging.debug(f"Placing buy order for {amount} BTC at current price {current_price}")
                                order = await exchange.create_market_buy_order('BTC/USD', amount)
                                logging.info(f"Buy order placed: {order}")
                                
                                # Set stop-loss and take-profit
                                entry_price = current_price
                                stop_loss_price = entry_price * 0.95  # 5% below entry (adjust based on strategy)
                                take_profit_price = entry_price * 1.05  # 5% above entry (adjust based on strategy)
                                await set_stop_loss(exchange, 'BTC/USD', amount, entry_price, stop_loss_price)
                                await set_take_profit(exchange, 'BTC/USD', amount, entry_price, take_profit_price)
                            elif signal == -1:  # Sell signal
                                logging.debug(f"Placing sell order for {amount} BTC at current price {current_price}")
                                order = await exchange.create_market_sell_order('BTC/USD', amount)
                                logging.info(f"Sell order placed: {order}")
                        else:
                            logging.info("Not enough funds or assets for trading")
                        await exchange.close()  # Close the exchange after use
                        logging.debug("Exchange instance closed")
    except Exception as e:
        logging.error(f"Error in on_message: {e}")
        if 'exchange' in locals():
            await exchange.close()
            logging.debug("Exchange instance closed due to exception")
        raise

def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info(f"Closed connection with status {close_status_code}: {close_msg}")

def on_open(ws):
    logging.info("Connected to WebSocket")
    
    # Subscribe to market data
    subscription_message = {
        "type": "subscribe",
        "channels": [{"name": "live_trades", "symbols": ["BTCUSD"]}]
    }
    logging.debug(f"Sending subscription message: {subscription_message}")
    ws.send(json.dumps(subscription_message))

async def start_websocket():
    """Asynchronously start the WebSocket connection."""
    global ws, model, context_buffer
    try:
        # Set SelectorEventLoop policy for Windows to support WebSocket (optional, but kept for WebSocket compatibility)
        logging.debug(f"Current event loop: {asyncio.get_event_loop().__class__.__name__}")
        if sys.platform == "win32":
            loop = asyncio.get_event_loop()
            if loop.__class__.__name__ == "ProactorEventLoop":
                from asyncio import SelectorEventLoop
                asyncio.set_event_loop_policy(SelectorEventLoop())
                logging.info("Switched to SelectorEventLoop for Windows compatibility with WebSocket")

        # Load model and initialize context buffer before starting WebSocket
        model = await load_model()
        await initialize_context_buffer('BTC/USD')
        
        logging.debug(f"Initializing WebSocketApp with URL: {WS_URL}")
        ws = websocket.WebSocketApp(
            WS_URL,
            on_message=lambda ws, msg: asyncio.run(on_message(ws, msg)),
            on_error=on_error,
            on_close=on_close
        )
        ws.on_open = on_open
        # Run WebSocket in the current event loop
        logging.debug("Starting WebSocket run_forever in thread")
        await asyncio.to_thread(ws.run_forever)
    except Exception as e:
        logging.error(f"Error in start_websocket: {e}")
        raise

async def signal_handler(sig, frame):
    """Asynchronously handle shutdown signals."""
    global ws
    logging.info('Shutting down the bot...')
    if ws:
        ws.close()
        logging.debug("WebSocket connection closed on shutdown")
    sys.exit(0)

async def main():
    """Main async function to run the WebSocket bot."""
    try:
        logging.debug("Starting WebSocket bot")
        
        # Register signal handler
        loop = asyncio.get_event_loop()
        logging.debug(f"Current event loop policy: {loop.__class__.__name__}")
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(signal_handler(sig, None)))
        
        # Start WebSocket
        await start_websocket()
    except Exception as e:
        logging.error(f"Error running WebSocket bot: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot terminated by user")
    except Exception as e:
        logging.error(f"Error running WebSocket bot: {e}")