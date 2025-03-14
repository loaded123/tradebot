# src/data/gemini_websocket.py

import asyncio
import sys
import os
import torch
import json
import logging
import signal
from typing import Optional
import ccxt.async_support as ccxt
from dotenv import load_dotenv
import websocket  # Explicitly import websocket for debugging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add `src/` directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.transformer_model import TransformerPredictor
from src.data.data_preprocessor import preprocess_data
from src.data.data_fetcher import fetch_historical_data
from src.strategy.signal_generator import generate_signals
from src.strategy.position_sizer import kelly_criterion
from src.strategy.risk_manager import manage_risk
from src.api.gemini import create_gemini_exchange
from src.constants import FEATURE_COLUMNS
from src.utils.config import Config

# Setup logging for better visibility into WebSocket events
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# WebSocket URL for Gemini
WS_URL = "wss://api.gemini.com/v1/marketdata/BTCUSD"

# Global variables for WebSocket and model
ws = None
model = None
context_buffer = None

async def load_model():
    """Asynchronously load the trained TransformerPredictor model."""
    root_dir = Path(__file__).parent.parent.parent
    model_path = root_dir / 'best_model.pth'
    
    logger.debug(f"Attempting to load model from: {model_path}")
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Please ensure the file exists at the root directory.")
        raise FileNotFoundError(f"No such file or directory: '{model_path}'")
    
    model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS), d_model=128, n_heads=8, n_layers=4, dropout=0.7)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        logger.info(f"TransformerPredictor loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    return model

async def initialize_context_buffer(symbol: str = Config.TRADING_PAIR):
    """Initialize the context buffer with historical data for context_length=24."""
    global context_buffer
    historical_data = await fetch_historical_data(symbol, limit=24)
    preprocessed_data = preprocess_data(historical_data)
    context_buffer = preprocessed_data[FEATURE_COLUMNS].tail(24).values  # Shape [24, 22]
    if len(context_buffer) < 24:
        logger.warning(f"Insufficient historical data ({len(context_buffer)} < 24). Using available data.")
        context_buffer = np.pad(context_buffer, ((24 - len(context_buffer), 0), (0, 0)), mode='edge')  # Use edge values
    logger.debug(f"Initialized context_buffer shape: {context_buffer.shape}")

async def calculate_atr(exchange: ccxt.Exchange, symbol: str, window: int = 14) -> float:
    """Asynchronously calculate Average True Range (ATR) from historical data."""
    historical_data = await fetch_historical_data(symbol)
    preprocessed_data = preprocess_data(historical_data)
    high = preprocessed_data['high'].tail(window).values
    low = preprocessed_data['low'].tail(window).values
    close = preprocessed_data['close'].tail(window).values
    tr = np.max([high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))], axis=0)[1:]
    atr = np.mean(tr)
    logger.debug(f"Calculated ATR for {symbol}: {atr}")
    return atr

async def calculate_trade_amount(exchange: ccxt.Exchange, symbol: str = Config.TRADING_PAIR) -> float:
    """Asynchronously calculate the amount to trade based on current balance and risk management."""
    try:
        logger.debug(f"Fetching balance for {symbol}")
        balance = await exchange.fetch_balance()
        btc_balance = balance['free'].get('BTC', 0.0)
        usd_balance = balance['free'].get('USD', 0.0)
        
        ticker = await exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        atr = await calculate_atr(exchange, symbol)
        
        # Use Kelly Criterion with config risk per trade
        position_size = kelly_criterion(0.5, 2.0, Config.INITIAL_CAPITAL, atr, current_price, Config.RISK_PER_TRADE)
        logger.debug(f"Calculated position size: {position_size}")
        
        if Config.SIMULATION_MODE:
            return min(position_size, 0.001)  # Limit to 0.001 BTC in simulation mode
        elif btc_balance > position_size:
            return position_size
        elif usd_balance > (position_size * current_price):
            return min(position_size, usd_balance / current_price)
        return 0
    except Exception as e:
        logger.error(f"Error calculating trade amount: {e}")
        return 0

async def set_stop_loss(exchange: ccxt.Exchange, symbol: str, amount: float, entry_price: float, stop_loss_price: float):
    """Asynchronously place a stop-loss order"""
    try:
        logger.debug(f"Setting stop-loss for {symbol} at {stop_loss_price}")
        if not exchange.has['createStopLimitOrder']:
            logger.warning("Gemini does not support stop-limit orders directly; using limit order instead.")
            limit_order = await exchange.create_limit_sell_order(symbol, amount, stop_loss_price)
            logger.info(f"Stop-loss limit order set at {stop_loss_price}: {limit_order}")
        else:
            stop_loss_order = await exchange.create_stop_limit_order(symbol, 'sell', amount, stop_loss_price, stop_loss_price)
            logger.info(f"Stop-loss order set at {stop_loss_price}: {stop_loss_order}")
    except Exception as e:
        logger.error(f"Failed to set stop-loss: {e}")

async def set_take_profit(exchange: ccxt.Exchange, symbol: str, amount: float, entry_price: float, take_profit_price: float):
    """Asynchronously place a take-profit order"""
    try:
        logger.debug(f"Setting take-profit for {symbol} at {take_profit_price}")
        take_profit_order = await exchange.create_limit_sell_order(symbol, amount, take_profit_price)
        logger.info(f"Take-profit limit order set at {take_profit_price}: {take_profit_order}")
    except Exception as e:
        logger.error(f"Failed to set take-profit: {e}")

async def on_message(ws, message):
    """Asynchronously handle incoming WebSocket messages."""
    global model, context_buffer
    try:
        logger.debug(f"Received WebSocket message: {message}")
        data = json.loads(message)
        
        if 'events' in data:
            for event in data['events']:
                if event['type'] == 'trade':
                    logger.info(f"Trade event received: {event}")
                    
                    price = float(event['price'])
                    timestamp = pd.to_datetime(event['timestamp'], unit='ms')

                    # Prepare latest data
                    latest_data = pd.DataFrame({
                        'open': [price], 'high': [price], 'low': [price], 'close': [price], 'volume': [0]
                    }, index=[timestamp])
                    preprocessed_data = preprocess_data(latest_data)

                    # Update context_buffer with the latest data
                    context_buffer = np.roll(context_buffer, -1, axis=0)
                    context_buffer[-1] = preprocessed_data[FEATURE_COLUMNS].values[0]

                    # Prepare data for prediction
                    X = context_buffer.reshape(1, 24, len(FEATURE_COLUMNS))
                    X_tensor = torch.FloatTensor(X)
                    device = torch.device('cpu')
                    with torch.no_grad():
                        past_time_features = torch.FloatTensor(np.zeros((1, 24, 1)))
                        past_observed_mask = torch.ones((1, 24, len(FEATURE_COLUMNS)))
                        future_values = torch.FloatTensor(np.zeros((1, 1, 1)))
                        future_time_features = torch.FloatTensor(np.zeros((1, 1, 1)))

                        predictions, confidence = model.predict(
                            X_tensor.to(device),
                            past_time_features=past_time_features.to(device),
                            past_observed_mask=past_observed_mask.to(device),
                            future_values=future_values.to(device),
                            future_time_features=future_time_features.to(device)
                        )

                    predicted_price = predictions.cpu().numpy().flatten()[0]
                    current_price = price

                    # Generate signals using the full strategy
                    signal_data = pd.DataFrame({
                        'close': [current_price],
                        'predicted_price': [predicted_price],
                        'confidence': [confidence[0][0]]  # Assuming confidence shape [1, 1]
                    }, index=[timestamp])
                    signals = await generate_signals(
                        signal_data,
                        preprocessed_data.tail(24),
                        model,
                        FEATURE_COLUMNS,
                        joblib.load('feature_scaler.pkl'),
                        joblib.load('target_scaler.pkl'),
                        rsi_threshold=30,
                        macd_fast=12,
                        macd_slow=26,
                        atr_multiplier=2.0,
                        max_risk_pct=Config.RISK_PER_TRADE
                    )
                    signal = signals['signal'].iloc[0] if not signals.empty else 0

                    if signal != 0:
                        exchange = await create_gemini_exchange()
                        amount = await calculate_trade_amount(exchange, Config.TRADING_PAIR)
                        if amount > 0:
                            if signal == 1:  # Buy signal
                                logger.debug(f"Placing buy order for {amount} BTC at current price {current_price}")
                                order = await exchange.create_market_buy_order(Config.TRADING_PAIR, amount)
                                logger.info(f"Buy order placed: {order}")
                                # Use precomputed levels from generate_signals and risk_manager
                                current_balance = Config.INITIAL_CAPITAL if Config.SIMULATION_MODE else exchange.fetch_balance()['total']['USD']
                                signals, _ = manage_risk(signals, current_balance, max_drawdown_pct=0.10, atr_multiplier=2.0, recovery_volatility_factor=0.15, max_risk_pct=Config.RISK_PER_TRADE, min_position_size=0.001)
                                entry_price = current_price
                                stop_loss_price = signals['stop_loss'].iloc[0] if 'stop_loss' in signals.columns else entry_price * (1 - Config.RISK_PER_TRADE)
                                take_profit_price = signals['take_profit'].iloc[0] if 'take_profit' in signals.columns else entry_price * 1.05
                                await set_stop_loss(exchange, Config.TRADING_PAIR, amount, entry_price, stop_loss_price)
                                await set_take_profit(exchange, Config.TRADING_PAIR, amount, entry_price, take_profit_price)
                            elif signal == -1:  # Sell signal
                                logger.debug(f"Placing sell order for {amount} BTC at current price {current_price}")
                                order = await exchange.create_market_sell_order(Config.TRADING_PAIR, amount)
                                logger.info(f"Sell order placed: {order}")
                        else:
                            logger.info("Not enough funds or assets for trading")
                        await exchange.close()
                        logger.debug("Exchange instance closed")
    except Exception as e:
        logger.error(f"Error in on_message: {e}")
        if 'exchange' in locals():
            await exchange.close()
            logger.debug("Exchange instance closed due to exception")
        raise

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info(f"Closed connection with status {close_status_code}: {close_msg}")

def on_open(ws):
    logger.info("Connected to WebSocket")
    subscription_message = {
        "type": "subscribe",
        "channels": [{"name": "live_trades", "symbols": [Config.TRADING_PAIR.replace('/', '')]}]
    }
    logger.debug(f"Sending subscription message: {subscription_message}")
    ws.send(json.dumps(subscription_message))

async def start_websocket():
    """Asynchronously start the WebSocket connection with reconnection logic."""
    global ws, model, context_buffer
    while True:
        try:
            logger.debug(f"Current event loop: {asyncio.get_event_loop().__class__.__name__}")
            if sys.platform == "win32":
                from asyncio import SelectorEventLoop
                asyncio.set_event_loop_policy(SelectorEventLoop())
                logger.info("Switched to SelectorEventLoop for Windows compatibility with WebSocket")

            model = await load_model()
            await initialize_context_buffer()

            logger.debug(f"Initializing WebSocketApp with URL: {WS_URL}")
            ws = websocket.WebSocketApp(
                WS_URL,
                on_message=lambda ws, msg: asyncio.run(on_message(ws, msg)),
                on_error=on_error,
                on_close=on_close
            )
            ws.on_open = on_open
            logger.debug("Starting WebSocket run_forever in thread")
            await asyncio.to_thread(ws.run_forever)
        except Exception as e:
            logger.error(f"WebSocket failed: {e}. Retrying in 5 seconds...")
            await asyncio.sleep(5)

async def signal_handler(sig, frame):
    """Asynchronously handle shutdown signals."""
    global ws
    logger.info('Shutting down the bot...')
    if ws:
        ws.close()
        logger.debug("WebSocket connection closed on shutdown")
    sys.exit(0)

async def main():
    """Main async function to run the WebSocket bot."""
    try:
        logger.debug("Starting WebSocket bot")
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(signal_handler(sig, None)))
        await start_websocket()
    except Exception as e:
        logger.error(f"Error running WebSocket bot: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot terminated by user")
    except Exception as e:
        logger.error(f"Error running WebSocket bot: {e}")