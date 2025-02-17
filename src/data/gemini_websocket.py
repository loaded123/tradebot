import sys
import os
import websocket
import json
import threading
import numpy as np
import torch
import logging
import signal

# Add `src/` directory to Python's module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.model_predictor import predict_live_price
from src.data.data_preprocessor import feature_columns, scaler
from src.strategy.strategy_generator import process_trade_event
from src.models.lstm_model import LSTMModel

# Setup logging for better visibility into WebSocket events
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# WebSocket URL for Gemini
WS_URL = "wss://api.gemini.com/v1/marketdata/BTCUSD"

def load_model():
    """Load the trained LSTM model."""
    model_path = os.path.join(os.path.dirname(__file__), '../models/trained_lstm_model.pth')
    model = LSTMModel(input_size=len(feature_columns), hidden_layer_size=50, output_size=1)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        logging.info(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Please ensure the file exists.")
        raise
    
    return model

def calculate_trade_amount():
    """Calculate the amount to trade based on current balance."""
    exchange = create_gemini_exchange()
    balance = exchange.fetch_balance()
    btc_balance = balance['free']['BTC']
    usd_balance = balance['free']['USD']
    
    # Example: Buy or sell 0.01 BTC if available
    if btc_balance > 0.01:
        return 0.01
    elif usd_balance > 100:  # Assuming 1 BTC = 10000 USD for simplicity
        return usd_balance / 10000  # This would convert USD to BTC
    
    return 0  # If neither condition is met, return 0

def set_stop_loss(exchange, symbol, amount, entry_price, stop_loss_price):
    """Place a stop-loss order"""
    try:
        stop_loss_order = exchange.create_order(symbol, 'stop', 'sell', amount, stop_loss_price)
        logging.info(f"Stop-loss order set at {stop_loss_price}: {stop_loss_order}")
    except Exception as e:
        logging.error(f"Failed to set stop-loss: {e}")

def set_take_profit(exchange, symbol, amount, entry_price, take_profit_price):
    """Place a take-profit order"""
    try:
        take_profit_order = exchange.create_order(symbol, 'take_profit', 'sell', amount, take_profit_price)
        logging.info(f"Take-profit order set at {take_profit_price}: {take_profit_order}")
    except Exception as e:
        logging.error(f"Failed to set take-profit: {e}")

def on_message(ws, message):
    """Callback function to handle incoming messages."""
    data = json.loads(message)
    
    if 'events' in data:
        for event in data['events']:
            if event['type'] == 'trade':
                logging.info(f"Trade event received: {event}")
                
                # Extract the price for prediction
                price = float(event['price'])

                # Prepare data for prediction
                current_data = np.array([[price]])

                # Predict next price
                try:
                    next_price = predict_live_price(model, current_data, feature_columns, scaler)
                    logging.info(f"Predicted next price: {next_price}")
                except Exception as e:
                    logging.error(f"Exception in processing trade event: {e}")

                # Use the strategy to decide whether to trade
                try:
                    signal = process_trade_event(event, next_price)
                    if signal == 1:  # Buy signal
                        exchange = create_gemini_exchange()
                        amount = calculate_trade_amount()
                        if amount > 0:
                            order = exchange.create_market_buy_order('BTC/USD', amount)
                            logging.info(f"Buy order placed: {order}")
                            
                            # Set stop-loss and take-profit
                            entry_price = float(event['price'])
                            stop_loss_price = entry_price * 0.95  # 5% below entry
                            take_profit_price = entry_price * 1.05  # 5% above entry
                            set_stop_loss(exchange, 'BTC/USD', amount, entry_price, stop_loss_price)
                            set_take_profit(exchange, 'BTC/USD', amount, entry_price, take_profit_price)
                        else:
                            logging.info("Not enough funds for buying")
                    elif signal == -1:  # Sell signal
                        exchange = create_gemini_exchange()
                        amount = calculate_trade_amount()
                        if amount > 0:
                            order = exchange.create_market_sell_order('BTC/USD', amount)
                            logging.info(f"Sell order placed: {order}")
                        else:
                            logging.info("Not enough assets for selling")
                    else:
                        logging.info("No action taken based on signal")
                except Exception as e:
                    logging.error(f"Error in placing order: {e}")

def on_error(ws, error):
    logging.error(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info("Closed connection")

def on_open(ws):
    logging.info("Connected to WebSocket")
    
    # Subscribe to market data
    subscription_message = {
        "type": "subscribe",
        "channels": [{"name": "live_trades", "symbols": ["BTCUSD"]}]
    }
    ws.send(json.dumps(subscription_message))

def start_websocket():
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    ws.run_forever()

def signal_handler(sig, frame):
    logging.info('Shutting down the bot...')
    ws.close()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Load the model before starting the WebSocket
model = load_model()

# Run WebSocket in a separate thread
thread = threading.Thread(target=start_websocket)
thread.start()