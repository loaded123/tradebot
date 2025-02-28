# src/trading/trader_new.py

import asyncio
from ccxt import gemini
import pandas as pd
import torch
import numpy as np
from src.models.transformer_model import TransformerPredictor
from src.data.data_preprocessor import preprocess_data
from src.data.data_fetcher import fetch_historical_data, fetch_real_time_data
from src.strategy.strategy_generator import generate_signals
from src.utils.logger import logger
from src.utils.config import GEMINI_API_KEY, GEMINI_API_SECRET
from src.constants import FEATURE_COLUMNS  # Import feature columns for consistency

# Initialize Gemini client
exchange = gemini({
    'apiKey': GEMINI_API_KEY,
    'secret': GEMINI_API_SECRET,
    'enableRateLimit': True,
})

async def execute_trade(symbol, signal, amount, stop_loss=None, take_profit=None):
    """
    Execute a trade on Gemini with risk management features.
    """
    try:
        order_type = 'market'
        side = 'buy' if signal == 1 else 'sell'
        
        # Wrap blocking call in `asyncio.to_thread()`
        order = await asyncio.to_thread(exchange.create_order, symbol, order_type, side, amount)

        logger.info(f"{side.capitalize()} order placed: {order}")

        if stop_loss or take_profit:
            logger.info(f"Setting stop-loss at {stop_loss} and take-profit at {take_profit}")
            # TODO: Implement stop-loss and take-profit logic using Gemini's order types (e.g., stop-limit)

        return order
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return None

async def monitor_market(symbol, model, feature_columns, trade_amount, stop_loss_pct=0.02, take_profit_pct=0.04):
    """
    Monitor the market and execute trades based on Transformer model predictions.
    """
    try:
        # Load historical data to initialize the model context (optional, for better predictions)
        historical_data = await fetch_historical_data(symbol)
        preprocessed_historical = preprocess_data(historical_data)
        
        # Initialize a buffer for the last 10 time steps to match context_length=10
        context_buffer = preprocessed_historical[FEATURE_COLUMNS].tail(10).values  # Shape [10, 17]
        if len(context_buffer) < 10:
            context_buffer = np.pad(context_buffer, ((10 - len(context_buffer), 0), (0, 0)), mode='constant')  # Pad with zeros

        async for data in fetch_real_time_data(symbol):
            latest_data = pd.DataFrame([data], columns=['open', 'high', 'low', 'close', 'volume'])
            preprocessed_data = preprocess_data(latest_data)

            # Update context_buffer with the latest data
            context_buffer = np.roll(context_buffer, -1, axis=0)  # Shift buffer
            context_buffer[-1] = preprocessed_data[FEATURE_COLUMNS].values[0]  # Add latest data

            # Prepare data for prediction (shape [1, 10, 17] to match Transformer input)
            X = context_buffer.reshape(1, 10, len(FEATURE_COLUMNS))  # Shape [1, 10, 17]
            X_tensor = torch.FloatTensor(X)

            # Predict using Transformer model
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
            current_price = latest_data['close'].iloc[0]

            # Generate signals based on predicted price (simplified logic for real-time)
            signal = 1 if predicted_price > current_price * 1.001 else -1 if predicted_price < current_price * 0.999 else 0  # 0.1% threshold

            if signal != 0:
                stop_loss = current_price * (1 - stop_loss_pct) if signal == 1 else current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct) if signal == 1 else current_price * (1 - take_profit_pct)
                
                await execute_trade(symbol, signal, trade_amount, stop_loss, take_profit)
                logger.info(f"Executed {('buy' if signal == 1 else 'sell')} signal at price {current_price}, predicted {predicted_price}")
    except Exception as e:
        logger.error(f"An error occurred while monitoring market: {e}")

if __name__ == "__main__":
    symbol = 'BTC/USD'
    trade_amount = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained Transformer model
    model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS), d_model=64, n_heads=4, n_layers=2, dropout=0.1).to(device)
    model_path = os.path.join(os.path.dirname(__file__), '../../best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Loaded Transformer model from {model_path}")
    else:
        logger.error(f"Model file not found at {model_path}. Training a new model or using fallback.")
        # Optionally train a new model or raise an error

    historical_data = asyncio.run(fetch_historical_data(symbol))
    preprocessed_historical_data = preprocess_data(historical_data)
    feature_columns = FEATURE_COLUMNS  # Use constants for consistency
    
    asyncio.run(monitor_market(symbol, model, feature_columns, trade_amount))