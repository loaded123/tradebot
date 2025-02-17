import asyncio
from ccxt import gemini
import pandas as pd
from models.model_trainer import LSTMModel
from data.data_preprocessor import preprocess_data
from data.data_fetcher import fetch_historical_data, fetch_real_time_data  # ✅ Fixed Import
from strategy.strategy_generator import generate_signals
from utils.logger import logger
from utils.config import GEMINI_API_KEY, GEMINI_API_SECRET

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
            # TODO: Implement stop-loss and take-profit logic

        return order
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return None


async def monitor_market(symbol, model, feature_columns, trade_amount, stop_loss_pct=0.02, take_profit_pct=0.04):
    """
    Monitor the market and execute trades based on model predictions.
    """
    try:
        async for data in fetch_real_time_data(symbol):
            latest_data = pd.DataFrame([data], columns=['open', 'high', 'low', 'close', 'volume'])
            preprocessed_data = preprocess_data(latest_data)  # ✅ Fixed Function Call
            signal_df = generate_signals(preprocessed_data, model, feature_columns)
            signal = signal_df['signal'].iloc[0]
            current_price = latest_data['close'].iloc[0]

            if signal != 0:
                stop_loss = current_price * (1 - stop_loss_pct) if signal == 1 else current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct) if signal == 1 else current_price * (1 - take_profit_pct)
                
                # ✅ Fixed Incorrect Function Call (removed `current_price`)
                await execute_trade(symbol, signal, trade_amount, stop_loss, take_profit)
    except Exception as e:
        logger.error(f"An error occurred while monitoring market: {e}")


if __name__ == "__main__":
    symbol = 'BTC/USD'
    trade_amount = 0.001
    model = LSTMModel(input_dim=5, hidden_dim=32, layer_dim=1, output_dim=1)
    
    # ✅ Fixed `fetch_historical_data` Import Issue
    historical_data = fetch_historical_data(symbol)
    preprocessed_historical_data = preprocess_data(historical_data)
    feature_columns = preprocessed_historical_data.columns.drop('target')
    
    asyncio.run(monitor_market(symbol, model, feature_columns, trade_amount))
