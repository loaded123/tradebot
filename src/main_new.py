import asyncio
import ccxt
import time
from data.data_fetcher import fetch_historical_data, fetch_real_time_data  # Import real-time data fetch
from data.data_preprocessor import preprocess_data
from models.model_trainer import train_model
from models.model_predictor import predict_live_price
from strategy.strategy_generator import generate_signals
from strategy.backtester import backtest_strategy
from trading.trader_new import execute_trade
from utils.config import Config
from utils.logger import logger
from utils.performance import PerformanceTracker
from utils.database import Database
from strategy.strategy_selector import StrategySelector
from src.data.gemini_websocket import start_websocket

config = Config()

async def run_trading_bot():
    performance_tracker = PerformanceTracker()
    db = Database()
    
    try:
        logger.info("Starting trading bot...")
        performance_tracker.start()

        # Fetch and process historical data
        logger.info("Fetching historical data...")
        data = fetch_historical_data('COINBASEPRO', config.TRADING_PAIR, config.TIME_FRAME, config.HISTORICAL_DATA_POINTS)
        processed_data = preprocess_data(data)
        logger.info("Data fetched and processed successfully.")

        # Train model
        logger.info("Training model...")
        model = train_model(processed_data[['features']], processed_data['target'])
        logger.info("Model trained.")

        # Backtest strategy
        logger.info("Backtesting strategy...")
        strategy = StrategySelector(config.STRATEGY_NAME).get_strategy()
        
        # Generate signals for backtest (pass processed data and model)
        signals = strategy.generate_signals(processed_data, model)
        
        # Run backtest
        backtest_results = backtest_strategy(processed_data, initial_capital=10000, risk_per_trade=0.02)
        logger.info(f"Backtest completed. Results: {backtest_results[['total', 'cumulative_returns']]}")

        # Initialize exchange - simulation or real
        if config.SIMULATION_MODE:
            logger.info("Running in simulation mode.")
            exchange = ccxt.coinbasepro({'apiKey': '', 'secret': '', 'password': '', 'enableRateLimit': True})
        else:
            logger.info("Running in live trading mode.")
            exchange = ccxt.coinbasepro({
                'apiKey': config.API_KEY,
                'secret': config.API_SECRET,
                'password': config.API_PASS
            })

        # Real-time trading loop
        logger.info("Entering real-time trading loop.")
        while True:
            try:
                logger.info("Fetching latest market data...")
                latest_data = await fetch_real_time_data(exchange, config.TRADING_PAIR)  # Use async fetch
                if latest_data is None:
                    continue
                processed_latest_data = preprocess_data(latest_data)

                # Predict next price
                prediction = predict_live_price(model, processed_latest_data)

                # Generate trading signals
                signal = strategy.generate_signals(processed_latest_data, model, prediction)

                # Execute trade if conditions are met
                if signal == 'BUY':
                    trade_result = execute_trade(exchange, 'buy', config.TRADING_PAIR, amount=0.01)  # Example amount
                    logger.info(f"BUY executed: {trade_result}")
                    db.insert_trade(trade_result)
                elif signal == 'SELL':
                    trade_result = execute_trade(exchange, 'sell', config.TRADING_PAIR, amount=0.01)
                    logger.info(f"SELL executed: {trade_result}")
                    db.insert_trade(trade_result)
                else:
                    logger.info(f"No action taken. Signal: {signal}")

                time.sleep(config.POLLING_INTERVAL)
            except Exception as e:
                logger.error(f"Error in real-time trading loop: {e}")
                time.sleep(config.ERROR_DELAY)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        performance_tracker.stop()
        performance_tracker.log_performance()
        logger.info("Bot execution completed.")

if __name__ == "__main__":
    start_websocket()

    # Run the trading bot asynchronously
    asyncio.run(run_trading_bot())  # Use asyncio to run the async trading bot loop
