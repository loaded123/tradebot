# src/data/data_fetcher.py
import asyncio
import sys
import ccxt.async_support as ccxt
import pandas as pd
import logging
from ccxt.base.errors import NetworkError, ExchangeError

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def fetch_historical_data(symbol, timeframe='1h', limit=3000, exchange_name='gemini'):
    """
    Fetch historical OHLCV data asynchronously from a specified exchange.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        timeframe (str): Timeframe for OHLCV (e.g., '1h')
        limit (int): Number of data points to fetch
        exchange_name (str): Name of the exchange (e.g., 'gemini', 'binance', 'kraken')
    
    Returns:
        pd.DataFrame: OHLCV data with unscaled prices
    """
    exchanges = ['gemini', 'kraken', 'binance']  # Order of preference
    for current_exchange in exchanges:
        try:
            # Set SelectorEventLoop policy for Windows to support aiodns in ccxt.async_support
            if sys.platform == "win32":
                loop = asyncio.get_event_loop()
                if loop.__class__.__name__ == "ProactorEventLoop":
                    from asyncio import SelectorEventLoop
                    asyncio.set_event_loop_policy(SelectorEventLoop())
                    logging.info("Switched to SelectorEventLoop for Windows compatibility with aiodns in ccxt.async_support")
                logging.debug(f"Using asyncio event loop: {asyncio.get_event_loop().__class__.__name__}")

            # Initialize exchange with default event loop (let ccxt.async_support manage it)
            exchange = getattr(ccxt, current_exchange)({
                'enableRateLimit': True,
                'enableWebSocket': False,  # Use REST for historical data
            })

            # Adjust symbol for each exchange
            adjusted_symbol = symbol
            if current_exchange.lower() == 'binance' and symbol == 'BTC/USD':
                adjusted_symbol = 'BTC/USDT'  # Binance uses USDT for stable pricing close to USD
                logging.info(f"Adjusted symbol to {adjusted_symbol} for {current_exchange} compatibility")
            elif current_exchange.lower() == 'kraken' and symbol == 'BTC/USD':
                adjusted_symbol = 'XXBT/ZUSD'  # Kraken uses XXBT/ZUSD for BTC/USD
                logging.info(f"Adjusted symbol to {adjusted_symbol} for {current_exchange} compatibility")

            # Fetch OHLCV data with retries for network issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = await exchange.fetch_ohlcv(adjusted_symbol, timeframe, limit=limit)
                    break
                except (NetworkError, ExchangeError) as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Retry {attempt + 1}/{max_retries} for {adjusted_symbol} on {current_exchange} due to {e}. Waiting 5 seconds...")
                        await asyncio.sleep(5)
                    else:
                        logging.error(f"Failed to fetch OHLCV data from {current_exchange} after {max_retries} attempts: {e}")
                        raise
            else:
                continue  # Try the next exchange if retries fail

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Validate price range to ensure unscaled, realistic BTC prices
            if not df.empty:
                price_min = df[['open', 'high', 'low', 'close']].min().min()
                price_max = df[['open', 'high', 'low', 'close']].max().max()
                if price_min < 10000 or price_max > 200000:  # Typical BTC range as of Feb 2025
                    logging.warning(f"Prices appear scaled or invalid on {current_exchange}: min={price_min}, max={price_max}. Recalculating...")
                    # Adjust scaling if needed (e.g., divide by 1000 if scaled by mistake)
                    if price_max > 200000:
                        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] / 1000
                        logging.info(f"Adjusted prices by dividing by 1000 on {current_exchange} to match BTC/USD range")
                    elif price_min < 10000:
                        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] * 1000
                        logging.info(f"Adjusted prices by multiplying by 1000 on {current_exchange} to match BTC/USD range")
            
            logging.info(f"Fetched {len(df)} rows from {current_exchange} for {adjusted_symbol}")
            logging.info(f"Columns: {df.columns}")
            logging.info(f"Price range: min={df['close'].min()}, max={df['close'].max()}")
            
            if len(df) < 20:
                logging.warning(f"Data has fewer than 20 rows ({len(df)}) on {current_exchange}, some indicators may fail.")
            
            return df

        except Exception as e:
            logging.error(f"Error fetching historical data from {current_exchange}: {e}")
            continue  # Try the next exchange
        finally:
            if 'exchange' in locals():
                await exchange.close()

    logging.error(f"Failed to fetch data from all exchanges: {exchanges}")
    return pd.DataFrame()

async def fetch_real_time_ws(symbol, exchange_name='gemini'):
    """
    Fetch real-time OHLCV data asynchronously via WebSocket from a specified exchange.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        exchange_name (str): Name of the exchange (e.g., 'gemini', 'binance')
    
    Yields:
        dict: Real-time OHLCV data
    """
    exchanges = ['gemini', 'kraken', 'binance']  # Order of preference
    for current_exchange in exchanges:
        try:
            exchange = getattr(ccxt, current_exchange)({
                'enableRateLimit': True,
                'enableWebSocket': True,  # Enable WebSocket for real-time
            })

            # Adjust symbol for each exchange
            adjusted_symbol = symbol
            if current_exchange.lower() == 'binance' and symbol == 'BTC/USD':
                adjusted_symbol = 'BTC/USDT'
                logging.info(f"Adjusted symbol to {adjusted_symbol} for {current_exchange} WebSocket compatibility")
            elif current_exchange.lower() == 'kraken' and symbol == 'BTC/USD':
                adjusted_symbol = 'XXBT/ZUSD'
                logging.info(f"Adjusted symbol to {adjusted_symbol} for {current_exchange} WebSocket compatibility")

            ws = await exchange.watch_ohlcv(adjusted_symbol, '1m')  # Use 1-minute candles for real-time
            while True:
                try:
                    ohlcv = await ws.recv()
                    if ohlcv:
                        data = {
                            'timestamp': pd.Timestamp(ohlcv[0], unit='ms'),
                            'open': ohlcv[1],
                            'high': ohlcv[2],
                            'low': ohlcv[3],
                            'close': ohlcv[4],
                            'volume': ohlcv[5]
                        }
                        # Validate price range
                        if data['close'] < 10000 or data['close'] > 200000:
                            logging.warning(f"Real-time price appears scaled or invalid on {current_exchange}: {data['close']}. Skipping...")
                            continue
                        yield data
                except Exception as e:
                    logging.error(f"Error fetching real-time data from {current_exchange} WebSocket: {e}")
                await asyncio.sleep(1)  # Wait 1 second between retries
        except Exception as e:
            logging.error(f"Error initializing WebSocket for {current_exchange}: {e}")
            continue  # Try the next exchange
        finally:
            await exchange.close()

    logging.error(f"Failed to fetch real-time data from all exchanges: {exchanges}")
    yield None

if __name__ == "__main__":
    async def test():
        # Test historical data
        df = await fetch_historical_data('BTC/USD', '1h', 5)
        logging.info(f"Historical data:\n{df.head()}")
        logging.info(f"Price range: min={df['close'].min()}, max={df['close'].max()}")

        # Test real-time data (limit to 5 updates for testing)
        async for i, data in enumerate(fetch_real_time_ws('BTC/USD')):
            if i >= 5:
                break
            if data:
                logging.info(f"Real-time data {i + 1}: {data}")

    asyncio.run(test())