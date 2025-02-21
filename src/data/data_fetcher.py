# src/data/data_fetcher.py

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def fetch_historical_data(symbol, timeframe='1h', limit=3000, exchange_name='gemini'):
    """
    Fetch historical OHLCV data asynchronously from a specified exchange.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        timeframe (str): Timeframe for OHLCV (e.g., '1h')
        limit (int): Number of data points to fetch
        exchange_name (str): Name of the exchange (e.g., 'gemini', 'binance')
    
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        # Initialize exchange
        exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
            'asyncio_loop': asyncio.get_event_loop()
        })
        
        # Fetch OHLCV data
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logging.info(f"Fetched {len(df)} rows from {exchange_name} for {symbol}")
        logging.info(f"Columns: {df.columns}")
        
        if len(df) < 20:
            logging.warning(f"Data has fewer than 20 rows ({len(df)}), some indicators may fail.")
        
        return df

    except Exception as e:
        logging.error(f"Error fetching historical data from {exchange_name}: {e}")
        return pd.DataFrame()

    finally:
        await exchange.close()

async def fetch_real_time_ws(symbol, exchange_name='gemini'):
    exchange = getattr(ccxt, exchange_name)({'enableRateLimit': True})
    async for ohlcv in exchange.watch_ohlcv(symbol, '1m'):
        yield ohlcv
    
    try:
        while True:
            try:
                ticker = await exchange.fetch_ticker(symbol)
                data = {
                    'timestamp': pd.Timestamp.now(),
                    'open': ticker.get('open', ticker['last']),
                    'high': ticker['high'],
                    'low': ticker['low'],
                    'close': ticker['last'],
                    'volume': ticker['baseVolume']
                }
                yield data
            except Exception as e:
                logging.error(f"Error fetching real-time data from {exchange_name}: {e}")
            await asyncio.sleep(interval)
    finally:
        await exchange.close()

if __name__ == "__main__":
    async def test():
        # Test historical data
        df = await fetch_historical_data('BTC/USD', '1h', 5)
        logging.info(f"Historical data:\n{df.head()}")

        # Test real-time data
        async for data in fetch_real_time_data('BTC/USD'):
            logging.info(f"Real-time data: {data}")
            await asyncio.sleep(1)  # Limit output rate

    asyncio.run(test())