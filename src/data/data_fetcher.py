# src/data/data_fetcher.py
import asyncio
import pandas as pd
import ccxt.async_support as ccxt
import logging
import os

# Configure logging
script_dir = os.path.abspath(os.path.dirname(__file__))
log_dir = os.path.join(script_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'data_fetcher.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_fetcher')

async def fetch_historical_data(symbol, timeframe='1h', limit=3000, exchange_name='gemini'):
    """
    Fetch historical OHLCV data asynchronously from a specified exchange, ensuring full date range from Jan 1, 2025, to Mar 1, 2025,
    with robust timestamp handling for millisecond timestamps on Gemini and Kraken.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        timeframe (str): Timeframe for OHLCV (e.g., '1h')
        limit (int): Number of data points to fetch
        exchange_name (str): Name of the exchange (e.g., 'gemini', 'kraken')
    
    Returns:
        pd.DataFrame: OHLCV data with unscaled prices and full date range
    """
    exchanges = ['gemini', 'kraken']
    for current_exchange in exchanges:
        try:
            exchange = getattr(ccxt, current_exchange)({
                'enableRateLimit': True,
                'timeout': 30000
            })

            adjusted_symbol = symbol
            if current_exchange.lower() == 'kraken' and symbol == 'BTC/USD':
                adjusted_symbol = 'XXBTZUSD'
                logger.info(f"Adjusted symbol to {adjusted_symbol} for {current_exchange}")

            start_date = pd.to_datetime('2025-01-01 00:00:00', utc=True).tz_localize(None)
            end_date = pd.to_datetime('2025-03-01 23:00:00', utc=True).tz_localize(None)
            start_timestamp = start_date.timestamp() * 1000
            end_timestamp = end_date.timestamp() * 1000

            all_ohlcv = []
            current_timestamp = start_timestamp
            while current_timestamp < end_timestamp:
                attempt = 0
                max_retries = 5
                while attempt < max_retries:
                    try:
                        async with asyncio.timeout(30):
                            ohlcv = await exchange.fetch_ohlcv(adjusted_symbol, timeframe, int(current_timestamp), limit=min(limit, 1000))
                        if not ohlcv:
                            logger.warning(f"No OHLCV data returned for {adjusted_symbol} on {current_exchange}")
                            break
                        logger.debug(f"Sample OHLCV data for {adjusted_symbol} on {current_exchange}: {ohlcv[:5]}")
                        all_ohlcv.extend(ohlcv)
                        current_timestamp = ohlcv[-1][0] + (60 * 60 * 1000)
                        break
                    except asyncio.TimeoutError as e:
                        logger.error(f"Timeout fetching OHLCV data for {adjusted_symbol} on {current_exchange} (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(5)
                        else:
                            raise
                    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                        logger.error(f"Network/Exchange error for {adjusted_symbol} on {current_exchange} (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(5)
                        else:
                            raise
                    attempt += 1

            if not all_ohlcv:
                raise ValueError(f"No OHLCV data fetched for {adjusted_symbol} on {current_exchange}")

            # Convert to DataFrame with robust timestamp handling
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            logger.debug(f"DataFrame shape: {df.shape}")
            logger.debug(f"Raw timestamp data type: {df['timestamp'].dtype}")
            logger.debug(f"Sample raw timestamps: {df['timestamp'].head().tolist()}")

            # Convert timestamps to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            logger.debug(f"After conversion, timestamp dtype: {df['timestamp'].dtype}")
            logger.debug(f"Sample converted timestamps: {df['timestamp'].head().tolist()}")

            # Check for NaT values
            if df['timestamp'].isna().any():
                logger.warning(f"NaT values found in timestamp column: {df['timestamp'].isna().sum()} rows")
                df = df.dropna(subset=['timestamp'])

            # Remove timezone to ensure naive datetime
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            logger.debug(f"After tz_localize(None), timestamp dtype: {df['timestamp'].dtype}")
            logger.debug(f"Sample naive timestamps: {df['timestamp'].head().tolist()}")

            # Set timestamp as index
            df = df.set_index('timestamp')  # Avoid inplace=True for clarity
            logger.debug(f"After set_index, index type: {type(df.index)}")
            logger.debug(f"Sample index values: {df.index[:5].tolist()}")

            # Check if index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error(f"Index is not a DatetimeIndex: {type(df.index)}, Sample: {df.index[:5].tolist()}")
                raise ValueError(f"Failed to create DatetimeIndex for {adjusted_symbol} on {current_exchange}")

            # Reindex to full date range
            full_range = pd.date_range(start=start_date, end=end_date, freq='h')
            df = df.reindex(full_range, method='ffill').fillna({
                'open': 78877.88, 'high': 79367.5, 'low': 78186.98, 'close': 78877.88, 'volume': 1000.0
            })

            logger.info(f"Fetched {len(df)} rows from {current_exchange} for {adjusted_symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data from {current_exchange}: {e}")
            continue
        finally:
            if 'exchange' in locals():
                await exchange.close()

    logger.error(f"Failed to fetch data from all exchanges: {exchanges}")
    return pd.DataFrame()

async def fetch_real_time_ws(symbol, exchange_name='gemini'):
    """
    Fetch real-time data via WebSocket from the specified exchange.
    
    This is a placeholder function. Implement the actual WebSocket logic here.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        exchange_name (str): Name of the exchange (e.g., 'gemini', 'kraken')
    
    Yields:
        dict: Real-time data (e.g., {'timestamp': ..., 'price': ...})
    """
    logger.info(f"Starting real-time data fetch for {symbol} from {exchange_name}")
    try:
        # Placeholder: Implement actual WebSocket connection here
        exchange = getattr(ccxt, exchange_name)({
            'enableRateLimit': True,
        })
        while True:
            # Dummy data as a placeholder
            data = {
                'timestamp': pd.Timestamp.now(),
                'price': 12345.67  # Replace with actual WebSocket data
            }
            logger.debug(f"Yielding real-time data: {data}")
            yield data
            await asyncio.sleep(1)  # Simulate real-time updates
    except Exception as e:
        logger.error(f"Error in fetch_real_time_ws: {e}")
    finally:
        await exchange.close()