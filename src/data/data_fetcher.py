# src/data/data_fetcher.py
import asyncio
import pandas as pd
import ccxt.async_support as ccxt
import logging
import os
from datetime import datetime
import pytz
from src.constants import DEFAULT_OPEN, DEFAULT_CLOSE, DEFAULT_HIGH, DEFAULT_LOW, DEFAULT_VOLUME

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

async def fetch_historical_data(symbol, timeframe='1h', limit=3000, exchange_name='bitfinex', csv_path=None, min_volume_threshold=1.0):
    """
    Fetch historical OHLCV data asynchronously from a specified exchange or load from a CSV file.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        timeframe (str): Timeframe for OHLCV (e.g., '1h')
        limit (int): Maximum number of data points to fetch per request
        exchange_name (str): Name of the exchange (e.g., 'bitfinex', 'gemini')
        csv_path (str, optional): Path to a local CSV file with OHLCV data
        min_volume_threshold (float): Minimum volume threshold to filter low-volume periods (default: 1.0)
    
    Returns:
        pd.DataFrame: OHLCV data with unscaled prices, continuous timestamps, and filtered low-volume periods
    """
    current_date = datetime.now(pytz.UTC).replace(hour=23, minute=0, second=0, microsecond=0)  # Ensure UTC timezone

    # Prioritize CSV if provided and exists
    if csv_path and os.path.exists(csv_path):
        logger.info(f"Loading data from CSV: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file must contain the following columns: {required_columns}")
            # Handle various timestamp formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', int]:
                try:
                    if df['timestamp'].dtype == 'int64' or df['timestamp'].dtype == 'float64':
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s' if df['timestamp'].dtype == 'int64' else 'ms', utc=True, errors='coerce')
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt, utc=True, errors='coerce')
                    break
                except ValueError:
                    continue
            if df['timestamp'].isna().all():
                raise ValueError("Unable to parse timestamp column in CSV")
            df = df.dropna(subset=['timestamp'])

            # Check for duplicate timestamps
            if df['timestamp'].duplicated().any():
                logger.warning(f"Found {df['timestamp'].duplicated().sum()} duplicate timestamps. Removing duplicates.")
                df = df.drop_duplicates(subset=['timestamp'], keep='first')

            df = df.set_index('timestamp')
            # Ensure the index is timezone-aware (UTC)
            if df.index.tz is None:
                logger.warning("Index timezone is None, localizing to UTC")
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz != pytz.UTC:
                logger.warning(f"Index timezone {df.index.tz} differs from UTC, converting to UTC")
                df.index = df.index.tz_convert('UTC')

            # Ensure continuous timestamps
            full_range = pd.date_range(start=df.index.min(), end=current_date, freq='h', tz='UTC')
            if len(full_range) != len(df):
                logger.warning(f"Timestamps are not continuous. Expected {len(full_range)} timestamps, got {len(df)}. Reindexing.")
                df = df.reindex(full_range)
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].ffill()
                df['open'] = df['open'].fillna(df['open'].mean() if not df['open'].isna().all() else DEFAULT_OPEN)
                df['high'] = df['high'].fillna(df['high'].mean() if not df['high'].isna().all() else DEFAULT_HIGH)
                df['low'] = df['low'].fillna(df['low'].mean() if not df['low'].isna().all() else DEFAULT_LOW)
                df['close'] = df['close'].fillna(df['close'].mean() if not df['close'].isna().all() else DEFAULT_CLOSE)
                df['volume'] = df['volume'].fillna(df['volume'].mean() if not df['volume'].isna().all() else DEFAULT_VOLUME)

            # Filter low-volume periods
            low_volume_mask = df['volume'] < min_volume_threshold
            if low_volume_mask.any():
                logger.warning(f"Found {low_volume_mask.sum()} periods with volume below {min_volume_threshold}. Imputing volume.")
                df.loc[low_volume_mask, 'volume'] = df['volume'].rolling(window=24, min_periods=1).mean()[low_volume_mask]

            logger.info(f"Loaded {len(df)} rows from CSV file")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}")
            raise ValueError(f"Failed to load data from CSV file: {e}")

    # Fallback to exchange if no CSV is provided
    logger.warning("No valid CSV path provided, falling back to exchange data fetching")
    exchanges = ['bitfinex', 'gemini']
    max_iterations = 1000  # Prevent infinite looping

    for current_exchange in exchanges:
        try:
            exchange = getattr(ccxt, current_exchange)({
                'enableRateLimit': True,
                'timeout': 30000
            })

            adjusted_symbol = symbol
            if current_exchange.lower() == 'bitfinex' and symbol == 'BTC/USD':
                adjusted_symbol = 'BTC/USD'
                logger.info(f"Adjusted symbol to {adjusted_symbol} for {current_exchange}")
            elif current_exchange.lower() == 'gemini' and symbol == 'BTC/USD':
                adjusted_symbol = 'BTC/USD'
                logger.info(f"Adjusted symbol to {adjusted_symbol} for {current_exchange}")

            # Initial fetch to determine the earliest available data
            initial_timestamp = pd.to_datetime('2020-01-01 00:00:00', utc=True).timestamp() * 1000
            try:
                initial_ohlcv = await exchange.fetch_ohlcv(adjusted_symbol, timeframe, int(initial_timestamp), 1)
                if initial_ohlcv:
                    start_timestamp = initial_ohlcv[0][0]
                    logger.info(f"Earliest available timestamp from {current_exchange}: {pd.to_datetime(start_timestamp, unit='ms')}")
                else:
                    start_timestamp = pd.to_datetime('2024-12-01 00:00:00', utc=True).timestamp() * 1000
                    logger.warning(f"No initial data from {current_exchange}, using fallback start: {pd.to_datetime(start_timestamp, unit='ms')}")
            except Exception as e:
                logger.error(f"Failed to fetch initial timestamp from {current_exchange}: {e}")
                start_timestamp = pd.to_datetime('2024-12-01 00:00:00', utc=True).timestamp() * 1000
                logger.warning(f"Using fallback start timestamp: {pd.to_datetime(start_timestamp, unit='ms')}")

            end_timestamp = current_date.timestamp() * 1000
            all_ohlcv = []
            current_timestamp = start_timestamp
            chunk_size = 100
            last_fetched_timestamp = start_timestamp
            rate_limit_delay = 10
            iteration_count = 0

            while current_timestamp < end_timestamp and iteration_count < max_iterations:
                attempt = 0
                max_retries = 3
                fetched_data = False
                while attempt < max_retries:
                    try:
                        async with asyncio.timeout(30):
                            ohlcv = await exchange.fetch_ohlcv(adjusted_symbol, timeframe, int(current_timestamp), chunk_size)
                        if not ohlcv:
                            logger.warning(f"No OHLCV data returned for {adjusted_symbol} on {current_exchange} at timestamp {current_timestamp}")
                            break
                        logger.debug(f"Sample OHLCV data for {adjusted_symbol} on {current_exchange}: {ohlcv[:5]}")
                        all_ohlcv.extend(ohlcv)
                        last_fetched_timestamp = ohlcv[-1][0]
                        current_timestamp = last_fetched_timestamp + (60 * 60 * 1000)
                        fetched_data = True
                        break
                    except ccxt.BadSymbol as e:
                        logger.error(f"BadSymbol error for {adjusted_symbol} on {current_exchange}: {e}")
                        raise
                    except (asyncio.TimeoutError, ccxt.NetworkError, ccxt.ExchangeError) as e:
                        logger.error(f"Error fetching OHLCV data for {adjusted_symbol} on {current_exchange} (attempt {attempt + 1}/{max_retries}): {e}")
                        attempt += 1
                        if attempt < max_retries:
                            await asyncio.sleep(5)
                        else:
                            logger.error(f"Max retries reached for timestamp {current_timestamp}, advancing timestamp")
                            break
                    except asyncio.CancelledError as e:
                        logger.error(f"CancelledError for {adjusted_symbol} on {current_exchange} at timestamp {current_timestamp}: {e}")
                        raise

                if not fetched_data:
                    current_timestamp += chunk_size * 3600000
                    logger.warning(f"No data fetched, advancing timestamp to {current_timestamp}")
                await asyncio.sleep(rate_limit_delay)
                iteration_count += 1

            if not all_ohlcv:
                raise ValueError(f"No OHLCV data fetched for {adjusted_symbol} on {current_exchange}")

            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
            df = df.dropna(subset=['timestamp'])

            # Check for duplicate timestamps
            if df['timestamp'].duplicated().any():
                logger.warning(f"Found {df['timestamp'].duplicated().sum()} duplicate timestamps. Removing duplicates.")
                df = df.drop_duplicates(subset=['timestamp'], keep='first')

            df = df.set_index('timestamp')
            # Ensure continuous timestamps
            full_range = pd.date_range(start=pd.to_datetime(start_timestamp, unit='ms', utc=True), end=current_date, freq='h', tz='UTC')
            if len(full_range) != len(df):
                logger.warning(f"Timestamps are not continuous. Expected {len(full_range)} timestamps, got {len(df)}. Reindexing.")
                df = df.reindex(full_range)
                df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].ffill()
                df['open'] = df['open'].fillna(df['open'].mean() if not df['open'].isna().all() else DEFAULT_OPEN)
                df['high'] = df['high'].fillna(df['high'].mean() if not df['high'].isna().all() else DEFAULT_HIGH)
                df['low'] = df['low'].fillna(df['low'].mean() if not df['low'].isna().all() else DEFAULT_LOW)
                df['close'] = df['close'].fillna(df['close'].mean() if not df['close'].isna().all() else DEFAULT_CLOSE)
                df['volume'] = df['volume'].fillna(df['volume'].mean() if not df['volume'].isna().all() else DEFAULT_VOLUME)

            # Filter low-volume periods
            low_volume_mask = df['volume'] < min_volume_threshold
            if low_volume_mask.any():
                logger.warning(f"Found {low_volume_mask.sum()} periods with volume below {min_volume_threshold}. Imputing volume.")
                df.loc[low_volume_mask, 'volume'] = df['volume'].rolling(window=24, min_periods=1).mean()[low_volume_mask]

            logger.info(f"Fetched {len(df)} rows from {current_exchange} for {adjusted_symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data from {current_exchange}: {e}")
            continue
        finally:
            if 'exchange' in locals():
                await exchange.close()

    logger.error("Failed to fetch data from all exchanges and no valid CSV path provided")
    raise ValueError("Unable to fetch data from exchanges or CSV file")

async def fetch_real_time_ws(symbol, exchange_name='bitfinex'):
    """
    Fetch real-time data via WebSocket from the specified exchange.
    
    This is a placeholder function. Implement the actual WebSocket logic here.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USD')
        exchange_name (str): Name of the exchange (e.g., 'bitfinex', 'gemini')
    
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
                'timestamp': pd.Timestamp.now(tz='UTC'),
                'price': 12345.67  # Replace with actual WebSocket data
            }
            logger.debug(f"Yielding real-time data: {data}")
            yield data
            await asyncio.sleep(1)  # Simulate real-time updates
    except Exception as e:
        logger.error(f"Error in fetch_real_time_ws: {e}")
    finally:
        await exchange.close()