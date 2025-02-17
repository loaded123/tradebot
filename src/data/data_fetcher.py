import asyncio
import ccxt.async_support as ccxt  # Async version of CCXT
import pandas as pd
import numpy as np  # Add numpy import
import talib  # Add TA-Lib import for technical indicators
from datetime import datetime

async def fetch_historical_data(symbol, timeframe='1h', limit=1000):
    """
    Fetch historical OHLCV data from Gemini asynchronously.
    
    :param symbol: Trading pair symbol (e.g., 'BTC/USD')
    :param timeframe: Timeframe for OHLCV (e.g., '1h')
    :param limit: Number of data points to fetch
    :return: Pandas DataFrame with OHLCV data
    """
    exchange = ccxt.gemini()
    
    try:
        # Fetch OHLCV data directly
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Debug: Print fetched data columns
        print("Fetched data columns:", df.columns)
        
        # Check if there are enough rows for ATR calculation (at least 14)
        if len(df) < 14:
            raise ValueError("DataFrame must contain at least 14 rows to calculate ATR.")
        
        # Feature Engineering: Adding technical indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_volatility'] = df['close'].rolling(window=14).std()
        df['atr'] = df['high'].subtract(df['low']).rolling(window=14).mean()  # ATR for volatility
        df['momentum_rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['trend_macd'], _, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Target variable for supervised learning
        df['target'] = df['close'].shift(-1)
        
        # Drop rows with NaN values (e.g., due to shifting or rolling calculations)
        df.dropna(inplace=True)

        # Debug: Print columns before scaling
        print("Columns before scaling:", df.columns)

        return df

    except Exception as e:
        print(f"Error fetching data from Gemini: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

    finally:
        await exchange.close()  # Ensure exchange is closed

async def fetch_real_time_data(symbol):
    """
    Fetch real-time market data for a given symbol.

    :param symbol: Trading pair symbol (e.g., 'BTC/USD')
    :yield: Dictionary containing real-time price data
    """
    exchange = ccxt.gemini()

    while True:
        try:
            ticker = await exchange.fetch_ticker(symbol)
            yield {
                'open': ticker.get('open', ticker['last']),  
                'high': ticker['high'],
                'low': ticker['low'],
                'close': ticker['last'],
                'volume': ticker['baseVolume']
            }
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
        
        await asyncio.sleep(1)  # Poll every second

if __name__ == "__main__":
    async def main():
        # Fetch historical data example
        df = await fetch_historical_data('BTC/USD')
        print(df.head())

        # Example usage for real-time data
        async for data in fetch_real_time_data('BTC/USD'):
            print(data)
            # Implement your processing logic here
            await asyncio.sleep(1)  # Adjust the sleep time as needed

    asyncio.run(main())
