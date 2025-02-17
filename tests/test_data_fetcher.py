import requests
import pandas as pd
import time

def fetch_gemini_data(symbol, start_time, end_time, interval='1h'):
    """
    Fetch historical OHLCV data from Gemini.

    :param symbol: Trading pair (e.g., 'BTCUSD')
    :param start_time: Start time for the data (timestamp in seconds)
    :param end_time: End time for the data (timestamp in seconds)
    :param interval: Data interval (e.g., '1h', '15m')
    :return: DataFrame with OHLCV data
    """
    url = f'https://api.gemini.com/v1/candles/{symbol}/{interval}'
    params = {
        'start': start_time,
        'end': end_time,
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# Example usage
symbol = 'BTCUSD'
start_time = int(time.mktime(time.strptime('2025-01-01', '%Y-%m-%d')))  # Convert to timestamp
end_time = int(time.mktime(time.strptime('2025-02-01', '%Y-%m-%d')))  # Convert to timestamp

data = fetch_gemini_data(symbol, start_time, end_time)
if data is not None:
    print(data.head())  # Preview the data
