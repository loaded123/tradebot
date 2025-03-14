# fetch_btc_usd_data.py
import requests
import pandas as pd
import time
import csv

# Define the time range
start_date = pd.to_datetime('2020-01-01 00:00:00', utc=True)
end_date = pd.to_datetime('2025-03-10 23:00:00', utc=True)
total_hours = int((end_date - start_date).total_seconds() / 3600)  # ~18624 hours

# CryptoCompare API settings
base_url = "https://min-api.cryptocompare.com/data/histohour"
fsym = "BTC"
tsym = "USD"
exchange = "Bitfinex"
limit = 2000  # Max allowed by CryptoCompare
chunk_size = limit

# Initialize variables
all_data = []
current_end_ts = int(end_date.timestamp())

# Fetch data in chunks
while current_end_ts > int(start_date.timestamp()):
    params = {
        'fsym': fsym,
        'tsym': tsym,
        'limit': limit,
        'aggregate': 1,
        'e': exchange,
        'toTs': current_end_ts
    }
    print(f"Fetching data up to {pd.to_datetime(current_end_ts, unit='s', utc=True)}...")
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        break
    
    data = response.json()
    if data['Response'] != 'Success':
        print(f"Error: {data['Message']}")
        break
    
    rows = data['Data']
    if not rows:
        print("No more data available.")
        break
    
    all_data.extend(rows)
    earliest_ts = rows[0]['time']
    current_end_ts = earliest_ts  # Move to the earliest timestamp of this chunk
    time.sleep(1)  # Avoid hitting rate limits (CryptoCompare allows ~20 requests/minute)

# Convert to DataFrame and save as CSV
if all_data:
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.drop(columns=['time', 'volumefrom', 'conversionType', 'conversionSymbol'])
    df = df.rename(columns={'volumeto': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('timestamp')  # Ensure chronological order
    df = df.drop_duplicates(subset=['timestamp'])  # Remove duplicates if any
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    output_path = 'btc_usd_historical.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
else:
    print("No data fetched.")