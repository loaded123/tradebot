# fetch_btc_usd_data.py
import requests
import pandas as pd
import time
import csv
import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('fetch_data.log'),
        logging.StreamHandler()
    ]
)

# Define the time range
# CryptoCompare's earliest hourly data for BTC/USD on Bitfinex starts around July 17, 2010
start_date = pd.to_datetime('2010-07-17 00:00:00', utc=True)  # Earliest available date
end_date = pd.to_datetime('2025-03-10 23:00:00', utc=True)
total_hours = int((end_date - start_date).total_seconds() / 3600)
logging.info(f"Fetching data from {start_date} to {end_date} ({total_hours} hours)")

# CryptoCompare API settings
base_url = "https://min-api.cryptocompare.com/data/histohour"
fsym = "BTC"
tsym = "USD"
exchange = "Bitfinex"
limit = 2000  # Max allowed by CryptoCompare per request
chunk_size = limit
output_path = 'btc_usd_historical.csv'
min_valid_rows_threshold = limit * 0.1  # 10% of limit (200 rows) as a threshold for sparse data

# Initialize variables
all_data = []
current_end_ts = int(end_date.timestamp())
fetched_timestamps = set()  # Track fetched timestamps to detect duplicates
consecutive_sparse_chunks = 0  # Counter for consecutive sparse/invalid chunks
max_consecutive_sparse = 5  # Stop if we get too many sparse chunks
consecutive_duplicates = 0  # Counter for consecutive duplicate chunks
max_consecutive_duplicates = 3  # Stop if too many duplicate chunks

# Load existing data if the file exists (for resuming)
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)
    all_data = existing_df.to_dict('records')
    fetched_timestamps = set(existing_df['timestamp'].astype(int) // 10**9)  # Convert to Unix timestamps
    earliest_existing = existing_df['timestamp'].min()
    current_end_ts = int(earliest_existing.timestamp()) - 1  # Start just before the earliest existing
    logging.info(f"Resuming from existing data. Earliest timestamp: {earliest_existing}, {len(all_data)} rows loaded.")

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
    logging.info(f"Fetching data up to {pd.to_datetime(current_end_ts, unit='s', utc=True)}...")

    # Retry logic with exponential backoff
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            break
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached. Skipping this chunk.")
                current_end_ts -= limit * 3600  # Move back anyway to avoid infinite loop
                time.sleep(5)
                continue
            time.sleep(2 ** attempt)  # Exponential backoff

    if response.status_code != 200:
        logging.error(f"HTTP error: {response.status_code}")
        time.sleep(10)
        continue

    data = response.json()
    if data['Response'] != 'Success':
        logging.error(f"API error: {data['Message']}")
        if "no data for the symbol" in data['Message'].lower():
            logging.warning("No more historical data available. Breaking loop.")
            break
        time.sleep(10)  # Wait before retrying
        continue

    rows = data['Data']
    logging.info(f"Raw rows fetched: {len(rows)}")
    if not rows:
        logging.warning("No more data available in this chunk.")
        consecutive_sparse_chunks += 1
        if consecutive_sparse_chunks >= max_consecutive_sparse:
            logging.warning(f"Reached {max_consecutive_sparse} consecutive sparse chunks. Stopping fetch.")
            break
        current_end_ts -= limit * 3600  # Move back by the chunk size
        time.sleep(2)
        continue

    # Filter out rows with zero volume or invalid prices
    filtered_rows = [row for row in rows if row['volumeto'] > 0 and row['close'] > 0]
    logging.info(f"Valid rows after filtering: {len(filtered_rows)}")
    if not filtered_rows:
        logging.warning(f"No valid data in this chunk (up to {pd.to_datetime(current_end_ts, unit='s', utc=True)}). Skipping.")
        consecutive_sparse_chunks += 1
        if consecutive_sparse_chunks >= max_consecutive_sparse:
            logging.warning(f"Reached {max_consecutive_sparse} consecutive sparse chunks. Stopping fetch.")
            break
        current_end_ts -= limit * 3600  # Move back by the chunk size
        time.sleep(2)
        continue

    # Check for sparse data
    if len(filtered_rows) < min_valid_rows_threshold:
        logging.warning(f"Sparse data: Only {len(filtered_rows)} valid rows (threshold: {min_valid_rows_threshold}). Forcing timestamp advance.")
        consecutive_sparse_chunks += 1
        if consecutive_sparse_chunks >= max_consecutive_sparse:
            logging.warning(f"Reached {max_consecutive_sparse} consecutive sparse chunks. Stopping fetch.")
            break
        earliest_ts = filtered_rows[0]['time']
        forced_step = max(limit * 3600, len(filtered_rows) * 3600)
        current_end_ts = earliest_ts - forced_step
        logging.info(f"Forced timestamp advance to {pd.to_datetime(current_end_ts, unit='s', utc=True)}")
        time.sleep(2)
        continue

    # Check for duplicates
    new_timestamps = {row['time'] for row in filtered_rows}
    duplicates = new_timestamps & fetched_timestamps
    if duplicates:
        logging.warning(f"Found {len(duplicates)} duplicate timestamps in this chunk: {duplicates}")
        consecutive_duplicates += 1
        if consecutive_duplicates >= max_consecutive_duplicates:
            logging.warning(f"Reached {max_consecutive_duplicates} consecutive duplicate chunks. Stopping fetch.")
            break
        # Force a larger step back to avoid the duplicate range
        current_end_ts -= limit * 3600  # Move back by the chunk size
        time.sleep(2)
        continue
    else:
        consecutive_duplicates = 0  # Reset on a successful non-duplicate fetch

    # Add new data
    all_data.extend(filtered_rows)
    fetched_timestamps.update(new_timestamps)
    consecutive_sparse_chunks = 0  # Reset counter on successful fetch
    earliest_ts = filtered_rows[0]['time']
    current_end_ts = earliest_ts - 1  # Subtract 1 second to avoid overlap with toTs
    logging.info(f"Fetched {len(filtered_rows)} rows. Earliest timestamp: {pd.to_datetime(earliest_ts, unit='s', utc=True)}")

    # Save incrementally to avoid losing progress
    temp_df = pd.DataFrame(all_data)
    temp_df['timestamp'] = pd.to_datetime(temp_df['time'], unit='s', utc=True)
    temp_df = temp_df.drop(columns=['time', 'volumefrom', 'conversionType', 'conversionSymbol'])
    temp_df = temp_df.rename(columns={'volumeto': 'volume'})
    temp_df = temp_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    temp_df = temp_df.sort_values('timestamp')
    temp_df = temp_df.drop_duplicates(subset=['timestamp'])
    temp_df['timestamp'] = temp_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    temp_df.to_csv(output_path, index=False)
    logging.info(f"Incrementally saved {len(temp_df)} rows to {output_path}")

    time.sleep(2)  # Avoid hitting rate limits (CryptoCompare allows ~20 requests/minute)

# Final processing
if all_data:
    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.drop(columns=['time', 'volumefrom', 'conversionType', 'conversionSymbol'])
    df = df.rename(columns={'volumeto': 'volume'})
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('timestamp')  # Ensure chronological order
    df = df.drop_duplicates(subset=['timestamp'])  # Remove duplicates if any

    # Handle missing or invalid data
    df = df[(df['close'] > 0) & (df['volume'] > 0)]  # Remove invalid rows
    df = df.set_index('timestamp').reindex(
        pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='h', tz='UTC'),
        method='ffill'
    ).reset_index()  # Fill gaps in timestamps, updated to 'h'
    df = df.rename(columns={'index': 'timestamp'})
    df = df.ffill()  # Forward-fill any remaining NaNs, updated syntax
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # Ensure column order

    # Format timestamp for CSV
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(output_path, index=False)
    logging.info(f"Final save: Saved {len(df)} rows to {output_path}")
else:
    logging.error("No data fetched.")