# combine_data.py
import pandas as pd

# Load all datasets
bitfinex_df = pd.read_csv('btc_usd_historical.csv')  # Bitfinex data (2013-01-14 to 2025-03-10)
yahoo_df = pd.read_csv('btc_usd_historical_yahoo.csv')  # Yahoo Finance data (2012-06-01 to 2013-01-14)
mtgox_df = pd.read_csv('btc_usd_historical_mtgox.csv')  # MtGox data (2010-08-04 to 2012-06-01)

# Convert timestamps to datetime
bitfinex_df['timestamp'] = pd.to_datetime(bitfinex_df['timestamp'], utc=True)
yahoo_df['timestamp'] = pd.to_datetime(yahoo_df['timestamp'], utc=True)
mtgox_df['timestamp'] = pd.to_datetime(mtgox_df['timestamp'], utc=True)

# Concatenate and remove duplicates
combined_df = pd.concat([mtgox_df, yahoo_df, bitfinex_df])
combined_df = combined_df.sort_values('timestamp')
combined_df = combined_df.drop_duplicates(subset=['timestamp'])

# Fill gaps and ensure column order
combined_df = combined_df.set_index('timestamp').reindex(
    pd.date_range(start=combined_df['timestamp'].min(), end=combined_df['timestamp'].max(), freq='h', tz='UTC'),
    method='ffill'
).reset_index()
combined_df = combined_df.rename(columns={'index': 'timestamp'})
combined_df = combined_df.ffill()
combined_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# Format timestamp for CSV
combined_df['timestamp'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
combined_df.to_csv('btc_usd_historical_combined.csv', index=False)
print(f"Combined data saved to btc_usd_historical_combined.csv with {len(combined_df)} rows")