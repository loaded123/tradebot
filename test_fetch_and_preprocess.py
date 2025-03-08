# test_fetch_and_preprocessor.py
import asyncio
import logging
from src.data import data_fetcher, data_preprocessor

# Set logging level to DEBUG for detailed debugging
logging.getLogger('data_fetcher').setLevel(logging.DEBUG)
logging.getLogger('ccxt').setLevel(logging.DEBUG)

async def test_fetch_and_preprocess():
    # Fetch historical data for full range
    df = await data_fetcher.fetch_historical_data('BTC/USD', '1h')
    print(f"Fetched data:\n{df.head()}")
    print(f"Index type: {type(df.index)}, Index sample: {df.index[:5]}, last: {df.index[-5:]}")

    # Preprocess the data
    processed_df = data_preprocessor.preprocess_data(df)
    print(f"Preprocessed data:\n{processed_df.head()}")
    print(f"Processed index: {processed_df.index[:5]}, last: {processed_df.index[-5:]}")
    print(f"Feature columns: {processed_df.columns.tolist()}")
    print(f"Price range: min_close={processed_df['close'].min()}, max_close={processed_df['close'].max()}")

if __name__ == "__main__":
    asyncio.run(test_fetch_and_preprocess())