# test_real_time.py
import asyncio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.data.data_fetcher import fetch_real_time_ws

async def test_real_time():
    i = 0
    async for data in fetch_real_time_ws('BTC/USD', exchange_name='gemini'):
        print(f"Real-time data {i + 1}: {data}")
        i += 1
        if i >= 10:  # Limit to 10 iterations for testing
            break

if __name__ == "__main__":
    asyncio.run(test_real_time())