import sys
import ccxt
from src.api.gemini import create_gemini_exchange

def test_api_connection():
    print("Script is executing")
    print("Checking environment variables...")
    
    try:
        # Create a Gemini exchange instance
        exchange = create_gemini_exchange()
        
        print("Fetching markets...")
        markets = exchange.fetch_markets()
        if markets:
            print(f"Successfully fetched markets. First market: {markets[0]['symbol']}")
        else:
            print("No markets fetched.")
        
        # Optionally, test fetching balance for more robust authentication check
        print("Fetching balance...")
        balance = exchange.fetch_balance()
        print(f"Account balance: {balance}")
        
        print("API connection test passed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_api_connection()