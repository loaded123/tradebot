import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

def create_gemini_exchange():
    # Fetch credentials from the environment
    api_key = os.getenv('GEMINI_API_KEY')
    api_secret = os.getenv('GEMINI_API_SECRET')
    
    # Ensure the API keys are set correctly
    if not api_key or not api_secret:
        raise ValueError("API Key or Secret is missing. Please check your .env file.")

    # Initialize the Gemini exchange object
    exchange = ccxt.gemini({
        'apiKey': api_key,
        'secret': api_secret,
    })
    return exchange