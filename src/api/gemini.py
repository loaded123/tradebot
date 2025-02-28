import ccxt.async_support as ccxt
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

async def create_gemini_exchange():
    """
    Asynchronously create and return a Gemini exchange instance with API credentials.
    
    Returns:
        ccxt.async_support.gemini: Gemini exchange instance
    
    Raises:
        ValueError: If API keys are missing
        ccxt.AuthenticationError: If credentials are invalid
    """
    # Fetch credentials from the environment
    api_key = os.getenv('GEMINI_API_KEY')
    api_secret = os.getenv('GEMINI_API_SECRET')
    
    # Ensure the API keys are set correctly
    if not api_key or not api_secret:
        raise ValueError("API Key or Secret is missing. Please check your .env file.")

    try:
        # Initialize the Gemini exchange object (async version)
        exchange = ccxt.gemini({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        logging.info("Gemini exchange initialized successfully")
        return exchange
    except ccxt.AuthenticationError as e:
        logging.error(f"Authentication error for Gemini: {e}")
        raise
    except Exception as e:
        logging.error(f"Error initializing Gemini exchange: {e}")
        raise

# Example synchronous wrapper for testing (optional)
def get_gemini_exchange():
    return asyncio.run(create_gemini_exchange())