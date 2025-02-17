import ccxt
from src.utils.config import COINBASE_API_KEY, COINBASE_API_SECRET

def create_coinbase_exchange():
    return ccxt.coinbaseadvanced({
        'apiKey': COINBASE_API_KEY,
        'secret': COINBASE_API_SECRET,
    })
