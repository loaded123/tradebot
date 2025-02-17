import time
import logging
import pandas as pd
from datetime import datetime
from data.data_fetcher import fetch_historical_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperTradingEngine:
    def __init__(self, initial_capital=10000, fee_rate=0.001, slippage_rate=0.0005):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.cash = initial_capital
        self.positions = {}
        self.trade_log = []
        self.market_data = None

    def fetch_market_data(self, symbol, timeframe='1h', limit=1000):
        """Fetch and store market data"""
        self.market_data = fetch_historical_data(symbol, timeframe, limit)
        logging.info(f"Fetched {len(self.market_data)} data points for {symbol}.")

    def execute_trade(self, symbol, size, trade_type, price):
        """Simulate trade execution with slippage and fees"""
        if trade_type == 'buy':
            execution_price = price * (1 + self.slippage_rate)
            cost = execution_price * size
            fee = cost * self.fee_rate
            total_cost = cost + fee

            if total_cost > self.cash:
                logging.warning("Insufficient funds for trade.")
                return
            
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + size

        elif trade_type == 'sell':
            if self.positions.get(symbol, 0) < size:
                logging.warning("Not enough holdings to sell.")
                return
            
            execution_price = price * (1 - self.slippage_rate)
            proceeds = execution_price * size
            fee = proceeds * self.fee_rate
            net_proceeds = proceeds - fee

            self.cash += net_proceeds
            self.positions[symbol] -= size
            if self.positions[symbol] == 0:
                del self.positions[symbol]

        self.trade_log.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'size': size,
            'trade_type': trade_type,
            'execution_price': execution_price,
            'cash_balance': self.cash
        })
        logging.info(f"Executed {trade_type.upper()} order for {size} {symbol} at {execution_price:.2f}")

    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        holdings_value = sum(size * current_prices[symbol] for symbol, size in self.positions.items())
        return self.cash + holdings_value

    def run_backtest(self, strategy_signals):
        """Simulate trading based on generated signals"""
        for i, row in strategy_signals.iterrows():
            symbol = row['symbol']
            price = row['close']
            signal = row['signal']
            size = self.cash * 0.02 / price  # Risk 2% per trade

            if signal == 1:
                self.execute_trade(symbol, size, 'buy', price)
            elif signal == -1:
                self.execute_trade(symbol, size, 'sell', price)

        logging.info(f"Final Portfolio Value: {self.get_portfolio_value(strategy_signals['close'].iloc[-1])}")
        return pd.DataFrame(self.trade_log)

if __name__ == "__main__":
    engine = PaperTradingEngine()
    engine.fetch_market_data('BTC/USD')
