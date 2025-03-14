# src/trading/paper_trader.py

import time
import logging
import pandas as pd
from datetime import datetime
import asyncio
import numpy as np
import torch
import joblib
import os

from constants import FEATURE_COLUMNS  # Added missing import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add `src/` directory to Python's module search path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data.data_fetcher import fetch_historical_data
from src.models.transformer_model import TransformerPredictor
from src.strategy.signal_generator import generate_signals
from src.utils.config import Config

class PaperTradingEngine:
    def __init__(self, initial_capital=Config.INITIAL_CAPITAL, fee_rate=0.001, slippage_rate=0.0005):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.cash = initial_capital
        self.positions = {}
        self.trade_log = []
        self.market_data = None
        self.model = self._load_model()
        self.feature_scaler = joblib.load('feature_scaler.pkl')
        self.target_scaler = joblib.load('target_scaler.pkl')

    def _load_model(self):
        """Load the trained Transformer model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS), d_model=128, n_heads=8, n_layers=4, dropout=0.7).to(device)
        model_path = os.path.join(os.path.dirname(__file__), '../../best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            logger.info(f"Loaded Transformer model from {model_path}")
        else:
            logger.error(f"Model file not found at {model_path}. Using untrained model.")
        return model

    async def fetch_market_data(self, symbol=Config.TRADING_PAIR, timeframe=Config.TIME_FRAME, limit=Config.HISTORICAL_DATA_POINTS):
        """Fetch and store market data"""
        self.market_data = await fetch_historical_data(symbol, timeframe, limit)
        logger.info(f"Fetched {len(self.market_data)} data points for {symbol}.")

    def execute_trade(self, symbol, size, trade_type, price, stop_loss=None, take_profit=None):
        """Simulate trade execution with slippage, fees, stop-loss, and take-profit"""
        execution_price = price * (1 + self.slippage_rate) if trade_type == 'buy' else price * (1 - self.slippage_rate)
        cost = execution_price * size
        fee = cost * self.fee_rate
        total_cost = cost + fee

        if trade_type == 'buy':
            if total_cost > self.cash:
                logger.warning("Insufficient funds for trade.")
                return False
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + size
        elif trade_type == 'sell':
            if self.positions.get(symbol, 0) < size:
                logger.warning("Not enough holdings to sell.")
                return False
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
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'cash_balance': self.cash
        })
        logger.info(f"Executed {trade_type.upper()} order for {size} {symbol} at {execution_price:.2f} "
                    f"(Stop-loss: {stop_loss}, Take-profit: {take_profit})")
        return True

    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        holdings_value = sum(size * current_prices[symbol] for symbol, size in self.positions.items())
        return self.cash + holdings_value

    async def run_backtest(self, strategy_signals):
        """Simulate trading based on generated signals"""
        for i, row in strategy_signals.iterrows():
            symbol = row['symbol'] if 'symbol' in row else Config.TRADING_PAIR
            price = row['close']
            signal = row['signal']
            current_balance = self.cash + self.get_portfolio_value({symbol: price})

            # Generate signals with model
            signal_data = pd.DataFrame({'close': [price]}, index=[i])
            signals = await generate_signals(
                signal_data,
                self.market_data.tail(24),
                self.model,
                FEATURE_COLUMNS,
                self.feature_scaler,
                self.target_scaler,
                rsi_threshold=30,
                macd_fast=12,
                macd_slow=26,
                atr_multiplier=2.0,
                max_risk_pct=Config.RISK_PER_TRADE
            )
            signal = signals['signal'].iloc[0] if not signals.empty else 0
            size = current_balance * Config.RISK_PER_TRADE / price if signal != 0 else 0

            if signal == 1:
                stop_loss = signals['stop_loss'].iloc[0] if 'stop_loss' in signals.columns else price * (1 - Config.RISK_PER_TRADE)
                take_profit = signals['take_profit'].iloc[0] if 'take_profit' in signals.columns else price * 1.05
                self.execute_trade(symbol, size, 'buy', price, stop_loss, take_profit)
            elif signal == -1:
                stop_loss = signals['stop_loss'].iloc[0] if 'stop_loss' in signals.columns else price * (1 + Config.RISK_PER_TRADE)
                take_profit = signals['take_profit'].iloc[0] if 'take_profit' in signals.columns else price * 0.95
                self.execute_trade(symbol, size, 'sell', price, stop_loss, take_profit)

        final_value = self.get_portfolio_value({Config.TRADING_PAIR: strategy_signals['close'].iloc[-1]})
        logger.info(f"Final Portfolio Value: {final_value}")
        return pd.DataFrame(self.trade_log)

if __name__ == "__main__":
    engine = PaperTradingEngine()
    asyncio.run(engine.fetch_market_data())
    # Example signals (replace with actual signals from your strategy)
    signals = engine.market_data[['close']].copy()
    signals['symbol'] = Config.TRADING_PAIR
    signals['signal'] = np.random.choice([0, 1, -1], size=len(signals))  # Random signals for demo
    asyncio.run(engine.run_backtest(signals))