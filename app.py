# app.py

from flask import Flask, jsonify, render_template
import pandas as pd
from datetime import datetime
import logging
import os
import sys
import asyncio
import numpy as np

# Add src/ to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.trading.paper_trader import PaperTradingEngine
from src.utils.config import Config

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize paper trading engine
engine = PaperTradingEngine()
asyncio.run(engine.fetch_market_data())

# Mock signals for initial backtest (replace with actual signals from signal_generator)
signals = engine.market_data[['close']].copy()
signals['symbol'] = Config.TRADING_PAIR
signals['signal'] = np.random.choice([0, 1, -1], size=len(signals))  # Random signals for demo
trade_log = asyncio.run(engine.run_backtest(signals))

# In-memory storage for logs
logs = []

# Add a stream handler to capture logs
class LogCaptureHandler(logging.Handler):
    def emit(self, record):
        logs.append(self.format(record))
        if len(logs) > 100:  # Keep last 100 logs
            logs.pop(0)

log_handler = LogCaptureHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(log_handler)

@app.route('/')
def dashboard():
    # Calculate performance metrics
    if not trade_log.empty:
        total_pnl = trade_log['profit'].sum() if 'profit' in trade_log.columns else 0.0
        num_trades = len(trade_log)
        win_rate = len(trade_log[trade_log['profit'] > 0]) / num_trades if num_trades > 0 else 0.0
    else:
        total_pnl = 0.0
        num_trades = 0
        win_rate = 0.0

    # Placeholder metrics (replace with backtest results)
    sharpe_ratio = 2.0  # Placeholder
    sortino_ratio = 3.0  # Placeholder
    max_drawdown = 15.0  # Placeholder

    performance = {
        'pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    trades = trade_log.to_dict('records') if not trade_log.empty else []

    return render_template('dashboard.html', performance=performance, trades=trades, logs=logs)

@app.route('/api/performance')
def api_performance():
    if not trade_log.empty:
        total_pnl = trade_log['profit'].sum() if 'profit' in trade_log.columns else 0.0
        num_trades = len(trade_log)
        win_rate = len(trade_log[trade_log['profit'] > 0]) / num_trades if num_trades > 0 else 0.0
        # Mock P&L over time (replace with actual portfolio value history)
        timestamps = trade_log['timestamp'].astype(str).tolist()
        pnl_values = trade_log['profit'].cumsum().tolist()
    else:
        total_pnl = 0.0
        num_trades = 0
        win_rate = 0.0
        timestamps = []
        pnl_values = []

    # Placeholder metrics (replace with backtest results)
    sharpe_ratio = 2.0
    sortino_ratio = 3.0
    max_drawdown = 15.0

    return jsonify({
        'pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pnl_timestamps': timestamps,
        'pnl_values': pnl_values
    })

@app.route('/api/trades')
def api_trades():
    trades = trade_log.to_dict('records') if not trade_log.empty else []
    return jsonify({'trades': trades})

@app.route('/api/logs')
def api_logs():
    return jsonify({'logs': logs})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)