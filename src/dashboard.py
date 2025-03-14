# dashboard.py
from flask import Flask, render_template, jsonify
from datetime import datetime
import json
import os
import pandas as pd

app = Flask(__name__)

# Paths to backtest results and trade data
BACKTEST_RESULTS_FILE = "backtest_results.json"
TRADE_DATA_FILE = "trade_data.json"

def load_performance_data():
    """Load performance data from the backtest results file."""
    default_data = {
        'pnl': 0.0,
        'num_trades': 0,
        'win_rate': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'max_drawdown': 0.0,
        'last_update': datetime.now().isoformat(),
        'pnl_timestamps': [],
        'pnl_values': []
    }
    if os.path.exists(BACKTEST_RESULTS_FILE):
        try:
            with open(BACKTEST_RESULTS_FILE, 'r') as f:
                data = json.load(f)
            # Ensure PNL data is included (to be updated in backtest)
            if 'pnl_timestamps' not in data:
                data['pnl_timestamps'] = []
            if 'pnl_values' not in data:
                data['pnl_values'] = []
            return data
        except Exception as e:
            print(f"Error loading performance data: {e}")
    return default_data

def load_trade_data():
    """Load trade data from the trade data file."""
    default_trades = {'trades': []}
    if os.path.exists(TRADE_DATA_FILE):
        try:
            with open(TRADE_DATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading trade data: {e}")
    return default_trades

@app.route('/')
def dashboard():
    performance = load_performance_data()
    trades = load_trade_data()
    log_file = 'trading_bot.log'
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.read().splitlines()[-100:]  # Last 100 lines
    return render_template('dashboard.html', performance=performance, trades=trades['trades'], logs=logs)

@app.route('/api/performance')
def get_performance():
    """API endpoint to get performance metrics for real-time updates."""
    data = load_performance_data()
    return jsonify(data)

@app.route('/api/trades')
def get_trades():
    """API endpoint to get trade data for real-time updates."""
    data = load_trade_data()
    return jsonify(data)

@app.route('/api/logs')
def get_logs():
    """API endpoint to get logs for real-time updates."""
    log_file = 'trading_bot.log'
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.read().splitlines()[-100:]
    return jsonify({'logs': logs})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)