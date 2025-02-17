from flask import Flask, render_template, jsonify
from .utils.logger import logger
import json
from datetime import datetime, timedelta

app = Flask(__name__)

# Mock data - In a real scenario, you'd fetch this from your trading bot's database or API
performance_data = {
    'pnl': 1234.56,
    'num_trades': 45,
    'win_rate': 0.67,
    'last_update': (datetime.now() - timedelta(minutes=5)).isoformat()
}

@app.route('/')
def dashboard():
    # Fetch latest logs or performance data
    with open('trading_bot.log', 'r') as f:
        logs = f.read().splitlines()[-100:]  # Last 100 lines for example
    
    # Here, we're using mock data. In practice, you'd fetch this from your data source.
    return render_template('dashboard.html', logs=logs, performance=performance_data)

@app.route('/api/performance')
def get_performance():
    """API endpoint to get performance metrics for real-time updates."""
    return jsonify(performance_data)

@app.route('/api/logs')
def get_logs():
    """API endpoint to get logs for real-time updates."""
    with open('trading_bot.log', 'r') as f:
        logs = f.read().splitlines()[-100:]
    return jsonify({'logs': logs})

if __name__ == '__main__':
    app.run(debug=True)