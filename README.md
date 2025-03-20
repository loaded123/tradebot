# TradeBot: Transformer-Based BTC/USD Trading System

## Overview

TradeBot is a machine learning-driven trading system designed for BTC/USD price prediction and automated trading on an hourly timeframe. It leverages a Transformer model to predict price movements, incorporating advanced technical indicators and signals from tools like LuxAlgo, TrendSpider, SMRT Algo, HassOnline arbitrage, and MetaStock trend slope analysis. The system includes modules for data fetching, model training, signal generation, risk management, position sizing, and backtesting with visualization.

### Key Features

- **Transformer Model**: A custom Transformer architecture (`TransformerPredictor`) with Monte Carlo Dropout for uncertainty estimation, predicting log returns for the next hour.
- **Feature Set**: 22 technical features, including RSI, MACD, Bollinger Bands, ATR, VPVR (Volume Profile Visible Range), and advanced signals (LuxAlgo, TrendSpider, SMRT, HassOnline, MetaStock).
- **Market Regime Detection**: Dynamically adjusts trading parameters based on market conditions (bullish, bearish, neutral, high/low volatility).
- **Risk Management**: Implements dynamic position sizing, drawdown limits, and ATR-based stop-loss/take-profit levels.
- **Backtesting and Visualization**: Comprehensive backtesting with equity curve, drawdown, and signal visualization.
- **Modular Design**: Organized into separate modules for data fetching, model training, signal generation, and strategy execution.

### Project Structure
    TradeBot/
    │
    ├── src/
    │   ├── data/
    │   │   ├── data_fetcher.py         # Fetches historical and live data (currently CSV-based)
    │   │   ├── data_preprocessor.py    # Preprocesses raw data for model input
    │   │   └── btc_usd_historical.csv  # Historical BTC/USD data (update with your own data)
    │   │
    │   ├── models/
    │   │   ├── model_predictor.py      # Handles live price prediction using the trained model
    │   │   ├── train_transformer_model.py # Training pipeline for the Transformer model
    │   │   └── transformer_model.py    # TransformerPredictor model definition
    │   │
    │   ├── strategy/
    │   │   ├── backtest_visualizer_ultimate.py # Backtesting and visualization of trading strategy
    │   │   ├── execution.py            # Placeholder for live trade execution (to be implemented)
    │   │   ├── indicators.py           # Technical indicators (RSI, MACD, VPVR, LuxAlgo, etc.)
    │   │   ├── market_regime.py        # Detects market regimes (bullish, bearish, etc.)
    │   │   ├── position_sizer.py       # Calculates position sizes (e.g., Kelly Criterion)
    │   │   ├── risk_manager.py         # Manages risk (drawdown limits, stop-loss, etc.)
    │   │   └── signal_generator.py     # Generates trading signals using model predictions
    │   │
    │   ├── utils/
    │   │   └── sequence_utils.py       # Utilities for creating sequences for model input
    │   │
    │   └── constants.py                # Constants (e.g., FEATURE_COLUMNS, weights for signals)
    │
    ├── best_model.pth                  # Trained Transformer model weights
    ├── feature_scaler.pkl              # Scaler for input features
    ├── target_scaler.pkl               # Scaler for target variable
    └── README.md                       # Project documentation


## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch (for model training and inference)
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`, `asyncio`, `winloop` (for Windows event loop)

### Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/TradeBot.git
   cd TradeBot

2. **Install Dependencies**: Create a virtual environment and install the required packages:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install torch numpy pandas matplotlib scikit-learn joblib asyncio winloop 

3. **Prepare Data**:   
   Place your historical BTC/USD data in src/data/btc_usd_historical.csv.
   The CSV should have columns: timestamp, open, high, low, close, volume.
   Ensure timestamps are in a parseable format (e.g., YYYY-MM-DD HH:MM:SS).

### Usage

1. **Train the Model**:
   python -m src.models.train_transformer_model

   Output:
   Trained model saved as best_model.pth.
   Feature and target scalers saved as feature_scaler.pkl and target_scaler.pkl.
   Logs will display training progress, including loss, learning rate, and test metrics (MSE, MAE, MAPE).

2. **Run a Backtest**:
   Backtest the strategy and visualize results:
   python -m src.strategy.backtest_visualizer_ultimate

   Output:
   Backtest metrics (Sharpe ratio, Sortino ratio, max drawdown, etc.) logged to the console.
   Visualizations saved as BTC-USD_backtest_results.png, including equity curve, drawdown, and signal plots.

3. **Live Trading (Not Yet Implemented)**:
   Live trading is currently a placeholder in execution.py. To enable live trading:

   Integrate an exchange API (e.g., Binance) using ccxt.
   Implement order execution logic in execution.py.
   Update model_predictor.py to fetch live data and generate real-time signals.

3. **Performance Metrics (Pre-Feature Additions)**:

   Based on the last training run (before adding LuxAlgo, TrendSpider, SMRT, HassOnline, and MetaStock features):

   MAPE: 0.36% (highly competitive, indicating excellent predictive accuracy).
   MAE (Price Space): 285.75 USD (reasonable for BTC/USD prices in the $67,000-$69,000 range).
   MSE (Price Space): 206,927.54 (high, but less relevant than MAPE for interpretability).

   Sample Predictions:
   Actual: [67675.48, 68912.80, 67974.02, 67213.79, 69021.20]
   Predicted: [67499.72, 68252.71, 67903.07, 68171.60, 68238.62]
   Confidences: [0.9962, 0.9967, 0.9966, 0.9963, 0.9955] (very high, indicating strong model certainty).
   

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a pull request with a detailed description of your changes.
Development Guidelines
Follow PEP 8 style guidelines for Python code.
Add inline comments to explain complex logic.
Update this README if you add new features or change usage instructions.
Test your changes by running training and backtesting scripts.
Future Improvements
Live Trading: Implement live trading with ccxt integration in execution.py.
Alternative Data: Add on-chain data (e.g., transaction volume) or real sentiment analysis (e.g., via paid X API).
Scalability: Refactor to support multiple assets and higher frequencies (e.g., minute-level trading).
Performance Optimization: Optimize the Transformer model for faster training/inference (e.g., reduce d_model, use model pruning).
UI: Develop a web-based UI (e.g., using Flask or Dash) for real-time monitoring.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or support, contact dromback@gmail.com or open an issue on GitHub.