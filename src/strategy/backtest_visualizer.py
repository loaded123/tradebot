import asyncio
import sys
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import torch
import joblib
from src.models.transformer_model import TransformerPredictor
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data
from src.constants import FEATURE_COLUMNS
from src.strategy.backtester import backtest_strategy
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logging.debug(f"numpy version in backtest_visualizer: {__import__('numpy').__version__ if 'numpy' in globals() else 'Not loaded'}")
import src.strategy.strategy_generator as strategy_generator  # Changed to import the entire module
from src.strategy.market_regime import detect_market_regime
from src.strategy.position_sizer import calculate_position_size
from src.strategy.risk_manager import manage_risk
import importlib

if sys.platform == "win32":
    import winloop
    asyncio.set_event_loop_policy(winloop.EventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_backtest_results(results):
    """Plot backtest results including equity curve, returns, drawdowns, and trades."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

    axes[0].plot(results.index, results['total'], label='Equity Curve', color='blue')
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].set_title("Equity Curve")
    axes[0].legend()
    axes[0].grid()

    cum_returns = results['cumulative_returns']
    axes[1].plot(results.index, cum_returns, label='Cumulative Returns', color='green')
    axes[1].set_ylabel("Cumulative Return (%)")
    axes[1].set_title(f"Cumulative Returns (Total: {cum_returns.iloc[-1]:.2f}%)")
    axes[1].legend()
    axes[1].grid()

    drawdown = (results['total'] / results['total'].cummax() - 1) * 100
    axes[2].fill_between(results.index, drawdown, color='red', alpha=0.3, label='Drawdown')
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_title(f"Max Drawdown: {drawdown.min():.2f}%")
    axes[2].legend()
    axes[2].grid()

    axes[3].plot(results.index, results['close'], label='Price', color='black')
    buy_signals = results[results['signal'] == 1]
    sell_signals = results[results['signal'] == -1]
    axes[3].plot(buy_signals.index, buy_signals['close'], '^', markersize=8, color='g', label='Buy')
    axes[3].plot(sell_signals.index, sell_signals['close'], 'v', markersize=8, color='r', label='Sell')
    axes[3].set_ylabel("Price")
    axes[3].set_title("Price with Buy/Sell Signals")
    axes[3].legend()
    axes[3].grid()

    daily_returns = results['returns'].dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    axes[4].plot(results.index, daily_returns.cumsum(), label=f'Daily Returns (Sharpe: {sharpe:.2f})', color='purple')
    axes[4].set_ylabel("Cumulative Daily Returns")
    axes[4].set_title("Daily Returns")
    axes[4].legend()
    axes[4].grid()

    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.show()

def load_model(model_class):
    """Load the trained transformer model with enhanced debugging and compatibility checks."""
    model_path = "best_model.pth"
    input_dim = len(FEATURE_COLUMNS)  # 17, matching training
    model = TransformerPredictor(input_dim=input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1).to(device)
    
    # Enhanced debugging for model loading, ensuring all logs execute
    logging.debug(f"Attempting to load model - PyTorch version: {torch.__version__}")
    logging.debug(f"Attempting to load model from: {os.path.abspath(model_path)}")
    
    try:
        # Check if file exists before attempting to load
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at: {os.path.abspath(model_path)}")
            return None
        
        logging.debug(f"Model path exists: {os.path.exists(model_path)}")
        logging.debug(f"Model input_dim: {input_dim}, device: {device}")
        
        # Attempt to load state dict with detailed logging
        state_dict = torch.load(model_path, map_location=device)
        logging.debug(f"Loaded state dict keys: {list(state_dict.keys())}")
        
        # Try loading with strict=False to handle potential mismatches, logging any issues
        try:
            model.load_state_dict(state_dict, strict=False)
            logging.debug("Model state dict loaded successfully with strict=False")
        except RuntimeError as e:
            logging.error(f"Partial load failed - state dict mismatch: {e}")
            # Try to identify mismatched keys
            model_dict = model.state_dict()
            missing_keys = [k for k in state_dict.keys() if k not in model_dict]
            unexpected_keys = [k for k in model_dict.keys() if k not in state_dict]
            logging.debug(f"Missing keys: {missing_keys}")
            logging.debug(f"Unexpected keys: {unexpected_keys}")
            return None
        
        model.eval()
        logging.info(f"Loaded model from {model_path} with input_dim={input_dim}")
        return model
    
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {e}")
        return None
    except RuntimeError as e:
        logging.error(f"Error loading state dict: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading model: {e}")
        return None

def filter_signals(signal_data, min_hold_period=5):
    """Filter signals to enforce minimum hold period."""
    last_signal_date = None
    last_signal = 0
    signal_data['filtered_signal'] = signal_data['signal']
    
    for i, row in signal_data.iterrows():
        current_date = row.name
        current_signal = row['signal']
        if (last_signal_date is None or 
            (current_date - last_signal_date).total_seconds() / 3600 >= min_hold_period):
            last_signal_date = current_date
            last_signal = current_signal
        else:
            signal_data.loc[current_date, 'filtered_signal'] = last_signal
    signal_data['signal'] = signal_data['filtered_signal']
    signal_data.drop(columns=['filtered_signal'], inplace=True)
    return signal_data

async def main():
    """Run backtest and visualize results."""
    symbol = 'BTC/USD'
    try:
        # Fetch and preprocess data
        historical_data = await fetch_historical_data(symbol)
        if historical_data.empty:
            raise ValueError("No historical data fetched.")
        
        preprocessed_data = preprocess_data(historical_data)
        logging.info(f"Preprocessed data shape: {preprocessed_data.shape}")

        # Load model
        model = load_model(TransformerPredictor)
        if model is None:
            raise ValueError("Model loading failed.")

        # Prepare scaled data with all training features
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
        train_columns = FEATURE_COLUMNS + ['target']
        
        # Ensure feature names match during scaling (if MinMaxScaler was fitted with feature names)
        if hasattr(feature_scaler, 'feature_names_in_') and feature_scaler.feature_names_in_ is not None:
            scaled_data = feature_scaler.transform(preprocessed_data[FEATURE_COLUMNS])
        else:
            scaled_data = feature_scaler.transform(preprocessed_data[FEATURE_COLUMNS].values)
        scaled_df = pd.DataFrame(scaled_data, columns=FEATURE_COLUMNS, index=preprocessed_data.index)
        scaled_df['target'] = target_scaler.transform(preprocessed_data[['target']])
        scaled_df['close'] = preprocessed_data['close']
        scaled_df['price_volatility'] = preprocessed_data['price_volatility']
        logging.info(f"Scaled DataFrame columns: {scaled_df.columns}")

        # Market regime and signals
        regime = detect_market_regime(preprocessed_data)
        logging.info(f"Detected market regime: {regime}")

        regime_params = {
            'Bullish Low Volatility': {'rsi_threshold': 60, 'macd_fast': 10, 'macd_slow': 20, 'atr_multiplier': 1.5, 'max_risk_pct': 0.05},
            'Bullish High Volatility': {'rsi_threshold': 70, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2.5, 'max_risk_pct': 0.05},
            'Bearish Low Volatility': {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2, 'max_risk_pct': 0.05},
            'Bearish High Volatility': {'rsi_threshold': 40, 'macd_fast': 15, 'macd_slow': 30, 'atr_multiplier': 3, 'max_risk_pct': 0.05},
        }
        params = regime_params.get(regime, {'rsi_threshold': 50, 'macd_fast': 12, 'macd_slow': 26, 'atr_multiplier': 2, 'max_risk_pct': 0.05})

        # Force reload of strategy_generator to bypass caching issues
        importlib.reload(strategy_generator)

        signal_data = strategy_generator.generate_signals(scaled_df, model, train_columns, feature_scaler, target_scaler, 
                                                        rsi_threshold=params['rsi_threshold'], macd_fast=params['macd_fast'], 
                                                        macd_slow=params['macd_slow'], atr_multiplier=params['atr_multiplier'], 
                                                        max_risk_pct=params['max_risk_pct'])
        logging.info(f"Signal data columns before filtering: {signal_data.columns.tolist()}")
        signal_data = filter_signals(signal_data, min_hold_period=5)

        # Risk management with price_volatility
        current_balance = 10000
        signal_data = manage_risk(signal_data, current_balance, atr_multiplier=params['atr_multiplier'])

        logging.info(f"Signal data columns before backtest: {signal_data.columns.tolist()}")
        results = backtest_strategy(signal_data, initial_capital=current_balance)
        plot_backtest_results(results)

    except Exception as e:
        logging.error(f"Error in backtest: {e}")

if __name__ == "__main__":
    asyncio.run(main())