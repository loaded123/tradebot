# src/scheduler_new.py

import schedule
import time
import os
import torch
import asyncio
from src.utils.logger import logger
from src.models.transformer_model import TransformerPredictor
from src.models.train_transformer_model import train_model  # Use Transformer training
from src.data.data_fetcher import fetch_historical_data
from src.data.data_preprocessor import preprocess_data, split_data
from src.constants import FEATURE_COLUMNS  # Import feature columns for consistency

def retrain_model():
    """
    Fetch new data, preprocess it, train the Transformer model, and save it.
    """
    logger.info("Starting daily model retraining...")
    
    # Fetch new data (you might want to define how much historical data you need)
    crypto = 'BTC/USD'
    historical_data = asyncio.run(fetch_historical_data(crypto))
    
    # Preprocess data
    preprocessed_df = preprocess_data(historical_data)
    X, y = create_sequences(preprocessed_df[FEATURE_COLUMNS].values, preprocessed_df['target'].values, seq_length=13)
    # Split data (simplified, assuming train_transformer_model handles splitting internally)
    # For now, use a simple split or rely on train_transformer_model's logic
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), train_size=0.7, shuffle=False)
    
    # Train the Transformer model
    model = TransformerPredictor(input_dim=len(FEATURE_COLUMNS), d_model=64, n_heads=4, n_layers=2, dropout=0.1)
    trained_model = train_model(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train),
        torch.FloatTensor(X_test), torch.FloatTensor(y_test),
        epochs=200, batch_size=32, learning_rate=0.001, patience=10
    )
    
    # Save the model to both root and src/models for compatibility
    model_path_root = os.path.join(os.path.dirname(__file__), '../../best_model.pth')
    model_path_src = os.path.join(os.path.dirname(__file__), 'src/models/best_model.pth')
    
    try:
        torch.save(trained_model.state_dict(), model_path_root)
        logger.info(f"Model retrained and saved at {model_path_root}")
        if not os.path.exists(os.path.dirname(model_path_src)):
            os.makedirs(os.path.dirname(model_path_src))
        torch.save(trained_model.state_dict(), model_path_src)
        logger.info(f"Model also saved at {model_path_src} for fallback")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def job():
    """
    Wrapper for scheduled tasks including model retraining.
    """
    retrain_model()
    # Add other daily tasks here if needed

# Schedule the job to run daily at midnight
schedule.every().day.at("00:00").do(job)

if __name__ == "__main__":
    logger.info("Scheduler started")
    while True:
        schedule.run_pending()
        time.sleep(1)