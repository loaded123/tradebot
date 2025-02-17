import schedule
import time
import os
from utils.logger import logger
from models.model_trainer import train_model  # Assuming this function saves the model
from data.data_fetcher import fetch_historical_data
from data.data_preprocessor import preprocess_data, split_data

def retrain_model():
    """
    Fetch new data, preprocess it, train the model, and save it.
    """
    logger.info("Starting daily model retraining...")
    
    # Fetch new data (you might want to define how much historical data you need)
    crypto = 'BTC-USD'
    df = fetch_historical_data(crypto)
    
    # Preprocess data
    preprocessed_df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(preprocessed_df)
    
    # Train the model - assuming LSTMModel and train_model are set up for this
    from models.model_trainer import LSTMModel
    model = LSTMModel(input_dim=len(preprocessed_df.columns) - 1, hidden_dim=32, layer_dim=1, output_dim=1)
    trained_model = train_model(X_train, y_train)
    
    # Save the model - you'll need to implement this in train_model or here
    # Here's a basic example:
    model_path = 'models/trained_model.pth'
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Model retrained and saved at {model_path}")

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