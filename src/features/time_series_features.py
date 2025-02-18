import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # You can add more features like is_weekend, etc.
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    logging.info("Added time-based features to the dataset.")
    
    return df

if __name__ == "__main__":
    # Load your data here
    from src.data.load_data import load_data
    df = load_data()  # You need to implement or adjust this function to load your data
    
    df_with_time_features = add_time_features(df)
    # Continue with your preprocessing pipeline