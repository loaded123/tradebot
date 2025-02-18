import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_feature_importance(X, y):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    logging.info("Feature Importance Analysis:\n" + importance_df.to_string())
    
    return importance_df

if __name__ == "__main__":
    # Load your data here
    # Assuming 'X' and 'y' are your features and target
    # For demonstration, let's assume you have a function load_data() that returns X and y
    from src.data.load_data import load_data
    X, y = load_data()  # You need to implement or adjust this function to load your data
    
    importance_df = compute_feature_importance(X, y)
    # Optionally save or plot this data