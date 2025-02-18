import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def select_features(df, target_col, n_features=10):
    """
    Select top n features based on Random Forest Importance.

    :param df: DataFrame containing features
    :param target_col: Column name for the target variable
    :param n_features: Number of top features to select
    :return: DataFrame with selected features
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col].values
    
    # Initialize Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Use SelectFromModel to select features
    selector = SelectFromModel(rf, prefit=True, max_features=n_features)
    selected_features = list(X.columns[selector.get_support()])
    
    # Return DataFrame with only selected features and target
    return df[selected_features + [target_col]]

def preprocess_data(df, feature_scaler=None, target_scaler=None):
    if len(df) < 14:
        raise ValueError("DataFrame must contain at least 14 rows to calculate ATR.")
    
    # Initialize scalers if not provided
    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
    if target_scaler is None:
        target_scaler = MinMaxScaler()

    logging.info("After calculating returns:\n%s", df['returns'].head())
    logging.info("After calculating ATR:\n%s", df['atr'].head())
    logging.info("After calculating RSI:\n%s", df['momentum_rsi'].head())
    logging.info("After calculating MACD:\n%s", df['trend_macd'].head())
    
    # Check for all zero features
    for feature in ['momentum_rsi', 'trend_macd', 'atr', 'price_volatility']:
        if df[feature].abs().sum() == 0:
            logging.warning(f"{feature} is all zeros; check data or calculation.")

    # Select only the features listed in FEATURE_COLUMNS
    df = df[FEATURE_COLUMNS + ['target']]

    # Convert DataFrame to numpy array for fitting
    df_features = df[FEATURE_COLUMNS].values
    df_target = df[['target']].values

    # Fit scalers with numpy arrays
    feature_scaler.fit(df_features)
    target_scaler.fit(df_target)

    # Apply scaling 
    df[FEATURE_COLUMNS] = feature_scaler.transform(df_features)
    df['target'] = target_scaler.transform(df_target).flatten()  # Ensure target is 1D after scaling

    logging.info("After feature scaling:\n%s", df[FEATURE_COLUMNS].head())
    logging.info("After target scaling:\n%s", df['target'].head())

    # Save scalers
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

    logging.info("Final DataFrame NaN check:\n%s", df.isnull().sum())

    return df

def split_data(df, train_ratio=0.8):
    """
    Split the data into training and testing sets.

    :param df: Preprocessed DataFrame
    :param train_ratio: Ratio of data to use for training
    :return: Tuple of X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=['target'])
    y = df['target']
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

def prepare_data_for_training(df, train_ratio=0.8, feature_scaler=None, target_scaler=None):
    """
    Prepare the data for training by preprocessing, feature selection, and splitting.

    :param df: Raw DataFrame with OHLCV data
    :param train_ratio: Ratio of data to use for training
    :param feature_scaler: Scaler for feature normalization (e.g., MinMaxScaler)
    :param target_scaler: Scaler for target normalization (e.g., MinMaxScaler)
    :return: Tuple of X_train, X_test, y_train, y_test
    """
    preprocessed_df = preprocess_data(df, feature_scaler, target_scaler)
    return split_data(preprocessed_df, train_ratio)