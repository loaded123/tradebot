# File: src/data/data_preprocessor.py

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
import joblib
import logging
from src.strategy.indicators import compute_vwap, compute_adx  # Import indicators functions

logging.basicConfig(level=logging.INFO)

def remove_outliers(df, columns):
    """Remove outliers from the DataFrame using the Interquartile Range (IQR) method."""
    df_cleaned = df.copy()
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned

def select_features(df, target_col, n_features=10):
    """Select top n features based on Random Forest Importance."""
    X = df.drop(target_col, axis=1)
    y = df[target_col].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    selector = SelectFromModel(rf, prefit=True, max_features=n_features)
    selected_features = list(X.columns[selector.get_support()])

    return df[selected_features + [target_col]]

def preprocess_data(df, feature_scaler=None, target_scaler=None):
    """Preprocesses the data."""

    if len(df) < 14:
        raise ValueError("DataFrame must contain at least 14 rows to calculate ATR.")

    # Calculate indicators *before* outlier removal and feature selection
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['price_volatility'] = df['log_returns'].rolling(window=14).std() * (252**0.5)  # Annualized Volatility
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['atr'] = df['high'].rolling(window=14).apply(lambda x: max(abs(x[1:] - x[:-1])), raw=True)
    df['vwap'] = compute_vwap(df)
    df['adx'] = compute_adx(df, period=20)  # ADX period can be adjusted here
    df['target'] = df['close'].shift(-1)  # Next period's closing price
    df.fillna(0, inplace=True) # Fill NaN values

    # Remove outliers (now includes the calculated indicators)
    columns_to_check = FEATURE_COLUMNS + ['target', 'vwap', 'adx', 'atr']  # Include new columns
    df = remove_outliers(df, columns_to_check)

    logging.info("After calculating returns:\n%s", df['returns'].head())
    logging.info("After calculating ATR:\n%s", df['atr'].head())
    logging.info("After calculating VWAP:\n%s", df['vwap'].head())
    logging.info("After calculating ADX:\n%s", df['adx'].head())

    # Check for all zero features (after outlier removal)
    for feature in ['momentum_rsi', 'trend_macd', 'atr', 'price_volatility', 'vwap', 'adx']:
        if df[feature].abs().sum() == 0:
            logging.warning(f"{feature} is all zeros; check data or calculation.")

    # Select features (after indicator calculation and outlier removal)
    df = df[FEATURE_COLUMNS + ['target']]  # Keep only the selected features

    # ... (rest of the scaling and saving logic remains the same)
    # Convert DataFrame to numpy array for fitting
    df_features = df[FEATURE_COLUMNS].values
    df_target = df[['target']].values

    # Initialize scalers if not provided
    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
    if target_scaler is None:
        target_scaler = MinMaxScaler()

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