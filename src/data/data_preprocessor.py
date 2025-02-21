# src/data/data_preprocessor.py

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def remove_outliers(df, columns):
    """Remove outliers using IQR method."""
    df_cleaned = df.copy()
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned

def preprocess_data(df, feature_scaler=None, target_scaler=None):
    """Preprocess OHLCV data for training."""
    if len(df) < 20:  # Increased minimum rows for SMA_20 and ADX
        raise ValueError("DataFrame must contain at least 20 rows for indicator calculations.")

    df_processed = df.copy()

    # Calculate technical indicators
    df_processed['returns'] = df_processed['close'].pct_change()
    df_processed['log_returns'] = np.log1p(df_processed['returns'])
    df_processed['price_volatility'] = df_processed['log_returns'].rolling(window=14).std() * (252 ** 0.5)
    df_processed['sma_20'] = df_processed['close'].rolling(window=20).mean()
    df_processed['atr'] = ta.atr(df_processed['high'], df_processed['low'], df_processed['close'], length=14)
    df_processed['vwap'] = ta.vwap(df_processed['high'], df_processed['low'], df_processed['close'], df_processed['volume'])
    df_processed['adx'] = ta.adx(df_processed['high'], df_processed['low'], df_processed['close'], length=20)['ADX_20']
    df_processed['momentum_rsi'] = ta.rsi(df_processed['close'], length=14)
    df_processed['trend_macd'] = ta.macd(df_processed['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    df_processed['ema_50'] = df_processed['close'].ewm(span=50).mean()
    
    # Calculate Bollinger Bands and dynamically assign columns
    bbands = ta.bbands(df_processed['close'], length=20, std=2)
    logging.info(f"Bollinger Bands columns: {bbands.columns.tolist()}")  # Debug column names
    if 'BBU_20_2.0' in bbands.columns:
        df_processed['bollinger_upper'] = bbands['BBU_20_2.0']
    elif 'BBU_20_2' in bbands.columns:
        df_processed['bollinger_upper'] = bbands['BBU_20_2']
    else:
        raise KeyError(f"Unexpected Bollinger Bands upper band column. Found: {bbands.columns.tolist()}")
    
    if 'BBL_20_2.0' in bbands.columns:
        df_processed['bollinger_lower'] = bbands['BBL_20_2.0']
    elif 'BBL_20_2' in bbands.columns:
        df_processed['bollinger_lower'] = bbands['BBL_20_2']
    else:
        raise KeyError(f"Unexpected Bollinger Bands lower band column. Found: {bbands.columns.tolist()}")

    df_processed['target'] = df_processed['close'].shift(-1)

    # Drop NaNs from rolling calculations
    df_processed.dropna(inplace=True)

    # Remove outliers
    columns_to_check = FEATURE_COLUMNS + ['target']
    df_processed = remove_outliers(df_processed, columns_to_check)

    # Ensure no all-zero features
    for feature in FEATURE_COLUMNS:
        if df_processed[feature].abs().sum() == 0:
            logging.warning(f"{feature} is all zeros; check data or calculation.")

    # Subset to required columns
    df_processed = df_processed[FEATURE_COLUMNS + ['target']]

    # Scaling with 17 features
    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
    if target_scaler is None:
        target_scaler = MinMaxScaler()

    df_features = df_processed[FEATURE_COLUMNS].values  # 17 features
    df_target = df_processed[['target']].values

    feature_scaler.fit(df_features)
    target_scaler.fit(df_target)

    df_processed[FEATURE_COLUMNS] = feature_scaler.transform(df_features)
    df_processed['target'] = target_scaler.transform(df_target).flatten()

    # Save scalers
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

    logging.info(f"Final DataFrame shape: {df_processed.shape}")
    logging.info(f"NaN check:\n{df_processed.isnull().sum()}")

    return df_processed

def split_data(df, train_ratio=0.8):
    """Split data into train and test sets."""
    X = df.drop(columns=['target'])
    y = df['target']
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test