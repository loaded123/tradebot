import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
import joblib

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

    # Calculate returns and other features
    df['returns'] = df['close'].pct_change().fillna(0)
    df['log_returns'] = np.log1p(df['returns']).fillna(0)
    df['price_volatility'] = df['returns'].rolling(10).std().fillna(0)

    from ta.volatility import AverageTrueRange
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
    df['atr'] = atr_indicator.average_true_range()

    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    df['momentum_rsi'] = RSIIndicator(close=df['close']).rsi().fillna(0)
    macd = MACD(close=df['close'])
    df['trend_macd'] = macd.macd().fillna(0)

    df['sma_20'] = df['close'].rolling(window=20).mean().fillna(df['close'])  # Replace NaNs with close price

    df['volume'] = np.log1p(df['volume'])

    df['target'] = df['close'].shift(-1)
    df.dropna(inplace=True)

    # Debugging: Ensure no NaNs
    print("Final DataFrame NaN check:\n", df.isnull().sum())

    # Select only the features listed in FEATURE_COLUMNS
    df = df[FEATURE_COLUMNS + ['target']]

    # Apply feature scaling 
    df[FEATURE_COLUMNS] = feature_scaler.fit_transform(df[FEATURE_COLUMNS])

    # Apply target scaling
    df['target'] = target_scaler.fit_transform(df[['target']])

    # Save scalers
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

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