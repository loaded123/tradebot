# src/data/data_preprocessor.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Define the 17 features explicitly for consistency
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'volume', 'returns', 'log_returns', 
    'price_volatility', 'sma_20', 'atr', 'vwap', 'adx', 
    'momentum_rsi', 'trend_macd', 'ema_50', 'bollinger_upper', 
    'bollinger_lower', 'bollinger_middle'
]

feature_columns = FEATURE_COLUMNS

def remove_outliers(df, columns):
    """Remove outliers using IQR method, preserving unscaled BTC prices."""
    df_cleaned = df.copy()
    for column in columns:
        if column in ['open', 'high', 'low', 'close']:  # Skip price columns for outlier removal to preserve real values
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned

def calculate_adx(high, low, close, window=14):
    """Calculate ADX manually to ensure compatibility with unscaled prices, handling NaNs."""
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    
    # Handle NaN values in input data
    df = df.ffill().bfill()  # Forward-fill then back-fill to minimize NaNs (using modern pandas methods)
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    
    # Handle NaNs in intermediate calculations
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).fillna(0)
    dm_plus = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    dm_minus = np.where(df['low'] < df['low'].shift(1), df['low'].shift(1) - df['low'], 0)
    
    # Use rolling sums with min_periods=1 to handle initial NaNs
    tr_sum = tr.rolling(window=window, min_periods=1).sum().fillna(0)
    dm_plus_sum = pd.Series(dm_plus).rolling(window=window, min_periods=1).sum().fillna(0)
    dm_minus_sum = pd.Series(dm_minus).rolling(window=window, min_periods=1).sum().fillna(0)
    
    # Avoid division by zero
    di_plus = 100 * (dm_plus_sum / (tr_sum + 1e-10))  # Add small epsilon to prevent division by zero
    di_minus = 100 * (dm_minus_sum / (tr_sum + 1e-10))
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = dx.rolling(window=window, min_periods=1).mean().fillna(0)  # Fill remaining NaNs with 0
    return adx

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP manually to ensure compatibility with unscaled prices."""
    typical_price = (high + low + close) / 3
    cumulative_price_volume = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()
    vwap = cumulative_price_volume / (cumulative_volume + 1e-10)  # Add small epsilon to prevent division by zero
    return vwap.fillna(0)  # Fill NaNs with 0

def preprocess_data(df: pd.DataFrame, feature_scaler=None, target_scaler=None) -> pd.DataFrame:
    """Preprocess historical OHLCV data for model training or prediction, preserving unscaled BTC prices and handling NaNs."""
    try:
        logging.info(f"Initial data shape: {df.shape}")
        logging.info(f"Initial columns: {df.columns.tolist()}")

        if len(df) < 20:
            raise ValueError("DataFrame must contain at least 20 rows for indicator calculations.")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        logging.info("Required columns check passed")

        if not isinstance(df['close'], pd.Series):
            raise TypeError(f"df['close'] is not a Series, got {type(df['close'])}")

        df_processed = df.copy()

        # Ensure prices are unscaled BTC/USD values (e.g., $50,000+)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if df_processed[col].min() < 10000 or df_processed[col].max() > 200000:
                logging.warning(f"Price column {col} appears scaled or invalid: min={df_processed[col].min()}, max={df_processed[col].max()}")
                if df_processed[col].max() > 200000:
                    df_processed[col] = df_processed[col] / 1000  # Adjust if scaled by 1000
                elif df_processed[col].max() < 10000:
                    df_processed[col] = df_processed[col] * 1000  # Adjust if scaled down
            logging.info(f"Adjusted {col} range: min={df_processed[col].min()}, max={df_processed[col].max()}")

        # Calculate financial and technical indicators with unscaled prices, filling NaNs
        if 'returns' not in df_processed.columns:
            df_processed['returns'] = df_processed['close'].pct_change().fillna(0)  # Fill NaN with 0
            logging.info(f"Returns calculated, shape: {df_processed.shape}, NaNs: {df_processed['returns'].isna().sum()}")

        if 'log_returns' not in df_processed.columns:
            df_processed['log_returns'] = np.log1p(df_processed['returns']).fillna(0)  # Fill NaN with 0
            logging.info(f"Log returns calculated, shape: {df_processed.shape}, NaNs: {df_processed['log_returns'].isna().sum()}")

        if 'price_volatility' not in df_processed.columns:
            df_processed['price_volatility'] = df_processed['log_returns'].rolling(window=14, min_periods=1).std() * np.sqrt(252)
            df_processed['price_volatility'] = df_processed['price_volatility'].fillna(df_processed['price_volatility'].mean())  # Fill NaN with mean
            logging.info(f"Price volatility calculated, shape: {df_processed.shape}, NaNs: {df_processed['price_volatility'].isna().sum()}")

        if 'sma_20' not in df_processed.columns:
            df_processed['sma_20'] = df_processed['close'].rolling(window=20, min_periods=1).mean().fillna(df_processed['close'].mean())  # Fill NaN with mean
            logging.info(f"SMA_20 calculated, shape: {df_processed.shape}, NaNs: {df_processed['sma_20'].isna().sum()}")

        if 'atr' not in df_processed.columns:
            df_processed['atr'] = ta.atr(df_processed['high'], df_processed['low'], df_processed['close'], length=14).fillna(500.0)  # Default to typical BTC ATR (~$500)
            logging.info(f"ATR calculated, shape: {df_processed.shape}, NaNs: {df_processed['atr'].isna().sum()}")

        if 'vwap' not in df_processed.columns:
            df_processed['vwap'] = calculate_vwap(df_processed['high'], df_processed['low'], df_processed['close'], df_processed['volume'])
            df_processed['vwap'] = df_processed['vwap'].fillna(df_processed['vwap'].mean())  # Fill NaN with mean
            logging.info(f"VWAP calculated, shape: {df_processed.shape}, NaNs: {df_processed['vwap'].isna().sum()}")

        if 'adx' not in df_processed.columns:
            df_processed['adx'] = calculate_adx(df_processed['high'], df_processed['low'], df_processed['close'], window=14)
            df_processed['adx'] = df_processed['adx'].fillna(0)  # Fill NaN with 0 (neutral ADX)
            logging.info(f"ADX calculated, shape: {df_processed.shape}, NaNs: {df_processed['adx'].isna().sum()}")

        if 'momentum_rsi' not in df_processed.columns:
            df_processed['momentum_rsi'] = ta.rsi(df_processed['close'], length=14).fillna(50.0)  # Default to neutral RSI (50)
            logging.info(f"RSI calculated, shape: {df_processed.shape}, NaNs: {df_processed['momentum_rsi'].isna().sum()}")

        if 'trend_macd' not in df_processed.columns:
            macd_result = ta.macd(df_processed['close'], fast=12, slow=26, signal=9)
            df_processed['trend_macd'] = macd_result['MACD_12_26_9'].fillna(0)  # Fill NaN with 0
            logging.info(f"MACD calculated, shape: {df_processed.shape}, NaNs: {df_processed['trend_macd'].isna().sum()}")

        if 'ema_50' not in df_processed.columns:
            df_processed['ema_50'] = df_processed['close'].ewm(span=50, adjust=False).mean().fillna(df_processed['close'].mean())  # Fill NaN with mean
            logging.info(f"EMA_50 calculated, shape: {df_processed.shape}, NaNs: {df_processed['ema_50'].isna().sum()}")

        if 'bollinger_upper' not in df_processed.columns or 'bollinger_lower' not in df_processed.columns or 'bollinger_middle' not in df_processed.columns:
            bbands = ta.bbands(df_processed['close'], length=20, std=2)
            logging.info(f"Bollinger Bands columns: {bbands.columns.tolist()}")
            if 'BBU_20_2.0' in bbands.columns:
                df_processed['bollinger_upper'] = bbands['BBU_20_2.0'].fillna(df_processed['close'].mean() + 2 * df_processed['close'].std())  # Default to mean + 2*std
            elif 'BBU_20_2' in bbands.columns:
                df_processed['bollinger_upper'] = bbands['BBU_20_2'].fillna(df_processed['close'].mean() + 2 * df_processed['close'].std())
            else:
                raise KeyError(f"Unexpected Bollinger Bands upper band column. Found: {bbands.columns.tolist()}")
            if 'BBL_20_2.0' in bbands.columns:
                df_processed['bollinger_lower'] = bbands['BBL_20_2.0'].fillna(df_processed['close'].mean() - 2 * df_processed['close'].std())  # Default to mean - 2*std
            elif 'BBL_20_2' in bbands.columns:
                df_processed['bollinger_lower'] = bbands['BBL_20_2'].fillna(df_processed['close'].mean() - 2 * df_processed['close'].std())
            else:
                raise KeyError(f"Unexpected Bollinger Bands lower band column. Found: {bbands.columns.tolist()}")
            if 'BBM_20_2.0' in bbands.columns:
                df_processed['bollinger_middle'] = bbands['BBM_20_2.0'].fillna(df_processed['close'].mean())  # Default to mean
            elif 'BBM_20_2' in bbands.columns:
                df_processed['bollinger_middle'] = bbands['BBM_20_2'].fillna(df_processed['close'].mean())
            else:
                raise KeyError(f"Unexpected Bollinger Bands middle band column. Found: {bbands.columns.tolist()}")
            logging.info(f"Bollinger Bands calculated, shape: {df_processed.shape}, NaNs upper: {df_processed['bollinger_upper'].isna().sum()}, lower: {df_processed['bollinger_lower'].isna().sum()}")

        logging.info(f"df['close'] type before target: {type(df_processed['close'])}")
        # Preserve unscaled 'close' as target, shifting for next period
        df_processed['target'] = df_processed['close'].shift(-1).ffill()  # Forward-fill target NaNs (using modern pandas methods)
        logging.info(f"Target added, shape: {df_processed.shape}, NaNs: {df_processed['target'].isna().sum()}")

        # Drop rows only if all price-related columns are NaN, retaining rows with valid prices
        logging.info(f"Shape before selective dropna: {df_processed.shape}")
        price_columns = ['open', 'high', 'low', 'close']
        mask = df_processed[price_columns].isna().all(axis=1)  # Drop rows only if all price columns are NaN
        df_processed = df_processed[~mask]
        logging.info(f"Shape after selective dropna: {df_processed.shape}")
        if df_processed.empty:
            raise ValueError("DataFrame is empty after selective dropping of NaN price rows")

        # Only scale technical indicators and target, keeping price columns unscaled
        price_columns = ['open', 'high', 'low', 'close']
        indicator_columns = [col for col in feature_columns if col not in price_columns]  # 13 indicators + volume = 14
        target_column = ['target']

        logging.info(f"Scaling indicators: {indicator_columns} (count: {len(indicator_columns)}), keeping prices unscaled: {price_columns}")
        
        if feature_scaler is None:
            feature_scaler = MinMaxScaler()
        if target_scaler is None:
            target_scaler = MinMaxScaler()

        # Scale only indicators (14 features, including volume), not prices, and ensure 17 total features
        df_features_indicators = df_processed[indicator_columns].values
        df_target = df_processed[target_column].values

        feature_scaler.fit(df_features_indicators)
        target_scaler.fit(df_target)

        df_processed[indicator_columns] = feature_scaler.transform(df_features_indicators)
        df_processed[target_column] = target_scaler.transform(df_target)

        # Ensure price columns (4 features) remain unscaled
        df_processed[price_columns] = df_processed[price_columns]

        # Save the scaler for backtesting, logging the number of features
        joblib.dump(feature_scaler, 'feature_scaler.pkl')
        joblib.dump(target_scaler, 'target_scaler.pkl')

        logging.info(f"Final DataFrame shape: {df_processed.shape}")
        logging.info(f"NaN check:\n{df_processed.isnull().sum()}")
        logging.info(f"Price range after processing: min_close={df_processed['close'].min()}, max_close={df_processed['close'].max()}")
        logging.info(f"Feature count after scaling: {len(feature_columns)} (should be 17)")

        # Subset to required columns, including unscaled prices, scaled indicators, and 'close' for backtesting
        df_processed = df_processed[feature_columns + ['target', 'close']]
        logging.info(f"Shape after subsetting: {df_processed.shape}")
        logging.info(f"Feature columns for training: {feature_columns}, including 'close' for backtesting: {df_processed.columns.tolist()}")

        return df_processed

    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()

def split_data(df, train_ratio=0.8):
    """Split data into train and test sets, preserving unscaled prices and scaled indicators, including 'close' for backtesting."""
    X = df.drop(columns=['target', 'close'])  # Keep 'close' separate for backtesting
    y = df['target']
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    close_train, close_test = df['close'][:train_size], df['close'][train_size:]
    return X_train, X_test, y_train, y_test, close_train, close_test

if __name__ == "__main__":
    # Use realistic BTC/USD dummy data for testing
    dummy_data = pd.DataFrame({
        'open': np.linspace(80000, 81000, 100),  # Unscaled BTC prices (~$80,000)
        'high': np.linspace(80100, 81100, 100),
        'low': np.linspace(79900, 80900, 100),
        'close': np.linspace(80000, 81000, 100),
        'volume': [1000] * 100
    }, index=pd.date_range("2025-02-28", periods=100, freq="H"))
    preprocessed = preprocess_data(dummy_data)
    print(f"Preprocessed data:\n{preprocessed.head()}")
    print(f"Feature columns: {feature_columns}, including 'close': {preprocessed.columns.tolist()}")
    print(f"Price range: min_close={preprocessed['close'].min()}, max_close={preprocessed['close'].max()}")