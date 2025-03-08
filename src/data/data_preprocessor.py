# src/data/data_preprocessor.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from src.constants import FEATURE_COLUMNS
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

FEATURE_COLUMNS = [
    'open', 'high', 'low', 'volume', 'returns', 'log_returns',
    'price_volatility', 'sma_20', 'atr', 'vwap', 'adx',
    'momentum_rsi', 'trend_macd', 'ema_50', 'bollinger_upper',
    'bollinger_lower', 'bollinger_middle'
]

def remove_outliers(df, columns):
    """Remove outliers using IQR method, preserving unscaled BTC prices."""
    df_cleaned = df.copy()
    for column in columns:
        if column in ['open', 'high', 'low', 'close']:
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[column] >= lower_bound) & (df_cleaned[column] <= upper_bound)]
    return df_cleaned

def calculate_adx(high, low, close, window=14):
    """Calculate ADX manually."""
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    df = df.ffill().bfill().fillna({'high': 79367.5, 'low': 78186.98, 'close': 78877.88})
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).fillna(500.0)
    dm_plus = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    dm_plus = pd.Series(dm_plus).fillna(0)
    dm_minus = np.where(df['low'] < df['low'].shift(1), df['low'].shift(1) - df['low'], 0)
    dm_minus = pd.Series(dm_minus).fillna(0)
    tr_sum = tr.rolling(window=window, min_periods=1).sum().fillna(500.0)
    dm_plus_sum = dm_plus.rolling(window=window, min_periods=1).sum().fillna(0)
    dm_minus_sum = dm_minus.rolling(window=window, min_periods=1).sum().fillna(0)
    di_plus = 100 * (dm_plus_sum / (tr_sum + 1e-10))
    di_minus = 100 * (dm_minus_sum / (tr_sum + 1e-10))
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = dx.rolling(window=window, min_periods=1).mean().fillna(25.0)
    return adx

def calculate_vwap(high, low, close, volume):
    """Calculate VWAP manually."""
    typical_price = (high + low + close) / 3
    cumulative_price_volume = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()
    vwap = cumulative_price_volume / (cumulative_volume + 1e-10)
    return vwap.fillna(78877.88)

def calculate_atr(high, low, close, period=14):
    """Calculate ATR manually."""
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).fillna(500.0)
    atr = true_range.rolling(window=period, min_periods=14).mean().fillna(500.0)  # Use min_periods=14
    return atr

def preprocess_data(df: pd.DataFrame, feature_scaler=None, target_scaler=None) -> pd.DataFrame:
    """Preprocess historical OHLCV data."""
    try:
        logging.info(f"Initial data shape: {df.shape}")
        if len(df) < 20:
            raise ValueError("DataFrame must contain at least 20 rows.")
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df_processed = df.copy()
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed.index = pd.to_datetime(df_processed.index, utc=True).tz_localize(None)
        expected_start = pd.to_datetime('2025-01-01 00:00:00', utc=True).tz_localize(None)
        expected_end = pd.to_datetime('2025-03-01 23:00:00', utc=True).tz_localize(None)
        full_range = pd.date_range(start=expected_start, end=expected_end, freq='h')
        df_processed = df_processed.reindex(full_range, method='ffill').fillna({
            'open': 78877.88, 'high': 79367.5, 'low': 78186.98, 'close': 78877.88, 'volume': 1000.0
        })

        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if df_processed[col].min() < 10000 or df_processed[col].max() > 200000:
                if df_processed[col].max() > 200000:
                    df_processed[col] /= 1000
                elif df_processed[col].min() < 10000:
                    df_processed[col] *= 1000
            df_processed[col] = df_processed[col].fillna(78877.88)

        df_processed['returns'] = df_processed['close'].pct_change().fillna(0)
        df_processed['log_returns'] = np.log1p(df_processed['returns']).fillna(0)
        df_processed['price_volatility'] = df_processed['log_returns'].rolling(window=14, min_periods=1).std() * np.sqrt(252)
        df_processed['price_volatility'] = df_processed['price_volatility'].fillna(df_processed['price_volatility'].mean())
        df_processed['sma_20'] = df_processed['close'].rolling(window=20, min_periods=1).mean().fillna(df_processed['close'].mean())
        df_processed['atr'] = calculate_atr(df_processed['high'], df_processed['low'], df_processed['close']).fillna(500.0)
        df_processed['vwap'] = calculate_vwap(df_processed['high'], df_processed['low'], df_processed['close'], df_processed['volume'])
        df_processed['vwap'] = df_processed['vwap'].fillna(78877.88)
        df_processed['adx'] = calculate_adx(df_processed['high'], df_processed['low'], df_processed['close']).fillna(25.0)
        df_processed['momentum_rsi'] = ta.rsi(df_processed['close'], length=14).fillna(50.0)
        macd_result = ta.macd(df_processed['close'], fast=12, slow=26, signal=9)
        df_processed['trend_macd'] = macd_result['MACD_12_26_9'].fillna(0)
        df_processed['ema_50'] = df_processed['close'].ewm(span=50, adjust=False).mean().fillna(df_processed['close'].mean())
        bbands = ta.bbands(df_processed['close'], length=20, std=2)
        df_processed['bollinger_upper'] = bbands['BBU_20_2.0'].fillna(df_processed['close'].mean() + 2 * df_processed['close'].std())
        df_processed['bollinger_lower'] = bbands['BBL_20_2.0'].fillna(df_processed['close'].mean() - 2 * df_processed['close'].std())
        df_processed['bollinger_middle'] = bbands['BBM_20_2.0'].fillna(df_processed['close'].mean())

        df_processed['target'] = df_processed['close'].shift(-1).ffill().fillna(78877.88)
        price_columns = ['open', 'high', 'low', 'close']
        mask = df_processed[price_columns].isna().all(axis=1)
        df_processed = df_processed[~mask]

        # Ensure all feature columns are present
        for col in FEATURE_COLUMNS:
            if col not in df_processed.columns:
                df_processed[col] = np.nan  # Add missing columns as NaN

        # Fill NaN values for indicators
        indicator_columns = [col for col in FEATURE_COLUMNS if col not in price_columns]
        df_processed[indicator_columns] = df_processed[indicator_columns].fillna(df_processed[indicator_columns].mean())

        # Scale indicators and target
        if feature_scaler is None:
            feature_scaler = MinMaxScaler()
        if target_scaler is None:
            target_scaler = MinMaxScaler()

        scaled_indicators = pd.DataFrame(
            feature_scaler.fit_transform(df_processed[indicator_columns]),
            index=df_processed.index,
            columns=indicator_columns
        )
        scaled_target = pd.DataFrame(
            target_scaler.fit_transform(df_processed[['target']]),
            index=df_processed.index,
            columns=['target']
        )

        # Concatenate price columns, scaled indicators, and scaled target
        df_processed = pd.concat([df_processed[price_columns], scaled_indicators, scaled_target], axis=1)

        # Ensure 'close' column is preserved
        if 'close' not in df_processed.columns:
            raise ValueError("'close' column missing after preprocessing")

        # Select only the required columns
        final_columns = [col for col in FEATURE_COLUMNS + ['target', 'close'] if col in df_processed.columns]
        df_processed = df_processed[final_columns]

        joblib.dump(feature_scaler, 'feature_scaler.pkl')
        joblib.dump(target_scaler, 'target_scaler.pkl')

        logging.info(f"Final DataFrame shape: {df_processed.shape}")
        return df_processed

    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return pd.DataFrame()

def split_data(df, train_ratio=0.8):
    """Split data into train and test sets."""
    X = df.drop(columns=['target', 'close'])
    y = df['target']
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    close_train, close_test = df['close'][:train_size], df['close'][train_size:]
    return X_train, X_test, y_train, y_test, close_train, close_test

if __name__ == "__main__":
    dummy_data = pd.DataFrame({
        'open': np.linspace(80000, 81000, 1440),
        'high': np.linspace(80100, 81100, 1440),
        'low': np.linspace(79900, 80900, 1440),
        'close': np.linspace(80000, 81000, 1440),
        'volume': [1000] * 1440
    }, index=pd.date_range("2025-01-01", periods=1440, freq="h"))
    preprocessed = preprocess_data(dummy_data)
    print(f"Preprocessed data:\n{preprocessed.head()}")