import pandas as pd
import numpy as np

def generate_synthetic_data(rows=1000):
    np.random.seed(42)  # For reproducibility
    timestamps = pd.date_range(start="2024-01-01", periods=rows, freq="T")  # 1-minute intervals

    prices = np.cumsum(np.random.randn(rows) * 2) + 50000  # Simulated BTC price
    high = prices + np.random.rand(rows) * 10
    low = prices - np.random.rand(rows) * 10
    open_ = prices + np.random.randn(rows) * 2
    close = prices + np.random.randn(rows) * 2
    volume = np.random.rand(rows) * 5

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("src/data/synthetic_ohlcv.csv", index=False)
    print("Synthetic data saved to src/data/synthetic_ohlcv.csv")
