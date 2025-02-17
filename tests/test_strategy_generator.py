import pytest
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from strategy.strategy_generator import generate_signals

# Generate some sample data for testing
def generate_sample_data():
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'price': np.random.rand(100)
    }
    return pd.DataFrame(data)

def test_generate_signals():
    data = generate_sample_data()

    # Simulated model with random predictions
    class MockModel:
        def predict(self, X):
            batch_size = X.shape[0]
            return torch.tensor(np.random.rand(batch_size, 1), dtype=torch.float32)  

        def __call__(self, X):
            return self.predict(X)

        def eval(self):  
            pass  # Mock eval method

    model = MockModel()

    # Ensure that 'data' is a DataFrame with the correct feature columns
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=['feature1', 'feature2', 'price'])

    # Create and fit a mock scaler
    scaler = StandardScaler()
    scaler.fit(data[['feature1', 'feature2']])

    feature_columns = ['feature1', 'feature2']

    # Generate signals with the scaler
    signals = generate_signals(data, model, feature_columns, scaler=scaler)

    # Check that the generated signals have the correct length
    assert len(signals) == len(data) - 10  # Adjusted for time_steps
