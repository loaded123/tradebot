from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
import torch

config = TimeSeriesTransformerConfig(
    input_size=17,
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.1,
    prediction_length=1,
    context_length=10,
    num_time_features=1,
    lags_sequence=[1, 2, 3]
)
model = TimeSeriesTransformerModel(config)
past_values = torch.randn(1, 13, 17)  # Batch size 1, history 13, features 17
past_time_features = torch.randn(1, 13, 1)  # Batch size 1, history 13, time features 1
past_observed_mask = torch.ones(1, 13, 17)  # Batch size 1, history 13, features 17
outputs = model(past_values, past_time_features, past_observed_mask)
print(outputs)