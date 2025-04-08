import torch
import torch.nn as nn
import math

# Encoder-only Transformer inspired by "A Time Series is  
# Worth 64 Words" (https://arxiv.org/abs/2211.14730)
#
class Transformer_Encoder_Only(nn.Module):
    
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(Transformer_Encoder_Only, self).__init__()
        self.isPytorchModel = True
        self.num_of_features = num_of_features
        self.forecast_horizon = 24
        
        # Fixed dense, for better comparison to other models
        hidden_dimension_dense1 = 30
        hidden_dimension_dense2 = 20
        
        # Finetune the XLSTM config variables
        if model_size == "1k":
            num_heads=2
            num_layers=1
            dim_feedforward=10
            d_model=10
        elif model_size == "2k":
            num_heads=2
            num_layers=1
            dim_feedforward=16
            d_model=14
        elif model_size == "5k":
            num_heads=4
            num_layers=1
            dim_feedforward=90
            d_model=20
        elif model_size == "10k":
            num_heads=4
            num_layers=1
            dim_feedforward=200
            d_model=20
        elif model_size == "20k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=20
        elif model_size == "40k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=40
        elif model_size == "80k":
            num_heads=8
            num_layers=2
            dim_feedforward=400
            d_model=40
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Transformer Encoder Layers
        self.input_projection = nn.Linear(num_of_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, timesteps=self.forecast_horizon)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(d_model, hidden_dimension_dense1)
        self.dense2 = nn.Linear(hidden_dimension_dense1, hidden_dimension_dense2)
        self.output_layer = nn.Linear(hidden_dimension_dense2, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)
        return x

# This implementation of positional encoding is based on the
# "Attention Is All You Need" paper, and is conceptually similar to:
# https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, timesteps=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(timesteps, d_model)  # [timesteps, d_model]
        position = torch.arange(0, timesteps, dtype=torch.float).unsqueeze(1)  # [timesteps, 1]
        _2i = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the array

        pe = pe.unsqueeze(0)  # [1, timesteps, d_model]
        self.register_buffer('pe', pe)  # Save as a non-learnable buffer

    def forward(self, x):
        
        batches, timesteps, features = x.shape
        assert (self.pe.size(1) == timesteps), f"Expected timesteps: {self.pe.size(1)}, received timesteps: {timesteps}"
        assert (self.pe.size(2) == features), f"Expected features: {self.pe.size(2)}, received features: {features}"
        
        x = x + self.pe # Add positional encoding
        
        return x
