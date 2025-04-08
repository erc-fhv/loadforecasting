import torch
import torch.nn as nn
import math

# Use a full pytorch transformer for timeseries prediction
#
class Transformer_Full(nn.Module):
    
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(Transformer_Full, self).__init__()
        self.isPytorchModel = True
        self.num_of_features = num_of_features
        self.forecast_horizon = 24
        
        # Finetune the XLSTM config variables
        if model_size == "1k":
            num_heads=2
            num_layers=1
            dim_feedforward=6
            d_model=6
        elif model_size == "2k":
            num_heads=2
            num_layers=1
            dim_feedforward=10
            d_model=10
        elif model_size == "5k":
            num_heads=2
            num_layers=1
            dim_feedforward=16
            d_model=14
        elif model_size == "10k":
            num_heads=4
            num_layers=1
            dim_feedforward=90
            d_model=20
        elif model_size == "20k":
            num_heads=4
            num_layers=1
            dim_feedforward=200
            d_model=20
        elif model_size == "40k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=20
        elif model_size == "80k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=40
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Project input features to transformer dimension
        self.input_projection = nn.Linear(num_of_features, d_model)        
        self.tgt_projection = nn.Linear(1, d_model)        
        self.positional_encoding = PositionalEncoding(d_model, timesteps=self.forecast_horizon)

        # Transformer Encoder-Decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        # Final output projection (to 1)
        self.output_layer = nn.Linear(d_model, 1)
    
    def forward(self, x):
        
        # Take the latest available lagged loads as target-input
        lagged_load_feature = 11
        tgt = x[:,:, lagged_load_feature].unsqueeze(-1)
        
        # Input and tgt projection
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        tgt = self.tgt_projection(tgt)
        tgt = self.positional_encoding(tgt)

        # Run the full transformer model
        out = self.transformer(x, tgt)
        out = self.output_layer(out)
        
        return out
    

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

