
import torch
import torch.nn as nn
import math

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class xLSTM(nn.Module):
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(xLSTM, self).__init__()
        self.isPytorchModel = True
        self.forecast_horizon = 24
        
        # The following xLSTM config variables are as as provided by the xLSTM authors:
        conv1d_kernel_size=4
        num_heads=4
        qkv_proj_blocksize=4
        proj_factor=1.3
        num_blocks=7
        slstm_at=[1]
        
        # Finetune the XLSTM config variables
        if model_size == "0.1k":
            num_blocks=1
            num_heads=1
            d_model=1
            slstm_at=[0]
        elif  model_size == "0.2k":
            num_blocks=1
            num_heads=1
            d_model=1
            slstm_at=[0]
        elif model_size == "0.5k":
            num_blocks=1
            num_heads=2
            d_model=2
            slstm_at=[0]
        elif model_size == "1k":
            num_blocks=1
            num_heads=2
            d_model=4
            slstm_at=[0]
        elif model_size == "2k":
            num_blocks=1
            num_heads=4
            d_model=8
            slstm_at=[0]
        elif model_size == "5k":
            num_blocks=2
            num_heads=4
            d_model=8
            slstm_at=[1]
        elif model_size == "10k":
            num_blocks=2
            num_heads=4
            d_model=16
            slstm_at=[1]
        elif model_size == "20k":
            num_blocks=2
            num_heads=4
            d_model=32
            slstm_at=[1]
        elif model_size == "40k":
            num_blocks=4
            num_heads=4
            d_model=32
            slstm_at=[1]
        elif model_size == "80k":
            num_blocks=4
            num_heads=8
            d_model=40
            slstm_at=[1]
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"
        
        # Configuration for the xLSTM Block
        self.cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, qkv_proj_blocksize=qkv_proj_blocksize, num_heads=num_heads
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",  # For now run at CPU. Changed from "cuda".
                    num_heads=num_heads,
                    conv1d_kernel_size=conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=proj_factor, act_fn="gelu"),
            ),
            context_length=256,
            num_blocks=num_blocks,
            embedding_dim=d_model,
            slstm_at=slstm_at,
        )
        self.xlstm_stack = xLSTMBlockStack(self.cfg)

        # Adding none-xlstm layers
        self.input_projection = nn.Linear(num_of_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, timesteps=self.forecast_horizon)
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.xlstm_stack(x)
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
