import torch
from loadforecasting_models.interfaces import (
    MachineLearningModelInitParams,
    MachineLearningModelTrainParams,
    MachineLearningPredictionParams, )
from loadforecasting_models.pytorch_helpers import PytorchHelper, PositionalEncoding


class Transformer_Full(torch.nn.Module):
    """Use a full pytorch transformer for timeseries prediction"""    
    
    def __init__(
        self,
        params: MachineLearningModelInitParams,
        ) -> None:
        
        super().__init__()
        
        # Finetune the XLSTM config variables
        if params.model_size == "1k":
            num_heads=2
            num_layers=1
            dim_feedforward=6
            d_model=6
        elif params.model_size == "2k":
            num_heads=2
            num_layers=1
            dim_feedforward=10
            d_model=10
        elif params.model_size == "5k":
            num_heads=2
            num_layers=1
            dim_feedforward=16
            d_model=14
        elif params.model_size == "10k":
            num_heads=4
            num_layers=1
            dim_feedforward=90
            d_model=20
        elif params.model_size == "20k":
            num_heads=4
            num_layers=1
            dim_feedforward=200
            d_model=20
        elif params.model_size == "40k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=20
        elif params.model_size == "80k":
            num_heads=4
            num_layers=1
            dim_feedforward=400
            d_model=40
        else:
            assert False, f"Unimplemented params.model_size parameter given: {params.model_size}"

        # Project input features to transformer dimension
        self.input_projection = torch.nn.Linear(params.num_of_features, d_model)
        self.tgt_projection = torch.nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, timesteps=self.forecast_horizon)

        # Transformer Encoder-Decoder
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        # Final output projection (to 1)
        self.output_layer = torch.nn.Linear(d_model, 1)

    def train_model(
        self,
        params: MachineLearningModelTrainParams
        ) -> dict:
        """
        Train this model.
        """

        history = PytorchHelper.train(self, params)

        return history

    def forward(
        self,
        params: MachineLearningPredictionParams
        ) -> torch.Tensor:
        """Model forward pass."""

        # Take the latest available lagged loads as target-input
        lagged_load_feature = 11
        x = params.x.float()
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
