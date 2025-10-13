import torch
from loadforecasting_models.interfaces import (
    MachineLearningModelInitParams,
    MachineLearningModelTrainParams,
    MachineLearningPredictionParams, )
from loadforecasting_models.pytorch_helpers import PytorchHelper, PositionalEncoding


class Transformer(torch.nn.Module):
    """
    Encoder-only Transformer inspired by "A Time Series is
    Worth 64 Words" (https://arxiv.org/abs/2211.14730)
    """

    def __init__(
        self,
        params: MachineLearningModelInitParams,
        ) -> None:

        super().__init__()

        # Finetune the config variables
        if params.model_size == "0.1k":
            num_layers=1
            num_heads=2
            dim_feedforward=5
            d_model = 2
        elif params.model_size == "0.2k":
            num_layers=1
            num_heads=2
            dim_feedforward=5
            d_model=4
        elif params.model_size == "0.5k":
            num_layers=1
            num_heads=2
            dim_feedforward=6
            d_model=6
        elif params.model_size == "1k":
            num_layers=1
            num_heads=2
            dim_feedforward=10
            d_model=10
        elif params.model_size == "2k":
            num_layers=1
            num_heads=2
            dim_feedforward=16
            d_model=14
        elif params.model_size == "5k":
            num_layers=1
            num_heads=4
            dim_feedforward=90
            d_model=20
        elif params.model_size == "10k":
            num_layers=1
            num_heads=4
            dim_feedforward=200
            d_model=20
        elif params.model_size == "20k":
            num_layers=1
            num_heads=4
            dim_feedforward=400
            d_model=20
        elif params.model_size == "40k":
            num_layers=1
            num_heads=4
            dim_feedforward=400
            d_model=40
        elif params.model_size == "80k":
            num_layers=2
            num_heads=8
            dim_feedforward=400
            d_model=40
        else:
            assert False, f"Unimplemented params.model_size parameter given: {params.model_size}"

        # Transformer Encoder Layers
        self.input_projection = torch.nn.Linear(params.num_of_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, timesteps=params.forecast_horizon)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
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

        x = self.input_projection(params.x.float())
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.output_layer(x)
        return x
