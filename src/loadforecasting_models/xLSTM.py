import torch
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
from loadforecasting_models.interfaces import (
    MachineLearningModelInitParams,
    MachineLearningModelTrainParams,
    MachineLearningPredictionParams, )
from loadforecasting_models.pytorch_helpers import PytorchHelper, PositionalEncoding


class xLSTM(torch.nn.Module):
    """xLSTM configuration as provided by the xLSTM authors."""

    def __init__(
        self,
        params: MachineLearningModelInitParams,
        ) -> None:
        
        super().__init__()
        
        # The following xLSTM config variables are overtaken from the xLSTM authors
        conv1d_kernel_size=4
        num_heads=4
        qkv_proj_blocksize=4
        proj_factor=1.3
        num_blocks=7
        slstm_at=[1]

        # Finetune the XLSTM config variables
        if params.model_size == "0.1k":
            num_blocks=1
            num_heads=1
            d_model=1
            slstm_at=[0]
        elif  params.model_size == "0.2k":
            num_blocks=1
            num_heads=1
            d_model=1
            slstm_at=[0]
        elif params.model_size == "0.5k":
            num_blocks=1
            num_heads=2
            d_model=2
            slstm_at=[0]
        elif params.model_size == "1k":
            num_blocks=1
            num_heads=2
            d_model=4
            slstm_at=[0]
        elif params.model_size == "2k":
            num_blocks=1
            num_heads=4
            d_model=8
            slstm_at=[0]
        elif params.model_size == "5k":
            num_blocks=2
            num_heads=4
            d_model=8
            slstm_at=[1]
        elif params.model_size == "10k":
            num_blocks=2
            num_heads=4
            d_model=16
            slstm_at=[1]
        elif params.model_size == "20k":
            num_blocks=2
            num_heads=4
            d_model=32
            slstm_at=[1]
        elif params.model_size == "40k":
            num_blocks=4
            num_heads=4
            d_model=32
            slstm_at=[1]
        elif params.model_size == "80k":
            num_blocks=4
            num_heads=8
            d_model=40
            slstm_at=[1]
        else:
            assert False, f"Unimplemented params.model_size parameter given: {params.model_size}"

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
        self.input_projection = torch.nn.Linear(params.num_of_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, timesteps=params.forecast_horizon)
        self.output_layer = torch.nn.Linear(d_model, 1)

    def train_model(
        self,
        params: MachineLearningModelTrainParams
        ) -> dict:
        """Train this model."""

        history = PytorchHelper.train(self, params)

        return history

    def forward(
        self,
        params: MachineLearningPredictionParams
        ) -> torch.Tensor:
        """Model forward pass."""

        x = self.input_projection(params.x.float())
        x = self.positional_encoding(x)
        x = self.xlstm_stack(x)
        x = self.output_layer(x)
        return x
