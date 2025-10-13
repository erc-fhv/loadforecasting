import torch
from loadforecasting_models.interfaces import (
    MachineLearningModelInitParams,
    MachineLearningModelTrainParams,
    MachineLearningPredictionParams, )
from loadforecasting_models.pytorch_helpers import PytorchHelper

class LSTM(torch.nn.Module):
    def __init__(
        self,
        params: MachineLearningModelInitParams,
        ) -> None:
        
        super().__init__()
        
        # Define the LSTM size
        if params.model_size == "0.1k":
            bidirectional=False
            hidden_dimension_lstm1 = 1
            hidden_dimension_lstm2 = 1
            hidden_dimension_dense1 = 4
            hidden_dimension_dense2 = 4
        elif  params.model_size == "0.2k":
            bidirectional=True
            hidden_dimension_lstm1 = 1
            hidden_dimension_lstm2 = 1
            hidden_dimension_dense1 = 4
            hidden_dimension_dense2 = 4
        elif params.model_size == "0.5k":
            bidirectional=True
            hidden_dimension_lstm1 = 2
            hidden_dimension_lstm2 = 2
            hidden_dimension_dense1 = 5
            hidden_dimension_dense2 = 5
        elif params.model_size == "1k":
            bidirectional=True
            hidden_dimension_lstm1 = 3
            hidden_dimension_lstm2 = 3
            hidden_dimension_dense1 = 10
            hidden_dimension_dense2 = 10
        elif params.model_size == "2k":
            bidirectional=True
            hidden_dimension_lstm1 = 5
            hidden_dimension_lstm2 = 5
            hidden_dimension_dense1 = 15
            hidden_dimension_dense2 = 10
        elif params.model_size == "5k":
            bidirectional=True
            hidden_dimension_lstm1 = 8
            hidden_dimension_lstm2 = 9
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif params.model_size == "10k":
            bidirectional=True
            hidden_dimension_lstm1 = 10
            hidden_dimension_lstm2 = 18
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif params.model_size == "20k":
            bidirectional=True
            hidden_dimension_lstm1 = 22
            hidden_dimension_lstm2 = 20
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif params.model_size == "40k":
            bidirectional=True
            hidden_dimension_lstm1 = 42
            hidden_dimension_lstm2 = 20
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        elif params.model_size == "80k":
            bidirectional=True
            hidden_dimension_lstm1 = 70
            hidden_dimension_lstm2 = 21
            hidden_dimension_dense1 = 30
            hidden_dimension_dense2 = 20
        else:
            assert False, f"Unimplemented params.model_size parameter given: {params.model_size}"

        if bidirectional:
            bidirectional_factor = 2
        else:
            bidirectional_factor = 1

        self.lstm1 = torch.nn.LSTM(input_size=params.num_of_features, hidden_size=hidden_dimension_lstm1, batch_first=True, bidirectional=bidirectional)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_dimension_lstm1*bidirectional_factor, hidden_size=hidden_dimension_lstm2, batch_first=True, bidirectional=bidirectional)

        # Adding additional dense layers
        self.activation = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(hidden_dimension_lstm2*bidirectional_factor, hidden_dimension_dense1)
        self.dense2 = torch.nn.Linear(hidden_dimension_dense1, hidden_dimension_dense2)
        self.output_layer = torch.nn.Linear(hidden_dimension_dense2, 1)

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

        x, _ = self.lstm1(params.x.float())
        x, _ = self.lstm2(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)

        return x

    def get_nr_of_parameters(self):
        """
        Return the number of parameters of this model
        """

        total_params = sum(p.numel() for p in self.parameters())

        return total_params

