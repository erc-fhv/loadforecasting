import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, model_size, num_of_features, modelAdapter):
        super(LSTM, self).__init__()
        self.isPytorchModel = True
        self.forecast_horizon = 24
        
        # Fixed dense, for better comparison
        hidden_dimension_dense1 = 30
        hidden_dimension_dense2 = 20
        
        # Define the LSTM size
        if model_size == "1k":
            hidden_dimension_lstm1 = 1
            hidden_dimension_lstm2 = 1
        elif model_size == "2k":
            hidden_dimension_lstm1 = 4
            hidden_dimension_lstm2 = 4
        elif model_size == "5k":
            hidden_dimension_lstm1 = 8
            hidden_dimension_lstm2 = 9
        elif model_size == "10k":
            hidden_dimension_lstm1 = 10
            hidden_dimension_lstm2 = 18
        elif model_size == "20k":
            hidden_dimension_lstm1 = 22
            hidden_dimension_lstm2 = 20
        elif model_size == "40k":
            hidden_dimension_lstm1 = 42
            hidden_dimension_lstm2 = 20
        elif model_size == "80k":
            hidden_dimension_lstm1 = 70
            hidden_dimension_lstm2 = 21
        else:
            assert False, f"Unimplemented model_size parameter given: {model_size}"

        # Fixed dense, for better comparison to other models
        self.lstm1 = nn.LSTM(input_size=num_of_features, hidden_size=hidden_dimension_lstm1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dimension_lstm1*2, hidden_size=hidden_dimension_lstm2, batch_first=True, bidirectional=True)
        
        # Adding additional dense layers
        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(hidden_dimension_lstm2*2, hidden_dimension_dense1)
        self.dense2 = nn.Linear(hidden_dimension_dense1, hidden_dimension_dense2)
        self.output_layer = nn.Linear(hidden_dimension_dense2, 1)          
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.output_layer(x)
        return x
