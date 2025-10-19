"""
Wrapper class that provides a unified interface and additional utilities for a single prediction model.
"""

import importlib
import torch

# Import all used forecasting Models
#
from loadforecasting_models.Perfect import Perfect
from loadforecasting_models.interfaces import ModelParams, EvaluationParams
from loadforecasting_models.pytorch_helpers import PytorchHelper


class Model:
    """
    Wrapper class that provides a unified interface to different model types.
    Each Model instance encapsulates exactly one machine learning or heuristic model.
    """

    def __init__(
        self,
        params: ModelParams,
        ) -> None:
        """
        Initialize the Model.
        """

        # Import and instantiate the given model
        try:
            model = importlib.import_module(f"loadforecasting_models.{params.model_type}")
            my_model_class = getattr(model, params.model_type)
            self.my_model = my_model_class(params)

        except AttributeError as e:

            # No class with name model_type was found
            print(f"Unexpected 'model_type' parameter received: {params.model_type}")
            print(f"Detailed error description : {e}")

    def train_model(
        self,
        params: ModelParams
        ) -> dict:
        """
        Train the model on the given training data.

        Args:
            params (ModelParams): Training parameters for the given model type.

        Returns:
            dict: Key = Data-Set. Value = Traing losses per epoch.
        """

        history = self.my_model.train(params)

        return history

    def predict(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Predict y from the given x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_len, sequence_len, features) 
                containing the features for which predictions are to be made.

        Returns:
            torch.Tensor: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        if isinstance(self.my_model, torch.nn.Module):
            # Machine Learning Model
            self.my_model.eval()
            with torch.no_grad():
                y = self.my_model(x)

        else:
            # Simple models
            y = self.my_model.forward(x)

        return y

    def evaluate(
        self,
        params: EvaluationParams,
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        if self.my_model.isPytorchModel == False:   # Simple, parameter free models

            # Predict
            if isinstance(self.my_model, Perfect):
                # The Perfect prediction model just gets and returns 
                # the perfect profile (used for reference).
                output = self.predict(Y_test)
            else:
                output = self.predict(X_test)
            assert output.shape == Y_test.shape, \
                f"Shape mismatch: got {output.shape}, expected {Y_test.shape})"

            # Unnormalize the target variable, if wished.
            if deNormalize == True:
                assert self.model_adapter != None, "No model_adapter given."
                Y_test = self.model_adapter.deNormalizeY(Y_test)
                output = self.model_adapter.deNormalizeY(output)

            # Compute Loss
            loss = self.loss_fn(output, Y_test)
            results['test_loss'] = [loss.item()]
            metric = PytorchHelper.smape(output, Y_test)
            results['test_sMAPE'] = [metric]
            reference = float(torch.mean(Y_test))
            results['test_loss_relative'] = [100.0*loss.item()/reference]            
            results['predicted_profile'] = output

        else:   # Pytorch models
            results = PytorchHelper.evaluate()
        
        return results
    
    
