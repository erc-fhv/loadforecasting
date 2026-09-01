from typing import Callable, Union
import numpy as np
import torch
from .helpers import SklearnOptunaHelper
from .normalizer import Normalizer
from pygam.terms import TermList
from pygam import LinearGAM

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class Gam():
    """
    GAM (Generalized Additive Model) model for timeseries prediction.
    """

    def __init__(
        self,
        all_gam_terms:TermList,
        normalizer: Union[Normalizer, None] = None,
        lam:float = 0.5,
        fit_intercept:bool = True,
        loss_relative_to: str = "",
        ) -> None:
        """
        Args:
            all_gam_terms (TermList): List of GAM terms to be used in the model.
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
            lam (float): Regularization parameter for the GAM model.
            fit_intercept (bool): Whether to fit an intercept term in the GAM model.
            loss_relative_to (str): Reference for relative loss calculation. Default: "".
        """

        self.normalizer = normalizer
        self.loss_relative_to = loss_relative_to
        self.all_gam_terms = all_gam_terms
        self.x_train = torch.Tensor([])
        self.y_train = torch.Tensor([])

        self.gam = LinearGAM(
            all_gam_terms,
            lam=lam,
            fit_intercept=fit_intercept,
            )

    def predict(
        self, x: ArrayLike,
        ) -> ArrayLike:
        """
        Given an input x, return the predicted y.
        Args:
            x: Input features of shape (batch_len, sequence_len, features).
        Returns:
            ArrayLike: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        # Convert numpy to torch if needed
        input_was_numpy = isinstance(x, np.ndarray)
        if input_was_numpy:
            x_tensor  = torch.from_numpy(x).float()
        else:
            x_tensor  = x.float()

        # Unexpected number of dimensions for x (should be either 2 or 3)
        if x_tensor.ndim == 3:
            # Convert from (batch_len, seq, features) to (batch_len * seq, features)
            output_shape = (x_tensor.shape[0], x_tensor.shape[1], 1)
            x_tensor = x_tensor.reshape(-1, x_tensor.shape[2])
        elif x_tensor.ndim == 2:
            output_shape = (x_tensor.shape[0],)
        else:
            raise ValueError(f"Unexpected number of dimensions for x: {x_tensor.ndim}")

        # Prediction on new data
        y_pred = self.gam.predict(x_tensor)

        # Reshape back if needed
        y_pred = y_pred.reshape(output_shape)

        # Convert back to torch if needed
        if input_was_numpy is False:
            y_pred = torch.from_numpy(y_pred).float()

        return y_pred

    def train_model(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        ) -> dict:
        """
        Train this model.
        Args:
            X_train (torch.Tensor): Training input features of shape (batch_len, sequence_len,
                features).
            Y_train (torch.Tensor): Training labels of shape (batch_len, sequence_len, 1).
        Returns:
            dict: Training history containing loss values.
        """

        # Convert numpy to torch if needed
        if isinstance(x_train, np.ndarray):
            x_train  = torch.from_numpy(x_train).float()
        if isinstance(y_train, np.ndarray):
            y_train  = torch.from_numpy(y_train.copy()).float()

        # Store training features
        if x_train.ndim == 3:
            # Convert from (batch_len, seq, features) to (batch_len * seq, features)
            self.x_train = x_train.reshape(-1, x_train.shape[2])
        elif x_train.ndim == 2:
            self.x_train = x_train
        else:
            raise ValueError(f"Unexpected number of dimensions for x_train: {x_train.ndim}")

        # Store training target
        if y_train.ndim in [2, 3]:
            # Convert from (batch_len, seq, 1) to (batch_len * seq)
            self.y_train = y_train.flatten()
        elif y_train.ndim == 1:
            self.y_train = y_train
        else:
            raise ValueError(f"Unexpected number of dimensions for y_train: {y_train.ndim}")

        # Fit the GAM model
        self.gam.fit(self.x_train, self.y_train)

        # Evaluate the training loss
        history = {}
        history['loss'] = self.evaluate(x_train, y_train)['test_loss']

        return history

    def train_model_auto(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        term_candidates: Union[list, None] = None,
        param_resolver: Union[callable, None] = None,
        n_trials: int = 50,
        k_folds: int = 3,
        storage_path: Union[str, None] = None,
        study_name: Union[str, None] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Tune this model's hyperparameters (lam) with Optuna and TimeSeriesSplit
        cross-validation, then refit on the full training data with the best
        settings found.

        Note: unlike the other scikit-learn-style models, Gam does not support
        feature-group search, since each term in all_gam_terms is already bound to
        specific feature indices when the model is constructed. Instead, if
        term_candidates is given (a list of whole TermList objects, e.g. with/without
        an interaction term, a different lag day feeding the trend term, or
        different n_splines choices), Optuna additionally picks which candidate term
        set to use, letting you compare structurally different models without
        hand-picking one. (n_splines can't be tuned as a separate global dimension
        the way lam can - pyGAM only accepts it per-term at construction time - so
        it has to be baked into the term_candidates themselves.)

        Args:
            x_train: Input features of shape (batch_len, sequence_len, features).
            y_train: Target values of shape (batch_len, sequence_len, 1).
            term_candidates: Optional list of TermList objects to choose from. If
                None, only self.all_gam_terms is used (as before).
            param_resolver: Optional callable(dict) -> dict that resolves a raw
                suggested/stored params dict (containing 'term_set_index') into
                actual constructor kwargs (containing 'all_gam_terms'). If not
                given but term_candidates is, a resolver mapping term_set_index
                into term_candidates[term_set_index] is built automatically. Pass
                your own if you need this resolution to also happen outside of
                tuning (e.g. when reusing a stored term_set_index).
            n_trials (int): Number of Optuna trials for hyperparameter search.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation.
            storage_path: Optional sqlite file path for the Optuna study storage.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.

        Returns:
            dict: Training history and best hyperparameters. If term_candidates was
            given, best_params['term_set_index'] names the winning candidate.
        """

        if param_resolver is None and term_candidates:
            def param_resolver(params: dict) -> dict:
                resolved = dict(params)
                term_set_index = resolved.pop('term_set_index', None)
                resolved['all_gam_terms'] = term_candidates[term_set_index] \
                    if term_set_index is not None else self.all_gam_terms
                return resolved

        tuner = SklearnOptunaHelper(self)
        return tuner.train_auto(
            x_train=x_train,
            y_train=y_train,
            n_trials=n_trials,
            k_folds=k_folds,
            suggest_params_kwargs={'term_candidates': term_candidates},
            param_resolver=param_resolver,
            storage_path=storage_path,
            study_name=study_name,
            verbose=verbose,
            )

    @staticmethod
    def suggest_params(trial, term_candidates=None) -> dict:
        """Optuna search space for this model's hyperparameters."""
        params = {
            'lam': trial.suggest_float('lam', 1e-3, 1e3, log=True),
        }
        if term_candidates:
            params['term_set_index'] = trial.suggest_int(
                'term_set_index', 0, len(term_candidates) - 1)
        return params

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Union[dict, None] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        loss_relative_to: str = "",
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.
        """

        if results is None:
            results = {}

        # Convert numpy to torch if needed
        if isinstance(x_test, np.ndarray):
            x_tensor  = torch.from_numpy(x_test).float()
        else:
            x_tensor  = x_test.float()
        if isinstance(y_test, np.ndarray):
            y_tensor  = torch.from_numpy(y_test).float()
        else:
            y_tensor  = y_test.float()

        # Get model output
        output = self.predict(x_tensor)

        assert output.shape == y_tensor.shape, \
            f"Shape mismatch: got {output.shape}, expected {y_tensor.shape})"

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.normalizer is not None, "No normalizer given."
            y_tensor = self.normalizer.de_normalize_y(y_tensor)
            output = self.normalizer.de_normalize_y(output)
            assert isinstance(y_tensor, torch.Tensor), "Denormalized y_tensor is not a torch.Tensor"

        # Set reference for relative loss if 'loss_relative_to' is an empty string.
        if loss_relative_to == "":
            if self.loss_relative_to != "":
                loss_relative_to = self.loss_relative_to
            else:
                loss_relative_to = "mean"

        # Compute Loss
        if loss_relative_to == "mean":
            reference = float(torch.abs(torch.mean(y_tensor)))
        elif loss_relative_to == "max":
            reference = float(torch.abs(torch.max(y_tensor)))
        elif loss_relative_to == "range":
            reference = float(torch.max(y_tensor) - torch.min(y_tensor))
        else:
            raise ValueError(f"Unexpected parameter: loss_relative_to = {loss_relative_to}")

        assert isinstance(y_tensor, torch.Tensor), "Model target is not a torch.Tensor"
        output = torch.Tensor(output)
        loss = eval_fn(output, y_tensor)
        results['test_loss'] = [loss.item()]
        results['test_loss_relative'] = [100.0 * loss.item() / reference]
        results['predicted_profile'] = output

        return results

    def state_dict(self):
        """Store the persistent parameters of this model."""
        state_dict = {}
        state_dict['x_train'] = self.x_train
        state_dict['y_train'] = self.y_train
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the persistent parameters of this model and re-trigger the fitting."""
        self.gam.fit(state_dict['x_train'], state_dict['y_train'])
