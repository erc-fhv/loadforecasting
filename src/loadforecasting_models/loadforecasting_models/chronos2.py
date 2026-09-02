from typing import Optional, Callable, Sequence, Union
import numpy as np
import torch
from .helpers import SklearnOptunaHelper
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class Chronos2:
    """
    Zero-shot forecasting with the Chronos-2 foundation model (Amazon).

    No training is performed. Instead, train_model() stores the training data as
    historic context, which is later used as the target history (and covariate
    history) for the zero-shot forecasts.

    Context reconstruction assumes that the training windows are contiguous in
    time and immediately precede the test windows (i.e. data splits with
    dev_set = 0 and train_set_2 = 0).
    """

    def __init__(self,
        normalizer: Normalizer,
        future_covariate_indices: Optional[list] = None,
        ckpt: str = "amazon/chronos-2",
        device: Optional[str] = None,
        batch_size: int = 64,
        context_length: Optional[int] = None,
        eval_stride: int = 1,
        loss_relative_to: str = "",
        ) -> None:
        """
        Args:
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
            future_covariate_indices (list | None): Feature indices of the input tensor
                to be passed to Chronos-2 as future-known covariates (e.g. calendar and
                weather features). If None, a univariate (target-only) forecast is done.
            ckpt (str): Chronos-2 checkpoint (Hugging Face repo id or local directory).
            device (str | None): 'cpu', 'cuda' or 'mps'. If None, cuda is used when
                available, else cpu.
            batch_size (int): Batch size used for the Chronos-2 forecast calls.
            context_length (int | None): Optional cap on the number of historic
                timesteps used as context. If None, the model's default context
                length (8192 for Chronos-2) is used.
            eval_stride (int): In evaluate(), only forecast/score every Nth window
                (default 1 = every window). The context built for each scored window
                still includes every true value that came before it in x_test/y_test
                (built from the full, unstrided data) - only the *selection* of which
                windows are actually sent to the model and scored is subsampled, so
                the context stays gapless. predict() always uses every window,
                regardless of this setting. Useful to cut wall-clock during
                hyperparameter tuning, where evaluate() is the dominant cost (one
                sequential forecast call per window).
            loss_relative_to (str): Reference for relative loss calculation. Default is "".
        """
        self.normalizer = normalizer
        self.future_covariate_indices = future_covariate_indices
        self.ckpt = ckpt
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.context_length = context_length
        assert eval_stride >= 1, f"eval_stride must be >= 1, got {eval_stride}"
        self.eval_stride = eval_stride
        self.loss_relative_to = loss_relative_to

        self._model = None      # Lazy-loaded Chronos2Pipeline
        self._x_history = None  # Stored covariate history, shape (days, timesteps, features)
        self._y_history = None  # Stored target history, shape (days, timesteps, 1)

    def __getstate__(self) -> dict:
        """Drop the heavy foundation model when pickling (it is re-loadable from the checkpoint)."""
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    # Loaded checkpoints, shared across instances and keyed by (ckpt, device). Needed
    # because context_length/future_covariate_indices only affect inference, not the
    # loaded weights, so hyperparameter tuning (which constructs a fresh instance per
    # trial/fold) would otherwise reload the checkpoint from scratch every time.
    _MODEL_CACHE: dict = {}

    def _load_model(self):
        """Lazy-load the Chronos-2 checkpoint (cached per checkpoint+device)."""
        cache_key = (self.ckpt, self.device)
        if cache_key not in Chronos2._MODEL_CACHE:
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
            from chronos import Chronos2Pipeline
            Chronos2._MODEL_CACHE[cache_key] = \
                Chronos2Pipeline.from_pretrained(self.ckpt, device_map=self.device)
        self._model = Chronos2._MODEL_CACHE[cache_key]
        return self._model

    def train_model(self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        ) -> dict:
        """
        No training is done for this zero-shot model. The given training data is
        stored as historic context for the later forecasts.
        """

        x_tensor = self.normalizer.convert_to_torch_tensor(x_train).float() \
            if self.normalizer is not None else torch.as_tensor(x_train).float()
        y_tensor = self.normalizer.convert_to_torch_tensor(y_train).float() \
            if self.normalizer is not None else torch.as_tensor(y_train).float()

        assert x_tensor.shape[0] == y_tensor.shape[0], \
            f"Day count mismatch: x_train has {x_tensor.shape[0]} days, " \
            f"y_train has {y_tensor.shape[0]} days."
        assert x_tensor.shape[1] == y_tensor.shape[1], \
            f"Timestep count mismatch: x_train has {x_tensor.shape[1]} timesteps, " \
            f"y_train has {y_tensor.shape[1]} timesteps."

        self._x_history = x_tensor
        self._y_history = y_tensor

        history = {}
        history['loss'] = [0.0]

        return history

    def train_model_auto(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike,
        n_trials: int = 20,
        k_folds: int = 2,
        feature_index_groups: Union[Sequence[Sequence[int]], None] = None,
        eval_stride: int = 1,
        storage_path: Union[str, None] = None,
        study_name: Union[str, None] = None,
        verbose: int = 1,
        ) -> dict:
        """
        Tune context_length and, if feature_index_groups is given, which feature
        groups to use as future_covariate_indices, with Optuna and TimeSeriesSplit
        cross-validation. No weights are fitted (this is a zero-shot model), so
        tuning searches directly over evaluate() loss for each candidate setting.

        Note: unlike the column-slicing models (Knn, Ridge, ...), this model only
        ever consumes the specific columns named by future_covariate_indices, not
        all of x's columns - so an empty selection (a univariate forecast) is a
        valid choice here, and the full x is always passed through unsliced.

        Defaults for n_trials/k_folds are lower than the other models': each trial
        runs actual foundation-model inference (no cheap proxy loss available), and
        model construction is comparatively expensive even with checkpoint caching.

        Args:
            x_train: Input features of shape (batch_len, sequence_len, features).
            y_train: Target values of shape (batch_len, sequence_len, 1).
            n_trials (int): Number of Optuna trials for hyperparameter search.
            k_folds (int): Number of TimeSeriesSplit folds used for cross-validation.
            feature_index_groups: Optional list of column-index groups (one group per
                named feature) to choose future_covariate_indices from.
            eval_stride: Passed as a fixed (untuned) constructor kwarg to every
                trial's model - see Chronos2.__init__'s eval_stride for what it
                does. Default 1 = no subsampling. Cuts per-trial wall-clock since
                evaluate() forecasts sequentially, one call per scored window.
            storage_path: Optional sqlite file path for the Optuna study storage.
            verbose (int): Verbosity level. 0: silent, 1: dots, 2: full.

        Returns:
            dict: Training history and best hyperparameters.
        """

        tuner = SklearnOptunaHelper(self)
        return tuner.train_auto(
            x_train=x_train,
            y_train=y_train,
            n_trials=n_trials,
            k_folds=k_folds,
            feature_index_groups=feature_index_groups,
            covariate_param_name='future_covariate_indices',
            fixed_kwargs={'eval_stride': eval_stride},
            storage_path=storage_path,
            study_name=study_name,
            verbose=verbose,
            )

    @staticmethod
    def suggest_params(trial) -> dict:
        """Optuna search space for this model's hyperparameters."""
        return {
            # Chronos-2's own max context length is 8192.
            'context_length': trial.suggest_int('context_length', 168, 2048, log=True),
        }

    def _build_inputs(self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor],
        ) -> list:
        """
        Build one Chronos-2 input dict per forecast window in x.

        The context of window i is the stored training history plus (if y_true is
        given) the true values of the previous windows 0..i-1 - just as a real
        day-ahead forecaster would have access to all past actuals. Each covariate
        is z-scored with the statistics of its own context part before being passed
        to the model.
        """

        assert self._y_history is not None, "train_model() must be called before predict()."

        nr_of_batches, _, _ = x.shape
        y_hist_flat = self._y_history[..., 0].reshape(-1)             # (days*timesteps,)
        x_hist = self._x_history                                      # (days, timesteps, features)

        inputs = []
        for i in range(nr_of_batches):

            # Target context: stored history plus true values of previous windows
            if y_true is not None and i > 0:
                target = torch.cat([y_hist_flat, y_true[:i, :, 0].reshape(-1)])
                x_context = torch.cat([x_hist, x[:i]], dim=0)
            else:
                target = y_hist_flat
                x_context = x_hist

            # Optionally cap the context length
            if self.context_length is not None and target.shape[0] > self.context_length:
                target = target[-self.context_length:]

            item = {'target': target}

            if self.future_covariate_indices is not None:
                # Past covariate values, one flattened series per feature
                past_cov = x_context[:, :, self.future_covariate_indices] \
                    .reshape(-1, len(self.future_covariate_indices))          # (context, F)
                future_cov = x[i][:, self.future_covariate_indices]           # (horizon, F)

                if self.context_length is not None and past_cov.shape[0] > self.context_length:
                    past_cov = past_cov[-self.context_length:]

                # Z-score each covariate with the statistics of its context part
                mean = past_cov.mean(dim=0, keepdim=True)
                std = past_cov.std(dim=0, keepdim=True).clamp(min=1e-8)
                past_cov = (past_cov - mean) / std
                future_cov = (future_cov - mean) / std

                names = [f'cov_{j}' for j in range(len(self.future_covariate_indices))]
                item['past_covariates'] = {name: past_cov[:, j] for j, name in enumerate(names)}
                item['future_covariates'] = {name: future_cov[:, j] for j, name in enumerate(names)}

            inputs.append(item)

        return inputs

    def _forecast(self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        stride: int = 1,
        ) -> torch.Tensor:
        """
        Forecast every `stride`-th window in x and return the median quantile
        predictions with shape (ceil(batch_len/stride), sequence_len, 1).

        The context for each forecast window is still built from the full,
        unstrided x/y_true (see _build_inputs) - stride only subsamples which of
        the resulting, already-correctly-contextualized windows are actually sent
        to the (expensive) model.predict() call, so the context never has gaps
        even when stride > 1.
        """

        model = self._load_model()
        horizon = x.shape[1]
        inputs = self._build_inputs(x, y_true)
        if stride > 1:
            inputs = inputs[::stride]

        predictions = model.predict(
            inputs=inputs,
            prediction_length=horizon,
            batch_size=self.batch_size,
            context_length=self.context_length,
            )

        # Find the median (0.5) quantile index
        quantile_levels = [round(float(q), 6) for q in model.quantiles]
        if 0.5 in quantile_levels:
            median_index = quantile_levels.index(0.5)
        else:
            median_index = len(quantile_levels) // 2

        # Each prediction has shape (n_variates, n_quantiles, prediction_length)
        y_pred = torch.stack([p[0, median_index, :] for p in predictions])   # (batch, horizon)
        y_pred = y_pred.float().unsqueeze(-1)                                # (batch, horizon, 1)

        return y_pred

    def predict(self,
            x: ArrayLike,
            ) -> ArrayLike:
        """
        Predict the upcoming load profiles with the zero-shot Chronos-2 model.

        Args:
            x (ArrayLike): Model input tensor of shape (batch_len, sequence_len,
                features). Only the covariate features (see future_covariate_indices)
                are used; the target context is the history stored by train_model().

        Returns:
            ArrayLike: Predicted y tensor of shape (batch_len, sequence_len, 1).
        """

        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float()

        return self._forecast(x_tensor, y_true=None)

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        loss_relative_to: str = "",
        ) -> dict:
        """
        Evaluate the model on the given x_test and y_test.

        The true values of the previous test windows are used as part of the
        forecast context, as a real day-ahead forecaster would have access to
        all past actuals. If self.eval_stride > 1, only every eval_stride-th
        window is actually forecast and scored (see _forecast) - y_test is
        subsampled the same way before computing the loss, so shapes still match.
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

        output = self._forecast(x_tensor, y_true=y_tensor, stride=self.eval_stride)
        if self.eval_stride > 1:
            y_tensor = y_tensor[::self.eval_stride]

        assert output.shape == y_tensor.shape, \
            f"Shape mismatch: got {output.shape}, expected {y_tensor.shape})"

        # Unnormalize the target variable, if wished.
        if de_normalize:
            assert self.normalizer is not None, "No model_adapter given."
            y_tensor = self.normalizer.de_normalize_y(y_tensor)
            assert isinstance(y_tensor, torch.Tensor), "Denormalized y_tensor is not a torch.Tensor"
            output = self.normalizer.de_normalize_y(output)
            assert isinstance(output, torch.Tensor), "Denormalized output is not a torch.Tensor"

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
        loss = eval_fn(output, y_tensor)
        results['test_loss'] = [loss.item()]
        results['test_loss_relative'] = [100.0*loss.item()/reference]
        results['predicted_profile'] = output

        return results

    def state_dict(self) -> dict:
        """No persistent parameter needed for this zero-shot model."""
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict) -> None:
        """No persistent parameter needed for this zero-shot model."""
