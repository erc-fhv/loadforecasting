from typing import Optional, Callable, Union
import numpy as np
import torch
from .normalizer import Normalizer

# Define a type that can be either a torch Tensor or a numpy ndarray
ArrayLike = Union[torch.Tensor, np.ndarray]

class Tirex2:
    """
    Zero-shot forecasting with the TiRex-2 foundation model (NX-AI).

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
        ckpt: str = "NX-AI/TiRex-2",
        device: Optional[str] = None,
        batch_size: int = 64,
        context_length: Optional[int] = None,
        loss_relative_to: str = "",
        ) -> None:
        """
        Args:
            normalizer (Normalizer): Used for X and Y normalization and denormalization.
            future_covariate_indices (list | None): Feature indices of the input tensor
                to be passed to TiRex-2 as future-known covariates (e.g. calendar and
                weather features). If None, a univariate (target-only) forecast is done.
            ckpt (str): TiRex-2 checkpoint (Hugging Face repo id or local directory).
            device (str | None): 'cpu', 'cuda' or 'mps'. If None, cuda is used when
                available, else cpu.
            batch_size (int): Batch size used for the TiRex-2 forecast calls.
            context_length (int | None): Optional cap on the number of historic
                timesteps used as context. If None, all available history is used.
            loss_relative_to (str): Reference for relative loss calculation. Default: "".
        """
        self.normalizer = normalizer
        self.future_covariate_indices = future_covariate_indices
        self.ckpt = ckpt
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.context_length = context_length
        self.loss_relative_to = loss_relative_to

        self._model = None      # Lazy-loaded TiRex-2 ForecastModel
        self._x_history = None  # Stored covariate history, shape (days, timesteps, features)
        self._y_history = None  # Stored target history, shape (days, timesteps, 1)

    def __getstate__(self) -> dict:
        """Drop the heavy foundation model when pickling (it is re-loadable from the checkpoint)."""
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    def _load_model(self):
        """Lazy-load the TiRex-2 checkpoint."""
        if self._model is None:
            from tirex2 import load_model
            self._model = load_model(self.ckpt, device=self.device)
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

    def _build_timeseries(self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor],
        ) -> list:
        """
        Build one tirex2.TimeseriesType per forecast window in x.

        The context of window i is the stored training history plus (if y_true is
        given) the true values of the previous windows 0..i-1 - just as a real
        day-ahead forecaster would have access to all past actuals.
        """

        from tirex2 import TimeseriesType

        assert self._y_history is not None, "train_model() must be called before predict()."

        nr_of_batches, horizon, _ = x.shape
        y_hist_flat = self._y_history[..., 0].reshape(-1)             # (days*timesteps,)
        x_hist = self._x_history                                      # (days, timesteps, features)

        timeseries = []
        for i in range(nr_of_batches):

            # Target context: stored history plus true values of previous windows
            if y_true is not None and i > 0:
                target = torch.cat([y_hist_flat, y_true[:i, :, 0].reshape(-1)])
                x_context = torch.cat([x_hist, x[:i]], dim=0)
            else:
                target = y_hist_flat
                x_context = x_hist

            # Future covariates: must cover context + prediction horizon
            future_covariates = None
            if self.future_covariate_indices is not None:
                cov = torch.cat([x_context, x[i:i+1]], dim=0)         # (days, timesteps, features)
                cov = cov[:, :, self.future_covariate_indices]
                future_covariates = cov.reshape(-1, len(self.future_covariate_indices))

            # Optionally cap the context length (covariates keep the horizon part)
            if self.context_length is not None and target.shape[0] > self.context_length:
                target = target[-self.context_length:]
                if future_covariates is not None:
                    future_covariates = future_covariates[-(self.context_length + horizon):, :]

            # Z-score each covariate with the statistics of its context part
            if future_covariates is not None:
                context_part = future_covariates[:target.shape[0]]
                mean = context_part.mean(dim=0, keepdim=True)
                std = context_part.std(dim=0, keepdim=True).clamp(min=1e-8)
                future_covariates = (future_covariates - mean) / std
                future_covariates = future_covariates.T.contiguous()  # (F, context+horizon)

            timeseries.append(TimeseriesType(
                target=target.unsqueeze(0),                             # (1, context)
                past_covariates=None,
                future_covariates=future_covariates,
                ))

        return timeseries

    def _forecast(self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """
        Forecast all windows in x and return the median quantile predictions
        with shape (batch_len, sequence_len, 1).
        """

        model = self._load_model()
        horizon = x.shape[1]
        timeseries = self._build_timeseries(x, y_true)

        forecasts = model.forecast(
            timeseries=timeseries,
            prediction_length=horizon,
            output_type="numpy",
            batch_size=self.batch_size,
            )

        # Find the median (0.5) quantile index
        quantile_levels = [round(float(q), 6) for q in model.quantiles]
        if 0.5 in quantile_levels:
            median_index = quantile_levels.index(0.5)
        else:
            median_index = len(quantile_levels) // 2

        # Each forecast has shape (n_target_variates, n_quantiles, prediction_length)
        y_pred = np.stack([f[0, median_index, :] for f in forecasts])   # (batch, horizon)
        y_pred = torch.from_numpy(y_pred).float().unsqueeze(-1)         # (batch, horizon, 1)

        return y_pred

    def predict(self,
            x: ArrayLike,
            ) -> ArrayLike:
        """
        Predict the upcoming load profiles with the zero-shot TiRex-2 model.

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
        all past actuals.
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

        output = self._forecast(x_tensor, y_true=y_tensor)

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
