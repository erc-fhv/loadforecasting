from typing import Optional
import torch
from .chronos2 import Chronos2
from .normalizer import Normalizer

from huggingface_hub.utils import disable_progress_bars
from timestral import TimestralForecaster


class Timestral(Chronos2):
    """
    Zero-shot forecasting with the Timestral foundation model (Mistral AI).

    Shares the context handling with Chronos2 (stored training history, reporting
    delay as NaN mask, per-context z-scored covariates, Optuna tuning, evaluation);
    only model loading and the forecast call differ. The same assumption on
    contiguous training/test windows applies (see Chronos2).
    """

    MAX_CONTEXT_LENGTH = 32768  # Timestral's own maximum

    # Loaded checkpoints, shared across instances and keyed by (ckpt, device).
    _MODEL_CACHE: dict = {}

    def __init__(self,
        normalizer: Normalizer,
        future_covariate_indices: Optional[list] = None,
        ckpt: str = "mistralai/Timestral-209M",
        device: Optional[str] = None,
        batch_size: int = 64,
        context_length: Optional[int] = None,
        power_data_delay_steps: int = 0,
        eval_stride: int = 1,
        loss_relative_to: str = "",
        ) -> None:
        """
        Args: see Chronos2.__init__. ckpt is a Timestral checkpoint (Hugging Face
        repo id, e.g. 'mistralai/Timestral-5M' ... 'mistralai/Timestral-209M', or a
        local directory). If context_length is None, Timestral's maximum of 32768 is
        used.
        """
        super().__init__(
            normalizer=normalizer,
            future_covariate_indices=future_covariate_indices,
            ckpt=ckpt,
            device=device,
            batch_size=batch_size,
            context_length=context_length,
            power_data_delay_steps=power_data_delay_steps,
            eval_stride=eval_stride,
            loss_relative_to=loss_relative_to,
            )

    def _load_model(self):
        """Lazy-load the Timestral checkpoint (cached per checkpoint+device)."""
        cache_key = (self.ckpt, self.device)
        if cache_key not in Timestral._MODEL_CACHE:
            disable_progress_bars()
            Timestral._MODEL_CACHE[cache_key] = \
                TimestralForecaster.from_pretrained(self.ckpt, device=self.device)
        self._model = Timestral._MODEL_CACHE[cache_key]
        return self._model

    @staticmethod
    def _left_pad(series: list, length: int) -> torch.Tensor:
        """Stack 1D series of different lengths, left-padded with NaN (= missing)."""
        out = torch.full((len(series), length), float('nan'))
        for i, s in enumerate(series):
            out[i, length - s.shape[0]:] = s
        return out

    def _forecast(self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        stride: int = 1,
        ) -> torch.Tensor:
        """
        Forecast every `stride`-th window in x and return the median predictions
        with shape (ceil(batch_len/stride), sequence_len, 1).
        """

        model = self._load_model()
        horizon = x.shape[1]
        inputs = self._build_inputs(x, y_true)
        if stride > 1:
            inputs = inputs[::stride]

        y_pred = []
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start:start + self.batch_size]

            # Timestral needs one common context length T per call.
            length = max(item['target'].shape[0] for item in chunk)
            context = self._left_pad([item['target'] for item in chunk], length)  # (B, T)

            known_covariates = None
            if self.future_covariate_indices is not None:
                names = list(chunk[0]['past_covariates'].keys())
                rows = []
                for name in names:
                    # Past part padded to T, then the horizon part: (B, T + H)
                    past = self._left_pad([item['past_covariates'][name] for item in chunk], length)
                    future = torch.stack([item['future_covariates'][name] for item in chunk])
                    rows.append(torch.cat([past, future], dim=1))
                known_covariates = torch.stack(rows, dim=1)                     # (B, C, T + H)

            forecast = model.forecast(
                context=context,
                prediction_length=horizon,
                known_covariates=known_covariates,
                # Net load can be negative (PV feed-in), so never clip at zero.
                clip_nonnegative=False,
                )
            y_pred.append(forecast.median[:, 0, :].detach().cpu())              # (B, H)

        return torch.cat(y_pred).float().unsqueeze(-1)                          # (batch, H, 1)
