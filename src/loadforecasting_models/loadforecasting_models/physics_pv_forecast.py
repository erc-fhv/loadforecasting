"""
Physics-based PV prediction combined with 7-day persistence for net load forecasting.

Net load is decomposed into two components:
  - Household load (positive part): predicted via 7-day persistence
  - PV generation  (negative part): predicted via physics model

Net load prediction formula:
    netload_pred = (netload_7d_ago + pv_physics_7d_ago) - pv_physics_today
"""

from typing import Callable, Optional, Union

import numpy as np
import torch

from .normalizer import Normalizer

# Accepted input types
ArrayLike = Union[torch.Tensor, np.ndarray]


class PhysicsPvForecast:
    """
    Net load forecasting model combining a physics-based PV prediction with
    7-day persistence for the household load component.

    The model requires two sets of weather feature indices in the input tensor ``x``:
    one for the current prediction day and one for the same day 7 days ago (lagged
    weather).  The calling application is responsible for providing these features.

    Units: PV system parameters must be in the same power unit as the net load
    (e.g. all in W, or all in kW).
    """

    def __init__(
        self,
        total_pv_peak_power_wp: float,
        feature_index_radiation: int,
        feature_index_ambient_temperature: int,
        feature_index_radiation_7d_ago: int,
        feature_index_ambient_temperature_7d_ago: int,
        feature_index_netload_7d_ago: int = 0,
        inverter_max_power_w: Optional[float] = None,
        normalizer: Optional[Normalizer] = None,
        temp_coefficient_pct_per_degc: float = -0.35,
        inverter_efficiency: float = 0.95,
        cell_temp_model: str = "King",
        feature_index_wind_speed: Optional[int] = None,
        feature_index_wind_speed_7d_ago: Optional[int] = None,
    ) -> None:
        """
        Args:
            total_pv_peak_power_wp:
                Total PV system peak power at Standard Test Conditions (STC)
                in Wp (1000 W/m², 25 °C cell temperature).
            inverter_max_power_w:
                Maximum AC output power of the inverter in W (clips AC output).
                If None, defaults to ``total_pv_peak_power_wp`` (no clipping beyond peak power).
            feature_index_netload_7d_ago:
                Feature index in ``x`` that contains the 7-days-ago net load.
            feature_index_radiation:
                Feature index for global radiation on the tilted module surface
                for the current prediction day (W/m²).
            feature_index_ambient_temperature:
                Feature index for outdoor ambient air temperature for the current
                prediction day (°C).
            feature_index_radiation_7d_ago:
                Feature index for global radiation on the tilted module surface
                7 days ago (W/m²).
            feature_index_ambient_temperature_7d_ago:
                Feature index for outdoor ambient air temperature 7 days ago (°C).
            normalizer:
                Used for X and Y normalization / denormalization.
                If None, x is assumed to already be in physical units and the
                predicted net load is returned without normalization.
            temp_coefficient_pct_per_degc:
                PV module power temperature coefficient in %/°C.
                Typical crystalline silicon: -0.35 %/°C (default).
            inverter_efficiency:
                Inverter efficiency in the range [0, 1]. Default: 0.95.
            cell_temp_model:
                Cell temperature correlation to use. One of:
                  ``'King'``     - accounts for wind speed (default, recommended).
                  ``'Kurtz'``    - accounts for wind speed.
                  ``'Ross'``     - no wind speed dependency.
                  ``'Skoplaki'`` - accounts for wind speed; mounting-dependent.
            feature_index_wind_speed:
                Feature index for wind speed (m/s) for the current prediction day.
                Required for ``'King'``, ``'Kurtz'``, and ``'Skoplaki'`` models.
            feature_index_wind_speed_7d_ago:
                Feature index for wind speed (m/s) 7 days ago.
                Required for ``'King'``, ``'Kurtz'``, and ``'Skoplaki'`` models.
        """
        if cell_temp_model not in ("King", "Kurtz", "Ross", "Skoplaki"):
            raise ValueError(
                f"Unknown cell_temp_model '{cell_temp_model}'. "
                "Choose one of: 'King', 'Kurtz', 'Ross', 'Skoplaki'."
            )
        if cell_temp_model in ("King", "Kurtz", "Skoplaki") and (
            feature_index_wind_speed is None or feature_index_wind_speed_7d_ago is None
        ):
            raise ValueError(
                f"cell_temp_model='{cell_temp_model}' requires wind speed features. "
                "Provide 'feature_index_wind_speed' and 'feature_index_wind_speed_7d_ago'."
            )

        self.normalizer = normalizer
        self.total_pv_peak_power_wp = total_pv_peak_power_wp
        self.inverter_max_power_w = inverter_max_power_w if inverter_max_power_w is not None else total_pv_peak_power_wp
        self.feature_index_netload_7d_ago = feature_index_netload_7d_ago
        self.feature_index_radiation = feature_index_radiation
        self.feature_index_ambient_temperature = feature_index_ambient_temperature
        self.feature_index_radiation_7d_ago = feature_index_radiation_7d_ago
        self.feature_index_ambient_temperature_7d_ago = feature_index_ambient_temperature_7d_ago
        self.temp_coefficient_pct_per_degc = temp_coefficient_pct_per_degc
        self.inverter_efficiency = inverter_efficiency
        self.cell_temp_model = cell_temp_model
        self.feature_index_wind_speed = feature_index_wind_speed
        self.feature_index_wind_speed_7d_ago = feature_index_wind_speed_7d_ago

    # ------------------------------------------------------------------
    # Cell temperature correlations
    # ------------------------------------------------------------------

    def _cell_temp_king(
        self,
        t_ambient: np.ndarray,
        radiation: np.ndarray,
        wind_speed: np.ndarray,
    ) -> np.ndarray:
        """King et al. (2004) - doi.org/10.2172/919131 - Eqs. 11 & 12."""
        a = -3.56
        b = -0.075
        delta_temperature = 3.0
        t_backsheet = t_ambient + radiation * np.exp(a + b * wind_speed)
        return t_backsheet + radiation / 1000.0 * delta_temperature

    def _cell_temp_kurtz(
        self,
        t_ambient: np.ndarray,
        radiation: np.ndarray,
        wind_speed: np.ndarray,
    ) -> np.ndarray:
        """Kurtz et al. (2009) - doi.org/10.1109/PVSC.2009.5411307 - Eq. 3."""
        return t_ambient + radiation * np.exp(-3.473 - 0.0594 * wind_speed)

    def _cell_temp_ross(
        self,
        t_ambient: np.ndarray,
        radiation: np.ndarray,
    ) -> np.ndarray:
        """
        Ross (1981) -
        Design techniques for flat-plate photovoltaic arrays

        See also:
        pvlib-python.readthedocs.io/en/latest/reference/generated/pvlib.temperature.ross.html
        """
        k = 0.0563        # Sloped roof, non-ventilated
        return t_ambient + radiation * k

    def _cell_temp_skoplaki(
        self,
        t_ambient: np.ndarray,
        radiation: np.ndarray,
        wind_speed: np.ndarray,
    ) -> np.ndarray:
        """
        Skoplaki et al. (2008) -
        doi.org/10.1016/j.solmat.2008.05.016 -
        Eq. 25, sloped roof mounting (omega=1.8).
        """
        omega = 1.8  # mounting coefficient: sloped roof
        return t_ambient + omega * (0.32 / (8.91 + 2.0 * wind_speed)) * radiation

    def _cell_temperature(
        self,
        t_ambient: np.ndarray,
        radiation: np.ndarray,
        wind_speed: Optional[np.ndarray],
    ) -> np.ndarray:
        """Dispatch to the selected cell temperature model."""
        if self.cell_temp_model == "King":
            assert wind_speed is not None
            return self._cell_temp_king(t_ambient, radiation, wind_speed)
        elif self.cell_temp_model == "Kurtz":
            assert wind_speed is not None
            return self._cell_temp_kurtz(t_ambient, radiation, wind_speed)
        elif self.cell_temp_model == "Ross":
            return self._cell_temp_ross(t_ambient, radiation)
        else:  # Skoplaki
            assert wind_speed is not None
            return self._cell_temp_skoplaki(t_ambient, radiation, wind_speed)

    # ------------------------------------------------------------------
    # PV power model
    # ------------------------------------------------------------------

    def _pv_ac_power(
        self,
        radiation: np.ndarray,
        t_ambient: np.ndarray,
        wind_speed: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Compute AC PV output power via linear STC extrapolation.

        Returns non-negative power clipped at ``inverter_max_power_w``.

        Args:
            radiation: Global radiation on tilted module surface (W/m²).
            t_ambient: Outdoor air temperature (°C).
            wind_speed: Wind speed (m/s). Required for King / Kurtz / Skoplaki.

        Returns:
            AC PV output power (same unit as ``total_pv_peak_power_wp``).
        """
        t_cell = self._cell_temperature(t_ambient, radiation, wind_speed)

        dc_power = (
            self.total_pv_peak_power_wp
            * (radiation / 1000.0)
            * ((t_cell - 25.0) * self.temp_coefficient_pct_per_degc / 100.0 + 1.0)
        )

        ac_power = np.minimum(dc_power * self.inverter_efficiency, self.inverter_max_power_w)

        # Clamp to zero: no reverse power flow through the model
        return np.maximum(ac_power, 0.0)

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def predict(self, x: ArrayLike) -> ArrayLike:
        """
        Predict net load using physics-based PV and 7-day household persistence.

        Formula::

            netload_pred = (netload_7d_ago + pv_physics_7d_ago) - pv_physics_today

        The term ``(netload_7d_ago + pv_physics_7d_ago)`` removes the modeled PV
        contribution from 7 days ago and thereby isolates the household load component.
        Subtracting ``pv_physics_today`` yields the predicted net load.

        Args:
            x: Normalised input tensor of shape ``(batch_len, sequence_len, features)``.

        Returns:
            Predicted y tensor of shape ``(batch_len, sequence_len, 1)``.
        """

        input_was_tensor = isinstance(x, torch.Tensor)

        if input_was_tensor:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float64)

        # --- Extract 7-day-ago net load ---
        netload_7d = x_np[:, :, self.feature_index_netload_7d_ago]

        # --- Extract weather features (today) ---
        radiation_today = x_np[:, :, self.feature_index_radiation]
        temp_today = x_np[:, :, self.feature_index_ambient_temperature]
        wind_today = (
            x_np[:, :, self.feature_index_wind_speed]
            if self.feature_index_wind_speed is not None
            else None
        )

        # --- Extract weather features (7 days ago) ---
        radiation_7d = x_np[:, :, self.feature_index_radiation_7d_ago]
        temp_7d = x_np[:, :, self.feature_index_ambient_temperature_7d_ago]
        wind_7d = (
            x_np[:, :, self.feature_index_wind_speed_7d_ago]
            if self.feature_index_wind_speed_7d_ago is not None
            else None
        )

        # --- Physics-based PV power (positive W) ---
        pv_today = self._pv_ac_power(radiation_today, temp_today, wind_today)
        pv_7d = self._pv_ac_power(radiation_7d, temp_7d, wind_7d)

        # --- Net load prediction ---
        # Add back modeled PV from 7d ago → household load
        # Subtract today's modeled PV → net load
        netload_pred = (netload_7d + pv_7d) - pv_today

        # Reshape to (batch_len, sequence_len, 1)
        netload_pred = netload_pred[:, :, np.newaxis].astype(np.float32)

        if self.normalizer is not None:
            y_out = self.normalizer.normalize_y(
                torch.from_numpy(netload_pred), training=False
            )
        else:
            y_out = torch.from_numpy(netload_pred)

        if input_was_tensor:
            return y_out
        else:
            if isinstance(y_out, torch.Tensor):
                return y_out.numpy()
            return y_out

    def train_model(self) -> dict:
        """No training required for this physics-based model."""
        return {"loss": [0.0]}

    def evaluate(
        self,
        x_test: ArrayLike,
        y_test: ArrayLike,
        results: Optional[dict] = None,
        de_normalize: bool = False,
        eval_fn: Callable[..., torch.Tensor] = torch.nn.L1Loss(),
        loss_relative_to: str = "mean",
    ) -> dict:
        """
        Evaluate the model on the given test data.

        Args:
            x_test: Normalised input tensor of shape ``(batch_len, sequence_len, features)``.
            y_test: Normalised target tensor of shape ``(batch_len, sequence_len, 1)``.
            results: Optional dict to extend with the evaluation results.
            de_normalize: If True, de-normalize predictions and targets before
                computing the loss.
            eval_fn: Loss function. Default: MAE (``torch.nn.L1Loss``).
            loss_relative_to: Reference for the relative loss.
                One of ``'mean'``, ``'max'``, ``'range'``.

        Returns:
            Dict with keys ``'test_loss'``, ``'test_loss_relative'``,
            ``'predicted_profile'``.
        """
        if results is None:
            results = {}

        if isinstance(x_test, np.ndarray):
            x_tensor = torch.from_numpy(x_test).float()
        else:
            x_tensor = x_test.float()
        if isinstance(y_test, np.ndarray):
            y_tensor = torch.from_numpy(y_test).float()
        else:
            y_tensor = y_test.float()

        output = self.predict(x_tensor)
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output).float()

        assert output.shape == y_tensor.shape, (
            f"Shape mismatch: got {output.shape}, expected {y_tensor.shape}"
        )

        if de_normalize:
            assert self.normalizer is not None, "No normalizer given."
            y_tensor = self.normalizer.de_normalize_y(y_tensor)
            assert isinstance(y_tensor, torch.Tensor), (
                "Denormalized y_tensor is not a torch.Tensor"
            )
            output = self.normalizer.de_normalize_y(output)
            assert isinstance(output, torch.Tensor), (
                "Denormalized output is not a torch.Tensor"
            )

        if loss_relative_to == "mean":
            reference = float(torch.abs(torch.mean(y_tensor)))
        elif loss_relative_to == "max":
            reference = float(torch.abs(torch.max(y_tensor)))
        elif loss_relative_to == "range":
            reference = float(torch.max(y_tensor) - torch.min(y_tensor))
        else:
            raise ValueError(
                f"Unexpected parameter: loss_relative_to = '{loss_relative_to}'"
            )

        loss = eval_fn(output, y_tensor)
        results["test_loss"] = [loss.item()]
        results["test_loss_relative"] = [100.0 * loss.item() / reference]
        results["predicted_profile"] = output

        return results

    def state_dict(self) -> dict:
        """No trainable parameters - returns an empty state dict."""
        return {}
