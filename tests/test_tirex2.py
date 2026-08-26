from unittest.mock import patch
import numpy as np
import torch
from loadforecasting_models import Tirex2, Normalizer

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

class FakeForecastModel:
    """Fake tirex2.ForecastModel: returns deterministic quantile forecasts."""

    def __init__(self):
        self.quantiles = torch.tensor(QUANTILE_LEVELS)
        self.seen_timeseries = None

    def forecast(self, timeseries, prediction_length, output_type="numpy", batch_size=512):
        self.seen_timeseries = timeseries
        forecasts = []
        for ts in timeseries:
            assert ts.target.ndim == 2 and ts.target.shape[0] == 1
            if ts.future_covariates is not None:
                # Covariates must cover context + prediction horizon
                assert ts.future_covariates.shape[1] == \
                    ts.target.shape[1] + prediction_length
            forecast = np.zeros((1, len(QUANTILE_LEVELS), prediction_length),
                dtype=np.float32)
            # Median (index 4) = 1.0, so it is distinguishable from other quantiles
            forecast[0, QUANTILE_LEVELS.index(0.5), :] = 1.0
            forecasts.append(forecast)
        return forecasts

def create_model(future_covariate_indices=None):
    normalizer = Normalizer()
    my_model = Tirex2(normalizer=normalizer,
        future_covariate_indices=future_covariate_indices,
        loss_relative_to="range")
    return my_model

def test_train_and_predict_shapes():
    """Test if storing the context and predicting runs without errors."""

    with patch('tirex2.load_model', return_value=FakeForecastModel()):
        my_model = create_model()

        x_train = torch.randn(30, 24, 10)
        y_train = torch.randn(30, 24, 1)
        history = my_model.train_model(x_train, y_train)
        assert history['loss'] == [0.0]

        x_test = torch.randn(7, 24, 10)
        y_pred = my_model.predict(x_test)
        assert y_pred.shape == (7, 24, 1)
        assert torch.allclose(y_pred, torch.ones_like(y_pred))

def test_evaluate():
    """Test if evaluating runs without errors and returns the expected keys."""

    with patch('tirex2.load_model', return_value=FakeForecastModel()):
        my_model = create_model()

        x_train = torch.randn(30, 24, 10)
        y_train = torch.randn(30, 24, 1)
        my_model.train_model(x_train, y_train)

        x_test = torch.randn(7, 24, 10)
        y_test = torch.randn(7, 24, 1)
        results = my_model.evaluate(x_test, y_test, de_normalize=True)

        assert 'test_loss' in results
        assert 'test_loss_relative' in results
        assert 'predicted_profile' in results
        assert results['predicted_profile'].shape == (7, 24, 1)

def test_future_covariates():
    """Test if the future covariates are built with the correct shape."""

    fake_model = FakeForecastModel()
    with patch('tirex2.load_model', return_value=fake_model):
        future_covariate_indices = [3, 4, 5]
        my_model = create_model(future_covariate_indices=future_covariate_indices)

        nr_of_train_days, timesteps = 30, 24
        x_train = torch.randn(nr_of_train_days, timesteps, 10)
        y_train = torch.randn(nr_of_train_days, timesteps, 1)
        my_model.train_model(x_train, y_train)

        x_test = torch.randn(7, timesteps, 10)
        y_test = torch.randn(7, timesteps, 1)
        my_model.evaluate(x_test, y_test)

        # Check the shapes of the TimeseriesType objects seen by the (fake) model
        for i, ts in enumerate(fake_model.seen_timeseries):
            expected_context = (nr_of_train_days + i) * timesteps
            assert ts.target.shape == (1, expected_context)
            assert ts.future_covariates.shape == \
                (len(future_covariate_indices), expected_context + timesteps)

def test_context_length_cap():
    """Test if the context length cap is applied to target and covariates."""

    fake_model = FakeForecastModel()
    with patch('tirex2.load_model', return_value=fake_model):
        my_model = create_model(future_covariate_indices=[3, 4, 5])
        my_model.context_length = 48

        x_train = torch.randn(30, 24, 10)
        y_train = torch.randn(30, 24, 1)
        my_model.train_model(x_train, y_train)

        x_test = torch.randn(3, 24, 10)
        my_model.predict(x_test)

        for ts in fake_model.seen_timeseries:
            assert ts.target.shape == (1, 48)
            assert ts.future_covariates.shape == (3, 48 + 24)

def test_numpy_input():
    """Test if numpy inputs are accepted as well."""

    with patch('tirex2.load_model', return_value=FakeForecastModel()):
        my_model = create_model()

        x_train = np.random.randn(30, 24, 10)
        y_train = np.random.randn(30, 24, 1)
        my_model.train_model(x_train, y_train)

        x_test = np.random.randn(7, 24, 10)
        y_pred = my_model.predict(x_test)
        assert y_pred.shape == (7, 24, 1)

def test_pickling_drops_foundation_model():
    """Test if pickling drops the heavy foundation model (re-loaded lazily)."""

    import pickle
    with patch('tirex2.load_model', return_value=FakeForecastModel()):
        my_model = create_model()
        my_model.train_model(torch.randn(30, 24, 10), torch.randn(30, 24, 1))
        my_model._load_model()
        assert my_model._model is not None

        my_restored = pickle.loads(pickle.dumps(my_model))
        assert my_restored._model is None
        assert my_restored._y_history is not None

        # Lazy re-load on next predict
        y_pred = my_restored.predict(torch.randn(2, 24, 10))
        assert y_pred.shape == (2, 24, 1)
