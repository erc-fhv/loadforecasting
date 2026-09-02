import math
from unittest.mock import patch
import numpy as np
import pytest
import torch
from sklearn.model_selection import TimeSeriesSplit
from loadforecasting_models import Tirex2, Normalizer

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

@pytest.fixture(autouse=True)
def clear_tirex2_model_cache():
    """
    Tirex2 caches loaded checkpoints at the class level, keyed by (ckpt, device), so
    that repeated instantiations during hyperparameter tuning don't reload the
    checkpoint from scratch. Without clearing it between tests, the first test's
    patched tirex2.load_model return value would leak into every later test that
    uses the same (default) ckpt/device.
    """
    Tirex2._MODEL_CACHE.clear()
    yield
    Tirex2._MODEL_CACHE.clear()

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

def create_model(future_covariate_indices=None, eval_stride=1):
    normalizer = Normalizer()
    my_model = Tirex2(normalizer=normalizer,
        future_covariate_indices=future_covariate_indices,
        eval_stride=eval_stride,
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

class RecordingFakeForecastModel(FakeForecastModel):
    """FakeForecastModel that also records how many windows each forecast() call saw."""

    def __init__(self):
        super().__init__()
        self.call_sizes = []

    def forecast(self, timeseries, prediction_length, output_type="numpy", batch_size=512):
        self.call_sizes.append(len(timeseries))
        return super().forecast(timeseries, prediction_length, output_type, batch_size)

def test_eval_stride_keeps_context_gapless():
    """
    Test if eval_stride only subsamples which windows are actually forecast/scored,
    while the context built for each scored window still includes every true day
    before it (built from the full, unstrided x_test/y_test) - i.e. no gaps.
    """

    fake_model = RecordingFakeForecastModel()
    with patch('tirex2.load_model', return_value=fake_model):
        my_model = create_model(eval_stride=7)

        nr_of_train_days, timesteps = 10, 24
        x_train = torch.randn(nr_of_train_days, timesteps, 10)
        y_train = torch.randn(nr_of_train_days, timesteps, 1)
        my_model.train_model(x_train, y_train)

        nr_of_val_days = 22
        x_test = torch.randn(nr_of_val_days, timesteps, 10)
        y_test = torch.randn(nr_of_val_days, timesteps, 1)
        results = my_model.evaluate(x_test, y_test)

        expected_selected = list(range(0, nr_of_val_days, 7))   # [0, 7, 14, 21]
        assert fake_model.call_sizes == [len(expected_selected)]
        assert results['predicted_profile'].shape == (len(expected_selected), timesteps, 1)

        # Each scored window's context must include every true day before it,
        # contiguous - not gapped - since only the *selection* of scored windows
        # is strided, not the context construction itself.
        for call_index, i in enumerate(expected_selected):
            ts = fake_model.seen_timeseries[call_index]
            expected_context_days = nr_of_train_days + i
            assert ts.target.shape == (1, expected_context_days * timesteps)

def test_eval_stride_default_scores_every_window():
    """Test if the default eval_stride=1 forecasts/scores every window."""

    fake_model = RecordingFakeForecastModel()
    with patch('tirex2.load_model', return_value=fake_model):
        my_model = create_model()   # eval_stride defaults to 1

        x_train = torch.randn(10, 24, 10)
        y_train = torch.randn(10, 24, 1)
        my_model.train_model(x_train, y_train)

        x_test = torch.randn(7, 24, 10)
        y_test = torch.randn(7, 24, 1)
        my_model.evaluate(x_test, y_test)

    assert fake_model.call_sizes == [7]

def test_eval_stride_does_not_affect_predict():
    """Test if predict() always forecasts every window, regardless of eval_stride."""

    fake_model = RecordingFakeForecastModel()
    with patch('tirex2.load_model', return_value=fake_model):
        my_model = create_model(eval_stride=7)

        x_train = torch.randn(10, 24, 10)
        y_train = torch.randn(10, 24, 1)
        my_model.train_model(x_train, y_train)

        x_test = torch.randn(9, 24, 10)
        y_pred = my_model.predict(x_test)

    assert fake_model.call_sizes == [9]
    assert y_pred.shape == (9, 24, 1)

def _expected_val_windows(n_samples, k_folds, dataset_index, eval_stride):
    """Mirror SklearnOptunaHelper's pooled fold-rotation + Tirex2's eval_stride selection."""

    splits = list(TimeSeriesSplit(n_splits=k_folds).split(np.arange(n_samples)))
    _, val_idx = splits[dataset_index % len(splits)]
    return math.ceil(len(val_idx) / eval_stride)

def test_train_model_auto_eval_stride(tmp_path):
    """
    Test if train_model_auto's eval_stride reaches every trial's model (via
    fixed_kwargs), so each pooled dataset's fold evaluation only forecasts/scores
    ceil(val_len/7) windows, not the full fold.
    """

    x1, y1 = torch.randn(40, 24, 10), torch.randn(40, 24, 1)
    x2, y2 = torch.randn(61, 24, 10), torch.randn(61, 24, 1)

    fake_model = RecordingFakeForecastModel()
    with patch('tirex2.load_model', return_value=fake_model):
        my_model = create_model()
        my_model.train_model_auto([x1, x2], [y1, y2], n_trials=1, k_folds=2,
            eval_stride=7, storage_path=str(tmp_path / "study.db"), verbose=0)

    assert fake_model.call_sizes == [
        _expected_val_windows(40, 2, 0, eval_stride=7),
        _expected_val_windows(61, 2, 1, eval_stride=7),
    ]
    # A stride longer than the fold must still evaluate at least one window.
    assert all(size >= 1 for size in fake_model.call_sizes)

def test_train_model_auto_default_eval_stride_is_unaffected(tmp_path):
    """Test if the default eval_stride=1 evaluates every row of the validation fold."""

    x1, y1 = torch.randn(40, 24, 10), torch.randn(40, 24, 1)
    x2, y2 = torch.randn(61, 24, 10), torch.randn(61, 24, 1)

    fake_model = RecordingFakeForecastModel()
    with patch('tirex2.load_model', return_value=fake_model):
        my_model = create_model()
        my_model.train_model_auto([x1, x2], [y1, y2], n_trials=1, k_folds=2,
            storage_path=str(tmp_path / "study.db"), verbose=0)

    assert fake_model.call_sizes == [
        _expected_val_windows(40, 2, 0, eval_stride=1),
        _expected_val_windows(61, 2, 1, eval_stride=1),
    ]
