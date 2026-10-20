import sys
import types
from unittest.mock import patch
import pytest
import torch
from loadforecasting_models import Timestral, Normalizer


class FakeForecast:
    def __init__(self, median):
        self.median = median


class FakeForecaster:
    """Fake timestral.TimestralForecaster: records its inputs, median = 1.0."""

    def __init__(self):
        self.calls = []

    def forecast(self, context, prediction_length, known_covariates=None,
            clip_nonnegative=True):
        self.calls.append({'context': context, 'known_covariates': known_covariates,
            'clip_nonnegative': clip_nonnegative})
        assert context.ndim == 2
        if known_covariates is not None:
            # Known covariates must cover context + horizon exactly.
            assert known_covariates.shape[0] == context.shape[0]
            assert known_covariates.shape[2] == context.shape[1] + prediction_length
        return FakeForecast(torch.ones(context.shape[0], 1, prediction_length))


@pytest.fixture
def fake_forecaster():
    """Inject a fake timestral module, so neither the package nor weights are needed."""
    forecaster = FakeForecaster()
    module = types.ModuleType('timestral')
    module.TimestralForecaster = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: forecaster)
    Timestral._MODEL_CACHE.clear()
    with patch.dict(sys.modules, {'timestral': module}):
        yield forecaster
    Timestral._MODEL_CACHE.clear()


def create_model(**kwargs):
    return Timestral(normalizer=Normalizer(), loss_relative_to="range", **kwargs)


def test_train_and_predict_shapes(fake_forecaster):
    my_model = create_model()
    history = my_model.train_model(torch.randn(30, 24, 10), torch.randn(30, 24, 1))
    assert history['loss'] == [0.0]

    y_pred = my_model.predict(torch.randn(7, 24, 10))
    assert y_pred.shape == (7, 24, 1)
    assert torch.allclose(y_pred, torch.ones_like(y_pred))
    # Net load may be negative, so the forecast must never be clipped at zero.
    assert all(not call['clip_nonnegative'] for call in fake_forecaster.calls)


def test_evaluate_pads_contexts_and_batches(fake_forecaster):
    """Growing evaluation contexts are left-padded with NaN to one common length."""

    my_model = create_model(future_covariate_indices=[0, 2], batch_size=3)
    my_model.train_model(torch.randn(5, 24, 10), torch.randn(5, 24, 1))
    results = my_model.evaluate(torch.randn(7, 24, 10), torch.randn(7, 24, 1))

    assert results['predicted_profile'].shape == (7, 24, 1)
    assert [call['context'].shape[0] for call in fake_forecaster.calls] == [3, 3, 1]

    first = fake_forecaster.calls[0]
    assert first['known_covariates'].shape == (3, 2, first['context'].shape[1] + 24)
    # Window 0 has the shortest context, so it carries the most leading NaNs.
    n_pad = 2 * 24
    assert torch.isnan(first['context'][0, :n_pad]).all()
    assert not torch.isnan(first['context'][0, n_pad:]).any()
    assert not torch.isnan(first['context'][2]).any()


def test_eval_stride(fake_forecaster):
    my_model = create_model(eval_stride=3)
    my_model.train_model(torch.randn(10, 24, 10), torch.randn(10, 24, 1))
    results = my_model.evaluate(torch.randn(7, 24, 10), torch.randn(7, 24, 1))
    assert results['predicted_profile'].shape == (3, 24, 1)


def test_missing_package_gives_install_hint():
    Timestral._MODEL_CACHE.clear()
    with patch.dict(sys.modules, {'timestral': None}):
        with pytest.raises(ImportError, match="pip install"):
            create_model()._load_model()
