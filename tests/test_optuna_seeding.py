"""Tests for SklearnOptunaHelper's/OptunaHelper's pre-tuning-default seed trial."""

from unittest.mock import patch
import numpy as np
import pytest
import torch
from loadforecasting_models import Knn, Tirex2, Transformer, Normalizer

@pytest.fixture(autouse=True)
def clear_tirex2_model_cache():
    """Tirex2 caches loaded checkpoints at the class level - see test_tirex2.py."""
    Tirex2._MODEL_CACHE.clear()
    yield
    Tirex2._MODEL_CACHE.clear()


def test_seed_default_hyperparams_and_all_feature_groups(tmp_path):
    """
    Test if the seed trial reproduces a column-slicing model's (Knn) pre-tuning
    default hyperparameters and enables every feature group - since such models
    had no feature selection before tuning existed (all of x was used).
    """

    normalizer = Normalizer()
    my_model = Knn(k=40, weights='distance', normalizer=normalizer, loss_relative_to="range")

    x_train = torch.randn(30, 24, 3)
    y_train = torch.randn(30, 24, 1)
    feature_index_groups = [[0], [1], [2]]

    # n_trials=1: with an enqueued seed trial, this is the only trial that runs.
    history = my_model.train_model_auto(x_train, y_train, n_trials=1, k_folds=2,
        feature_index_groups=feature_index_groups,
        storage_path=str(tmp_path / "study.db"), verbose=0)

    seeded_params = history['optuna_study'].trials[0].params
    assert seeded_params == {
        'k': 40,
        'weights': 'distance',
        'use_feature_group_0': True,
        'use_feature_group_1': True,
        'use_feature_group_2': True,
        }

def test_seed_default_hyperparams_out_of_range_is_skipped_not_crashed(tmp_path):
    """
    Test if a pre-tuning attribute value that falls outside suggest_params()'s
    declared range (here: k=200, outside Knn's [3, 100]) is silently left out of
    the seed instead of crashing study.optimize() with an out-of-distribution error.
    """

    normalizer = Normalizer()
    my_model = Knn(k=200, weights='distance', normalizer=normalizer, loss_relative_to="range")

    x_train = torch.randn(30, 24, 3)
    y_train = torch.randn(30, 24, 1)

    history = my_model.train_model_auto(x_train, y_train, n_trials=1, k_folds=2,
        storage_path=str(tmp_path / "study.db"), verbose=0)

    # k=200 was out of suggest_params()'s [3, 100] range, so it must not have been
    # seeded - the sampler picks its own (necessarily different, in-range) value.
    seeded_params = history['optuna_study'].trials[0].params
    assert seeded_params['k'] != 200
    assert seeded_params['weights'] == 'distance'

def test_seed_covariate_mode_matches_default_covariate_and_context_length(tmp_path):
    """
    Test if the seed trial for a covariate_param_name model (Tirex2) enables only
    the feature group(s) covering the model's current future_covariate_indices
    default (not every group), and seeds context_length from the model's current
    attribute value.
    """

    class FakeForecastModel:
        """Fake tirex2.ForecastModel: returns deterministic quantile forecasts."""

        quantiles = torch.tensor([0.1, 0.5, 0.9])

        def forecast(self, timeseries, prediction_length, output_type="numpy", batch_size=512):
            return [np.zeros((1, 3, prediction_length), dtype=np.float32) for _ in timeseries]

    with patch('tirex2.load_model', return_value=FakeForecastModel()):
        normalizer = Normalizer()
        my_model = Tirex2(normalizer=normalizer, future_covariate_indices=[2],
            context_length=2048, loss_relative_to="range")

        x_train = torch.randn(40, 24, 4)
        y_train = torch.randn(40, 24, 1)
        feature_index_groups = [[0], [1], [2], [3]]

        history = my_model.train_model_auto(x_train, y_train, n_trials=1, k_folds=2,
            feature_index_groups=feature_index_groups,
            storage_path=str(tmp_path / "study.db"), verbose=0)

    seeded_params = history['optuna_study'].trials[0].params
    assert seeded_params['context_length'] == 2048
    assert seeded_params['use_feature_group_2'] is True
    for group_id in (0, 1, 3):
        assert seeded_params[f'use_feature_group_{group_id}'] is False

def test_seed_default_feature_group_ids_for_dl_models(tmp_path):
    """
    Test if OptunaHelper's default_feature_group_ids seeds exactly the given
    groups as enabled (and every other group as disabled) in one guaranteed
    trial - this is what model_trainer.py uses to give the DL models (Transformer/
    Lstm/xLstm) a hand-picked default feature selection.
    """

    normalizer = Normalizer()
    my_model = Transformer('0.1k', normalizer=normalizer, loss_relative_to="range")

    batch_size, seq_len, n_features = 20, 24, 5
    x_train = torch.randn(batch_size, seq_len, n_features)
    y_train = torch.randn(batch_size, seq_len, 1)
    feature_index_groups = [[0], [1], [2], [3], [4]]

    history = my_model.train_model_auto(x_train, y_train, n_trials=1, k_folds=2,
        feature_index_groups=feature_index_groups, default_feature_group_ids=[1, 3],
        storage_path=str(tmp_path / "study.db"), verbose=0)

    seeded_params = history['optuna_study'].trials[0].params
    for group_id in range(5):
        assert seeded_params[f'use_feature_group_{group_id}'] is (group_id in (1, 3))
