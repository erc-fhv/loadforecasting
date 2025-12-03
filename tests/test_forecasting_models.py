from typing import Type, Union
import numpy as np
import torch
from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer

def test_if_models_are_running():
    """
    Test if creating, training and prediction runs without error,
    i.e. no accuracy is checked here.
    """

    for model_class in [Knn, Lstm, Transformer, xLstm, Persistence]:

        print(f'Test the {model_class.__name__} model.')

        x_train = torch.randn(365, 24, 10)
        y_train = torch.randn(365, 24, 1)
        normalizer = Normalizer()

        if model_class == Knn:
            my_model = Knn(k=40, weights = 'distance', normalizer=None)
            my_model.train_model(x_train, y_train)

        elif model_class == Persistence:
            my_model = Persistence(lagged_load_feature=3, normalizer=normalizer)
            my_model.train_model()

        else:   # Machine Learning Models

            model_class: Type[Union[Lstm, Transformer, xLstm]]   # Help the type checker
            my_model = model_class(model_size='5k', normalizer=normalizer)
            my_model.train_model(x_train, y_train, verbose=1,
                epochs=1) # epochs=1 for faster tests

        x_test = torch.randn(90, 24, 10)
        y_test = torch.randn(90, 24, 1)
        y_pred = my_model.evaluate(x_test, y_test)

        y_pred = my_model.predict(x_test)
        assert y_pred.shape == (90, 24, 1)

def test_if_models_are_running_w_numpy():
    """
    Do the same as test_if_models_are_running(), but with numpy arrays.
    """

    for model_class in [Knn, Lstm, Transformer, xLstm, Persistence]:

        print(f'Test the {model_class.__name__} model.')

        x_train = np.random.randn(365, 24, 10)
        y_train = np.random.randn(365, 24, 1)
        normalizer = Normalizer()

        if model_class == Knn:
            my_model = Knn(k=40, weights = 'distance', normalizer=None)
            my_model.train_model(x_train, y_train)

        elif model_class == Persistence:
            my_model = Persistence(lagged_load_feature=3, normalizer=normalizer)
            my_model.train_model()

        else:   # Machine Learning Models

            model_class: Type[Union[Lstm, Transformer, xLstm]]   # Help the type checker
            my_model = model_class(model_size='5k', normalizer=normalizer)
            my_model.train_model(x_train, y_train, verbose=1,
                epochs=1) # epochs=1 for faster tests

        x_test = np.random.randn(90, 24, 10)
        y_test = np.random.randn(90, 24, 1)

        y_pred = my_model.evaluate(x_test, y_test)

        y_pred = my_model.predict(x_test)
        assert y_pred.shape == (90, 24, 1)

def test_models_simple_prediction():
    """
    Test if models can learn and predict a simple sine wave and 
    produce the correct output shape.
    """

    for model_class in [Lstm, Transformer, xLstm, Knn]:

        for seq_len in [24, 24*4, 24*7]:

            print(f'Testing simple prediction with {model_class.__name__}' + \
                f' model and seq_len={seq_len}.')

            # Input data: Sine Values
            batch_size = 40
            n_features = 3
            full_periods = 5
            x_vals = torch.linspace(0, 1, batch_size * seq_len).reshape(batch_size, seq_len, 1)
            x_vals = torch.sin(full_periods * 2 * torch.pi * x_vals)
            noise = 0.1 * torch.randn(batch_size, seq_len, 1)
            y_train = 2 * x_vals + 1 + noise

            # Add noise features
            x_train = torch.cat([x_vals, torch.randn(size=(batch_size, seq_len, n_features - 1))],
                dim=2)

            # Normalize
            normalizer = Normalizer()
            x_train = normalizer.normalize_x(x_train, training=True)
            y_train = normalizer.normalize_y(y_train, training=True)

            # Train the model
            if model_class == Knn:
                my_model = Knn(k=40, weights = 'distance', normalizer=None)
                my_model.train_model(x_train, y_train)
            else:
                model_class: Type[Union[Lstm, Transformer, xLstm]]   # Help the type checker
                my_model = model_class('5k', normalizer=normalizer)
                my_model.train_model(x_train, y_train, verbose=1, epochs=100)

            # Test input data: Sine Values
            x_test_vals = torch.linspace(1, 2, batch_size * seq_len).reshape(batch_size, seq_len, 1)
            x_test_vals = torch.sin(full_periods * 2 * torch.pi * x_test_vals)
            y_test_true = 2 * x_test_vals + 1
            x_test = torch.cat([x_test_vals, torch.randn(batch_size, seq_len,
                n_features - 1)], dim=2)
            x_test = normalizer.normalize_x(x_test, training=False)

            # Normalize and predict
            y_pred = my_model.predict(x_test)
            y_pred = normalizer.de_normalize_y(y_pred)

            # 1. Check output shape
            assert y_pred.shape == (batch_size, seq_len, 1), \
                f"Unexpected output shape: {y_pred.shape}"

            # 2. Check relative accuracy
            treshold = 20.0 # [%]
            mae = torch.mean(torch.abs(torch.Tensor(y_pred - y_test_true)))
            mean_true = torch.mean(torch.abs(torch.Tensor(y_test_true)))
            n_mae = mae / mean_true
            print(f"Relative error for {model_class.__name__}: {n_mae:.2%}")
            assert n_mae < treshold/100., f"{model_class.__name__} too inaccurate: {n_mae:.2%}"

    print("All models passed the simple prediction test.")


def test_models_simple_prediction_w_optuna():
    """
    Similar test as test_models_simple_prediction(), but this time with hyperparameter
    optimization via Optuna.
    """

    for model_class in [Lstm, Transformer, xLstm]:

        print(f'Testing simple prediction with {model_class.__name__}.')

        # Input data: Sine Values
        batch_size = 40
        n_features = 3
        full_periods = 5
        seq_len = 24
        x_vals = torch.linspace(0, 1, batch_size * seq_len).reshape(batch_size, seq_len, 1)
        x_vals = torch.sin(full_periods * 2 * torch.pi * x_vals)
        noise = 0.1 * torch.randn(batch_size, seq_len, 1)
        y_train = 2 * x_vals + 1 + noise

        # Add noise features
        x_train = torch.cat([x_vals, torch.randn(size=(batch_size, seq_len, n_features - 1))],
            dim=2)

        # Normalize
        normalizer = Normalizer()
        x_train = normalizer.normalize_x(x_train, training=True)
        y_train = normalizer.normalize_y(y_train, training=True)

        # Train the model
        model_class: Type[Union[Lstm, Transformer, xLstm]]   # Help the type checker
        my_model = model_class(normalizer=normalizer)
        my_model.train_model_auto(x_train, y_train, n_trials=1, k_folds=1, verbose=0)

        # Test input data: Sine Values
        x_test_vals = torch.linspace(1, 2, batch_size * seq_len).reshape(batch_size, seq_len, 1)
        x_test_vals = torch.sin(full_periods * 2 * torch.pi * x_test_vals)
        y_test_true = 2 * x_test_vals + 1
        x_test = torch.cat([x_test_vals, torch.randn(batch_size, seq_len,
            n_features - 1)], dim=2)
        x_test = normalizer.normalize_x(x_test, training=False)

        # Normalize and predict
        y_pred = my_model.predict(x_test)
        y_pred = normalizer.de_normalize_y(y_pred)

        # Check output shape
        assert y_pred.shape == (batch_size, seq_len, 1), \
            f"Unexpected output shape: {y_pred.shape}"

    print("All models worked with optuna.")

def test_transfer_learning():
    """
    Test if transfer learning works
    """

    # Define a simple linear relationship: y = 2 * x + 1
    batch_size = 40
    n_features = 3
    seq_len = 24

    # Input data: Sine Values
    full_periods = 5
    x_vals = torch.linspace(0, 1, batch_size * seq_len).reshape(batch_size, seq_len, 1)
    x_vals = torch.sin(full_periods * 2 * torch.pi * x_vals)
    noise = 0.1 * torch.randn(batch_size, seq_len, 1)
    y_train = 2 * x_vals + 1 + noise

    # Add noise features
    x_train = torch.cat([x_vals, torch.randn(size=(batch_size, seq_len, n_features - 1))],
        dim=2)

    # Normalize
    normalizer = Normalizer()
    x_train = normalizer.normalize_x(x_train, training=True)
    y_train = normalizer.normalize_y(y_train, training=True)

    pretrain_model = Transformer('5k', normalizer=normalizer)
    pretrain_model.train_model(
        x_train,
        y_train,
        pretrain_now = True,
        finetune_now = False,
        verbose=1,
        epochs=100, # deteiled pre-training
        )

    my_model = Transformer('5k', normalizer=normalizer)
    my_model.train_model(
        x_train,
        y_train,
        pretrain_now = False,
        finetune_now = True,
        verbose=1,
        epochs=0, # skip fine-tuning
        )

    # Test input data: Sine Values
    x_test_vals = torch.linspace(1, 2, batch_size * seq_len).reshape(batch_size, seq_len, 1)
    x_test_vals = torch.sin(full_periods * 2 * torch.pi * x_test_vals)
    y_test_true = 2 * x_test_vals + 1
    x_test = torch.cat([x_test_vals, torch.randn(batch_size, seq_len,
        n_features - 1)], dim=2)
    x_test = normalizer.normalize_x(x_test, training=False)

    # Normalize and predict
    y_pred = my_model.predict(x_test)
    y_pred = normalizer.de_normalize_y(y_pred)

    # 1. Check output shape
    assert y_pred.shape == (batch_size, seq_len, 1), \
        f"Unexpected output shape: {y_pred.shape}"

    # 2. Check relative accuracy
    treshold = 20.0 # [%]
    mae = torch.mean(torch.abs(torch.Tensor(y_pred - y_test_true)))
    mean_true = torch.mean(torch.abs(y_test_true))
    n_mae = mae / mean_true
    print(f"Relative error : {n_mae:.2%}")
    assert n_mae < treshold/100., f"Model too inaccurate: {n_mae:.2%}"

if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            print(f"\nRunning {name}()")
            func()
