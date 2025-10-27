import torch
from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer

def test_model_prediction():
    """Test if models can be trained and make predictions with dummy data. """

    for model_class in [Knn, Lstm, Transformer, xLstm, Persistence]:

        print(f'Test the {model_class.__name__} model.')

        x_train = torch.randn(365, 24, 10)
        y_train = torch.randn(365, 24, 1)
        normalizer = Normalizer()

        if model_class == Knn:
            my_model = Knn(k=40, weights = 'distance', normalizer=None)
            history = my_model.train_model(x_train, y_train)

        elif model_class == Persistence:
            my_model = Persistence(lagged_load_feature=3, normalizer=normalizer)
            history = my_model.train_model()

        else:   # Machine Learning Models

            my_model = model_class('5k', x_train.size(2), normalizer=normalizer)
            history = my_model.train_model(x_train, y_train, verbose=2,
                epochs=1) # epochs=1 für schnellen Test

        x_test = torch.randn(90, 24, 10)
        y_pred = my_model.predict(x_test)

        assert y_pred.shape == (90, 24, 1)
