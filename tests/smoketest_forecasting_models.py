from loadforecasting_models import Model
import torch

def test_model_prediction():

    for model_type in ['KNN', 'LSTM', 'Transformer', 'xLSTM',]:

        print(f'Test the {model_type} model.')
        
        X_train = torch.randn(365, 24, 10)
        Y_train = torch.randn(365, 24, 1)
        model = Model(model_type, model_size='5k', num_of_features=X_train.shape[2])
        model.train_model(X_train, Y_train, pretrain_now=False, finetune_now=False, epochs=1, verbose=0)  # epochs=1 für schnelleren Test
    
        X_test = torch.randn(90, 24, 10)
        Y_pred = model.predict(X_test)
    
        assert Y_pred.shape == (90, 24, 1)
