# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

# for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Calculate the sMAPE
# @keras.saving.register_keras_serializable()
# def sMAPE(y_true, y_pred):
#     epsilon = 0.0001  # avoid division by zero
#     diff = K.abs(y_true - y_pred) / K.clip((K.abs(y_true) + K.abs(y_pred)) / 2, epsilon, None)
#     return 100. * K.mean(diff, axis=-1)

# # Calculate the RMSE
# @keras.saving.register_keras_serializable()
# def RMSE(y_true, y_pred):
#     return K.sqrt(keras.losses.mean_squared_error(y_true, y_pred))

class Optimizer:

    def __init__(
                self, 
                maxEpochs,
                set_learning_rates = [0.001],
                miniBatchSize=None,
                loss_type = 'MSE'
                ):
                
        self.loss_type = loss_type
        self.set_learning_rates = set_learning_rates
        self.maxEpochs = maxEpochs
        self.miniBatchSize = miniBatchSize
        self.optimizer = Adam

    def get_optimizer(self):
        return self.optimizer(
            learning_rate=self.set_learning_rates[0]
        )

    def get_early_stopping_callbacks(self):
        return keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    def get_lr_callback(self):

        lr_switching_points = np.flip(np.linspace(1, 0, len(self.set_learning_rates), endpoint=False))

        # Define a learning rate schedule function
        def lr_schedule(epoch):            

            # Iterate linearly through the set_learning_rates list, accord to the progress
            progress = epoch / self.maxEpochs
            
            for i, boundary in enumerate(lr_switching_points):
                if progress < boundary:
                    return self.set_learning_rates[i]

            # If progress is greater than or equal to 1, use the last learning rate
            return self.set_learning_rates[-1]

        # Create LearningRateScheduler callback
        lr_scheduler = LearningRateScheduler(lr_schedule)

        return lr_scheduler
    
    def get_model_checkpoint_callback(self):
        checkpoint = ModelCheckpoint(filepath='checkpoints/best_model.h5', 
                                    monitor='val_loss', 
                                    save_best_only=True, 
                                    save_weights_only=False, 
                                    mode='min',
                                    verbose=1)
        return checkpoint
    
    def get_all_callbacks(self):

        callbacks = [
                        self.get_lr_callback(), 
                        # self.get_early_stopping_callbacks()
                        # self.get_model_checkpoint_callback(), 
                    ]
        return callbacks

# Define the LSTM model
#
class Model:
    def __init__(self, optimizer, reg_strength, chosen_model='LSTM', shape=None):
        self.optimizer = optimizer
        
        if chosen_model == 'LSTM':
            model_layers = [
                keras.layers.Bidirectional(keras.layers.LSTM(units=10, return_sequences=True, kernel_regularizer=regularizers.l2(reg_strength))),
                keras.layers.BatchNormalization(),
                keras.layers.Bidirectional(keras.layers.LSTM(units=30, return_sequences=True, kernel_regularizer=regularizers.l2(reg_strength))),
                keras.layers.BatchNormalization(),
                keras.layers.Lambda(lambda x: x[:, -24:, :]),
                keras.layers.Dense(units=10, activation='relu', kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=10, activation='relu', kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=1, activation='linear'),
            ]
        elif chosen_model == 'LSTM_deep':
            model_layers = [
                keras.layers.Bidirectional(keras.layers.LSTM(units=10, return_sequences=True, kernel_regularizer=regularizers.l2(reg_strength))),
                keras.layers.BatchNormalization(),
                keras.layers.Lambda(lambda x: x[:, -24:, :]),
                keras.layers.Dense(units=20, kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=20, kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=20, kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=20, kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=20, kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=20, kernel_regularizer=regularizers.l2(reg_strength)),
                keras.layers.Dense(units=1, activation='linear'),
            ]
        else:
            raise ValueError("Unknown model choosen.")
        
        if shape != None:
            input_layer = keras.Input(shape=shape)
            model_layers.insert(0, input_layer)
        
        self.model = keras.Sequential(model_layers)

    def compile(self, metrics=[]):  # RMSE, sMAPE, 
        
        self.model.compile(optimizer=self.optimizer.get_optimizer(), \
                           loss=self.optimizer.loss_type,
                           metrics=metrics
                           )

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def predict(self, X, verbose='auto'):
        return self.model.predict(X, verbose=verbose)

