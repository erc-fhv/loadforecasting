# This file contains the configuration for an automated simulation run.
#

from collections import namedtuple

class DoPretraining():
    YES = True      # <= Baseline
    NO = False

class DoTransferLearning():
    YES = True      # <= Baseline
    NO = False

class ModelSize():
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"      # <= Baseline
    LARGE = "LARGE"

class Aggregation_Count():
    _1_HOUSEHOLD    = 'data/london_loadprofiles_1household_each.pkl'
    _10_HOUSEHOLDS  = 'data/london_loadprofiles_10households_each.pkl'
    _37_HOUSEHOLDS  = 'data/london_loadprofiles_37households_each.pkl'
    _74_HOUSEHOLDS  = 'data/london_loadprofiles_74households_each.pkl'  # <= Baseline

class NrOfComunities():
    _10 = 10
    _20 = 2      # <= Baseline

class TrainingHistory():
    _2_MONTH = 61
    _6_MONTH = 183
    _12_MONTH = 365
    _15_MONTH = 466      # <= Baseline

class InputSequenceLength():
    _48_HOURS = 48  # <= Baseline
    _7_DAYS = 24*7

class UsedModels():
    # ALL = ('xLSTM', 'LSTM', 'Transformer', 'KNN', 'PersistencePrediction')
    ALL = ('xLSTM',)

class Epochs():
    DEFAULT = 100

# Define all possible run settings
run_settings = ['modelSize', 'doPretraining', 'doTransferLearning', 'aggregation_Count', 'nrOfComunities', 
                'trainingHistory', 'inputSequenceLength', 'usedModels', 'epochs']
Config_of_one_run = namedtuple('Config_of_one_run', run_settings)

# Create test config for all simulation runs
configs = [
    Config_of_one_run(ModelSize.SMALL, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS,
            NrOfComunities._20, TrainingHistory._15_MONTH, InputSequenceLength._48_HOURS, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS,
            NrOfComunities._20, TrainingHistory._15_MONTH, InputSequenceLength._48_HOURS, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.LARGE, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS,
            NrOfComunities._20, TrainingHistory._15_MONTH, InputSequenceLength._48_HOURS, UsedModels.ALL, Epochs.DEFAULT),
]
