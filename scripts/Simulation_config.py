# This file contains the configuration for an automated simulation run.
#

from collections import namedtuple

class DoPretraining():
    YES = True      # <= Baseline
    NO = False
    
class ModelSize():
    SMALL = 1
    MEDIUM = 2      # <= Baseline
    LARGE = 3

class Aggregation_Count():
    _1_HOUSEHOLD    = 'data/london_loadprofiles_1household_each.pkl'
    _10_HOUSEHOLDS  = 'data/london_loadprofiles_10households_each.pkl'
    _37_HOUSEHOLDS  = 'data/london_loadprofiles_37households_each.pkl'
    _74_HOUSEHOLDS  = 'data/london_loadprofiles.pkl'                        # <= Baseline

class NrOfComunities():
    _10 = 10
    _20 = 20      # <= Baseline
    
class TrainingHistory():
    _2_MONTH = 1
    _6_MONTH = 2
    _12_MONTH = 3
    _18_MONTH = 4      # <= Baseline

class InputSequenceLength():
    _48_HOURS = 1      # <= Baseline
    _7_DAYS = 2
    
class DoTransferLearning():
    YES = True      # <= Baseline
    NO = False
    
class UsedModels():
    ALL = ('xLSTM', 'LSTM', 'Transformer', 'KNN', 'PersistencePrediction')

# Collect all classes from above
all_above_classes = [name for name, obj in globals().items() if isinstance(obj, type)]
Config_of_one_simulation = namedtuple('Config_of_one_simulation', all_above_classes)

# Create test config for all simulation runs
configs = [
    
    # Baseline:
    Config_of_one_simulation(DoPretraining.YES, ModelSize.MEDIUM, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20,
                             TrainingHistory._18_MONTH, InputSequenceLength._48_HOURS, DoTransferLearning.YES, UsedModels.ALL),
    
    # Variations:    
    # Config_of_one_simulation(DoPretraining.YES, ModelSize.MEDIUM, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._10,
    #                          TrainingHistory._18_MONTH, InputSequenceLength._48_HOURS, TransferLearning.YES),
]

print(configs[0].DoPretraining == DoPretraining.YES)
