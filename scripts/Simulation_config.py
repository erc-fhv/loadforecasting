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
    _20 = 20      # <= Baseline

class TrainingHistory():
    _2_MONTH = 61
    _6_MONTH = 183
    _12_MONTH = 365
    _15_MONTH = 466      # <= Baseline

class ModelInputHistory():
    _1_DAY = 24
    _2_DAYS = 24*2  # <= Baseline
    _7_DAYS = 24*8
    _21_DAYS = 24*22

class MeasurementDelay():
    """ Delay in hours to get the most recent measured smartmeter data from the DSO. """
    _1_HOUR = 1
    _1_DAY = 24  # <= Baseline
    _7_DAYS = 24*7

class UsedModels():
    ALL = ('KNN', 'PersistencePrediction', 'xLSTM', 'LSTM', 'Transformer', )

class Epochs():
    SMOKE_TEST = 1
    DEFAULT = 100     # <= Baseline

# Define all possible run settings
run_settings = ['modelSize', 'doPretraining', 'doTransferLearning', 'aggregation_Count', 'nrOfComunities', 
                'trainingHistory', 'modelInputHistory', 'measurementDelay', 'usedModels', 'epochs']
Config_of_one_run = namedtuple('Config_of_one_run', run_settings)


#########################################################################################################################
# Create test config for all simulation runs
#########################################################################################################################
configs = [
    # Baseline
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    
    # Vary the model sizes
    Config_of_one_run(ModelSize.SMALL, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.LARGE, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
        
    # Vary the community sizes
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._37_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._10_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._1_HOUSEHOLD, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),

    # Vary the train set size
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._12_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._6_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._2_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    
    # Vary the input length
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._1_DAY, MeasurementDelay._1_HOUR, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._7_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._21_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.YES, DoTransferLearning.YES, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._21_DAYS, MeasurementDelay._1_HOUR, UsedModels.ALL, Epochs.DEFAULT),



    # without transfer learning:
    #
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    
    # Vary the model sizes
    Config_of_one_run(ModelSize.SMALL, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.LARGE, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
        
    # Vary the community sizes
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._37_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._10_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._1_HOUSEHOLD, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),

    # Vary the train set size
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._12_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._6_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._2_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    
    # Vary the input length
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._1_DAY, MeasurementDelay._1_HOUR, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._2_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._7_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._21_DAYS, MeasurementDelay._1_DAY, UsedModels.ALL, Epochs.DEFAULT),
    Config_of_one_run(ModelSize.MEDIUM, DoPretraining.NO, DoTransferLearning.NO, Aggregation_Count._74_HOUSEHOLDS, NrOfComunities._20, 
            TrainingHistory._15_MONTH, ModelInputHistory._21_DAYS, MeasurementDelay._1_HOUR, UsedModels.ALL, Epochs.DEFAULT),
]
#########################################################################################################################

