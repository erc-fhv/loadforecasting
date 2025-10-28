"""
This file contains the configuration for an automated simulation run.
"""

from collections import namedtuple

class DoPretraining():
    """"Whether to do pretraining."""
    YES = True      # <= Baseline
    NO = False

class DoTransferLearning():
    """"Wheather to transfer learned weights from pretraining."""
    YES = True      # <= Baseline
    NO = False

class Model():
    """Different machine learning model sizes."""
    SIZE_0k1 = "0.1k"
    SIZE_0k2 = "0.2k"
    SIZE_0k5 = "0.5k"
    SIZE_1k = "1k"
    SIZE_2k = "2k"
    SIZE_5k = "5k"
    SIZE_10k = "10k"
    SIZE_20k = "20k"
    SIZE_40k = "40k"      # <= Baseline
    SIZE_80k = "80k"

class AggregationCount():
    """Different numbers of sizes of energy communities ."""
    ONE_HOUSEHOLD    = (1,   '../../data/london_loadprofiles_1households_each.pkl')
    TWO_HOUSEHOLDS   = (2,   '../../data/london_loadprofiles_2households_each.pkl')
    TEN_HOUSEHOLDS  = (10,  '../../data/london_loadprofiles_10households_each.pkl')
    FIFTY_HOUSEHOLDS  = (50,  '../../data/london_loadprofiles_50households_each.pkl')  # <= Baseline
    HUNDRED_HOUSEHOLDS = (100, '../../data/london_loadprofiles_100households_each.pkl')

class NrOfComunities():
    """Different numbers of communities to train on."""
    TEN = 10
    TWENTY = 20      # <= Baseline

class TrainSet1():
    """Different sizes of the past training history."""
    PAST_0_MONTH = 0
    PAST_2_MONTH = 61
    PAST_3_MONTH = 92
    PAST_4_MONTH = 122
    PAST_6_MONTH = 183
    PAST_9_MONTH = 275
    PAST_12_MONTH = 365     # <= Baseline
    PAST_15_MONTH = 447

class TestSize():
    """Different sizes of the test set."""
    NEXT_3_MONTH = 92   # <= Baseline
    NEXT_4_MONTH = 131

class DevSize():
    """Different sizes of the dev set."""
    NEXT_0_MONTH = 0
    NEXT_2_MONTH = 58   # <= Baseline

class TrainSet2():
    """Optionally: Use future data for training."""
    FUTURE_0_MONTH = 0     # <= Baseline
    FUTURE_3_MONTH = 92
    FUTURE_6_MONTH = 183
    FUTURE_9_MONTH = 275

class UsedModels():
    """Different models to be used for the load forecasting."""
    ALL = ('Perfect', 'Knn', 'Persistence', 'xLstm', 'Lstm', 'Transformer')

class Epochs():
    """Different numbers of training epochs."""
    SMOKE_TEST = 1
    DEFAULT = 100     # <= Baseline

# Define all possible run settings
run_settings = ['model_size', 'do_pretraining', 'do_transfer_learning', 'aggregation_count',
                'nr_of_comunities', 'training_history', 'test_size', 'train_set_future', 'dev_size',
                'used_models', 'epochs']
ConfigOfOneRun = namedtuple('Config_of_one_run', run_settings)


###################################################################################################
# Create test config for all simulation runs
###################################################################################################
configs = [
    # Baseline
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),

    # Vary the model sizes
    ConfigOfOneRun(Model.SIZE_0k1, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_0k2, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_0k5, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_1k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_2k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_10k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_20k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_40k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_80k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),

    # Vary the community sizes
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        ONE_HOUSEHOLD, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        TWO_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        TEN_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        HUNDRED_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),

    # Vary the train set size
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_2_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_4_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_6_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_9_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_15_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),

   # Vary the tested quartals
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_0_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_9_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_3_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_6_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_6_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_3_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_9_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
            
        
    # Without transfer learning:
    #
    
    # Baseline
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.NO, DoTransferLearning.NO, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),

    # Vary the train set size
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.NO, DoTransferLearning.NO, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_2_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.NO, DoTransferLearning.NO, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_4_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.NO, DoTransferLearning.NO, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_6_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.NO, DoTransferLearning.NO, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_9_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.NO, DoTransferLearning.NO, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_15_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.DEFAULT),

]
# #################################################################################################


###################################################################################################
# Simulation config, used only for testing on the Continous Integration (CI).
###################################################################################################
configs_for_the_ci = [
    # Test the Baseline
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.YES, DoTransferLearning.YES, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.SMOKE_TEST),

    # Test the Baseline, but without transfer learning
    ConfigOfOneRun(Model.SIZE_5k, DoPretraining.NO, DoTransferLearning.NO, AggregationCount.
        FIFTY_HOUSEHOLDS, NrOfComunities.TWENTY, TrainSet1.PAST_12_MONTH, TestSize.NEXT_3_MONTH,
        TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH, UsedModels.ALL, Epochs.SMOKE_TEST),
]
# #################################################################################################
