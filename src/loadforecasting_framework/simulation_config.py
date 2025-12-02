"""
This file contains the configuration for an automated simulation run.
"""

from typing import NamedTuple
from .interfaces import DataSplitType

class DoTransferLearning():
    """"Whether to transfer learned weights from pretraining."""
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

class NrOfCommunities():
    """Different numbers of communities to train on."""
    TEN = 10
    TWENTY = 20      # <= Baseline

class DataSplit():
    """Different train/test/dev/etc. data splits."""

    # Baseline:
    TRAIN_12_MONTH = DataSplitType(train_set_1=365, dev_set=0, test_set=92, train_set_2=0, pad=58)

    # Vary the historic training set size:
    TRAIN_2_MONTH = DataSplitType(train_set_1=61, dev_set=0, test_set=92, train_set_2=0, pad=58)
    TRAIN_4_MONTH = DataSplitType(train_set_1=122, dev_set=0, test_set=92, train_set_2=0, pad=58)
    TRAIN_6_MONTH = DataSplitType(train_set_1=183, dev_set=0, test_set=92, train_set_2=0, pad=58)
    TRAIN_9_MONTH = DataSplitType(train_set_1=275, dev_set=0, test_set=92, train_set_2=0, pad=58)
    TRAIN_15_MONTH = DataSplitType(train_set_1=447, dev_set=0, test_set=92, train_set_2=0, pad=58)

    # Vary the tested quartals:
    TEST_Q1 = DataSplitType(train_set_1=0, dev_set=0, test_set=92, train_set_2=275, pad=58)
    TEST_Q2 = DataSplitType(train_set_1=92, dev_set=0, test_set=92, train_set_2=183, pad=58)
    TEST_Q3 = DataSplitType(train_set_1=183, dev_set=0, test_set=92, train_set_2=92, pad=58)
    TEST_Q4 = DataSplitType(train_set_1=275, dev_set=0, test_set=92, train_set_2=0, pad=58)


# TrainSet1.PAST_2_MONTH, TestSize.NEXT_3_MONTH,TrainSet2.FUTURE_0_MONTH, DevSize.NEXT_2_MONTH

class UsedModels():
    """Different models to be used for the load forecasting."""
    ALL = ('Perfect', 'Knn', 'Persistence', 'Xlstm', 'Lstm', 'Transformer')

class Epochs():
    """Different numbers of training epochs."""
    SMOKE_TEST = 1
    DEFAULT = 100     # <= Baseline

class ConfigOfOneRun(NamedTuple):
    """ Class that defines all possible run settings """
    model_size: str
    do_transfer_learning: bool
    nr_of_communities: int
    used_models: tuple
    aggregation_count: tuple
    data_split: DataSplitType
    epochs: int

###################################################################################################
# Create test config for all simulation runs
###################################################################################################
configs = [
    # Baseline
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),

    # Vary the model sizes
    ConfigOfOneRun(Model.SIZE_0k2, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_0k1, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_0k5, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_1k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_2k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_10k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_20k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_40k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_80k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),

    # Vary the community sizes
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.ONE_HOUSEHOLD, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.TWO_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.TEN_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.HUNDRED_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),

    # Vary the train set size
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_2_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_4_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_6_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_9_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_15_MONTH, Epochs.DEFAULT),

   # Vary the tested quarters
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
         AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TEST_Q1, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TEST_Q2, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TEST_Q3, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TEST_Q4, Epochs.DEFAULT),


    # Without transfer learning:
    #

    # Baseline
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.NO, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.DEFAULT),

    # Vary the train set size
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.NO, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_2_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.NO, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_4_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.NO, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_6_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.NO, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_9_MONTH, Epochs.DEFAULT),
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.NO, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_15_MONTH, Epochs.DEFAULT),
]
# #################################################################################################


###################################################################################################
# Simulation config, used only for testing on the Continous Integration (CI).
###################################################################################################
configs_for_the_ci = [
    # Test the Baseline
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.YES, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.SMOKE_TEST),

    # Test the Baseline, but without transfer learning
    ConfigOfOneRun(Model.SIZE_5k, DoTransferLearning.NO, NrOfCommunities.TWENTY, UsedModels.ALL,
        AggregationCount.FIFTY_HOUSEHOLDS, DataSplit.TRAIN_12_MONTH, Epochs.SMOKE_TEST),
]
# #################################################################################################
