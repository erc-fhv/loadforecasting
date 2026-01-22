from typing import NamedTuple

class DataSplitType(NamedTuple):
    """
    Data split in days for load forecasting datasets.

    Attributes:
        train_set_1 (int): Number of historic (=past) training days. If set to -1, use
            all available historic data.
        dev_set (int): Number of future validation days.
        test_set (int): Number of future test days.
        train_set_2 (int): Number of future training days, after the test and dev sets.
        padding (int): Number of unused padding days at the end of the dataset.
    """
    train_set_1: int
    dev_set: int
    test_set: int
    train_set_2: int
    pad: int
