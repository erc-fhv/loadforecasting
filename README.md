# Framework for Short-Term Load Forecasting

## Summary

This repository provides a flexible and modular framework for short-term load forecasting (STLF), 
suitable for both research and real-world applications. It supports:

- Deep learning models: Transformer, Lstm, xLstm
- Baseline model: Knn, Persistence, Perfect
- Full pipeline for training, evaluation, and visualization
- Reproducibility of all experiments from the following paper

## Related Paper

More information about the models and the framework can be found in the following paper:  

> Moosbrugger et al. (2025), *Load Forecasting for Households and Energy Communities: Are Deep 
Learning Models Worth the Effort?*, [arXiv:2501.05000](https://arxiv.org/abs/2501.05000)



## Folder Structure

The repository is organized as follows:

```
├── data/                             # Preprocessed smart meter data
│   ├── *.pkl                         # Load profiles (varying community sizes per file)
│   ├── *.ipynb                       # Loadprofile preprocessing script
│
├── envs/                             # Conda environments
│   ├── env_linux.yml                 # Reproducible environment for the paper
│   └── env_from_nxai.yml             # Environment from xLstm authors
│   
├── src/      
│   ├── loadforecasting_models/       # All forecasting models
|   │   ├── pyproject.toml            # Description of the 'loadforecasting_models' package
│   │   └── *.py                      # Implementations of deep learning & baseline models
│   │
│   └── loadforecasting_framework/    # Evaluation framework and visualization
│       ├── simulation_config.py      # Config file for simulation runs
│       ├── model_trainer.py          # Training and evaluation loop
│       ├── data_preprocessr.py       # Data formatting and preprocessing
│       ├── paper_illustration.ipynb  # Plots and tables for the paper
│       └── case_study/               # Energy community MILP optimization
│
├── tests/                            # Automated unit and integration tests
│   └── *.py
|
├── pyproject.toml                    # Description of the 'loadforecasting_framework' package
├── LICENCE
└── README.md
```


## Code Structure

The main parts of the evaluation framework are connected as follows:

```

+-----------------------------+           +------------------------------+
| data                        |           | data_preprocessor            |
|-----------------------------|           |------------------------------|
| # Weather, load, standard-  +-----------+ transform_data()              |
|   load, and holidays.       |           | # Preprocesses the data      |
+-----------------------------+           +------------+-----------------+
                                                       |
                                                       |
+-----------------------------+           +------------+-----------------+
| simulation_config           |           | model_trainer                |
|-----------------------------|           |------------------------------|
| configs: list               |           | run()                        |
| # Parameterize the run      +-----------+ # Trains all models          |
| # loop.                     |           | # accord to the config.      |
+-----------------------------+           +------------+-----------------+
                                                       |
                                                       |
+-----------------------------+                        |
| normalizer                  |                        |
|-----------------------------|                        |
| normalize()                 +------------------------+
| de_normalize()              |                        |
+-----------------------------+                        |
                                                       |
                                                       |
       +-----------------+-------------+---------------+-----------------+
       |                 |             |               |                 |
       |                 |             |               |                 |
+------+------+ +--------+----+ +------+------+ +------+------+ +--------+----+
| Knn         | | Persistence | | xLstm       | | Lstm        | | Transformer |
|             | |             | |             | |             | |             |
|-------------| |-------------| |-------------| |-------------| |-------------|
|train_model()| |train_model()| |train_model()| |train_model()| |train_model()|
|predict()    | |predict()    | |predict()    | |predict()    | |predict()    |
|evaluate()   | |evaluate()   | |evaluate()   | |evaluate()   | |evaluate()   |
+------+------+ +--------+----+ +------+------+ +------+------+ +--------+----+
       |                 |             |               |                 |
       |                 |             |               |                 |
       +-----------------+-------------+---------------+-----------------+
                                                       |
                                                       |
                                          +------------+-----------------+
                                          | helpers                      |
                                          |------------------------------|
                                          | # Common (e.g. pytorch)      |
                                          | # models code.               |
                                          +------------------------------+

```

## Reusing only the Forecasting Models

Our forecasting models can be easily reused in other applications as shown below.

1. Install the package:
    ```bash
    pip install loadforecasting_models
    ```

2. Use of the machine learning models in Python:

    ```python
    from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer
    import torch

    # ------------------------------------------------------------------------------
    # Define dataset parameters
    # ------------------------------------------------------------------------------
    features = 10          # Number of input features
    seq_len = 24           # Sequence length (e.g., 24 hours)
    batches_train = 365    # Number of training samples (e.g., one year of daily sequences)
    batches_test = 90      # Number of test samples

    # ------------------------------------------------------------------------------
    # Prepare training data
    # ------------------------------------------------------------------------------
    normalizer = Normalizer()

    # Generate dummy training data (replace with your own)
    x_train = torch.randn(batches_train, seq_len, features)   # Shape: (batches, seq_len, features)
    y_train = torch.randn(batches_train, seq_len, 1)          # Shape: (batches, seq_len, 1)

    # Normalize data
    x_train = normalizer.normalize_x(x_train, training=True)
    y_train = normalizer.normalize_y(y_train, training=True)

    # ------------------------------------------------------------------------------
    # Initialize and train the model
    # ------------------------------------------------------------------------------

    # Available ML models: Transformer, LSTM, xLSTM
    myModel = Transformer(model_size='5k', normalizer=normalizer)
    myModel.train_model(x_train, y_train, epochs=100, verbose=0)

    # ------------------------------------------------------------------------------
    # Make predictions
    # ------------------------------------------------------------------------------
    x_test = torch.randn(batches_test, seq_len, features)
    x_test = normalizer.normalize_x(x_test, training=False)
    y_pred = myModel.predict(x_test)
    y_pred = normalizer.de_normalize_y(y_pred)

    print('\nOutput shape:', y_pred.shape)
    ```

3. Use of *non-machine-learning models*. For example the KNN model:

    ```python
    from loadforecasting_models import Knn, Lstm, Transformer, xLstm, Persistence, Normalizer
    import torch
    
    # Same setup as above
    # ...
    myModel = Knn(k=40, weights='distance', normalizer=normalizer)
    myModel.train_model(x_train, y_train)
    # ...
    ```

## Reusing the Forecasting Models AND the Preprocessing Module

1. Install the packages:
    ```bash
    pip install loadforecasting_models
    pip install -e git+https://github.com/erc-fhv/loadforecasting.git
    ```

2. Short example on how to use the models and preprocessing:

    ```python
    import pandas as pd
    import numpy as np

    from loadforecasting_models import Normalizer, Transformer
    from loadforecasting_framework import DataPreprocessor, ModelTrainer, DataSplitType

    # Read in a load profile, weather data and holiday with datetime index
    #
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2024-01-01')
    timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')

    df_load = pd.DataFrame({
        'timestamp': timestamps,
        'load': np.random.rand(len(timestamps)) * 1000
    }).set_index('timestamp')

    weather_data = pd.DataFrame({
        'date': timestamps,
        'precipitation': np.random.rand(len(timestamps)),
        'cloud_cover': np.random.rand(len(timestamps)) * 100,
        'global_tilted_irradiance': np.random.rand(len(timestamps)) * 1000,
    }).set_index('date')

    holidays = ModelTrainer().load_holidays(start_date, end_date, country="AT", subdiv="Vorarlberg")

    # Transform inputs
    #
    normalizer = Normalizer()
    pre = DataPreprocessor(
        normalizer=normalizer,
        add_lagged_profiles=(7, 14, 21),
        data_split = DataSplitType(
            train_set_1 = int(len(timestamps)*0.8), # 80% historic training data
            test_set=int(len(timestamps)*0.2),  # 20% future test data
            dev_set=0, train_set_2=0, pad=0),
    )

    x, y = pre.transform_data(
        power_profile=df_load["load"],
        weather_data=weather_data,
        public_holidays=holidays
    )

    # Train the model
    #
    model = Transformer("5k", normalizer = normalizer)
    model.train_model(x_train=x["train"], y_train=y["train"], epochs=5)

    # Evaluate the model
    #
    results = model.evaluate(
        x_test=x["test"],
        y_test=y["test"],
        de_normalize=True,
        loss_relative_to="mean"
    )

    print(results['test_loss'])             # Print MAE
    print(results['test_loss_relative'])    # Print nMAE
    print(results['predicted_profile'])     # Print the predicted load profile
    ```


## Reproduce the Complete Paper

The entire paper can be reproduced by following these steps.

1. Download the whole repository:
    ```bash
    git clone https://github.com/erc-fhv/loadforecasting.git
    ```

1. Set up the environment (Linux only):
    ```bash
    conda env create --name load_forecasting --file=envs/env_linux.yml -y
    conda activate load_forecasting
    ```

1. Install the local packages (without dependencies)
    ```bash
    # From the project root
    pip install --no-deps -e src/loadforecasting_models/
    pip install --no-deps -e .
    ```

1. Train the models:
    ```bash
    python src/loadforecasting_framework/model_trainer.py
    ```

1. Generate figures and tables or run the case study:

    Open and run either 
    ```
    src/loadforecasting_framework/paper_illustration.ipynb
    ```
    or    
    ```
    src/loadforecasting_framework/model_evaluation.ipynb
    ```
    or    
    ```
    src/loadforecasting_framework/case_study/CaseStudy.ipynb
    ```
## Citation

If you use this codebase, or find our work valuable, please cite the following paper:

```
@article{moosbrugger2025load,
  title={Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?},
  author={Moosbrugger, Lukas and Seiler, Valentin and Wohlgenannt, Philipp and Hegenbart, Sebastian and Ristov, Sashko and Eder, Elias and Kepplinger, Peter},
  journal={arXiv preprint},
  year={2025},
  doi={10.48550/arXiv.2501.05000}
}
```
