# Framework for Short-Term Load Forecasting

## Summary

This repository provides a flexible and modular framework for short-term load forecasting (STLF), suitable for both research and real-world applications. It supports:

- Deep learning models: Transformer, LSTM, xLSTM
- Baseline model: KNN, Perfect
- Full pipeline for training, evaluation, and visualization
- Reproducibility of all experiments from the subsequent paper

## Related Paper

More information about the models and the framework can be found in the following paper:  

> Moosbrugger et al. (2025), *Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?*, [arXiv:2501.05000](https://arxiv.org/abs/2501.05000)



## Folder Structure

The repository is organized as follows:

```
├── models/                       # All forecasting models
│   ├── Model.py                  # Unified model wrapper
│   ├── *.py                      # Implementations of deep learning & baseline models
│         
├── data/                         # Preprocessed smart meter and weather data
│   ├── *.pkl                     # Load profiles (varying community sizes per file)
│   ├── *.ipynb/.py               # Preprocessing scripts (e.g., weather, London data)
│         
├── scripts/                      # Evaluation framework and visualization
│   ├── Simulation_config.py      # Config file for simulation runs
│   ├── ModelTrainer.py           # Training and evaluation loop
│   ├── ModelAdapter.py           # Data formatting and preprocessing
│   ├── Paper_Illustration.ipynb  # Plots and tables for the paper
│   └── case_study/               # Energy community MILP optimization
│
├── envs/                         # Conda environments
│   ├── env_linux.yml             # Reproducible environment for the paper
│   └── env_from_nxai.yml         # Environment from xLSTM authors
```


## Code Structure

The main parts of the evaluation framework are connected as follows:

```

+-----------------------------+           +------------------------------+
| Data                        |           | ModelAdapter                 |
|-----------------------------|           |------------------------------|
| # Weather, load, standard-  |           | + transformData()            |
|   load, and holidays.       +-----------+ # Preprocesses the data      |
+-----------------------------+           +------------+-----------------+
                                                       |
                                                       |
+-----------------------------+           +------------+-----------------+
| Simulation_config           |           | ModelTrainer                 |
|-----------------------------|           |------------------------------|
| configs: list               |           | + run()                      |
| # Parameterize the run      +-----------+ # Trains all models          |
| # loop.                     |           | # accord to the config.      |
+-----------------------------+           +------------+-----------------+
                                                       |
                                                       |
                                          +------------+-----------------+
                                          | Model                        |
                                          |------------------------------|
                                          | my_model: (xLSTM to KNN)     |
                                          | + train_model()              |
                                          | + evaluate()                 |
                                          +------------+-----------------+
                                                       |            
                                                       |                 
       +-----------------+-------------+---------------+-----------------+
       |                 |             |               |                 |
       |                 |             |               |                 |
+------+------+ +--------+----+ +------+------+ +------+------+ +--------+----+
| KNN         | | Persistence | | xLSTM       | | LSTM        | | Transformer |
|             | |             | |             | |             | |             |
|-------------| |-------------| |-------------| |-------------| |-------------|
| + forward() | | + forward() | | + forward() | | + forward() | | + forward() |
+-------------+ +-------------+ +-------------+ +-------------+ +-------------+

```

## Reusing only the Forecasting Model

Our forecasting models can be easily reused in other applications as shown below.

1. Install the package:
    ```bash
    pip install git+https://github.com/erc-fhv/loadforecasting.git
    ```

2. Use in Python:
    ```python
    from models import Model
    import torch 


    # Train the sequence-to-sequence model
    #

    X_train = torch.randn(365, 24, 10)  # Your train features of shape (batch_len, sequence_len, features)
    Y_train = torch.randn(365, 24, 1)  # Your train target of shape (batch_len, sequence_len, 1)
    myModel = Model('Transformer', model_size='5k', num_of_features=X_train.shape[2])   # Alternative Models: 'LSTM', 'xLSTM', 'KNN'
    myModel.train_model(X_train, Y_train, pretrain_now=False, finetune_now=False, epochs=100, verbose=0)


    # Predict
    #

    X_test = torch.randn(90, 24, 10)  # Your test features of shape (batch_len, sequence_len, features)
    Y_pred = myModel.predict(X_test)
    print('\nOutput Shape = ', Y_pred.shape)

    ```

## Reproduce the Complete Paper

The entire paper can be reproduced by following these steps.

1. Set up the environment (Linux only):
    ```bash
    conda env create --name load_forecasting --file=envs/env_linux.yml
    conda activate load_forecasting
    ```

2. Train the models:
    ```bash
    python scripts/ModelTrainer.py
    ```

3. Generate figures and tables:

    Open and run either `scripts/Paper_Illustration.ipynb` or `scripts/Model_Evaluation.ipynb`.

## Citation

If you use this codebase, or find our work valuable please cite the following paper:

```
@article{moosbrugger2025load,
  title={Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?},
  author={Moosbrugger, Lukas and Seiler, Valentin and Wohlgenannt, Philipp and Hegenbart, Sebastian and Ristov, Sashko and Eder, Elias and Kepplinger, Peter},
  journal={arXiv preprint},
  year={2025},
  doi={10.48550/arXiv.2501.05000}
}
```
