# Framework for Short-Term Load Forecasting

## Summary

This repository provides a flexible and modular framework for short-term load forecasting (STLF), suitable for both research and real-world applications. It supports:

- Deep learning models: Transformer, LSTM, xLSTM
- Baseline model: KNN, Persistence, Perfect
- Full pipeline for training, evaluation, and visualization
- Reproducibility of all experiments from the following paper

## Related Paper

More information about the models and the framework can be found in the following paper:  

> Moosbrugger et al. (2025), *Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?*, [arXiv:2501.05000](https://arxiv.org/abs/2501.05000)



## Folder Structure

The repository is organized as follows:

```
├── data/                             # Preprocessed smart meter data
│   ├── *.pkl                         # Load profiles (varying community sizes per file)
│   ├── *.ipynb                       # Loadprofile preprocessing script
│
├── envs/                             # Conda environments
│   ├── env_linux.yml                 # Reproducible environment for the paper
│   └── env_from_nxai.yml             # Environment from xLSTM authors
│   
├── src/      
│   ├── loadforecasting_models/       # All forecasting models
|   │   ├── pyproject.toml            # Description of the 'loadforecasting_models' package
│   │   └── *.py                      # Implementations of deep learning & baseline models
│   │
│   ├── loadforecasting_framework/    # Evaluation framework and visualization
│   │   ├── Simulation_config.py      # Config file for simulation runs
│   │   ├── ModelTrainer.py           # Training and evaluation loop
│   │   ├── ModelAdapter.py           # Data formatting and preprocessing
│   │   ├── Paper_Illustration.ipynb  # Plots and tables for the paper
│   │   └── case_study/               # Energy community MILP optimization
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
| data                        |           | data_prprocessor             |
|-----------------------------|           |------------------------------|
| # Weather, load, standard-  |           | + transformData()            |
|   load, and holidays.       +-----------+ # Preprocesses the data      |
+-----------------------------+           +------------+-----------------+
                                                       |
                                                       |
+-----------------------------+           +------------+-----------------+
| simulation_config           |           | model_trainer                |
|-----------------------------|           |------------------------------|
| configs: list               |           | + run()                      |
| # Parameterize the run      +-----------+ # Trains all models          |
| # loop.                     |           | # accord to the config.      |
+-----------------------------+           +------------+-----------------+
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
+-------------+ +-------------+ +-------------+ +-------------+ +-------------+

```

## Reusing only the Forecasting Models

Our forecasting models can be easily reused in other applications as shown below.

1. Install the package:
    ```bash
    pip install loadforecasting_models
    ```

2. Use in Python:
    ```python
    from loadforecasting_models import Model
    import torch 


    # Train the sequence-to-sequence model
    #

    X_train = torch.randn(365, 24, 10)  # Your train features of shape (batch_len, sequence_len, features)
    Y_train = torch.randn(365, 24, 1)  # Your train target of shape (batch_len, sequence_len, 1)
    myModel = Model('Transformer', model_size='5k', num_of_features=X_train.shape[2])   # Alternative Models: 'LSTM', 'xLSTM', 'KNN', 'Perfect'
    myModel.train_model(X_train, Y_train, pretrain_now=False, finetune_now=False, epochs=100, verbose=0)


    # Predict
    #

    X_test = torch.randn(90, 24, 10)  # Your test features of shape (batch_len, sequence_len, features)
    Y_pred = myModel.predict(X_test)
    print('\nOutput Shape = ', Y_pred.shape)

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

1. Install the local packages
    ```bash
    # From the project root
    pip install -e .
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
