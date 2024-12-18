# Shortterm Loadforecasting with xLSTM

## Summary

This work uses transfer learning from standard load profiles together with state of the art models like LSTMs, xLSTMs and Transformers and compares them to very simple models.

## Software Design

The basic code design is as follows.
```

+-----------------------------+           +------------------------------+
| Data                        |           | ModelAdapter                 |
|-----------------------------|           |------------------------------|
| # Weather, load, standard-  |           | + transformData()            |
|   load, and holidays.       |---------->| # Preprocesses the data      |
+-----------------------------+           +------------------------------+
                                                       |
                                                       |
+-----------------------------+           +------------v-----------------+
| Simulation_config           |           | ModelTrainer                 |
|-----------------------------|           |------------------------------|
| configs: list               |           | + run()                      |
| # Parameterize the run      |---------> | # Trains all models          |
| # loop.                     |           | # accord to the config.      |
+-----------------------------+           +------------------------------+
                                                       |
                                                       |
                                          +------------v-----------------+
                                          | Model                        |
                                          |------------------------------|
                                          | my_model: (xLSTM to KNN)     |
                                          | + train_model()              |
                                          | + evaluate()                 |
                                          +------------------------------+
+-------------+                                        |
| LSTM        <--------+                               |
|             |        |                               |                 
|-------------|        |                               |                 
| + forward() |        |                               |                 
+-------------+        |                               |                 
                       |                               |                 
+-------------+        |                               |                 
| Transformer <--------+----------+---------------+----+------------+
|             |        |          |               |                 |
|-------------|        |          |               |                 |
| + forward() |        |          |               |                 |
+-------------+        |          |               |                 |
                       |          |               |                 |
+-------------+        |   +------v------+ +------v------+ +--------v----+
| xLSTM       <--------+   | Persistence | | KNN         | | Synthetic   |
|             |            | Prediction  | |             | | Load        |
|-------------|            |-------------| |-------------| |-------------|
| + forward() |            | + forward() | | + forward() | | + forward() |
+-------------+            +-------------+ +-------------+ +-------------+

```

<!-- ## Components: todo! -->

## How to Use

1. **Install the conda enviroment** on a linux system:
    ```
    conda env create --name load_forecsting --file=envs/env_linux.yml
    conda activate load_forecasting
    ```

2. **Train the model** using `ModelTrainer`:
    ```python
    python scripts/ModelTrainer.py
    ```

3. **Evaluate the results** e.g. within `scripts/model_evaluate.ipynb` or within `Paper_Illustration.ipynb`.

## Citation

If you use this codebase, or find our work valuable please cite the following paper:

```
@article{
  title={Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?},
  author={Lukas Moosbrugger, Valentin Seiler, Philipp Wohlgenannt, Sebastian Hegenbart, Sashko Ristov, Peter Kepplinger},
  journal={preprint},
  year={2024}
}
```

