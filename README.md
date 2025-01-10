# Load Forecasting for Households and Energy Communities

## Summary

This study explores the use of transfer learning with standard load profiles in combination with advanced models such as LSTMs, xLSTMs, and Transformers, comparing their performance against simpler baseline models. The findings are detailed in the paper: Load Forecasting for Households and Energy Communities: Are Deep Learning Models Worth the Effort?.

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
    conda env create --name load_forecasting --file=envs/env_linux.yml
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
  author={Moosbrugger, Lukas and Seiler, Valentin and Wohlgenannt, Philipp and Hegenbart, Sebastian and Ristov, Sashko and Kepplinger, Peter},
  journal={Preprint submitted to Energy and AI},
  year={2025},
  doi={10.48550/arXiv.2501.05000}
```

