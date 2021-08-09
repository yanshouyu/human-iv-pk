# human-iv-pk
Modeling human intravenous (IV) pharmacokinetic (PK) parameters from data collected by [*Lombardo, F et al. 2018*](https://dmd.aspetjournals.org/content/early/2018/08/16/dmd.118.082966).

## Project

### Project Target
In this project I've explored the PK data and tried to fit models on 2 PK parameters: 

- steady-state volumn of distribution (VD<sub>ss</sub>) 
- human clearance in plasma (CL<sub>p</sub>)
 
Although other PK parameters, e.g. half-life (t<sub>1/2</sub>), mean residence time (MRT) are not modeled here, the package `ivpk` is easily extendable to these targets.

### Repository Structure
This repository consists of several directories, jupyter notebooks and python scripts:
```
.
├── configs/
├── data/
├── doc/
├── figures/
├── ivpk/
├── models/
├── results/
└── submission/
└── *.ipynb
└── *.py
└── ...
```
#### Directories
- configs: YAML configuration on model and GridSearchCV.
- data: raw, intermidiate and example data
- doc: article, sup table, leaderboard
- figures: visualization results
- `ivpk`: python package for this project
- models: saved models
- results: prediction results
- submission: results predicted by selected models on test set as final submission

#### Notebooks
Notebooks following the order of analysis:
1. [EDA](EDA.ipynb): explore data, get a basic idea of the data and form strategy of preprocessing.
2. [Preprocessing](Preprocessing.ipynb): implementation and test of preprocessing methods.
3. [Train_val_test_split](Train_val_test_split.ipynb): split data strategy.
4. [Baseline_models_VDss](Baseline_models_VDss.ipynb) and [Baseline_models_CL](Baseline_models_CL.ipynb): simple models as baseline: linear regression, LASSO regression, Random Forest Regressor and SVR. The hyperparameters are not tuned.
5. [MLP_head](MLP_head.ipynb): try MLP as regressor.
6. [MLP_head_skorch](MLP_head_skorch.ipynb): wrap MLP by `skorch` for hyperparameter tuning and easy comparison with sci-kit learn regressors.
7. [Model_selection](Model_selection.ipynb): make prediction on test set, generate leaderboard, select best models for making submission.
8. [Submission](Submission.ipynb): take the selected models, make submission, analyze model and results.

#### Scripts
Command line tools for execution on cluster:
1. [gridsearch_cv.py](gridsearch_cv.py): grid search on train + validation set to find best hyperparameters, input config YAML saved in [configs/](configs/)
2. [train_mlp.py](train_mlp.py): train MLP head by given hyperparameters, train and val set defined by fixed seed for reproducibility.
3. [pred_from_raw.py](pred_from_raw.py): make prediction from raw data, using the best estimator so far.

## Data


## Methods

## Results

Leaderboard:

## Discussion
