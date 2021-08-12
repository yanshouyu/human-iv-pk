"""Model hyperparameter tuning by GridSearchCV on train + val set, 
note that test set is kept untouched.
Use YAML to config model setting and hyperparameter searching space. 
"""
import numpy as np
import argparse
import yaml
import pickle
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import torch
from torch import nn
from skorch import NeuralNetRegressor
import ivpk


def select_model(model_name: str):
    "return the name speicifed estimator object"
    if model_name == "Lasso":
        return Lasso()
    elif model_name == "RFreg":
        return RandomForestRegressor()
    elif model_name == "MLP":
        reg = NeuralNetRegressor(
            ivpk.models.SimpleRegHead, 
            module__in_dim=267, 
            criterion=nn.MSELoss, 
            optimizer=torch.optim.SGD, 
            # max_epochs=20, 
            # lr=0.01, 
            train_split=None, 
        )
        return reg
    else:
        raise NotImplementedError(f"{model_name} not implemented")


def main(args):
    # prepare model
    conf = yaml.safe_load(args.config)
    model_name = conf.pop("model")
    model = select_model(model_name)
    gs = GridSearchCV(model, n_jobs=args.n_jobs, refit=True, **conf)

    # prepare data, omit SMILES in x
    (x, _), y, _, _ = ivpk.data.all_datasets(
        target=args.target, validation_size=None, 
        fpType="morgan", fpSize=256)
    
    # fit and save
    if model_name == "MLP":
        x, y = x.astype(np.float32), y.reshape(-1, 1).astype(np.float32)
    gs.fit(x, y)
    print(gs.best_params_)
    print(gs.best_score_)
    pickle.dump(gs, args.save)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model optimization by GridSearchCV")
    parser.add_argument(
        "--config", type=argparse.FileType("r"), 
        help="model configuration YAML file")
    parser.add_argument(
        "--target", type=str, choices=["VDss", "CL"], 
        help="Prediction target")
    parser.add_argument(
        "--n_jobs", type=int, default=8, 
        help="Number of jobs to run in parallel")
    parser.add_argument(
        "--save", type=argparse.FileType("wb"), 
        help="Saved path for GridSearchCV object")
    args = parser.parse_args()
    main(args)
