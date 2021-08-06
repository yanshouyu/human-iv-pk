"""Module for evaluating sklearn estimator-like models."""
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict
import os
import yaml
from tqdm import tqdm
import pickle
from . import data

SEED = data.SEED


class Evaluation(object):
    """Evaluation on model and input strategy."""
    def __init__(
        self, model, 
        target: str, 
        validation_size: float, 
        smiles_func: Callable = data.get_fingerprints, 
        **kwargs) -> None:
        """Initialize object with model and data.
        Args:
            model: model object with `fit` and `predict` methods.
        Input specs:
            target: {VDss, CL}
            validation_size: float. Ratio of validation set in training. 
            fpType: {morgan, rdkit}, fingerprint type.
            fpSize: bit size for fingerprint.
            smiles_func: method to get fingerprints from SMILES. if None, no 
                smiles fingerprints.
            kwargs: args for getting dataset. 
        """
        self.model = model
        self.target = target
        self.validation_size = validation_size
        self.fpType = kwargs.get("fpType")
        self.fpSize = kwargs.get("fpSize")
        
        try:
            assert self.validation_size >0 and self.validation_size < 1
        except AssertionError:
            raise ValueError("Validation set required in Evaluation object")

        (
            self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test
        ) = data.all_datasets(
            target=self.target, validation_size=self.validation_size, 
            smiles_func=smiles_func, **kwargs
        )
        self.evaluation = None


    def evaluate(
        self, 
        use_smiles: bool, 
        nn_data: bool = False, 
        fit: bool = False, 
        test: bool = False, 
        cv: int = 5, 
        **kwargs) -> Dict:
        """
        Evaluation on validation set (and test set).
        Args:
            use_smiles: whether the 2nd element in x, smiles array shape (N, ), 
                should be used by model or not. For GNN models which extract 
                graph from SMILES, `use_smiles` must be True.
            nn_data: whether the data is for skorch wrapped NN. If true:
                1. change dtype to np.float32 for x and y
                2. reshape y from (N, ) to (N, 1)
            fit: whether to fit the model. If False, only evaluate on 
                validation set and no K-fold CV.
            test: whether to evaluate on test set.
            cv: splitting strategy. if cv in (0, 1, None), no cross-validation
            kwargs: args for fitting if refit is True
        Returns:
            a dict of evaluation metrics
        """
        eval_dct = dict(
            # use_smiles=use_smiles, 
            # fit=fit, 
            # cv=cv
        )
        if not use_smiles:
            x_train, y_train = self.x_train[0], self.y_train
            x_val, y_val = self.x_val[0], self.y_val
            x_test, y_test = self.x_test[0], self.y_test
        if nn_data:
            x_train, y_train = x_train.astype(np.float32), y_train.reshape(-1, 1).astype(np.float32)
            x_val, y_val = x_val.astype(np.float32), y_val.reshape(-1, 1).astype(np.float32)
            x_test, y_test = x_test.astype(np.float32), y_test.reshape(-1, 1).astype(np.float32)
        if fit:
            self.model.fit(x_train, y_train, **kwargs)

        self.y_train_pred = self.model.predict(x_train)
        eval_dct['MAE_train'] = mean_absolute_error(y_train, self.y_train_pred)
        eval_dct['Pearsonr_train'] = pearsonr(y_train.reshape(-1), self.y_train_pred.reshape(-1))[0]
        self.y_val_pred = self.model.predict(x_val)
        eval_dct['MAE_val'] = mean_absolute_error(y_val, self.y_val_pred)
        eval_dct['Pearsonr_val'] = pearsonr(y_val.reshape(-1), self.y_val_pred.reshape(-1))[0]

        if test:
            self.y_test_pred = self.model.predict(x_test)
            eval_dct['MAE_test'] = mean_absolute_error(y_test, self.y_test_pred)
            eval_dct['Pearsonr_test'] = pearsonr(y_test.reshape(-1), self.y_test_pred.reshape(-1))[0]

        # cross validation on train + val, only for non-SMILES models
        if fit and cv not in (0, 1, None):
            order = np.arange(x_train.shape[0] + x_val.shape[0])
            rng = np.random.default_rng(SEED)
            rng.shuffle(order)
            self.x_train_val = np.r_[x_train, x_val][order]
            self.y_train_val = np.r_[y_train, y_val][order]
            self.y_train_val_pred = cross_val_predict(
                self.model, self.x_train_val, self.y_train_val, cv=cv, n_jobs=4
            )
            eval_dct[f"MAE_cv"] = mean_absolute_error(self.y_train_val, self.y_train_val_pred)
            eval_dct[f"Pearsonr_cv"] = pearsonr(
                self.y_train_val, self.y_train_val_pred
            )[0]
            # eval_dct[f"MAEs_Kfold"] = [-v for v in cross_val_score(
            #     self.model, self.x_train_val, self.y_train_val, cv=cv, n_jobs=4, 
            #     scoring='neg_mean_absolute_error'
            # )]
        self.evaluation = eval_dct
        return eval_dct

    def __repr__(self) -> str:
        return f"""
        target: {self.target}, 
        model: {self.model}, 
        evaluation: {self.evaluation}
        """

def eval_registered(
    target: str, 
    reg_path: Optional[str] = None, 
    filter: Optional[Dict[str, Tuple]] = None, 
    test: bool = False) -> Dict[str, Evaluation]:
    """Evaluate all registered models for a target. The evaluation results 
    will be used for model selection.
    Args:
        target: {VDss, CL}
        reg_path: file path for the registered model yaml
        filter: filter for the registered model. Example: 
            filter={"model": ("Lasso", "RFreg")}, only evaluate 
            Lasso and RFreg models.
            filter={"gridsearch": (True, )}, only evaluate gridsearch models
        test: whether to evaluatoin on test set
    Returns:
        named evaluation. Any evaluation contains:
            - model
            - data: {x, y}_{train, val(, test)} and the predicted ys
            - target
            - evaluation results, {MAE, Pearsonr}_{train, val(, test)} and 
                MAE_cv if the model is a GridSearchCV.
    """
    if reg_path is None:
        reg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            os.pardir, "models", "model_registration.yaml"
        )
    if not os.path.isfile(reg_path):
        raise FileNotFoundError("model register file not found")
    with open(reg_path, "r") as f:
        model_dct = yaml.safe_load(f)[target]
    
    # filter the registered models
    if filter: 
        rm_names = set()
        for name, info in model_dct.items():
            for k, v in filter.items():
                if info[k] not in v:
                    rm_names.add(name)
        for name in rm_names:
            _ = model_dct.pop(name)
    
    eval_dct = {}
    for name, info in tqdm(model_dct.items()):
        with open(info["path"], "rb") as f:
            model = pickle.load(f)
        
        if info['model'] == 'MLP':     # skorch.NeuralNetRegressor wrapped MLP
            nn_data = True
            # set batch size for memory usage
            if info['gridsearch']:
                model.best_estimator_.batch_size = 1
            else:
                model.batch_size = 1
        else:
            nn_data = False

        evl = Evaluation(
            model, 
            target=target, 
            validation_size=0.2, 
            fpType=info['fpType'], 
            fpSize=info['fpSize']
        )
        _ = evl.evaluate(use_smiles=False, nn_data=nn_data, fit=False, test=test)
        # update MAE_cv from GridSearchCV.best_score_
        if info['gridsearch']:
            evl.evaluation["MAE_cv"] = model.best_score_ * -1
        eval_dct[name] = evl
    return eval_dct
