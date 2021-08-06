import os
import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
from pathlib import Path
import pickle
from typing import Tuple, Callable, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, dataloader
from . import utils

SEED = 42

data_path = Path(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"
))
raw_file_path = data_path / "Supplemental_82966_revised_corrected.xlsx"

raw_df = pd.read_excel(raw_file_path, header=8)
SMILES_COL = 'SMILES'
X_COLS = [
    'MW', 'HBA', 'HBD', 'TPSA_NO', 'RotBondCount', 
    'moka_ionState7.4', 'MoKa.LogP', 'MoKa.LogD7.4'
]

#----------- X variables --------------#

def get_pruned_df(
    df: pd.DataFrame = raw_df, 
    drop_invalid_smiles: bool = True, 
    drop_missing_x: bool = True) -> pd.DataFrame:
    """Prune raw data for later processing.
    Column actions: drop some info columns
    Row actions: remove duplicated and / or SMILES, drop missing x
    """
    # df = df.drop(['CAS #', 'Reference', 'Comments ', 'Notes'], axis=1)
    df = df.loc[~raw_df.SMILES.duplicated(keep=False), ]
    if drop_invalid_smiles:
        df = df.loc[df["SMILES"].apply(utils.is_valid_smiles), ]
    if drop_missing_x:
        x_cols = X_COLS
        df = df.dropna(axis='index', subset=x_cols)
    return df

pruned_df = get_pruned_df()

def upper_clip_outliers(df: pd.DataFrame = pruned_df) -> pd.DataFrame:
    """Upper clip the outliers in 3 variables by 90% percentile:
    HBA - 15, HBD - 7, RBC - 13
    """
    cols = ['HBA', 'HBD', 'RotBondCount']
    newdf = df.drop(cols, axis=1)
    for colname in cols:
        upper_bound = round(np.percentile(df[colname], 90)) + 1
        newdf[colname] = df[colname].apply(
            lambda ele: ele if ele < upper_bound else upper_bound
        )
    return newdf

def process_log2(df: pd.DataFrame) -> pd.DataFrame:
    """Transform "MW", "TPSA_NO" by adding 1 and log10
    """
    cols = ['MW', 'TPSA_NO']
    newdf = df.drop(cols, axis=1)
    for colname in cols:
        newdf[colname] = np.log2(df[colname] + 1)
    return newdf

processed_df = process_log2(upper_clip_outliers(pruned_df))[
    [SMILES_COL] + X_COLS
]

x_preprocessor = ColumnTransformer([
    ("stdscl", StandardScaler(), ['MoKa.LogP', 'MoKa.LogD7.4', 'MW', 'TPSA_NO']),
    ("minmax", MinMaxScaler(), ['HBA', 'HBD', 'RotBondCount']), 
    ("onehot", OneHotEncoder(
        categories=[['anionic', 'cationic', 'neutral', 'zwitterionic']], 
        handle_unknown='ignore'
    ), ['moka_ionState7.4'])
], n_jobs=-1)

# fit the x_processor with global distribution
proc_x = x_preprocessor.fit_transform(processed_df)

#----------- Y variables --------------#
def get_processed_y(target: str, df: pd.DataFrame = pruned_df) -> pd.Series:
    """log2 transform VDss and CL, drop others.
    Args:
        target: {VDss, CL}
        df: data containing target column
    """
    assert target in ("VDss", "CL")
    y_cols_dct = {
        "VDss": 'human VDss (L/kg)', 
        "CL": 'human CL (mL/min/kg)'
    }
    colname = y_cols_dct[target]
    return df[colname].dropna().apply(np.log2)

#----------- SMILES --------------#
def get_fingerprints(df: pd.DataFrame, **kwargs) -> np.ndarray:
    """get the fingerprint in 2darray as features for prediction.
    Args:
        df: table containing SMILES column, assume all valid
        kwargs: kwargs for calculating fingerprint matrix
    """
    return utils.get_fp_mat(df["SMILES"].tolist(), **kwargs)

#----------- data methods --------------#
def get_x_y(
    target: str, 
    df: pd.DataFrame, 
    smiles_func: Optional[Callable] = get_fingerprints, 
    **kwargs) -> Tuple:
    """prepare modeling ready data in (X, y).
    Args:
        target: {VDss, CL}
        df: data with x values clipped and log2 transformed, y unchanged. (e.g. processed_df)
        smiles_func: callable for extract feature from SMILES
        kwargs: kwargs for smiles_func
    Returns:
        2 element tuple, (X, y). y shape (N, )
        X is a 2 element tuple: (x0 shape (N, C), x1 shape (N, ))
    """
    y_ser = get_processed_y(target, df)
    # drop non-target records
    df = df.loc[y_ser.index, :]
    num_x = x_preprocessor.transform(df[[SMILES_COL] + X_COLS])
    if smiles_func:
        smiles_x = smiles_func(df, **kwargs)
        x0 = np.c_[num_x, smiles_x]
    else:
        x0 = num_x
    x1 = df[SMILES_COL].to_numpy()
    return (x0, x1), y_ser.to_numpy()

def all_datasets(
    target: str, 
    validation_size: Optional[float] = 0.2, 
    validation_random_state: int = SEED, 
    test_split_year: int = 2000, 
    **kwargs) -> Tuple[np.ndarray]:
    """A turn-key method for train, val, test datasets. 
    Train-val split is random, seed set to 42 for reproducibility.
    Source data is the `processed_df` instance.
    Args:
        target: {VDss, CL}
        validation_size: validation size used in train-validation split. 
            If None, no validation set will be split out from training set.
        validation_random_state: if train-val split, seed value.
        test_split_year: "Year of first disclosure" to split test set.
            before - train & val; include & after - test. Set to 2000 according 
            to project requirement.
        kwargs: kwargs used in generating features.
    Returns:
        (x_train, y_train, (x_val, y_val,) x_test, y_test)
    """
    df_train = processed_df.loc[
        processed_df["Year of first disclosure"] < test_split_year, ]
    df_test = processed_df.loc[
        processed_df["Year of first disclosure"] >= test_split_year, ]
    x_test, y_test = get_x_y(target, df_test, **kwargs)
    x_train_val, y_train_val = get_x_y(target, df_train, **kwargs)
    if validation_size is not None:
        x_train_0, x_val_0, x_train_1, x_val_1, y_train, y_val = train_test_split(
            x_train_val[0], x_train_val[1], y_train_val, 
            test_size=validation_size, 
            random_state=validation_random_state
        )
        return (
            (x_train_0, x_train_1), 
            y_train, 
            (x_val_0, x_val_1), 
            y_val, 
            x_test, y_test)
    else:
        return x_train_val, y_train_val, x_test, y_test

def dump_dataset(
    file: str, 
    target: str, 
    fpType: str, 
    fpSize: int, 
    validation_size: Optional[float] = None, 
    **kwargs) -> None:
    """A turn-key method to dump modeling ready data. Requiring args explicitly.
    """
    datasets = all_datasets(
        target=target, validation_size=validation_size, 
        fpType=fpType, fpSize=fpSize, **kwargs
    )
    if len(datasets) == 6:
       x_train, y_train, x_val, y_val, x_test, y_test = datasets
    elif len(datasets) == 4:
        x_train, y_train, x_test, y_test = datasets
        x_val = None
        y_val = None
    # check whether fingerprint is used or not
    if kwargs.get("smiles_func", get_fingerprints) is None:
        fpType = None
        fpSize = 0

    with open(file, "wb") as f:
        pickle.dump(dict(
            target = target, 
            x_train = x_train, 
            y_train = y_train,
            x_val = x_val,
            y_val = y_val,
            x_test = x_test, 
            y_test = y_test,
            fpType = fpType, 
            fpSize = fpSize
        ), f)
        
#----------- NN methods --------------#
def get_dataloaders(
    target: str, 
    dests: Tuple[str] = ("train", "val"), 
    batch_size: int = 256, 
    **kwargs) -> Tuple[DataLoader]:
    """Wrap data into Dataloader(s). 
    Note: **SMILES is not included in x**.
    Args:
        target: {VDss, CL}
        dests: Tuple of {train, val, test}
        batch_size: batch_size for dataloader
        kwargs: args for preparing data
    Returns:
        Tuple of Dataloader as `sets` specified. 
    """
    def dl_from_ndarray(x, y, batch_size: int, shuffle: bool):
        ds = TensorDataset(
            torch.tensor(x, dtype=torch.float32), 
            # add a dimension to y, from (N, ) to (N, 1)
            torch.tensor(np.expand_dims(y, axis=1), dtype=torch.float32)
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
        return dl
    (
        (x_train, _), y_train, 
        (x_val, _), y_val, 
        (x_test, _), y_test
    ) = all_datasets(target, **kwargs)
    dl_tup = ()
    for dest in dests:
        if dest == "train":
            dl_tup += (
                dl_from_ndarray(x_train, y_train, batch_size, True)
            , )
        elif dest == "val":
            dl_tup += (
                dl_from_ndarray(x_val, y_val, batch_size, False)
            , )
        elif dest == "test":
            dl_tup += (
                dl_from_ndarray(x_test, y_test, batch_size, False)
            , )
    return dl_tup

