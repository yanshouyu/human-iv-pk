"""CML tool for making prediction on new data.
Required columns in csv input (`[ivpk.data.SMILES_COL] + ivpk.data.X_COLS`): 
[
    'SMILES', 
    'MW', 'HBA', 'HBD', 'TPSA_NO', 'RotBondCount', 
    'moka_ionState7.4', 'MoKa.LogP', 'MoKa.LogD7.4'
]
Column order is not strict.

Current best model for VDss: models/GridSearchCV/VDss_rfreg_gridsearch.pkl
Current best model for CL: models/CL_mlp.pkl
Note that the CL model is memory demanding on Mac. 
"""
import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import logging
from typing import Union, IO, Tuple
import ivpk

def get_logger() -> logging.Logger:
    logger = logging.Logger(name=__file__, level="INFO")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger()

def process_data(f: Union[str, IO]) -> Tuple[np.ndarray]:
    required_cols = [ivpk.data.SMILES_COL] + ivpk.data.X_COLS
    try:
        raw_df = pd.read_csv(f)[required_cols]
    except KeyError:
        logger.error("required columns (partially) missing")
        raise KeyError(
            f"required columns not in table, should have {required_cols}"
        )
    df = ivpk.data.get_pruned_df(raw_df)
    if raw_df.shape[0] != df.shape[0]:
        logger.warning(f"{raw_df.shape[0] - df.shape[0]} records were removed")
    df = ivpk.data.process_log2(ivpk.data.upper_clip_outliers(df))
    # remember to reset the column order
    df = df[required_cols]
    x_num = ivpk.data.x_preprocessor.transform(df)
    x_fp = ivpk.data.get_fingerprints(df)
    return np.c_[x_num, x_fp], df[ivpk.data.SMILES_COL].to_numpy()

def pred_vdss(x: np.ndarray) -> np.ndarray:
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "models", "GridSearchCV", "VDss_rfreg_gridsearch.pkl"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model.predict(x)

def pred_cl(x: np.ndarray) -> np.ndarray:
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "models", "CL_mlp.pkl"
    )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model.predict(x.astype(np.float32)).reshape(-1)
    

def main(args):
    x, smiles = process_data(args.csv)
    logger.info("data processing completed")
    vdss_pred = pred_vdss(x)
    logger.info("VDss prediction completed")
    cl_pred = pred_cl(x)
    logger.info("CL prediction completed")
    pred_df = pd.DataFrame({
        ivpk.data.SMILES_COL: smiles,
        "log2VDss": vdss_pred, 
        "log2CL": cl_pred
    })
    pred_df.to_csv(args.output, index=False)
    logger.info("result written. All completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict VDss and CL")
    parser.add_argument(
        "--csv", type=argparse.FileType("r"), help="input table in CSV")
    parser.add_argument(
        "--output", type=argparse.FileType("w"), default=sys.stdout, 
        help="output CSV file, default to stdout")
    args = parser.parse_args()
    main(args)
    