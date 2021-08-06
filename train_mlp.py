"""CML tool for training a MLP regressor on cluster. The model will be registered 
and evaluated for model selection.
"""
import ivpk
import numpy as np
import argparse
import pickle
import logging

def get_logger() -> logging.Logger:
    logger = logging.Logger(name=__file__, level="INFO")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel("INFO")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_parser():
    parser = argparse.ArgumentParser(description="fit MLP for regression")
    # data args
    parser.add_argument(
        "--target", type=str, choices=["VDss", "CL"], 
        help="Target to predict"
    )
    parser.add_argument("--fpSize", type=int, help="fingerprint size")
    parser.add_argument(
        "--fpType", choices=["morgan", "rdkit"], default="morgan", 
        help="fingerprint type")
    # nn.Module args
    parser.add_argument("--in_dim", type=int, help="input layer dimension")
    parser.add_argument("--hid_dim", type=int, help="hidden layer dimension")
    parser.add_argument("--dropout", type=float, help="dropout rate")
    # skorch.NeuralNetRegressor args
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=500, 
        help="max epochs to train, real epochs can be less due to earlystopping")
    # model save
    parser.add_argument(
        "--save", type=argparse.FileType("wb"), 
        help="path to save trained model")
    return parser

def load_x_y(args):
    (x, _), y, _, _ = ivpk.data.all_datasets(
        target=args.target, 
        validation_size=None, 
        fpType=args.fpType, fpSize=args.fpSize
    )
    x = x.astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    return x, y

def main(args):
    logger = get_logger()

    # load data
    x, y = load_x_y(args)
    logger.info("data loaded")
    
    # create regressor
    reg_params = dict(
        module__in_dim = args.in_dim, 
        module__hid_dim = args.hid_dim, 
        module__dropout = args.dropout, 
        lr = args.lr, 
        batch_size = args.batch_size, 
        max_epochs = args.max_epochs,
    )
    reg = ivpk.models.wrapped_regressor(**reg_params)
    logger.info("regressor created")

    reg.fit(x, y)
    logger.info("fitting completed")
    
    pickle.dump(reg, args.save)
    logger.info("model saved. All completed.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)