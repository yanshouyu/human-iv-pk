from skorch import callbacks
import torch
from torch import nn
import skorch
from .data import SEED

class SimpleRegHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int, 
                 dropout: float = 0., 
                 intersept: float = None):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hid_dim), 
            nn.GELU(), 
            nn.Dropout(dropout, inplace=False),
            nn.Linear(hid_dim, 1)
        )
        # set the bias if needed
        if intersept:
            with torch.no_grad():
                self.main[3].bias.fill_(intersept)

    def forward(self, x):
        return self.main(x)

def wrapped_regressor(
    module=SimpleRegHead, 
    **kwargs) -> skorch.NeuralNetRegressor:
    """skorch wrapped regressor"""
    cbs = [
        skorch.callbacks.EarlyStopping(patience=10), 
        skorch.callbacks.EpochScoring('r2', lower_is_better=False), 
        skorch.callbacks.EpochScoring('neg_mean_absolute_error', lower_is_better=False)
    ]
    reg = skorch.NeuralNetRegressor(
        module, 
        train_split=skorch.dataset.CVSplit(cv=5, random_state=SEED), 
        callbacks=cbs, 
        **kwargs
    )
    return reg
