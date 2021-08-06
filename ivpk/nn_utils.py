from typing import Callable, Optional
import numpy as np
import torch
from torch import nn
from tqdm import trange
from typing import Callable, Tuple, Optional

def loss_batch(
    model: nn.Module, 
    loss_fn: nn.Module, 
    xb: torch.tensor, 
    yb: torch.tensor, 
    opt: Optional[torch.optim.Optimizer] = None) -> Tuple:
    loss = loss_fn(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(num_epochs, model, loss_fn, optimizer, train_dl, val_dl):
    for epoch in trange(num_epochs):
        model.train()
        losses, nums = zip(
            *[loss_batch(model, loss_fn, Xb, yb, optimizer) for Xb, yb in train_dl]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_fn, Xb, yb) for Xb, yb in val_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        yield (epoch, train_loss, val_loss)