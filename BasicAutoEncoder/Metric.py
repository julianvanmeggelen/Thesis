from typing import Any
import torch as torch
import torch.nn as nn
import numpy as np
from statsmodels.multivariate.cancorr import CanCorr


class Metric(object):
    def __init__(self, key: str):
        self.key = key

    def compute(self, X: torch.Tensor, y: torch.Tensor, mod: nn.Module, mode:str):
        raise NotImplementedError
    
    def __call__(self, X: torch.Tensor, y: torch.Tensor, mod: nn.Module, mode:str) -> Any:
        if mode not in ('train', 'val'): return ValueError(f"Incorrect mode value provided: {mode}")
        return self.compute(X,y,mod,mode)
    

class CCACorr(Metric):
    def __init__(self, f_train: np.ndarray, f_val: np.ndarray):
        """
        f: True factor, needs to be provided as training loop does not have acess to this
        """
        self.f_train = f_train
        self.f_val= f_val
        super().__init__(key='CCA_mean_corr')

    def compute(self, X: torch.Tensor, y: torch.Tensor, mod: nn.Module, mode:str):
        cc = CCACorr_functional(self.f_train if mode == 'train' else self.f_val, y, mod)
        return cc


def CCACorr_functional(f:torch.Tensor, y: torch.Tensor, mod:nn.Module):
      if not isinstance(y, torch.Tensor):
            y = torch.Tensor(f).float()

      f_hat = mod.enc(y).detach().numpy()
      cc = np.mean(CanCorr(f_hat, f).cancorr)
      return cc
