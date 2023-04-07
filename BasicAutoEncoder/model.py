import torch
from torch import nn
import torch.functional as F
from torch import optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, hidden_dim: list = [500,200,100], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = True):
        super().__init__()
        self.n_hidden = len(hidden_dim)
        self.hidden_dim: list = hidden_dim
        self.activation: nn.Module = activation
        self.use_batchnorm: bool = use_batchnorm
        self.sequential = self._get_sequential()

    def _get_sequential(self): #compile to nn.Sequential
        res = nn.Sequential()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            res.append(nn.Linear(self.hidden_dim[i],self.hidden_dim[i+1]))
            res.append(self.activation)
            if self.use_batchnorm and i != self.n_hidden-2:
                res.append(nn.BatchNorm1d(self.hidden_dim[i+1]))
        return res

    def forward(self, x):
        out = self.sequential(x)
        return out

class Decoder(nn.Module):
    def __init__(self, linear: bool = False, hidden_dim: list = [100,200,500], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = False):
        super().__init__()
        assert (linear and len(hidden_dim) == 2) or (not linear)
        self.n_hidden = len(hidden_dim)
        self.hidden_dim: list = hidden_dim
        self.activation: nn.Module = activation
        self.use_batchnorm: bool = use_batchnorm
        self.sequential = self._get_sequential()

    def _get_sequential(self): #compile to nn.Sequential
        res = nn.Sequential()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            res.append(nn.Linear(self.hidden_dim[i],self.hidden_dim[i+1]))

            if i < self.n_hidden -2:
                res.append(self.activation)
            if self.use_batchnorm and i < self.n_hidden-2:
                res.append(nn.BatchNorm1d(self.hidden_dim[i+1]))
        return res

    def forward(self, x):
        out = self.sequential(x)
        return out



class AutoEncoder(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder):
        super().__init__()
        self.enc = enc
        self.dec = dec
        assert dec.hidden_dim[0] == enc.hidden_dim[-1] and dec.hidden_dim[-1] == enc.hidden_dim[0]
    
    def forward(self,x):
        out = self.enc(x)
        out = self.dec(out)
        return out
