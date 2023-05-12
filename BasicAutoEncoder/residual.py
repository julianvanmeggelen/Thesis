from BasicAutoEncoder.model import Encoder, Decoder
import torch.nn as nn
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, act):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.act = act
        self.linear = nn.Linear(dim_in, dim_out)
        self.act = act

    def forward(self, input):
        lin_out = self.linear(input)
        act_out = self.act(lin_out)
        out = act_out+lin_out
        return out

class ResidualEncoder(Encoder):
    def __init__(self, hidden_dim: list = [500,200,100], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = False, lastLayerLinear: bool=False, use_xavier:bool = True):
        super().__init__(hidden_dim, activation, use_batchnorm, lastLayerLinear, use_xavier)

    def _get_sequential(self): #compile to nn.Sequential
        res = OrderedDict()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            act = nn.Identity() if i == len(self.hidden_dim)-2 and self.lastLayerLinear else self.activation()
            res[f"res_{i}"] = ResidualBlock(self.hidden_dim[i],self.hidden_dim[i+1], act)
            if self.use_batchnorm and i != self.n_hidden-2:
                res[f"batchnorm_{i}"]  = nn.BatchNorm1d(self.hidden_dim[i+1])
        return nn.Sequential(res)

class ResidualDecoder(Decoder):
    def __init__(self, linear: bool = False, hidden_dim: list = [100,200,500], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = False, lastLayerLinear: bool=False, use_xavier: bool = False):
        super().__init__( linear, hidden_dim, activation, use_batchnorm, lastLayerLinear, use_xavier)

    def _get_sequential(self): #compile to nn.Sequential
        res = OrderedDict()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            act = nn.Identity() if i == len(self.hidden_dim)-2 and self.lastLayerLinear else self.activation()
            res[f"res_{i}"] = ResidualBlock(self.hidden_dim[i],self.hidden_dim[i+1], act)
            if self.use_batchnorm and i != self.n_hidden-2:
                res[f"batchnorm_{i}"]  = nn.BatchNorm1d(self.hidden_dim[i+1])
        return nn.Sequential(res)