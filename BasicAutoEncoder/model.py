import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
from collections import OrderedDict

import sys
sys.path.append('../')

from BasicAutoEncoder.Metric import Metric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OrthoLoss(nn.Module):
    def __init__(self, enc, alpha, trainHist=None):
        super().__init__()
        self.enc = enc
        self.reproductionLoss = nn.MSELoss()
        self.alpha = alpha
        self.trainHist = trainHist

    def set_hist(self, hist):
        self.trainHist = hist
        self.trainHist['loss_repr'] = []
        self.trainHist['loss_orth'] = []
    
    def forward(self, output, target):
        reprLoss = self.reproductionLoss(output, target)
        f = self.enc(target)
        orthogonalLoss = torch.mean(torch.sum(f*f, dim=-1)-1) #X.T @ X - I
        #print(reprLoss.item(), orthogonalLoss.item(), end='\r')
        if self.trainHist is not None:
            self.trainHist['loss_repr'].append(reprLoss.item())
            self.trainHist['loss_orth'].append(orthogonalLoss.item())
        return reprLoss + self.alpha * orthogonalLoss #+ torch.nn.functional.mse_loss(torch.mean(f, dim=0),torch.zeros(f.shape[1]))


class Encoder(nn.Module):
    def __init__(self, hidden_dim: list = [500,200,100], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = False, lastLayerLinear: bool=False, use_xavier:bool = True, init_function = None, dropout = 0.0):
        super().__init__()
        self.n_hidden = len(hidden_dim)
        self.hidden_dim: list = hidden_dim
        self.activation: nn.Module = activation if isinstance(activation, type) else type(activation) #get original object type, reinitialize at every layer, don't know if this is required
        self.use_batchnorm: bool = use_batchnorm
        self.lastLayerLinear = lastLayerLinear
        self.use_xavier = use_xavier
        self.dropout = dropout
        self.sequential = self._get_sequential(init_function=init_function)

    def _get_sequential(self, init_function): #compile to nn.Sequential
        res = OrderedDict()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            res[f"linear_{i}"] = nn.Linear(self.hidden_dim[i],self.hidden_dim[i+1]) 
            
            if init_function:
                init_function(res[f"linear_{i}"].weight)
            elif self.use_xavier:
                nn.init.xavier_uniform(res[f"linear_{i}"].weight)

            res[f"activation_{i}"] = nn.Identity() if i == len(self.hidden_dim)-2 and self.lastLayerLinear else self.activation()
            if self.dropout:
                res[f"dropout_{i}"] = nn.Dropout(self.dropout)

            if self.use_batchnorm and i != self.n_hidden-2:
                res[f"batchnorm_{i}"]  = nn.BatchNorm1d(self.hidden_dim[i+1])
        self.init_method = None #remove so object can be pickled
        return nn.Sequential(res)

    def forward(self, x):
        out = self.sequential(x)
        return out

class Decoder(nn.Module):
    def __init__(self, linear: bool = False, hidden_dim: list = [100,200,500], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = False, lastLayerLinear: bool=False, use_xavier: bool = False, init_function = None, dropout=0.0):
        super().__init__()
        assert (linear and len(hidden_dim) == 2) or (not linear)
        self.n_hidden = len(hidden_dim)
        self.hidden_dim: list = hidden_dim
        self.activation: nn.Module = activation if isinstance(activation, type) else type(activation) #get original object type, reinitialize at every layer, don't know if this is required
        self.use_batchnorm: bool = use_batchnorm
        self.lastLayerLinear = lastLayerLinear
        self.use_xavier = use_xavier
        self.dropout = dropout
        self.sequential = self._get_sequential(init_function=init_function)

    def _get_sequential(self, init_function): #compile to nn.Sequential
        res = OrderedDict()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            res[f"linear_{i}"] = nn.Linear(self.hidden_dim[i],self.hidden_dim[i+1]) 
            if init_function:
                init_function(res[f"linear_{i}"].weight)
            elif self.use_xavier:
                nn.init.xavier_uniform(res[f"linear_{i}"].weight)
                
            res[f"activation_{i}"] = nn.Identity() if i == len(self.hidden_dim)-2 and self.lastLayerLinear else self.activation()
            if self.dropout:
                res[f"dropout_{i}"] = nn.Dropout(self.dropout)

            #if i < self.n_hidden -2: #lastlayer always linear
                #res.append(self.activation)
            if self.use_batchnorm and i < self.n_hidden-2:
                res[f"batchnorm_{i}"] = nn.BatchNorm1d(self.hidden_dim[i+1])
        return nn.Sequential(res)

    def forward(self, x):
        out = self.sequential(x)
        return out

#this is a little experiment
class SelfAttentionEncoder(nn.Module):
    def __init__(self, hidden_dim: list = [500,200,100], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = True, lastLayerLinear: bool=False):
        super().__init__()
        self.n_hidden = len(hidden_dim)
        self.hidden_dim: list = hidden_dim
        self.activation: nn.Module = activation
        self.use_batchnorm: bool = use_batchnorm
        self.sequential = self._get_sequential()
        self.attention = nn.MultiheadAttention(embed_dim = hidden_dim[0], num_heads=1)
        

    def _get_sequential(self): #compile to nn.Sequential
        res = nn.Sequential()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            res.append(nn.Linear(self.hidden_dim[i],self.hidden_dim[i+1]))
            res.append(self.activation)
            if self.use_batchnorm and i != self.n_hidden-2:
                res.append(nn.BatchNorm1d(self.hidden_dim[i+1]))

        if self.lastLayerLinear:
                res[-1] = nn.Identity()   
        return res

    def forward(self, x):
        out = self.attention(x,x,x,need_weights=False)
        out = self.sequential(x)
        return out
    
class AutoEncoder(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder):
        super().__init__()
        self.enc = enc
        self.dec = dec
        assert dec.hidden_dim[0] == enc.hidden_dim[-1], f"{dec.hidden_dim[0]} {enc.hidden_dim[-1]}"
        assert dec.hidden_dim[-1] == enc.hidden_dim[0], f"{dec.hidden_dim[-1]} {enc.hidden_dim[0]}"
    
    def forward(self,x):
        out = self.enc(x)
        out = self.dec(out)
        return out


def get_trainable_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train_val_split_randok(X: torch.Tensor, val_split=0.3, seed=None, batch_size=64) -> DataLoader:
    generator = None
    if seed:
        generator = torch.Generator().manual_seed(seed)
    X_train, X_val = torch.utils.data.random_split(X, lengths= [1-val_split, val_split], generator=generator)
    return DataLoader(X_train, batch_size=batch_size, shuffle=True), DataLoader(X_val, batch_size=batch_size)


def train_val_split(X: torch.Tensor, val_split=0.3, seed=None, batch_size=64, loader=True) -> any:
    indices = list(range(X.shape[0]))
    split_index = int((1-val_split) * X.shape[0])
    X_val = X[split_index:-1]
    X_train = X[0:split_index]
    if not loader:
        return X_train, X_val
    return DataLoader(X_train, batch_size=batch_size, shuffle=True), DataLoader(X_val, batch_size=batch_size)

def val_mse(model: nn.Module, X: torch.utils.data.DataLoader):
    if not isinstance(X, torch.utils.data.DataLoader):
        return val_mse_tensor(model, X)
    res = 0.0
    for i, batch in enumerate(X):
        pred = model(batch)
        batch_loss = F.mse_loss(pred, batch)
        res += batch_loss 
    return res.item()/len(X)

def val_mse_tensor(model: nn.Module, X: torch.utils.data.DataLoader):
    pred = model(X)
    loss = F.mse_loss(pred, X)
    return loss.item()



def init_train_hist(metrics:list[Metric] = None) -> dict:
    train_hist = {'train_loss': [], 'val_loss':[]} #these are bare metrics reported 
    if metrics is not None:
        for metric in metrics:
            train_hist[f"train_{metric.key}"] = []
            train_hist[f"val_{metric.key}"] = []
    return train_hist

def append_train_hist(X_train: torch.Tensor, mod: nn.Module, train_hist: dict, metrics:list[Metric] = None, X_val:torch.Tensor = None) -> dict:
    mod.eval()
    if metrics is not None:
        for metric in metrics:
            try:
                train_hist[f"train_{metric.key}"].append(metric(X=X_train,y=X_train,mod=mod, mode='train'))
            except Exception as e:
                #print(F"Exception encountered while computing metric {metric.key}: {e}")
                train_hist[f"train_{metric.key}"].append(np.nan)
            if X_val is not None:
                try:
                    train_hist[f"val_{metric.key}"].append(metric(X=X_val,y=X_val,mod=mod, mode='val'))
                except Exception as e:
                    train_hist[f"val_{metric.key}"].append(np.nan)
                    #print(F"Exception encountered while computing metric {metric.key}: {e}")
    mod.train()
    return train_hist
                      
def train(X_train: torch.Tensor, model: AutoEncoder, n_epoch:int, X_val: torch.Tensor = None, optimizer: optim.Optimizer = optim.Adam, criterion: nn.Module = nn.MSELoss(), batch_size: int=256, lr: float = 0.0001, epoch_callback=None, verbose: bool = True, metrics: list[Metric] = None, train_hist=None):
    """
    Vanilla gradient descent using Adam
    """    
    use_val = (X_val is not None)
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.Tensor(X_train).float()
        if use_val:
            X_val = torch.Tensor(X_val).float()

    if not isinstance(X_train, DataLoader):
        X_train.to(DEVICE)
        X_train = DataLoader(X_train, batch_size=batch_size)
    if use_val and not isinstance(X_val, DataLoader):
        X_val.to(DEVICE)
        X_val = DataLoader(X_val, batch_size=batch_size)
   
    optimizer = optimizer(model.parameters(), lr=lr)

    if train_hist is None:
        train_hist = init_train_hist(metrics)
    
    if isinstance(criterion, OrthoLoss):
        criterion.set_hist(train_hist) 
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, batch in enumerate(X_train):
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if use_val:
                val_loss = val_mse(model=model, X=X_val)
                train_hist['val_loss'].append(val_loss)
        if epoch_callback: #callback for e.g hyperparamer optimization
            epoch_callback(train_hist)
        train_hist['train_loss'].append(running_loss/len(X_train))
        train_hist = append_train_hist(X_train=X_train,X_val=X_val,mod=model,train_hist=train_hist,metrics=metrics)
        if verbose:
            print(f"Epoch {epoch} | {train_hist['train_loss'][-1]}", end='\r')
    return train_hist


def trainDenoising(X_train: torch.Tensor, model: AutoEncoder, n_epoch:int, X_val: torch.Tensor = None, optimizer: optim.Optimizer = optim.Adam, criterion: nn.Module = nn.MSELoss(), batch_size: int=256, lr: float = 0.0001, epoch_callback=None, verbose: bool = True, metrics: list[Metric] = None, train_hist=None):
    """
    Vanilla gradient descent using Adam
    """
    use_val = (X_val is not None)

    if not isinstance(X_train, torch.Tensor):
        X_train = torch.Tensor(X_train).float()
        if use_val:
            X_val = torch.Tensor(X_val).float()
    if not isinstance(X_train, DataLoader):
        X_train = DataLoader(X_train, batch_size=batch_size)
    if use_val and not isinstance(X_val, DataLoader):
        X_val = DataLoader(X_val, batch_size=batch_size)
   
    optimizer = optimizer(model.parameters(), lr=lr)
    
    if train_hist is None:
        train_hist = init_train_hist(metrics)

    
    if isinstance(criterion, OrthoLoss):
        criterion.set_hist(train_hist) 
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, batch in enumerate(X_train):
            batch = batch + torch.rand(size=batch.shape)/100
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if use_val:
                val_loss = val_mse(model=model, X=X_val)
                train_hist['val_loss'].append(val_loss)
        if epoch_callback: #callback for e.g hyperparamer optimization
            epoch_callback(train_hist)
        train_hist['train_loss'].append(running_loss/len(X_train))
        train_hist = append_train_hist(X_train=X_train,X_val=X_val,mod=model,train_hist=train_hist,metrics=metrics)
        if verbose:
            print(f"Epoch {epoch} | {train_hist['train_loss'][-1]}", end='\r')
    return train_hist



class MaskedMSELoss(nn.Module):
    """
    test_real = torch.Tensor([[1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1,1,1,1]])
    test_target = torch.Tensor([[1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1]])
    test_mask = torch.Tensor([[1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1]])
    train_criterion = MaskedMSELoss(test_mask) -> 0.0

    print(train_criterion(test_real,test_target))
    test_real = torch.Tensor([[1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1]])
    test_target = torch.Tensor([[1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1,1,1,1,1,1,1],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0]])
    print(train_criterion(test_real,test_target)) -> 0.3333
    """
    def __init__(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.Tensor(mask).float()
        mask.requires_grad_(False)
        self.nonzero = mask.sum()
        self.mask = mask
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return functional_MaskedMSELoss(input,target,self.mask)
    
def functional_MaskedMSELoss(input: torch.Tensor, target: torch.Tensor, mask:torch.Tensor):
    if not isinstance(mask, torch.Tensor):
        mask = torch.Tensor(mask).float()
    mask.requires_grad_(False)
    nonzero = mask.sum()
    mse_loss = F.mse_loss(input, target, reduction='none')
    masked_loss = (mask * mse_loss).sum()
    return masked_loss/nonzero
        