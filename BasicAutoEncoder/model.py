import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset

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
        return reprLoss + self.alpha * orthogonalLoss + torch.nn.functional.mse_loss(torch.mean(f, dim=0),torch.zeros(f.shape[1]))
        
class Encoder(nn.Module):
    def __init__(self, hidden_dim: list = [500,200,100], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = False, lastLayerLinear: bool=False):
        super().__init__()
        self.n_hidden = len(hidden_dim)
        self.hidden_dim: list = hidden_dim
        self.activation: nn.Module = activation
        self.use_batchnorm: bool = use_batchnorm
        self.lastLayerLinear = lastLayerLinear
        self.sequential = self._get_sequential()


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
        out = self.sequential(x)
        return out

class Decoder(nn.Module):
    def __init__(self, linear: bool = False, hidden_dim: list = [100,200,500], activation: nn.Module = nn.Tanh(), use_batchnorm: bool = False, lastLayerLinear: bool=False):
        super().__init__()
        assert (linear and len(hidden_dim) == 2) or (not linear)
        self.n_hidden = len(hidden_dim)
        self.hidden_dim: list = hidden_dim
        self.activation: nn.Module = activation
        self.use_batchnorm: bool = use_batchnorm
        self.lastLayerLinear = lastLayerLinear
        self.sequential = self._get_sequential()

    def _get_sequential(self): #compile to nn.Sequential
        res = nn.Sequential()
        for i, lin in enumerate(self.hidden_dim[:-1]):
            res.append(nn.Linear(self.hidden_dim[i],self.hidden_dim[i+1]))

            if i < self.n_hidden -2:
                res.append(self.activation)
            if self.use_batchnorm and i < self.n_hidden-2:
                res.append(nn.BatchNorm1d(self.hidden_dim[i+1]))

        if self.lastLayerLinear:
            res[-1] = nn.Identity()
        return res

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
    res = 0.0
    for i, batch in enumerate(X):
        pred = model(batch)
        batch_loss = F.mse_loss(pred, batch)
        res += batch_loss 
    return res.item()/len(X)
    
def train(X: torch.Tensor, model: AutoEncoder, n_epoch:int, optimizer: optim.Optimizer = optim.Adam, criterion: nn.Module = nn.MSELoss(), batch_size: int=64, lr: float = 0.0001, val_split = 0.3, use_val = True, epoch_callback=None, verbose: bool = True):
    """
    Vanilla gradient descent using Adam
    """
    if use_val:
        X_train, X_val = train_val_split(X, val_split=val_split, batch_size=batch_size)
        print(len(X), len(X_val))
    else:
        X_train = DataLoader(X, batch_size=batch_size)
   
    optimizer = optimizer(model.parameters(), lr=lr)
    train_hist = {'loss': [], 'val_loss':[]} 
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
        train_hist['loss'].append(running_loss/len(X_train))
        if verbose:
            print(f"Epoch {epoch} | {train_hist['loss'][-1]}", end='\r')
    return train_hist