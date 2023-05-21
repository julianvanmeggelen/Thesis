import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
import sys
sys.path.append('../')
from BasicAutoEncoder.model import train_val_split, AutoEncoder, OrthoLoss, val_mse, init_train_hist, append_train_hist, functional_MaskedMSELoss
from BasicAutoEncoder.Metric import Metric

try:
    from mcmc.ErrorProcess import ErrorProcess
except:
    from ErrorProcess import ErrorProcess

def check_convergence(loss_hist: list, eps = 2e-10):
    if len(loss_hist) < 11:
        return False
    if len(loss_hist) >= 100:
        return True
    return ((np.array(loss_hist)[-11:-1] - np.array(loss_hist)[-10:]) < eps).all()

def trainMCMC(X_train: torch.Tensor, model: AutoEncoder, errorProcess: ErrorProcess, n_epoch:int, X_val:torch.Tensor, max_iter:int = None, optimizer: optim.Optimizer = optim.Adam, criterion: nn.Module = nn.MSELoss(), batch_size: int=64, lr: float = 0.0001, epoch_callback=None, verbose: bool = True,  metrics: list[Metric] = None, train_hist=None):
    """
    MCMC gradient descent
    """
    use_val = (X_val is not None)

    if not isinstance(X_train, torch.Tensor): #We don't want to use a DataLoader at this point
        X_train = torch.Tensor(X_train)
    if use_val and not isinstance(X_val, DataLoader):
        X_val= torch.Tensor(X_val)

    print(X_train.shape, X_val.shape if use_val else None)
   
    optimizer = optimizer(model.parameters(), lr=lr)
    if train_hist is None:
            train_hist = init_train_hist(metrics)
    if isinstance(criterion, OrthoLoss):
        criterion.set_hist(train_hist) 

    convergence = False
    X_arr = X_train.detach().numpy() 
    errorProcess.T = X_arr.shape[0]
    #errorProcess.initialize()
    iter=0
    while not convergence:
        y_tilde = X_arr - errorProcess.conditionalExpectation()
        for epoch in range(n_epoch):
            y_tilde_mc = y_tilde + errorProcess.sample()
            batch_dl = DataLoader(torch.Tensor(y_tilde_mc).float(), batch_size = batch_size, shuffle=True)
            running_loss = 0.0
            for i, batch in enumerate(batch_dl):
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
            train_hist['train_loss'].append(running_loss/len(batch_dl))
            train_hist = append_train_hist(X_train = X_train,X_val = X_val,mod=model,train_hist = train_hist, metrics = metrics)
            if verbose:
                print(f"Epoch {epoch} | {train_hist['train_loss'][-1]}", end='\r')
        eps = X_arr- model(X_train).detach().numpy()
        errorProcess.fit(eps)
        iter+=1
        convergence = iter > max_iter#check_convergence(train_hist['train_loss']) or (max_iter not None and iter > max_iter)
        if verbose:
            print( iter, train_hist['train_loss'][-1], end='\n')

    return train_hist


def trainMCMCMasked(X_train: torch.Tensor, weights_train:np.ndarray, model: AutoEncoder, errorProcess: ErrorProcess, n_epoch:int, X_val:torch.Tensor, weights_val:np.ndarray, max_iter:int = 50, optimizer: optim.Optimizer = optim.Adam, batch_size: int=64, lr: float = 0.0001, epoch_callback=None, verbose: bool = True,  metrics: list[Metric] = None, train_hist=None):
    """
    MCMC gradient descent
    """
    use_val = (X_val is not None)

    if not isinstance(X_train, torch.Tensor): #We don't want to use a DataLoader at this point
        X_train = torch.Tensor(X_train)
    if use_val and not isinstance(X_val, DataLoader):
        X_val= torch.Tensor(X_val)
        X_val.requires_grad_(False)
        weights_val = torch.Tensor(weights_val).float()

    print(X_train.shape, X_val.shape if use_val else None)
   
    optimizer = optimizer(model.parameters(), lr=lr)
    if train_hist is None:
            train_hist = init_train_hist(metrics)
    convergence = False
    X_arr = X_train.detach().numpy() 
    errorProcess.T = X_arr.shape[0]
    errorProcess.initialize()
    iter=0
    while not convergence:
        y_tilde = X_arr - errorProcess.conditionalExpectation()
        for epoch in range(n_epoch):
            y_tilde_mc = y_tilde + errorProcess.sample()
            tensorWithWeight  = np.stack([y_tilde_mc,weights_train])
            tensorWithWeight = np.swapaxes(tensorWithWeight,0,1) #construct correct shape for dataloader
            batch_dl = DataLoader(tensorWithWeight, batch_size = batch_size, shuffle=True)
            running_loss = 0.0
            for i, batch in enumerate(batch_dl):
                batch_weight = batch[:,1,:].float()
                batch_y = batch[:,0,:].float()
                optimizer.zero_grad()
                out = model(batch_y)
                loss = functional_MaskedMSELoss(out, batch_y, batch_weight)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if use_val:
                    val_loss = torch.mean(weights_val*F.mse_loss(model(X_val), X_val, reduction='none')).item()
                    train_hist['val_loss'].append(val_loss)
            if epoch_callback: #callback for e.g hyperparamer optimization
                epoch_callback(train_hist)
            train_hist['train_loss'].append(running_loss/len(batch_dl))
            train_hist = append_train_hist(X_train = X_train,X_val = X_val,mod=model,train_hist = train_hist, metrics = metrics)
            if verbose:
                print(f"Epoch {epoch} | {train_hist['train_loss'][-1]}", end='\r')
        eps = X_arr- model(X_train).detach().numpy()
        errorProcess.fit(eps)
        iter+=1
        convergence = iter > max_iter#check_convergence(train_hist['train_loss']) or (max_iter not None and iter > max_iter)
        if verbose:
            print( iter, train_hist['train_loss'][-1], end='\n')

    return train_hist