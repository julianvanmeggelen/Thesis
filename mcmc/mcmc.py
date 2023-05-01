import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
import sys
sys.path.append('../')
from BasicAutoEncoder.model import train_val_split, AutoEncoder, OrthoLoss, val_mse
from mcmc.ErrorProcess import ErrorProcess

def check_convergence(loss_hist: list, eps = 1e-10):
    if len(loss_hist) < 11:
        return False
    if len(loss_hist) > 500:
        return True
    return ((np.array(loss_hist)[-11:-1] - np.array(loss_hist)[-10:]) < eps).all()

def trainMCMC(X: torch.Tensor, model: AutoEncoder, errorProcess: ErrorProcess, n_epoch:int, optimizer: optim.Optimizer = optim.Adam, criterion: nn.Module = nn.MSELoss(), batch_size: int=64, lr: float = 0.0001, val_split = 0.3, use_val = True, epoch_callback=None, verbose: bool = True):
    """
    MCMC gradient descent
    """
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    if use_val:
        X_train, X_val = train_val_split(X, val_split=val_split, batch_size=batch_size, loader=False)
        print(len(X), len(X_val))
    else:
        X_train = X# DataLoader(X, batch_size=batch_size)
   
    optimizer = optimizer(model.parameters(), lr=lr)
    train_hist = {'loss': [], 'val_loss':[]} 
    if isinstance(criterion, OrthoLoss):
        criterion.set_hist(train_hist) 

    convergence = False
    X_arr = X_train.detach().numpy() 
    errorProcess.T = X_arr.shape[0]
    errorProcess.initialize()
    iter=0
    while not convergence:
        y_tilde = X_arr - errorProcess.conditionalExpectation()
        for epoch in range(n_epoch):
            y_tilde_mc = y_tilde + errorProcess.sample()
            batch_dl = DataLoader(torch.Tensor(y_tilde_mc).float())
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
            train_hist['loss'].append(running_loss/len(batch_dl))
            if verbose:
                print(f"Epoch {epoch} | {train_hist['loss'][-1]}", end='\r')
        eps = X_arr- model(X_train).detach().numpy()
        errorProcess.fit(eps)
        iter+=1
        convergence = check_convergence(train_hist['loss'])
        if verbose:
            print( iter, train_hist['loss'][-1], end='\n')

    return train_hist