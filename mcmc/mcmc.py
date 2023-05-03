import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset
import sys
sys.path.append('../')
from BasicAutoEncoder.model import train_val_split, AutoEncoder, OrthoLoss, val_mse, init_train_hist, append_train_hist
from BasicAutoEncoder.Metric import Metric

from mcmc.ErrorProcess import ErrorProcess

def check_convergence(loss_hist: list, eps = 1e-10):
    if len(loss_hist) < 11:
        return False
    if len(loss_hist) >= 100:
        return True
    return ((np.array(loss_hist)[-11:-1] - np.array(loss_hist)[-10:]) < eps).all()

def trainMCMC(X_train: torch.Tensor, model: AutoEncoder, errorProcess: ErrorProcess, n_epoch:int, X_val:torch.Tensor, optimizer: optim.Optimizer = optim.Adam, criterion: nn.Module = nn.MSELoss(), batch_size: int=64, lr: float = 0.0001, epoch_callback=None, verbose: bool = True,  metrics: list[Metric] = None):
    """
    MCMC gradient descent
    """
    use_val = (X_val is not None)
    if not isinstance(X_train, torch.Tensor): #We don't want to use a DataLoader at this point
        X_train = torch.Tensor(X_train)
    if use_val and not isinstance(X_val, DataLoader):
        X_val= torch.Tensor(X_val)
   
    optimizer = optimizer(model.parameters(), lr=lr)
    train_hist = init_train_hist(metrics)

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
            train_hist['train_loss'].append(running_loss/len(batch_dl))
            train_hist = append_train_hist(X_train = X_train,X_val = X_val,mod=model,train_hist = train_hist, metrics = metrics)
            if verbose:
                print(f"Epoch {epoch} | {train_hist['train_loss'][-1]}", end='\r')
        eps = X_arr- model(X_train).detach().numpy()
        errorProcess.fit(eps)
        iter+=1
        convergence = check_convergence(train_hist['train_loss'])
        if verbose:
            print( iter, train_hist['train_loss'][-1], end='\n')

    return train_hist