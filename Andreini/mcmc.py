%load_ext autoreload
%autoreload 2
import numpy as np
import sys
from statsmodels.tsa.arima.model import ARIMA 
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.functional as F
from torch import optim
from sklearn.decomposition import PCA

sys.path.append('../')

from BasicAutoEncoder.model import Encoder, Decoder, AutoEncoder, SelfAttentionEncoder, OrthoLoss
from Andreini_data.data import load_y
from torch.autograd.functional import jacobian


# A step-up version where error terms epsilon_t are iid instead of ar process
def check_convergence(loss_hist: list, eps = 1e-10):
    if len(loss_hist) < 11:
        return False
    if len(loss_hist) > 1000:
        return True
    return ((np.array(loss_hist)[-11:-1] - np.array(loss_hist)[-10:]) < eps).all()

def init_eps(T:int,n:int):
    """
    Initinitalize idiosyncratic error terms.
    """
    #TODO: How to do this lol?
    eps = np.random.normal(size = (T,n), scale=0.01)
    return eps, np.mean(eps,axis=0), np.var(eps, axis=0)

def d2FMMCMC_iid(y: np.ndarray, model: AutoEncoder, n_epoch:int, optimizer: optim.Optimizer, criterion: nn.Module):
    n = y.shape[1]
    T = y.shape[0]
    eps, mu_eps, sigma_eps = init_eps(T,n)
    convergence = False

    hist_keys = ['loss', 'eps', 'mu_eps', 'var_eps']
    train_hist = {k: [] for k in hist_keys}

    y_tensor = torch.from_numpy(y).float()
    i = 0
    while not convergence:
        y_tilde = y - mu_eps
        for epoch in range(n_epoch):
            y_tilde_mc = y_tilde + np.random.multivariate_normal(mu_eps, np.diag(sigma_eps), size=(T))
            optimizer.zero_grad()
            model_in = torch.from_numpy(y_tilde_mc).float()
            out = model(model_in)
            loss  = criterion(out, model_in)
            loss.backward()
            optimizer.step()
            train_hist['loss'].append(loss.item())
        eps = y - model(y_tensor).detach().numpy()
        sigma_eps = np.var(eps, axis=0)
        mu_eps = np.mean(eps,axis=0)
        train_hist['eps'].append(eps)
        train_hist['mu_eps'].append(mu_eps)
        train_hist['var_eps'].append(sigma_eps)
        print(i, train_hist['loss'][-1], end='\r')
        i+=1
        convergence = check_convergence(train_hist['loss'])
    return model, train_hist


def init_eps(T:int,n:int):
    """
    Initinitalize idiosyncratic error terms.
    """
    #TODO: How to do this lol?
    return np.random.normal(size = (T,n))

def check_convergence(loss_hist: list, eps = 1e-10):
    if len(loss_hist) < 11:
        return False
    if len(loss_hist) > 1000:
        return True
    return ((np.array(loss_hist)[-11:-1] - np.array(loss_hist)[-10:]) < eps).all()

def d2FMMCMC(y: np.ndarray, model: AutoEncoder, n_epoch:int, optimizer: optim.Optimizer, criterion: nn.Module):
    n = y.shape[1]
    T = y.shape[0]
    eps = init_eps(T,n)
    phi = ArProcess(T,n,d=1).fit(eps)
    convergence = False
    hist_keys = ['loss', 'coeff', 'sigma2']
    train_hist = {k: [] for k in hist_keys}    
    y_tensor = torch.from_numpy(y).float()
    i = 0
    while not convergence:
        #train_hist['coeff'].append(phi.coefficients)
        #train_hist['sigma2'].append(phi.sigma2)

        y_tilde = y - phi.conditionalExpectation()
        for epoch in range(n_epoch):
            y_tilde_mc = y_tilde+phi.simulate()

            optimizer.zero_grad()
            model_in = torch.from_numpy(y_tilde_mc).float()
            out = model(model_in)
            loss  = criterion(out, model_in)
            loss.backward()
            optimizer.step()
            train_hist['loss'].append(loss.item())
        eps = y - model(y_tensor).detach().numpy()
        phi = ArProcess(T,n,d=1).fit(eps)
        print(i, train_hist['loss'][-1], end='\r')
        i+=1
        convergence = check_convergence(train_hist['loss'])
    return phi, model, train_hist



class ErrorProcess(object):
    """
    Generic interface for the observation error process. Must have functionality to:
    - Initialize the process
    - Sample the conditionalExpectation of the process
    - Re-estimate the process
    """
    def __init__(self, n:int, T:int):
        """
        n: Number of vars
        T: Length of data
        """
        self.n = n
        self.T - T
    
    def initialize(self) -> None:
        """
        Initialize the process (without having data to fit)
        """
        raise NotImplementedError

    def fit(self, resids: np.ndarray) -> None:
        """
        Fit the process based on incoming residuals
        resids: np.ndarray with shape (n, T)
        """
        raise NotImplementedError
    
    def conditionalExpectation(self) -> np.ndarray:
        """
        Return conditional expectation E[eps_t|eps_{t-1}] for t =1,..T
        returns:
            res: np.ndarray with shape (n,T)
        """
        raise NotImplementedError
    
    def sample(self) -> np.ndarray:
        """
        Return sampled path eps_t for t =1,..T
        returns:
            res: np.ndarray with shape (n,T)
        """
        raise NotImplementedError

    



    