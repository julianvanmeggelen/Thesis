import sys
sys.path.append('../')

import torch.nn as nn
import torch
import numpy as np
from BasicAutoEncoder.model import Encoder, Decoder, AutoEncoder, SelfAttentionEncoder, OrthoLoss
from typing import TypedDict
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.util import varsim
from typing import Tuple

def getSaved(index: int = 1, T:int = 2048) -> Tuple[np.ndarray, np.ndarray, nn.Module]:
    y = np.load(f"../DGP//saved/y_{index}.npy")[:T]
    f = np.load(f"../DGP/saved/f_{index}.npy")[:T]
    dec = torch.load(f"../DGP/saved/dec_{index}.pt")
    return f,y,dec

def simulateVar1(x0, delta, lamda, T, mu, omega, warmup): #Var 1
    assert delta.shape[0] == x0.shape[0] == lamda.shape[0] == x0.shape[0] == omega.shape[0] == mu.shape[0], 'Incorrect shapes'
    eigval, _ = np.linalg.eig(lamda)
    if not (np.abs(eigval)<=1.).all(): print(f"Warning: this system is unstable. Eigenvalues: {eigval}")
    res = [x0]
    for t in range(T):
        residuals = np.random.multivariate_normal(mu,omega)
        xt = delta + lamda @ res[-1] + residuals
        res.append(xt)
    y = np.array(res)
    y= (y-y.min(axis=0))/(y.max(axis=0)-y.min(axis=0))
    return y[int(warmup*T):]

def simulateRandomVar1(k, T=100, warmup=0.1): #VAR 1
    D = np.diag(np.random.uniform(0,1, size=(k))) #eigenvalues
    V = np.random.normal(size=(k,k), scale = np.eye(k)+0.1)
    #V = np.random.multivariate_normal(np.zeros(k),  np.eye(k)+0.1)
    #lamda = np.linalg.inv(V) @ D @ V #similarity transform
    lamda = D
    #print(np.linalg.eig(lamda))
    x0 = np.random.normal(size=(k))
    delta = np.random.normal(size=(k))
    mu = np.zeros(k)
    omega = np.identity((k))
    res = simulateVar1(x0,delta,lamda,T,mu,omega,warmup)
    return res

class Var1Params(TypedDict):
    x0: np.ndarray
    delta: np.ndarray
    lamda: np.ndarray
    mu: np.ndarray
    omega: np.ndarray

def getSimulatedNonlinear(factor_dim:int, obs_dim:int, T:int, dec: Decoder = None, params: Var1Params = None, warmup: float=0.0, **kwargs):
    if not params:
        f = simulateRandomVar1(k=factor_dim, T=T, warmup=warmup)
    else:
        f = simulateVar1(T = T, warmup = warmup, **params)
    if not dec:
     dec = Decoder(hidden_dim = [factor_dim, obs_dim], **kwargs)
    print(dec)
    f_tensor = torch.Tensor(f).float()
    y = dec(f_tensor).detach().numpy()
    return f, y


def getSimulatedNonlinearVarP(factor_dim:int, obs_dim:int, T:int, dec: Decoder = None, warmup: float=0.0, p: int = 5, p_eps: int = 0, covar_eps:np.ndarray = 1/30, covar_factor: np.ndarray = None, centered:bool=True, normalized: bool =False):
    f = simulateRandomVarP(d=factor_dim,p=p,T=T,T_warmup=400, diagonal=True, covar=covar_factor)
    if not dec:
        dec = Decoder(hidden_dim = [factor_dim, obs_dim])
    f_tensor = torch.Tensor(f).float()

    y = dec(f_tensor).detach().numpy()

    if p_eps == 0:
        obs_residual = np.random.multivariate_normal(np.zeros(obs_dim), covar_eps, size=(T))
    else:
        obs_residual = simulateRandomVarP(d=obs_dim,p=p_eps,T=T,T_warmup=400, diagonal=True, covar=covar_eps)

    if centered:
        y = y-y.mean(axis=0)
    if normalized:
        y_min = y.min(axis=0)
        y_max = y.max(axis=0)
        y = (y-y_min)/(y_max-y_min)

    y += obs_residual
    return f, y, obs_residual


def getVarCoeffient(d:int, p: int, covar:np.ndarray=None, l:float=1.01, diagonal: bool = False):
    """
    Iteratively sample coefficients and adjust untill a stable system is obtained
    """
    stable = False
    locs = [np.random.uniform(0,1, size=d)/lag for lag in range(1,p+1)]
    scale = np.eye(d)+0.1 

    n_iter = 0
    if not diagonal:
        coeff = [np.eye(d)*3 + np.random.uniform(-(p-_+1),p-_+1, size=(d,d)) for _ in range(1,p+1)]
    else:
        coeff = [np.diag(np.random.uniform(-1/p,1/p, size=d))*10 for _ in range(1,p+1)]
        coeff = [np.eye(d)*3/_  for _ in range(1,p+1)]        
    #coeff = [np.eye(d)*3/_  for _ in range(1,p+1)]
    #coeff = [np.diag(np.random.uniform(-1/p,1/p, size=d))*10 for _ in range(1,p+1)]
    if covar is None:
        covar = np.eye(d) #this is not needed as we just want to check stability condition
    coefs = np.stack(coeff, axis=0)

    while not stable:
        print(n_iter,end='\r')
        #coeff = [np.random.multivariate_normal(loc, scale, size=(d)) for loc in locs]
        #coefs = np.stack(coeff, axis=0)
        proc = VARProcess(coefs = coefs, coefs_exog = np.array([0]), sigma_u = covar)
        stable = proc.is_stable(verbose=False)
        if not stable:
            coefs = coefs / l
            #locs = [loc/l for loc in locs]
        n_iter+=1
    print(f"Obtained stable system after {n_iter} iterations.")
    return coefs, proc

def simulateVarP(coefs: np.ndarray, covar:np.ndarray=None, initial_values=None, T=200):
    """
    coefs: (p,d,d)
    """
    p,d,_ = coefs.shape
    if initial_values is None:
        initial_values = np.random.normal(size=(d,p))
    if covar is None:
        covar = np.eye(d)
    assert initial_values.shape == (d,p)
    #res = np.zeros(shape=(d,T))
    res = np.random.multivariate_normal(np.zeros(d),covar, size=T+p).T
    res[:,0:p] = initial_values
    #resids = np.random.multivariate_normal(np.ones(d),covar, size=T)
    for t in range(p,T+p):
        res[:,t] += np.tensordot(coefs.T, res[:,t-p:t]) 
    return res[:,p:]

    
def simulateRandomVarP(d:int, p:int, T:int, T_warmup = 100, covar: np.ndarray = None, diagonal: bool = False):
    eps= 0
    if covar is None:
        covar = (np.eye(d) + eps)/(1+eps)/10
    coeff, proc = getVarCoeffient(d=d,p=p,diagonal=diagonal)
    return simulateVarP(coefs=coeff, covar=covar, T=T+T_warmup).T[T_warmup:,:]
    return varsim(coeff, intercept=np.zeros(d), sig_u=covar, steps=T, initvalues= np.random.multivariate_normal(np.zeros(d), covar, size=(d,p)))[T_warmup:,:]
    return proc.simulate_var(steps = T, initvalues = np.random.multivariate_normal(np.zeros(d), covar, size=d))

