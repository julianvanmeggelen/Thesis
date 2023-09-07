from ray import tune
from ray.air import session
from ray import air, tune
import numpy as np
from typing import TypedDict
import torch.nn as nn
from torch import optim
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONPATH'] = '/Users/julianvanmeggelen/Documents/Studie/2022:23/Thesis/'


import sys
sys.path.append('../')

from BasicAutoEncoder.model import Encoder, Decoder, AutoEncoder, train
from Simulation.defaultCfg import cfg as defaultCfg
from DGP import dgp
from ErrorProcess import IIDErrorProcess
from mcmc import trainMCMC




"""
This hyperparameter optimization is only focussed on the autoencoder nn, not on time series params s.a factor lags etc.
"""


#Type definitions
class HyperOptConfig(TypedDict):
    hidden: int
    activation: str
    lr: float
    dropout: float
    n_iter: int


DEFAULT_SEARCH_SPACE: HyperOptConfig = {
    'hidden': tune.grid_search([2, 4, 6, 8, 10]),
    'activation': tune.grid_search(["Tanh", "Sigmoid", "ReLU"]),
    'lr': tune.grid_search([0.0001, 0.0005, 0.001, 0.005]),
    'dropout': 0#tune.quniform(0.0,0.9,0.1),
    'n_iter': tune.quniform(0.0,100,10)
}

ACTIVATION_MAP  = {
    "Identity": nn.Identity(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "ReLU": nn.ReLU()
}

def train_hyperopt(hConfig:HyperOptConfig, cfg, y_train, y_test, y_val, f_train, f_test, f_val):
    #Cast hyperopt config to standard config: overwrite defaults by hyperconfig
    hidden = hConfig['hidden']
    cfg.enc_hidden_dim = list(np.linspace(cfg.obs_dim,cfg.factor_dim,hidden+2).astype(int)) #pyramid arch
    cfg.dec_hidden_dim = list(np.linspace(cfg.factor_dim,cfg.obs_dim,hidden+2).astype(int)) #pyramid arch
    cfg.enc_activation = ACTIVATION_MAP[hConfig['activation']]
    cfg.dec_activation = ACTIVATION_MAP[hConfig['activation']]

    cfg.lr = hConfig['lr']

    cfg.enc_last_layer_linear = False
    cfg.dec_last_layer_linear = True


    print(cfg.enc_hidden_dim)
    dec = Decoder(hidden_dim=cfg.dec_hidden_dim, activation=cfg.dec_activation, lastLayerLinear=cfg.dec_last_layer_linear)
    enc = Encoder(hidden_dim=cfg.enc_hidden_dim, activation=cfg.enc_activation, lastLayerLinear=cfg.enc_last_layer_linear)
    mod = AutoEncoder(enc=enc, dec=dec)
    errorProcess = IIDErrorProcess(n=cfg.obs_dim, T = cfg.T)

    #Training callback
    epoch_callback = lambda train_hist: session.report({'score': train_hist['val_loss'][-1]})

    train_hist = trainMCMC(X_train=y_train, X_val = y_val, model=mod, errorProcess = errorProcess, n_epoch=cfg.n_epoch, lr = cfg.lr, batch_size=cfg.batch_size, epoch_callback=epoch_callback, max_iter = cfg.max_iter, verbose=False)
   

def train_hyperopt_ew(hConfig:HyperOptConfig, cfg, y_train, y_test, y_val, f_train, f_test, f_val, stepsize=20, folds = 10):
    #Cast hyperopt config to standard config: overwrite defaults by hyperconfig
    hidden = hConfig['hidden']
    cfg.enc_hidden_dim = list(np.linspace(cfg.obs_dim,cfg.factor_dim,hidden+2).astype(int)) #pyramid arch
    cfg.dec_hidden_dim = list(np.linspace(cfg.factor_dim,cfg.obs_dim,hidden+2).astype(int)) #pyramid arch
    cfg.enc_activation = ACTIVATION_MAP[hConfig['activation']]
    cfg.dec_activation = ACTIVATION_MAP[hConfig['activation']]

    cfg.lr = hConfig['lr']

    cfg.enc_last_layer_linear = False
    cfg.dec_last_layer_linear = True

    #combine val and train set because we will be doing custom folds
    f_train = np.concatenate([f_train, f_val])
    y_train = np.concatenate([y_train, y_val])


    print(cfg.enc_hidden_dim)
    dec = Decoder(hidden_dim=cfg.dec_hidden_dim, activation=cfg.dec_activation, lastLayerLinear=cfg.dec_last_layer_linear)
    enc = Encoder(hidden_dim=cfg.enc_hidden_dim, activation=cfg.enc_activation, lastLayerLinear=cfg.enc_last_layer_linear)
    mod = AutoEncoder(enc=enc, dec=dec)
    errorProcess = IIDErrorProcess(n=cfg.obs_dim, T = cfg.T, init_mu = 0.0, init_sigma= 0.001)

    #Training callback
    epoch_callback = lambda train_hist: session.report({'val_loss': train_hist['val_loss'][-1]})

    T_total = y_train.shape[0]
    T_train_begin = T_total - folds * stepsize
    total_val_loss = 0.0
    folds_list = list(range(T_train_begin, T_total, stepsize))
    for t in folds_list:
        y_train_fold = y_train[0:t]
        y_val_fold = y_train[t:t+stepsize]
        train_hist = trainMCMC(X_train=y_train_fold, X_val = y_val_fold, model=mod, errorProcess = errorProcess, n_epoch=cfg.n_epoch, lr = cfg.lr, batch_size=cfg.batch_size, epoch_callback=epoch_callback, max_iter = cfg.max_iter, verbose=False)
        val_loss = train_hist['val_loss'][-1]
        total_val_loss += val_loss
    val_loss_mean = total_val_loss/folds
    session.report({'score': val_loss_mean})


   

def mcmc_hyperopt(dgpIndex: int, T_train=None, T_val=None, useFolds = False, stepsize=20,folds=10):
    cfg = defaultCfg
    f,y,dec = dgp.getSaved(dgpIndex, T=cfg.T)
    cfg.factor_dim = f.shape[1]
    cfg.obs_dim = y.shape[1]
    cfg.n_epoch = 1
    cfg.n_iter = 15

    if T_train is not None:
        cfg.T_train = T_train

    if T_val is not None:
        cfg.T_val = T_val

    f_train = f[0:cfg.T_train]
    f_val = f[cfg.T_train:cfg.T_train+cfg.T_val]
    f_test = f[cfg.T_train+cfg.T_val:]

    y_train = y[0:cfg.T_train]
    y_val = y[cfg.T_train:cfg.T_train+cfg.T_val]
    y_test = y[cfg.T_train+cfg.T_val:]

    tuner = None
    if useFolds:
         tuner = tune.Tuner(
            trainable = lambda config: train_hyperopt_ew(config, cfg, y_train, y_test, y_val, f_train, f_test, f_val, stepsize=stepsize,folds=folds),
            param_space=DEFAULT_SEARCH_SPACE
        )
    else:
        tuner = tune.Tuner(
            trainable = lambda config: train_hyperopt(config, cfg, y_train, y_test, y_val, f_train, f_test, f_val),
            param_space=DEFAULT_SEARCH_SPACE
        )
    results = tuner.fit()
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Hyperoptimize #hidden, activation and learning rate for dgp i')
    parser.add_argument('i', type=int,
                        help='Which dgp')

    args = parser.parse_args()
    i = args.i
    mcmc_hyperopt(i)