from ray import tune
from ray.air import session
from ray import air, tune
import numpy as np
from typing import TypedDict
import torch.nn as nn
from torch import optim
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONPATH'] = '/Users/julianvanmeggelen/Documents/Studie/2022:23/Thesis/mcmc'


import sys
sys.path.append('../')

import BasicAutoEncoder
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


DEFAULT_SEARCH_SPACE: HyperOptConfig = {
    'hidden': tune.grid_search([2, 3, 4, 5, 6]),
    'activation': tune.grid_search(["Identity", "Tanh", "Sigmoid", "ReLU"]),
    'lr': tune.grid_search([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
}

ACTIVATION_MAP  ={
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
   
def mcmc_hyperopt(dgpIndex: int):
    cfg = defaultCfg
    f,y,dec = dgp.getSaved(dgpIndex, T=cfg.T)
    cfg.factor_dim = f.shape[1]
    cfg.obs_dim = y.shape[1]
    f_train = f[0:cfg.T_train]
    f_val = f[cfg.T_train:cfg.T_train+cfg.T_val]
    f_test = f[cfg.T_train+cfg.T_val:]

    y_train = y[0:cfg.T_train]
    y_val = y[cfg.T_train:cfg.T_train+cfg.T_val]
    y_test = y[cfg.T_train+cfg.T_val:]


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