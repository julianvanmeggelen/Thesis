from ray import tune
from ray.air import session
from ray import air, tune
from typing import TypedDict
import torch.nn as nn
from torch import optim
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from model import Encoder, Decoder, AutoEncoder, train



"""
This hyperparameter optimization is only focussed on the autoencoder nn, not on time series params s.a factor lags etc.
"""


#Type definitions
class Config(TypedDict):
    enc_hidden: int
    dec_hidden: int
    factor_dim: int
    enc_activation: nn.Module
    dec_activation: nn.Module
    lr: float
    n_epoch: int
    optimizer: optim.Optimizer
    val_split: float
    batch_size: int


DEFAULT_SEARCH_SPACE: Config = {
    'enc_hidden': tune.grid_search([0, 20, 40]),
    'dec_hidden': tune.grid_search([0, 20, 40]),
    'factor_dim': 10,
    'enc_activation': tune.grid_search(["Identity", "Tanh", "Sigmoid"]),
    'dec_activation': tune.grid_search(["Identity", "Tanh", "Sigmoid"]),
    'lr': 0.0001, #tune.grid_search([0.0001, 0.0005, 0.001]),
    'n_epoch': 500,
    'optimizer': optim.Adam,
    'val_split': 0.3,
    'batch_size': 64
}

ACTIVATION_MAP  ={
    "Identity": nn.Identity(),
    "Tanh": nn.Tanh(),
    "Sigmoid": nn.Sigmoid(),
    "ReLU": nn.ReLU()
}

def train_hyperopt(config:Config, X:nn.Module):
    #Model specs
    input_dim = X.shape[1]
    enc_hidden = config['enc_hidden']
    dec_hidden = config['enc_hidden']
    factor_dim = config['factor_dim']
    enc_hidden_dim  = [input_dim, enc_hidden, factor_dim] if enc_hidden > 0 else [input_dim, factor_dim]
    dec_hidden_dim  = [factor_dim, dec_hidden, input_dim] if dec_hidden > 0 else [factor_dim, input_dim]
    enc_activation = ACTIVATION_MAP[config['enc_activation']]
    dec_activation = ACTIVATION_MAP[config['dec_activation']]

    enc = Encoder(hidden_dim = enc_hidden_dim, activation=enc_activation, lastLayerLinear=False) 
    dec = Decoder(hidden_dim = dec_hidden_dim, activation=dec_activation)
    model = AutoEncoder(enc=enc, dec=dec)

    #Training specs
    lr = config['lr']
    n_epoch = config['n_epoch']
    batch_size = config['batch_size']
    optimizer = config['optimizer']
    val_split = config['val_split']

    #Training callback
    epoch_callback = lambda train_hist: session.report({'score': train_hist['val_loss'][-1]})

    train_hist = train(X=X,model=model,n_epoch=n_epoch, optimizer=optimizer,batch_size = batch_size, lr=lr, val_split=val_split, use_val=True, epoch_callback=epoch_callback, criterion=nn.MSELoss(), verbose=False)


def hyper_optimizer(X, search_space: Config = DEFAULT_SEARCH_SPACE):
    tuner = tune.Tuner(
        trainable = lambda config: train_hyperopt(config, X),
        param_space=search_space
    )
    results = tuner.fit()
    return results
   