import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from BasicAutoEncoder.model import AutoEncoder

def get_trainable_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def plot_train_hist(train_hist: dict, log:bool=True):
    plt.figure(figsize=(5,3))
    plt.plot(np.log(train_hist['train_loss']) if log else train_hist['train_loss'], label='Train')
    plt.plot(np.log(train_hist['val_loss']) if log else train_hist['val_loss'], label='Val')
    plt.legend()

def plot_factor_estimates(model: AutoEncoder, X: np.ndarray):
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    plt.plot(model.enc(X).detach().numpy());
    plt.title('Deep factor')


def plot_reconstructed_y(model: AutoEncoder, X: np.ndarray, **kwargs):
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    plt.plot(model(X).detach().numpy(), **kwargs);
    plt.title('Reconstructed $y$')