"""
cProfile progam analysis
"""


import sys
sys.path.append('../')
import cProfile
from datetime import datetime
import argparse

import sys
import ml_collections as mlc
import torch.nn as nn
import matplotlib
import numpy as np

matplotlib.rcParams['figure.figsize'] = (25, 5)

from DGP import dgp 
from BasicAutoEncoder.model import Encoder, Decoder, AutoEncoder, train
import BasicAutoEncoder
from BasicAutoEncoder.util import plot_factor_estimates, plot_train_hist, plot_reconstructed_y, get_trainable_params
from ErrorProcess import IIDErrorProcess
from mcmc import trainMCMC


cfg = mlc.ConfigDict()
cfg.factor_dim = 5
cfg.obs_dim = 20
cfg.p_factor = 2
cfg.covar_factor = np.eye(cfg.factor_dim)/100
cfg.p_eps = 5
cfg.covar_eps = np.eye(cfg.obs_dim)/1000
cfg.T = 100

dec = Decoder(hidden_dim=[cfg.factor_dim, 30, cfg.obs_dim], activation=nn.Tanh(), lastLayerLinear=False)
f, y, obs_residual = dgp.getSimulatedNonlinearVarP(factor_dim=cfg.factor_dim,p=cfg.p_factor, obs_dim=cfg.obs_dim,T=cfg.T, dec=dec, covar_factor = cfg.covar_factor, p_eps = cfg.p_eps,covar_eps=cfg.covar_eps)





def main():
    dec = Decoder(hidden_dim=[cfg.factor_dim, 30, cfg.obs_dim], activation=nn.Tanh(), lastLayerLinear=False)
    enc = Encoder(hidden_dim=[cfg.obs_dim, 30, cfg.factor_dim], activation=nn.Tanh(), lastLayerLinear=False)
    mod = AutoEncoder(enc=enc, dec=dec)
    print(f"Number of trainanle paramaters {get_trainable_params(mod)}")
    errorProcess = IIDErrorProcess(n=cfg.obs_dim, T = cfg.T)
    train_hist = trainMCMC(X=y,model=mod, errorProcess = errorProcess, n_epoch=100, lr = 0.0001)


def printProfile(f):
    import pstats
    p = pstats.Stats('./profile/' + f)
    p.sort_stats('tottime').print_stats()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", help="increase output verbosity")
    args = parser.parse_args()

    filename = None

    if args.f:
        if args.f == '-1':
            import os
            print("yes")
            filenames  = os.listdir('./profile/')
            filename = sorted(filenames)[-1]
        else:
            filename = args.f
        printProfile(filename)

       

    else:
        now = datetime.now() # current date and time
        filename = f"profile_{now.strftime('%m%d%Y_%H%M%S')}"
        filedir = './profile/' + filename 
        cProfile.run('main()', filename=filedir)
        printProfile(filename)