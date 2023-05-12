import ml_collections as mlc
import numpy as np
import torch.nn as nn

cfg = mlc.ConfigDict()


#Data config
cfg.factor_dim = 10
cfg.obs_dim = 100
cfg.p_factor = 2
cfg.covar_factor = np.eye(cfg.factor_dim)/100
cfg.p_eps = 0
cfg.covar_eps = np.eye(cfg.obs_dim)/1000
cfg.T_train = 8192
cfg.T_val = 1024
#cfg.T = cfg.T_train + cfg.T_test + cfg.T_val
cfg.T = 16384
cfg.T_test = cfg.T - cfg.T_train-cfg.T_val

cfg.use_default_data = True
cfg.saved_index = 3 #gdp to be used


#Training config
cfg.batch_size = 512
cfg.n_epoch = 30
cfg.max_iter = 30
cfg.lr = 0.0001


#Model config: must be filled in by experiment
cfg.enc_hidden_dim : list[int] = None
cfg.dec_hidden_dim : list[int] = None
cfg.enc_activation: nn.Module = None
cfg.dec_activation: nn.Module = None
cfg.enc_last_layer_linear = False
cfg.dec_last_layer_linear = False
