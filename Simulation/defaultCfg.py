import ml_collections as mlc
import numpy as np
cfg = mlc.ConfigDict()
cfg.factor_dim = 10
cfg.obs_dim = 100
cfg.p_factor = 2
cfg.covar_factor = np.eye(cfg.factor_dim)/100
cfg.p_eps = 0
cfg.covar_eps = np.eye(cfg.obs_dim)/1000
cfg.T_train = 4096
cfg.T_test = 2048
cfg.T_val = 512
cfg.T = cfg.T_train + cfg.T_test + cfg.T_val
cfg.use_default_data = True
cfg.saved_index = 2 #gdp to be used


#Training config
cfg.batch_size = 512
cfg.n_epoch = 100
cfg.max_iter = 15
cfg.n_epoch = 20
cfg.lr = 0.0001