import ml_collections as mlc
cfg = mlc.ConfigDict()
cfg.factor_dim = 5
cfg.obs_dim = 100
cfg.p_factor = 2
cfg.covar_factor = np.eye(cfg.factor_dim)/100
cfg.p_eps = 0
cfg.covar_eps = np.eye(cfg.obs_dim)/1000
cfg.T_train = 1024
cfg.T_test = 1024
cfg.T = cfg.T_train + cfg.T_test
cfg.T = 16384