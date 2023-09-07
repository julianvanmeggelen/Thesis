from statsmodels.tsa.vector_ar.var_model import VAR
import numpy as np
import torch 
from Andreini_data.data import load_y
import ml_collections as mlc 
y, mask, index, columns= load_y(daterange=['1950-01-01', '2024-01-01'])
y[np.isnan(y)] = 0
cfg = mlc.ConfigDict()
cfg.factor_dim = 5
cfg.obs_dim = y.shape[1]
cfg.p_factor = 2
cfg.p_eps = 0
cfg.T_train = 600
cfg.T_test = 129
cfg.T_val = 100
cfg.T = y.shape[0] #529

#Training config
cfg.batch_size = 32
cfg.lr = 0.0001
cfg.n_epoch = 10
cfg.max_iter = 25


#Create train test val
y_train = y[0:cfg.T_train]
y_val = y[cfg.T_train:cfg.T_train+cfg.T_val]
y_test = y[cfg.T_train+cfg.T_val:]

y_trainval = y[:cfg.T_train+cfg.T_val]
weights_train = mask[0:cfg.T_train]
weights_val = mask[cfg.T_train:cfg.T_train+cfg.T_val]



def saveForecastError(mod, fn=None):
    f_trainval_hat = mod.enc(torch.Tensor(y_trainval).float()).detach().numpy()
    f_test_hat = mod.enc(torch.Tensor(y_test).float()).detach().numpy()

    model = VAR(f_trainval_hat)
    res = model.fit(maxlags=10, ic='aic')
    test_model = VAR(np.concatenate([f_trainval_hat, f_test_hat]))
    f_test_pred = test_model.predict(res.params, lags=res.k_ar)[cfg.T_train+cfg.T_val-res.k_ar:]
    print(f_test_pred.shape)
    y_test_pred = mod.dec(torch.Tensor(f_test_pred).float()).detach().numpy()
    mse = np.mean((y_test-y_test_pred)**2,axis=1)
    if fn is not None:
        np.save(arr=mse,file=fn)
    return y_test_pred



def saveForecastErrorPCA(mod, fn=None):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3).fit(y_train)
    f_trainval_hat = pca.transform(y_trainval)
    f_test_hat = pca.transform(y_test)

    model = VAR(f_trainval_hat)
    res = model.fit(maxlags=10, ic='aic')
    test_model = VAR(np.concatenate([f_trainval_hat, f_test_hat]))
    f_test_pred = test_model.predict(res.params, lags=res.k_ar)[cfg.T_train+cfg.T_val-res.k_ar:]
    print(f_test_pred.shape)
    y_test_pred = pca.inverse_transform(f_test_pred)
    mse = np.mean((y_test-y_test_pred)**2,axis=1)
    if fn is not None:
        np.save(arr=mse,file=fn)
    return y_test_pred



def saveForecastErrorAR1(mod, fn=None):
    model = VAR(y_trainval)
    res = model.fit(maxlags=2, ic='aic')
    test_model = VAR(np.concatenate([y_trainval, y_test]))
    y_test_pred = test_model.predict(res.params, lags=res.k_ar)[cfg.T_train+cfg.T_val-res.k_ar:]
    mse = np.mean((y_test-y_test_pred)**2,axis=1)
    if fn is not None:
        np.save(arr=mse,file=fn)
    return y_test_pred
