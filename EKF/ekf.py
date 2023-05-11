from filterpy.kalman import ExtendedKalmanFilter as ExtendedKalmanFilter
import torch
from torch.autograd.functional import jacobian

def filteredState():
    dim_x = cfg.factor_dim
    dim_z = cfg.obs_dim
    ekf = ExtendedKalmanFilter(dim_x, dim_z)
    ekf.R = np.diag(np.diag(obs_cov))
    #ekf.Q = res.sigma_u
    ekf.F = res.params[:-1]
    ekf.P = res.sigma_u
    ekf.x = f_train_hat[-1]
    decoder = mod.dec
    Hx = lambda state: decoder(torch.Tensor(state).float()).detach().numpy()
    HJacobian = lambda state: jacobian(decoder,torch.Tensor(state).float()).detach().numpy()
    xs = []
    zs = []
    for i in range(y.shape[0]):
        z = y[i,:]
        ekf.update(z, HJacobian, Hx)
        xs.append(ekf.x)
        ekf.predict()
        zs.append(ekf.z)