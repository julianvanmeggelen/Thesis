from filterpy.kalman import ExtendedKalmanFilter as ExtendedKalmanFilter
import torch
from torch.autograd.functional import jacobian
import numpy as np

# def filteredState():
#     dim_x = cfg.factor_dim
#     dim_z = cfg.obs_dim
#     ekf = ExtendedKalmanFilter(dim_x, dim_z)
#     ekf.R = np.diag(np.diag(obs_cov))
#     #ekf.Q = res.sigma_u
#     ekf.F = res.params[:-1]
#     ekf.P = res.sigma_u
#     ekf.x = f_train_hat[-1]
#     decoder = mod.dec
#     Hx = lambda state: decoder(torch.Tensor(state).float()).detach().numpy()
#     HJacobian = lambda state: jacobian(decoder,torch.Tensor(state).float()).detach().numpy()
#     xs = []
#     zs = []
#     for i in range(y.shape[0]):
#         z = y[i,:]
#         ekf.update(z, HJacobian, Hx)
#         xs.append(ekf.x)
#         ekf.predict()
#         zs.append(ekf.z)

def unconditionalVariance2(A, cov_u):
    res = np.zeros((A.shape[0], A.shape[0]))
    for i in range(1000):
        Ai = A ** i
        res += Ai @ cov_u @ Ai.T
    return res
   
def unconditionalVariance(A, cov_u):
    res  = np.linalg.inv(np.identity(A.shape[0]**2) - np.kron(A,A)) @ cov_u.flatten('F')
    return res.reshape(A.shape[0], -1).T


def EKF(y,Z,Z_jacobian, H, T, R, Q, C=None, a0=None, P0=None):
    """
    y: Txn
    Z: callable: rx1 -> nx1
    Z_jacobian: callable: nx1 -> rx1
    H: nxn
    T: rxr
    R: rxr
    Q: rxr
    C: rx1
    a0: rx1
    P0: rxr
    """
    N = y.shape[0]
    n = y.shape[1]
    r = T.shape[0]

    v = np.zeros((N+1, n))
    F = np.zeros((N+1, n, n))
    P = np.zeros((N+1, r,r))
    a = np.zeros((N+1, r))
    K = np.zeros((N+1, r, n))
    a_filtered = np.zeros((N+1, r))  
    Z_dot = np.zeros((N, n, r))               

    if a0 is None:
        a0 = np.zeros(T.shape[0])
    if P0 is None:
        P0 = np.identity((T.shape[0]))
    if C is None:
        C = np.zeros((r))

    #Initialize 
    a[0] = a0
    P[0] = P0

    for t in range(N):
        Z_dot[t] = Z_jacobian(a[t])
        #d_t = Z(a[t]) - Z_dot_t @ a[t]
        #print(f"{Z_dot_t=}")
        v[t] = y[t] - Z(a[t])#y[t] - Z_dot_t @ a[t] - d_t
        #print(f"{y[t]=}, {Z(a[t])=}, {v[t]=}")
        F[t] = Z_dot[t] @ P[t] @ Z_dot[t].T + H
        #print(f"{F[t]=}")
        Ft_inv = np.linalg.inv(F[t])
        K[t] = T @ P[t] @ Z_dot[t].T @ Ft_inv
        #print(T.shape, a[t].shape, K[t].shape, F[t].shape, C.shape)
        a[t+1] = T @ a[t] + K[t] @ v[t] + C
        #P_tt = P[t] - P[t] @ Z_dot_t.T @ Ft_inv @ Z_dot_t @ P[t]
        #P[t+1] = T @ P_tt @ T.T + R @ Q @ R.T
        P[t+1] = T @ P[t] @ T.T + R @ Q @ R.T - K[t] @ F[t] @ K[t].T
        #P[t+1] = T @ P[t] @ (T - K[t] @ Z_dot_t).T + R @ Q @ R.T 
        a_filtered[t] = a[t] + P[t] @ Z_dot[t].T @ Ft_inv @ v[t]
        #a[t+1] = T @ a_filtered[t]

    return a[:-1], P[:-1], a_filtered, Z_dot, v, K, F


def EKS(Z_dot,F,v,a,P, T, K):
    """
    """
    N = v.shape[0]
    n = v.shape[1]
    r = a.shape[1]

    rr = np.zeros((N+1, r))
    L = np.zeros((N+1, n, n))
    a_smoothed = np.zeros((N,r))
    rr[N], L[N] = 0, 0

    a_filtered = np.zeros((N+1, r)) 
    for t in reversed(range(N-1)):
        Lt = T - K[t] @ Z_dot[t]
        Finv = np.linalg.inv(F[t])
        rr[t-1] = Z_dot[t].T @ Finv @ v[t]  + Lt.T  @ rr[t]
        a_smoothed[t] = a[t] + P[t] @ rr[t-1]
    return a_smoothed



def KF(y,Z, H, T, R, Q, C=None, a0=None, P0=None):
    """
    y: Txn
    Z: callable: rx1 -> nx1
    Z_jacobian: callable: nx1 -> rx1
    H: nxn
    T: rxr
    R: rxr
    Q: rxr
    C: rx1
    a0: rx1
    P0: rxr
    """
    N = y.shape[0]
    n = y.shape[1]
    r = T.shape[0]

    v = np.zeros((N+1, n))
    F = np.zeros((N+1, n, n))
    P = np.zeros((N+1, r,r))
    a = np.zeros((N+1, r))
    K = np.zeros((N+1, r, n))
    a_filtered = np.zeros((N+1, r))  

    if a0 is None:
        a0 = np.zeros(T.shape[0])
    if P0 is None:
        P0 = np.identity((T.shape[0]))
    if C is None:
        C = np.zeros((r))

    #Initialize 
    a[0] = a0
    P[0] = P0

    for t in range(N):
        #d_t = Z(a[t]) - Z_dot_t @ a[t]
        #print(f"{Z_dot_t=}")
        v[t] = y[t] - Z @ a[t]#y[t] - Z_dot_t @ a[t] - d_t
        #print(f"{y[t]=}, {Z(a[t])=}, {v[t]=}")
        F[t] = Z @ P[t] @ Z.T + H
        #print(f"{F[t]=}")
        Ft_inv = np.linalg.pinv(F[t])
        K[t] = T @ P[t] @ Z.T @ Ft_inv
        #print(T.shape, a[t].shape, K[t].shape, F[t].shape, C.shape)
        a[t+1] = T @ a[t] + K[t] @ v[t] + C
        #P_tt = P[t] - P[t] @ Z_dot_t.T @ Ft_inv @ Z_dot_t @ P[t]
        #P[t+1] = T @ P_tt @ T.T + R @ Q @ R.T
        #P[t+1] = T @ P[t] @ T.T + R @ Q @ R.T - K[t] @ F[t] @ K[t].T
        P[t+1] = T @ P[t] @ (T - K[t] @ Z).T + R @ Q @ R.T 
        a_filtered[t] = a[t] + P[t] @ Z.T @ Ft_inv @ v[t]
        #a[t+1] = T @ a_filtered[t]
        return a[:-1], P[:-1], a_filtered, v, K, F
