import numpy as np
from scipy.stats import chi2

def G(d, h):
    n = d.shape[0]
    dbar = np.mean(d, axis=1, keepdims=True)
    SCM = np.zeros((n, n))
    for t in range(h + 1, d.shape[1]):
        SCM += (d[:, t] - dbar) @ (d[:, t - h] - dbar).T
    SCM /= d.shape[1] - h - 1
    return SCM

def O(d, q):
    SLRV = G(d, 0)
    if q > 0:
        for h in range(1, q + 1):
            TEMP = G(d, h)
            SLRV += TEMP + TEMP.T
    return SLRV

def in_MDM_test(d, q, statistic):
    n = d.shape[0]
    dbar = np.mean(d, axis=1, keepdims=True)
    c = 1 - (1 + 2 * q) / d.shape[1] + q * (q + 1) / d.shape[1] ** 2
    Om = O(d, q)
    s = dbar.ravel() / np.sqrt(np.abs(np.diag(Om)) / d.shape[1])
    S = d.shape[1] * dbar.T @ np.linalg.solve(Om, dbar)
    if statistic == "Sc":
        S *= c
    pval = chi2.sf(S, df=n, loc=0, scale=1)
    ret = [s, {"statistic": S, "parameter": q, "alternative": "Equal predictive accuracy does not hold.",
               "p.value": pval, "method": "multivariate Diebold-Mariano test", "data.name": str(d)}]
    return ret

def MDM_test(realized, evaluated, q, statistic="Sc", loss_type="SE"):
    l = loss(realized, evaluated, loss_type)
    d = d_t(l)
    out = in_MDM_test(d, q, statistic)[1]
    out["data.name"] = f"{realized} and {evaluated}"
    return out

def loss(realized, evaluated, loss_type):
    e = realized - evaluated
    if loss_type == "SE":
        e = e ** 2
    elif loss_type == "AE":
        e = np.abs(e)
    elif loss_type == "SPE":
        e = e ** 2 / np.abs(realized)
    elif isinstance(loss_type, (float, int)):
        e = np.exp(loss_type * np.abs(e)) - 1 - loss_type * np.abs(e)
    return e

def d_t(e):
    d = np.zeros((e.shape[0] - 1, e.shape[1]))
    for j in range(e.shape[0] - 1):
        d[j] = np.sign(e[j]) * np.log(np.abs(e[j] / e[j + 1]))
    return d