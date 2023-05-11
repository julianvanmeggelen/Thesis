import numpy as np
import random
from statsmodels.multivariate.cancorr import CanCorr
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import torch
from tqdm import tqdm

import sys
sys.path.append('../')

from BasicAutoEncoder.model import AutoEncoder, train
from DGP import dgp
from results import loadResults, RESULT_DIR

def bootstrapCCA(f_hat, f_true, n_bootstrap = 100, n_sample=100):
    assert f_hat.shape == f_true.shape
    indices = list(range(f_hat.shape[0]))
    res = []
    for sample in range(n_bootstrap):
        sample = random.sample(indices,n_sample)
        f_true_sample = f_true[sample]
        f_hat_sample = f_hat[sample]
        cancorr = CanCorr(f_true_sample, f_hat_sample)
        cc = np.mean(cancorr.cancorr) 
        res.append(cc)
    return res

def expandingWindowForecast(mod: AutoEncoder, y_train:np.ndarray, y_test: np.ndarray, n_init_epochs:int=20, n_retrain_epochs: int =1, batch_size: int = 256, lr: float = 0.0005):
    y_train = np.copy(y_train)
    y_test= np.copy(y_test)

    T = y_test.shape[0]

    #train in validation set if desired
    train_hist = train(X_train=torch.Tensor(y_train).float(), model=mod,n_epoch=n_init_epochs,X_val=None,batch_size=batch_size,lr=lr,verbose=False)

    pred = []
    for t in tqdm(range(T)):

        #make one step ahead pred
        f_train_hat = mod.enc(torch.Tensor(y_train).float()).detach().numpy()
        f_test_hat = mod.enc(torch.Tensor(y_test).float()).detach().numpy()
        factorModel = VAR(f_train_hat)
        res = factorModel.fit(maxlags=10, ic='aic')
        test_model = VAR(np.concatenate([f_train_hat, f_test_hat[[0]]]))
        f_test_pred = test_model.predict(res.params, lags=res.k_ar)[[-1]]
        y_test_pred = mod.dec(torch.Tensor(f_test_pred).float()).detach().numpy()
        oneStepAheadForecast =  y_test_pred
        pred.append(oneStepAheadForecast)

        #add one observation from pred to test
        y_train = np.concatenate([y_train, y_test[[0]]])
        y_test = y_test[1:]

        #fit model for one epoch
        if t <= T:
            train_hist = train(X_train=torch.Tensor(y_train).float(), model=mod,n_epoch=1,X_val=None,batch_size=batch_size,lr=lr,verbose=False)
    return np.vstack(pred)


def addOneStepAheadForecast(dgpIndex,experimentName):
    train_hist, mod, cfg = loadResults(dgpIndex, experimentName)
    f,y,dec = dgp.getSaved(cfg.saved_index, T=cfg.T)
    y_train = y[0:cfg.T_train]
    y_val = y[cfg.T_train:cfg.T_train+cfg.T_val]
    y_test = y[cfg.T_train+cfg.T_val:]
    ewf = expandingWindowForecast(mod=mod,y_train = np.concatenate([y_train, y_val]), y_test = y_test)
    np.save(arr=ewf,file=f"{RESULT_DIR}{dgpIndex}/{experimentName}/expandingForecast.npy")

def evaluationProtocol(dgpIndex,experimentName):
    addOneStepAheadForecast(dgpIndex,experimentName)
    print('done')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('i', type=int,
                        help='Which dgp')
    parser.add_argument('name', type=str,
                        help='Experimentname')

    args = parser.parse_args()
    evaluationProtocol(args.i, args.name)




