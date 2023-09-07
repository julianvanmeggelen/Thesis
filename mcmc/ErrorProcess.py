import numpy as np
from statsmodels.tsa.arima.model import ARIMA 

class ErrorProcess(object):
    """
    Generic interface for the observation error process. Must have functionality to:
    - Initialize the process
    - Sample the conditionalExpectation of the process
    - Re-estimate the process
    - Store arbitrary data in fit_hist dict
    """
    def __init__(self, n:int, T:int):
        """
        n: Number of vars
        T: Length of data
        """
        self.n = n
        self.T = T
        self.fit_hist: dict = None
    
    def initialize(self) -> None:
        """
        Initialize the process (without having data to fit)
        """
        raise NotImplementedError

    def fit(self, resids: np.ndarray) -> None:
        """
        Fit the process based on incoming residuals
        resids: np.ndarray with shape (n, T)
        """
        raise NotImplementedError
    
    def conditionalExpectation(self) -> np.ndarray:
        """
        Return conditional expectation E[eps_t|eps_{t-1}] for t =1,..T
        returns:
            res: np.ndarray with shape (n,T)
        """
        raise NotImplementedError
    
    def sample(self) -> np.ndarray:
        """
        Return sampled path eps_t for t =1,..T
        returns:
            res: np.ndarray with shape (n,T)
        """
        raise NotImplementedError



class IIDErrorProcess(ErrorProcess):
    def __init__(self, n: int, T: int, diagonal: bool = False):
        """
        Diagonal: should cov matrix be diagonal?
        """
        super().__init__(n, T)
        self.mu = None
        self.cov = None
        self.diagonal = diagonal
        self.fit_hist = {'mu':[],'cov':[]}


    def initialize(self) -> None:
        eps_init = np.random.normal(size=(self.T,self.n), scale=0.01)
        self.mu = np.mean(eps_init, axis=0)
        self.cov = np.identity(self.n)*10#np.cov(eps_init.T) 
        if self.diagonal:
            self.cov = np.eye(self.n)*self.cov
        #print(self.mu.shape, self.cov.shape)
        return self
    
    def fit(self, resids: np.ndarray) -> None:
        self.mu = np.mean(resids, axis=0)
        self.cov = np.cov(resids.T)
        if self.diagonal:
            self.cov = np.eye(self.n)*self.cov
        self.fit_hist['mu'].append(self.mu)
        self.fit_hist['cov'].append(self.cov)

    def conditionalExpectation(self) -> np.ndarray:
        return self.mu

    def sample(self) -> np.ndarray:
        return np.random.multivariate_normal(self.mu,self.cov,size=(self.T))


class ArErrorProcess(ErrorProcess):
    """
    class containing the autoregressive processes of the idiosyncratic error terms \varepsilon_t
    This should be a var process if Q can be nondiagonal
    """
    def __init__(self, T:int, n:int, d:int):
        super().__init__(n,T)
        self.d = d #number of lags
        self.parameters = None
        x = np.zeros((T,n)) #start with empty data
        self.models = [ARIMA(order=(self.d,0,0), trend='n', endog=x[:,i]) for i in range(x.shape[1])]
        self.fit_hist = {'coeff':[],'var':[]}

    def fit(self, x):
        self.models = [ARIMA(order=(self.d,0,0), trend='n', endog=x[:,i]) for i in range(x.shape[1])]
        self.fitted = [mod.fit(method='yule_walker') for mod in self.models]
        self.fit_hist['var'].append(self.sigma2)
        self.fit_hist['coeff'].append(self.coefficients)

    @property
    def coefficients(self):
        """
        Return estimated coefficients of the ar() processes
        """
        return np.stack([_.params[:-1] for _ in self.fitted])
    
    @property
    def sigma2(self):
        """
        Return estimated residual sigma_2 of the processes
        """
        return np.stack([_.params[-1] for _ in self.fitted])

    def conditionalExpectation(self) -> np.ndarray:
        """
        obtain Phi(L)eps_t
        """
        return np.stack([mod.predict() for mod in self.fitted]).T
    
    def predict(self, eps: np.ndarray):
        """
        One step ahead forecast
        """
        return np.stack([mod.predict(eps[:,i]) for i, mod in enumerate(self.fitted)]).T

    def initialize(self):
        eps = np.random.normal(size = (self.T,self.n))
        self.fit(eps)
        #
        #for mod in self.models:
            #mod.initialize_stationary()
            
    def sample(self) -> np.ndarray:
        return np.stack([mod.simulate(nsimulations=self.T) for mod in self.fitted]).T