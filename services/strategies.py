import numpy as np
from services.estimators import *
from services.optimization import *
from sklearn.covariance import LedoitWolf


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns, factorReturns=None):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=60):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=60):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x


class LASSO_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=60):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q, adj_R2 = LASSO(returns, factRet)
        x = MVO(mu, Q)
        return x
    

class FactorRiskParity:
    def __init__(self, NumObs=60, c=1.0):
        self.NumObs = NumObs
        self.c      = c

    def execute_strategy(self, periodReturns, factorReturns):
        rets    = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]
        mu, Q   = OLS(rets, factRet)  


        
        x       = risk_parity(Q, c=self.c)
        return x


class RobustEllipMVO:
    def __init__(self, NumObs=60, alpha=0.9):
        self.NumObs = NumObs
        self.alpha  = alpha

    def execute_strategy(self, periodReturns, factorReturns):
        
        rets    = periodReturns.iloc[-self.NumObs:, :]
        factRet = factorReturns.iloc[-self.NumObs:, :]
        mu, Q,_ = LASSO(rets, factRet)
        # robust MVO
        x = robust_MVO_ellip(mu,Q,T = self.NumObs, 
                             alpha=self.alpha)
        return x
    
class OLS_PCA_MVO:
    def __init__(self, NumObs=60, n_pca=5):
        self.NumObs = NumObs
        self.n_pca  = n_pca

    def execute_strategy(self, periodReturns, factorReturns):
        rets = periodReturns.iloc[-self.NumObs:, :]
        facs = factorReturns.iloc[-self.NumObs:, :]
        mu, Q = OLS_with_PCA(rets, facs, self.n_pca)
        return MVO(mu, Q)
    



    
    