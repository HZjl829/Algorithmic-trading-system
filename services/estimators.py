import numpy as np
import gurobipy as gp
from sklearn.linear_model import Lasso

def OLS(returns, factRet):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q


def LASSO(returns, factRet, lambda_, K):
    """
    Use this function for the LASSO model. Note that you will not use K
    in this model (K is for BSS).

    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset regression
    """
    # ----------------------------------------------------------------------
    # Align on dates & drop missing
    data = returns.join(factRet, how='inner').dropna()

    # Factor matrix F (T×8) and compute its mean/covariance
    factor_cols = ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']
    F = data[factor_cols].values
    T, p = F.shape
    f_mean  = F.mean(axis=0)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)

    assets = returns.columns
    N = len(assets)

    # Storage
    alpha   = np.zeros(N)
    B       = np.zeros((N, p))
    eps_var = np.zeros(N)
    adj_R2  = np.zeros(N)

    # Fit a Lasso for each asset
    for i, asset in enumerate(assets):
        y = data[asset].values
        model = Lasso(alpha=lambda_, fit_intercept=True, max_iter=10000)
        model.fit(F, y)

        alpha[i] = model.intercept_
        B[i, :]  = model.coef_
        resid    = y - model.predict(F)
        eps_var[i] = np.var(resid, ddof=1)

        # Compute R²
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR / SST

        # count only non-zero factors
        p_eff = np.count_nonzero(model.coef_)

        # adjusted R² penalizes only the actually used predictors
        adj_R2[i] = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)

    # Expected returns
    mu = alpha + B.dot(f_mean)          # (n,)

    # Covariance
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)  # (n, n)
    # ----------------------------------------------------------------------

    return mu, Q



def FF(returns, factRet, lambda_, K):
    """
    Calibrate the Fama-French 3-factor model.

    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset regression
    """
    # ----------------------------------------------------------------------
    # align dates and drop any rows with missing data
    data = returns.join(factRet[['Mkt_RF','SMB','HML']], how='inner').dropna()
    
    # build design matrix X = [1, Mkt_RF, SMB, HML]
    F = data[['Mkt_RF','SMB','HML']].values    # (T, 3)
    T = F.shape[0]
    X = np.hstack([np.ones((T, 1)), F])        # (T, 4)
    
    assets = returns.columns
    N = len(assets)
    
    # storage
    B       = np.zeros((N, 3))
    alpha   = np.zeros(N)
    eps_var = np.zeros(N)
    adj_R2  = np.zeros(N)
    
    # run OLS for each asset
    for i, asset in enumerate(assets):
        y, *_ = data[asset].values, 
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        
        alpha[i] = coeffs[0]
        B[i, :]  = coeffs[1:]
        
        resid = y - X.dot(coeffs)
        eps_var[i] = resid.var(ddof=1)
        
        # compute R²
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR/SST
        
        # count only nonzero betas (exclude intercept)
        p_eff = np.count_nonzero(coeffs[1:])
        
        # adjusted R² with p_eff predictors
        adj_R2[i] = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)
    
    # expected returns
    f_mean = F.mean(axis=0)             # (3,)
    mu     = alpha + B.dot(f_mean)      # (N,)
    
    # factor-model covariance
    Sigma_f = np.cov(F, rowvar=False, ddof=1)  
    Q       = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)
    # ----------------------------------------------------------------------
    
    return mu, Q