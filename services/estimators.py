import numpy as np
import gurobipy as gp
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
import pandas as pd


def mu_js_shrink(sample_mu, T, prior=None):
    """
    sample_mu : (n,1) array of sample means
    T         : # observations
    prior     : (n,1) vector to shrink toward (default = zeros)
    """
    n = len(sample_mu)
    if prior is None:
        prior = np.zeros_like(sample_mu)

    # magnitude of shrinkage
    # constant τ can be tuned or set to (n-2)/T
    tau = (n - 2) / float(T)
    w   = max(0, 1 - tau)   # shrink factor
    return w * sample_mu + (1 - w) * prior

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


def LASSO(returns, factRet):
    """
    Use this function for the LASSO model. 

    We find the optimal lambda using cross-validation if lambdas is None.
    Otherwise, we use the provided lambda.
    Returns:
      mu      : n-vector of expected returns
      Q       : n×n asset covariance matrix
      adj_R2  : n-vector of adjusted R² for each asset
    """
    # 1) align & clean
    # strip space in column names
    returns.columns = [col.strip() for col in returns.columns]
    factRet.columns = [col.strip() for col in factRet.columns]
    # print(factRet.columns)
    data = returns.join(factRet, how='inner').dropna()
    factor_cols = ['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev']
    F = data[factor_cols].values
    T, p = F.shape
    f_mean  = F.mean(axis=0)
    Sigma_f = np.cov(F, rowvar=False, ddof=1)

    assets = returns.columns
    N = len(assets)
    
    lambda_ = np.logspace(-5, -1, 50) 
    
   

    # if doing CV, set up a time-series splitter
    tscv = TimeSeriesSplit(n_splits=3) 

    # 3) storage
    alpha   = np.zeros(N)
    B       = np.zeros((N, p))
    eps_var = np.zeros(N)
    adj_R2  = np.zeros(N)

    # 4) fit each asset
    for i, asset in enumerate(assets):
        y = data[asset].values

        
        # pick lamdba via CV
        model = LassoCV(alphas=lambda_,
                        cv=tscv,
                        fit_intercept=True,
                        max_iter=10000)

        model.fit(F, y)
        

        # store the CV‐chosen coefficients
        coefs = model.coef_.copy()
        intercept = model.intercept_

        # count how many non-zeros
        p_eff = np.count_nonzero(coefs)
        if p_eff < 3:
            # force at least 3 non-zeros by trying smaller penalties
            for lam in sorted(lambda_):   # smallest diag first → least shrinkage
                tmp = Lasso(alpha=lam,
                            fit_intercept=True,
                            max_iter=10000).fit(F, y)
                if np.count_nonzero(tmp.coef_) >= 3:
                    coefs = tmp.coef_
                    intercept = tmp.intercept_
                    break
            # at this point `coefs` has ≥3 non-zeros (or is your best fallback)

        # now assign back
        alpha[i] = intercept
        B[i, :]  = coefs
        
        # residual stats
        resid      = y - model.predict(F)
        eps_var[i] = np.var(resid, ddof=1)

        # compute adjusted R2
        SSR = np.sum(resid**2)
        SST = np.sum((y - y.mean())**2)
        R2  = 1 - SSR/SST
        p_eff      = np.count_nonzero(model.coef_)
        # print("p_eff", p_eff)
        adj_R2[i]  = 1 - (1 - R2) * (T - 1) / (T - p_eff - 1)

    # 5) build mu & Q
    mu = alpha + B.dot(f_mean)                 # (n,)
    Q  = B.dot(Sigma_f).dot(B.T) + np.diag(eps_var)  # (n,n)


    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2
    return mu, Q, adj_R2


def OLS_with_PCA(returns,
                 factRet,
                 n_components):
    """
    1) Align on dates & drop NaNs
    2) PCA-reduce factRet to `n_components`
    3) Call OLS(returns, pca_factors_df)
    Returns:
      mu, Q  as in original OLS
    """
    # 1) align
    data = returns.join(factRet, how="inner").dropna()
    R = data[returns.columns]
    F = data[factRet.columns]

    # 2) PCA on factors
    pca = PCA(n_components=n_components)
    F_pca = pca.fit_transform(F.values)        # shape (T, k)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(F_pca, index=F.index, columns=cols)

    # 3) call your OLS
    mu, Q = OLS(R, df_pca)
    return mu, Q

## test lasso usage, using random data

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    set_seed = 42
    np.random.seed(set_seed)

    # Generate random data
    n_samples = 1000
    n_features = 8
    n_targets = 5
    # Generate a random regression problem
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_targets=n_targets, noise=0.1, random_state=set_seed)
    # Convert to DataFrame
    factorReturns = pd.DataFrame(X, columns=['Mkt_RF','SMB','HML','RMW','CMA','Mom','ST_Rev','LT_Rev'])
    periodReturns = pd.DataFrame(y, columns=['Asset1', 'Asset2', 'Asset3', 'Asset4', 'Asset5'])

    # Call the function
    # mu, Q = OLS(periodReturns, factorReturns)
    mu, Q, adj_R2 = LASSO(periodReturns, factorReturns)
    print("Expected Returns (mu):", mu)
    print("Covariance Matrix (Q):", Q)
    print("Adjusted R2:", adj_R2)
    

    