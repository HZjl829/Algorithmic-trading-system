import cvxpy as cp
import numpy as np
from scipy.stats import chi2

def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value


def risk_parity(Q, c=1.0, solver=cp.SCS):
    """
    Solve min_y ½ yᵀQy - c ∑ᵢ ln(yᵢ)  s.t. y > 0,
    then return w = y / sum(y).
    """
    n = Q.shape[0]
    y = cp.Variable(n, pos=True)
    obj = 0.5*cp.quad_form(y, Q) - c*cp.sum(cp.log(y))
    prob = cp.Problem(cp.Minimize(obj),
                      [y>= 0])
    prob.solve(solver=solver, verbose=False)
    y_opt = y.value
    return y_opt / np.sum(y_opt)



def robust_MVO_ellip(mu, Q, T,
                     alpha):
    """
    Robust MVO under ellipsoidal uncertainty in mu:
      min_x x' Q x
      s.t. mu' x - eps2 * || sqrt(diag(Q)/T) * x ||_2 >= targetRet
           sum(x) == 1, x >= 0

    Args:
      mu        : (n,1) or (n,) vector of expected returns
      Q         : (n,n) covariance matrix
      T         : # observations used to estimate mu (for SE)
      targetRet : scalar, required return R; default = mean(mu)
      alpha     : confidence level for chi2 bound (ε2^2 = χ2_n(α))
      solver    : CVXPY solver to use

    Returns:
      x.value   : (n,) robust portfolio weights
    """
    mu = mu.flatten()
    n  = len(mu)

    # default target = average expected return
    
    targetRet = float(mu.mean())

    # construct Θ^(1/2) diagonal-vector
    theta_half = np.sqrt(np.diag(Q) / T)

    # compute eps2 = sqrt(chi2.ppf(alpha, df=n))
    eps2 = np.sqrt(chi2.ppf(alpha, df=n))
    # print(eps2)
    # decision variable
    x = cp.Variable(n)
    # print(T, eps2, mu)
    # constraints
    constraints = [
        mu @ x
          - eps2* cp.norm(cp.multiply(theta_half, x), 2)
              # || sqrt(diag(Q)/T) * x ||_2
          >= targetRet,
        cp.sum(x) == 1,
        x >= 0
    ]

    # objective = x' Q x
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Q)),
                      constraints)
    prob.solve(verbose=False)

   
    print("Status:", prob.status)
    # print(x.value)
    return x.value




# test the robust MVO with syntehtic data

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    from estimators import OLS
    set_seed = 42

    # 1) Simulate synthetic returns
    np.random.seed(42)
    T = 120                   # e.g. 120 “days”
    n_assets = 5

    # create a covariance matrix: daily vol ~1%, correlations ~0.2
    base_corr = 0.2
    Sigma = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(Sigma, 1.0)
    Sigma *= (0.01)**2        # scale to 1% vol

    # set “true” expected returns ~ 0.05% per day
    mu_true = np.array([0.0005, 0.0004, 0.0006, 0.00055, 0.00045])

    # draw T multivariate‐normal returns
    R = np.random.multivariate_normal(mu_true, Sigma, size=T)
    periodReturns = pd.DataFrame(R, columns=[f"Asset{i+1}" for i in range(n_assets)])

    # 2) Estimate sample mean & covariance
    mu_est = periodReturns.mean(axis=0).values       # shape (5,)
    Q_est  = periodReturns.cov().values             # shape (5×5)

    # 3) Call your robust MVO
    #    (ensure T is float or int; alpha is 95% conf.)

    
    Q = Q_est
    # 4) enforce PD + symmetry on Q
    Q = (Q + Q.T)/2

    weights = robust_MVO_ellip(mu_est, Q_est, T=T, alpha=0.95)

    print("Estimated μ:", np.round(mu_est, 6))
    print("Weights:", np.round(weights, 4))
    print("Sum of weights:", np.sum(weights))
