from services.strategies import *


def project_function(periodReturns, periodFactRet, X0=None):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    
    # Strategy = OLS_MVO()
    # Strategy = equal_weight()
    # Strategy = LASSO_MVO()
    Strategy = RobustEllipMVO()
    
    # x = Strategy.execute_strategy(periodReturns, periodFactRet)
    # Strategy = OLS_PCA_MVO()
    x = Strategy.execute_strategy(periodReturns, periodFactRet)

   
    



    # x = equal_weight(periodReturns)
    return x
