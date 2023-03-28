import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression(X, y, scale=True):
    """Given continuous data X,y in several contexts or groups, linear regression per group

    :param X: X, list of np.arrays
    :param y: y, continuous
    :return: regression coefficient per context/group
    """
    coef_k_x = np.array([LinearRegression().fit(X[pi_k], y[pi_k]).coef_
                      for pi_k in range(len(X))])
    if scale:
        return coef_rescale(coef_k_x, len(X),  X_n=len(coef_k_x[0])) # X_n=len(LinearRegression().fit(X[0], y[0]).coef_))

    return coef_k_x


def coef_rescale(coef_c_x, C_n, X_n):
    if X_n == 1:
        coef_c_x = coef_c_x.reshape(-1,1)
    #shape e.g. (5,2)
    else:
        c_n = coef_c_x.shape[0]
        x_n = coef_c_x.shape[1]
        assert x_n==X_n and  c_n == C_n
    return np.transpose([dim_rescale(coef_c_x[:, x_i]) for x_i in range(X_n)])

def dim_rescale(coef_c):
    offset = min(coef_c)
    coef_c = [coef_c[i] - offset for i in range(len(coef_c))]
    if (max(coef_c)==0): return([1 for i in range(len(coef_c))])

    scale = 100/max(coef_c)
    return [coef_c[i]* scale +1 for i in range(len(coef_c))]
