import scipy, cvxopt
from cvxopt.solvers import qp

import numpy as np
cimport numpy as np
cimport cython


"""
This file contains a list of algoritm that is used for estimation
    some optmisation should be done in future.
    Naming convention:
        function with no ec/ic prefix are estimator for unconstrained problem
        ec* prefix: this is for equality constraints.
        ic* prefixL this is for inequality constraints.
"""
DEBUG = 0

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def regression(X, y, W):
    """
    Return the estimated weight based on ordinary regression model

    :param X: Independent variable
    :type X: scipy.matrix<float>
    :param y: Dependent variable
    :type y: scipy.matrix<float>
    :param W: Weight matrix
    :type W: scipy.matrix<float>

    .. math::
       (X^T\; W \; X )^{-1} X^T W  y

    """
    return (X.T * W * X).I*(X.T * W * y)

def ecregression(X,y,W,D,d):
    r"""
    This return the estimated weight on the following regression problem
    
    .. math::
       y = X \beta

    constained to

    .. math::
       D \beta = d

    The problem is solved using Lagrangian Multiplier and 
    
    .. math::
       :nowrap:

       \begin{eqnarray*}
          E &=& (X^T W X)^{-1}\\
          \lambda &=& (D E D^T)^{-1} (D E X^T W y - d)\\
          \hat{\beta} &=& E (X^T W y - D^T \lambda)
       \end{eqnarray*}

    """
    covinv = (X.T * W * X).I
    lamb = (D * covinv * D.T).I * (D * covinv * X.T * W * y - d)
    return  covinv * (X.T * W * y - D.T * lamb)

def icregression(X,y, W, D,d,G,a,b,n):
    P = 2*X.T * W * X
    q = -2*X.T * W * y
    bigG = scipy.empty((2*n, n), dtype = DTYPE)
    h = scipy.empty((2*n, 1), dtype = DTYPE)
    bigG[:n, :] = -G
    bigG[n:, :] = G
    h[:n, :] = -a
    h[n:, :] = b    
    paraset = map(cvxopt.matrix , (P,q,bigG,h,D,d))
    return qp(*paraset)['x']
#============================================================
def kalman_predict(np.ndarray b,
                   np.ndarray V,
                   np.ndarray Phi,
                   np.ndarray S):
    (b, V, Phi, S) = map(scipy.matrix, (b, V, Phi, S))
    b = Phi * b
    V = Phi * V * Phi.T + S
    return b, V

cdef kalman_upd(np.ndarray beta,
                np.ndarray V,
                np.ndarray y,
                np.ndarray X,
                np.ndarray s,
                np.ndarray S,
                int flag = 0,
                np.ndarray D = None,
                np.ndarray d = None,
                np.ndarray G = None, 
                np.ndarray a = None,
                np.ndarray b = None):
    """switch:
        0 | no constraints
        1 | equality constraints
        2 | inequality constraints
    """

    e = y - X * beta
    K = V * X.T * ( s + X * V * X.T).I
    beta = beta + K * e
    if flag == 1:
        D = scipy.matrix(D, dtype = DTYPE)
        d = scipy.matrix(d, dtype = DTYPE)
        beta = beta - S * D.T * ( D * S * D.T).I * ( D * beta - d)
    elif flag == 2:
        G = scipy.matrix(G, dtype = DTYPE)
        a = scipy.matrix(a, dtype = DTYPE)
        b = scipy.matrix(b, dtype = DTYPE)
        n = len(beta)
        P = 2* V.I
        q = -2 * V.I.T * beta
        bigG = scipy.empty((2*n, n), dtype = DTYPE)
        h = scipy.empty((2*n, 1), dtype = DTYPE)
        bigG[:n, :] = -G
        bigG[n:, :] = G
        h[:n, :] = -a
        h[n:, :] = b
        paraset = map(cvxopt.matrix , (P,q,bigG,h,D,d))
        beta = qp(*paraset)['x']
    temp = K*X
    V = (scipy.identity(temp.shape[0]) - temp) * V
    return (beta,V, e,K)

@cython.boundscheck(False) # turn of bounds-checking for entire function
def kalman_filter(np.ndarray b, 
                  np.ndarray V,
                  np.ndarray Phi,
                  np.ndarray y,
                  np.ndarray X,
                  np.ndarray sigma,
                  np.ndarray Sigma,
                  int flag = 0,
                  np.ndarray D = None,
                  np.ndarray d = None,
                  np.ndarray G = None,
                  np.ndarray a = None,
                  np.ndarray c = None):
    cdef unsigned int i, x, t, n
    t = scipy.shape(X)[0]
    n = scipy.shape(X)[1]
    cdef np.ndarray beta, e, K
    beta = scipy.empty(scipy.shape(X), dtype = DTYPE)
    if D is None:
        D = scipy.ones((1,n), dtype = DTYPE)
    if d is None:
        d = scipy.matrix(1., dtype = DTYPE)
    if G is None:
        G = scipy.identity(n, dtype = DTYPE)
    if a is None:
        a = scipy.zeros((n,1), dtype = DTYPE)
    if c is None:
        c = scipy.ones((n,1), dtype = DTYPE)
    (b, V) = kalman_predict(b,V,Phi, Sigma)
    for i in range(t):
        beta[i] = scipy.array(b, dtype = DTYPE).T
        (b,V, e,K) = kalman_upd(b,V, y[i] ,X[i], sigma, Sigma, flag, D, d,G,a,c)
        (b, V) = kalman_predict(b,V,Phi, Sigma)
    return beta
