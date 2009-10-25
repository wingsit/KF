import scipy, cvxopt
from cvxopt.solvers import qp
from numpy import multiply as mlt
from numpy import mat

import numpy as np
cimport numpy as np

DTYPE = np.float32

cvxopt.solvers.options['show_progress'] = False

"""
This file contains a list of algoritm that is used for estimation
    some optmisation should be done in future.
    Naming convention:
        function with no ec/ic prefix are estimator for unconstrained problem
        ec* prefix: this is for equality constraints.
        ic* prefixL this is for inequality constraints.
"""
DEBUG = 0

def regression(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] y, np.ndarray[np.float64_t, ndim=2] W):
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
    if DEBUG:
        print "X: ", X
        print "y: ", y
        print "W: ", W
        print "(X.T * W * X): ", (X.T * W * X)
    return (X.T * W * X).I*(X.T * W * y)

def ecregression(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] y, np.ndarray[np.float64_t, ndim=2] W,np.ndarray[np.float64_t, ndim=2] D,np.ndarray[np.float64_t, ndim=2] d):
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
    if DEBUG:
        print "X: ", X
        print "y: ", y
        print "W: ", W
        print "D: ", D
        print "d: ", d
        print "(X.T * W * X).I",  (X.T * W * X).I
        print "(D * covinv * D.T).I: ", (D * (X.T * W * X).I * D.T).I
    covinv = (X.T * W * X).I
    lamb = (D * covinv * D.T).I * (D * covinv * X.T * W * y - d)
    return  covinv * (X.T * W * y - D.T * lamb)

def icregression(np.ndarray[np.float64_t, ndim=2] X,
                 np.ndarray[np.float64_t, ndim=2] y,
                 np.ndarray[np.float64_t, ndim=2] W,
                 np.ndarray[np.float64_t, ndim=2] D,
                 np.ndarray[np.float64_t, ndim=2] d,
                 np.ndarray[np.float64_t, ndim=2] G,
                 np.ndarray[np.float64_t, ndim=2] a,
                 np.ndarray[np.float64_t, ndim=2] b,
                 unsigned int n):
    r"""
    This return the estimated weight on the following regression problem
    
    .. math::
       y = X \beta

    constained to

    .. math::
       :nowrap:
       
       \begin{eqnarray*}
       D \beta &=& d\\
       a \leq G &\beta & \leq b
       \end{eqnarray*}


    This problem is translated nicely to quadratic programming problem. 
    """

    P = 2*X.T * W * X
    q = -2*X.T * W * y
    bigG = scipy.empty((2*n, n))
    h = scipy.empty((2*n, 1))
    bigG[:n, :] = -G
    bigG[n:, :] = G
    h[:n, :] = -a
    h[n:, :] = b    
    paraset = map(cvxopt.matrix , (P,q,bigG,h,D,d))
    return qp(*paraset)['x']
#============================================================
def kalman_predict(np.ndarray[np.float64_t, ndim=2] b,
                   np.ndarray[np.float64_t, ndim=2] V,
                   np.ndarray[np.float64_t, ndim=2] Phi,
                   np.ndarray[np.float64_t, ndim=2] S):
    r"""
    This fucntion return the predicted ..math: \beta, V
    """
    (b, V, Phi, S) = map(scipy.matrix, (b, V, Phi, S))
    b = Phi * b
    V = Phi * V * Phi.T + S
    return b, V

def kalman_upd( np.ndarray[np.float64_t, ndim=2] beta,
                np.ndarray[np.float64_t, ndim=2] V,
                np.ndarray[np.float64_t, ndim=2] y,
                np.ndarray[np.float64_t, ndim=2] X,
                double s,
                np.ndarray[np.float64_t, ndim=2] S,
                int switch = 0,
                np.ndarray[np.float64_t, ndim=2] D = None,
                np.ndarray[np.float64_t, ndim=2] d = None,
                np.ndarray[np.float64_t, ndim=2] G = None,
                np.ndarray[np.float64_t, ndim=2] a = None,
                np.ndarray[np.float64_t, ndim=2] b = None):
    """switch:
        0 | no constraints
        1 | equality constraints
        2 | inequality constraints
    """
    cdef np.ndarray[np.float64_t, ndim=2] e, K, P, q, bigG, h, temp
    cdef Py_ssize_t n
    e = y - X * beta
    K = V * X.T * ( s + X * V * X.T).I
    beta = beta + K * e
    if switch == 1:
#        D = scipy.matrix(D)
#        d = scipy.matrix(d)
        beta = beta - S * D.T * ( D * S * D.T).I * ( D * beta - d)
    elif switch == 2:
#        G = scipy.matrix(G)
#        a = scipy.matrix(a)
#        b = scipy.matrix(b)
        n = len(beta)
        P = 2* V.I
        q = -2 * V.I.T * beta
        bigG = scipy.empty((2*n, n))
        h = scipy.empty((2*n, 1))
        bigG[:n, :] = -G
        bigG[n:, :] = G
        h[:n, :] = -a
        h[n:, :] = b
        paraset = map(cvxopt.matrix , (P,q,bigG,h,D,d))
        beta = mat(qp(*paraset)['x'])
        
    temp = K*X
    V = (scipy.identity(temp.shape[0]) - temp) * V
    return (beta,V, e,K)

def kalman_filter(np.ndarray[np.float64_t, ndim=2] b,
                  np.ndarray[np.float64_t, ndim=2] V,
                  np.ndarray[np.float64_t, ndim=2] Phi,
                  np.ndarray[np.float64_t, ndim=2] y,
                  np.ndarray[np.float64_t, ndim=2] X,
                  double sigma,
                  np.ndarray[np.float64_t, ndim=2] Sigma,
                  int switch = 0,
                  np.ndarray[np.float64_t, ndim=2] D = None,
                  np.ndarray[np.float64_t, ndim=2] d = None,
                  np.ndarray[np.float64_t, ndim=2] G = None,
                  np.ndarray[np.float64_t, ndim=2] a = None,
                  np.ndarray[np.float64_t, ndim=2]c = None):
    cdef Py_ssize_t n = scipy.shape(X)[1]
    cdef np.ndarray[np.float64_t, ndim=2] beta = scipy.empty(scipy.shape(X))
#    n = len(b)
    if D is None:
        D = scipy.ones((1,n))
    if d is None:
        d = scipy.matrix(1.)
    if G is None:
        G = scipy.identity(n)
    if a is None:
        a = scipy.zeros((n,1))
    if c is None:
        c = scipy.ones((n,1))
#        import code; code.interact(local=locals())
    (b, V) = kalman_predict(b,V,Phi, Sigma)
    for i in xrange(len(X)):
        beta[i] = scipy.array(b).T
        (b,V, e,K) = kalman_upd(b,V, y[i] ,X[i], sigma, Sigma, switch, D, d,G,a,c)
        (b, V) = kalman_predict(b,V,Phi, Sigma)
    return beta

def constrainedflexibleleastsquare(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] y, double lamb, double W1, np.ndarray[np.float64_t, ndim=2] W2, np.ndarray[np.float64_t, ndim=2] Phi, np.ndarray[np.float64_t, ndim=2] D, d, np.ndarray[np.float64_t, ndim=2] smallG,  a,  b):
    cdef Py_ssize_t t,n,i
    cdef np.ndarray[np.float64_t, ndim=2] P, G, A, bigG
    cdef np.ndarray[np.float64_t, ndim=2] bigg, q
    t, n = scipy.shape(X)
    P = scipy.empty( (t * n, t * n), dtype = DTYPE)
    G = scipy.empty( (2*n, n), dtype = DTYPE)
    G[0:n, 0:n] = smallG
    G[n: 2*n, 0:n] = -smallG
    g = scipy.empty( (2*n, 1), dtype = DTYPE)
    g[0:n] = b
    g[n:2*n] = -a
####P#####
    for i in range(t):
        if i == 0:
            p = 2* W1* mlt(X[i].T, X[i]) + lamb * Phi.T * W2 * Phi
        elif i == t-1:
            p = 2* W1* mlt(X[i].T, X[i])
        else:
            p = 2* W1* mlt(X[i].T, X[i]) + lamb * (Phi.T * W2 * Phi + W2)
        if i < t-1:
            P[(i)*n:(i+1)*n, (i+1)*n:(i+2)*n] = -2 * lamb * Phi.T  *W2
            P[(i+1)*n:(i+2)*n, (i)*n:(i+1)*n] = -2 * lamb * W2 * Phi
        P[i*n:i*n+n, i*n:i*n+n] = p.copy()

##q##
    q = scipy.empty((t*n, 1))
    for i in xrange(t):
        q[i*n:(i+1)*n] = -2 * X[i].T * W1 * y[i]
#q = (-2 * W1 * y)
##bigG##
    gr, gc = scipy.shape(G)
    bigG = scipy.empty((gr*t, gc*t))
        
    for i in range(t):
        bigG[i*gr:(i+1)*gr, i*gc:(i+1)*gc] = G

    bigg = scipy.empty((gr*t,1))
    for i in range(t):
        bigg[i*gr:(i+1)*gr] = g
                
    dr, dc = scipy.shape(D)
    A = scipy.empty((t* dr,t* dc))
    for i in range(t):
        A[i*dr: (i+1) * dr, i*dc:(i+1)*dc] = D
        b = scipy.empty((t, 1))
    for i in range(t):
        b[i:(i+1)] = d

    paraset = map(cvxopt.matrix, (P, q, bigG, bigg, A, b))
    beta =  mat(qp(*paraset)['x']).reshape(t,n).tolist()
    return beta
