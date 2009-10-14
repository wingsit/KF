import scipy, cvxopt
from cvxopt.solvers import qp
"""This file contains a list of algoritm that is used for estimation
    some optmisation should be done in future.
    Naming convention:
        function with no ec/ic prefix are estimator for unconstrained problem
        ec* prefix: this is for equality constraints.
        ic* prefixL this is for inequality constraints."""
DEBUG = 0

def regression(X, y, W):
    return (X.T * W * X).I*(X.T * W * y)

def ecregression(X,y,W,D,d):
    covinv = (X.T * W * X).I
    lamb = (D * covinv * D.T).I * (D * covinv * X.T * W * y - d)
    return  covinv * (X.T * W * y - D.T * lamb)

def icregression(X,y, W, D,d,G,a,b,n):
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
def kalman_predict(b, V, Phi, S):
    (b, V, Phi, S) = map(scipy.matrix, (b, V, Phi, S))
    b = Phi * b
    V = Phi * V * Phi.T + S
    return b, V

def kalman_upd(beta, V, y, X, s, S, switch = 0,D = None, d = None, G = None, a = None, b = None):
    """switch:
        0 | no constraints
        1 | equality constraints
        2 | inequality constraints
    """
    e = y - X * beta
    K = V * X.T * ( s + X * V * X.T).I
    beta = beta + K * e
    if switch == 1:
        D = scipy.matrix(D)
        d = scipy.matrix(d)
        beta = beta - S * D.T * ( D * S * D.T).I * ( D * beta - d)
    elif switch == 2:
        G = scipy.matrix(G)
        a = scipy.matrix(a)
        b = scipy.matrix(b)
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
        beta = qp(*paraset)['x']
    temp = K*X
    V = (scipy.identity(temp.shape[0]) - temp) * V
    return (beta,V, e,K)

def kalman_filter(b, V, Phi,  y, X, sigma, Sigma, switch = 0,D = None, d = None, G = None, a = None, c = None):
    beta = scipy.empty(scipy.shape(X))
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
