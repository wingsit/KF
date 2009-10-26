"""
This module contains a list of algoritm that is used for estimation. These
algorithm are unsafe and all these type and bound check should be done in the wrapper
level.

List of algorithm that has been inplemented:

Ordinary Least Squares
Ordinary Least Squares with Linear Equality Constraints
Ordinary Least Squares with Linear Inequality Constraints

Stepwise Least Squares
Stepwise Least Squares with Linear Equality Constraints
Stepwise Least Squares with Linear Inequality Constraints

Kalman Filter
Kalman Smoother
Kalman Filter with Linear Equality Constraints
Kalman Filter with Linear Inequality Constraints

Flexible Least Squares with Linear Inequality Constraints

*UNTESTED:*
Kalman Smoother with Linear Equality Constraints
Kalman Smoother with Linear Inequality Constraints

A lot of matrix inversion are used in these function and singularity is not checked.
Potential optimisation includes more stable inversion, large scale QP solver for
Flexible Least Squares

Naming convention:
        function with no ec/ic prefix are estimator for unconstrained problem
        ec* prefix: this is for equality constraints.
        ic* prefixL this is for inequality constraints.
"""



import scipy, cvxopt
from cvxopt.solvers import qp
from numpy import multiply as mlt
from numpy import mat

cvxopt.solvers.options['show_progress'] = False

DEBUG = 0

def regression(X, y, W):
    """
    Return the estimated weight based on ordinary regression model. The algorithm used to solve for the weight is done by matrix inversion.
    

    
    :param X: Independent variable
    :type X: scipy.matrix<float>
    :param y: Dependent variable
    :type y: scipy.matrix<float>
    :param W: Weight matrix
    :type W: scipy.matrix<float>
    :return: :math:`\hat{\\beta}`
    :rtype: scipy.matrix<float>


    .. math::
       \hat{\beta} = (X^T\; W \; X )^{-1} X^T W  y

    """
    if DEBUG:
        print "X: ", X
        print "y: ", y
        print "W: ", W
        print "(X.T * W * X): ", (X.T * W * X)
    return (X.T * W * X).I*(X.T * W * y)

def ecregression(X, y, W, D, d):
    r"""

    :param X: Independent variable
    :type X: scipy.matrix<float>
    :param y: Dependent variable
    :type y: scipy.matrix<float>
    :param W: Weight matrix
    :type W: scipy.matrix<float>
    :param D: Equality constraint matrix
    :type D: scipy.matrix<float>
    :param d: Equality constraint vector
    :type d: scipy.matrix<float>
    :return: :math:`\hat{\beta}`
    :rtype: scipy.matrix<float>

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

def icregression(X, y, W, D, d, G, a, b, n):
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


    This problem is translated nicely to quadratic programming problem. CVXOPT package is used to as the quadratic solver engine. 

    :param X: Independent variable
    :type X: scipy.matrix<float>
    :param y: Dependent variable
    :type y: scipy.matrix<float>
    :param W: Weight matrix
    :type W: scipy.matrix<float>
    :param D: Equality constraint matrix
    :type D: scipy.matrix<float>
    :param d: Equality constraint vector
    :type d: scipy.matrix<float>
    :param G: Inequality constraint matrix
    :type G: scipy.matrix<float>
    :param a b: Lower and upper bound of the inequality constraints
    :type a b: scipy.matrix<float>
    :return: :math:`\hat{\beta}`
    :rtype: scipy.matrix<float>

    """

    P = 2*X.T * W * X
    q = -2*X.T * W * y
    bigG = scipy.empty((2*n, n))
    h = scipy.empty((2*n, 1))
    bigG[:n, :] = -G
    bigG[n:, :] = G
    h[:n, :] = -a
    h[n:, :] = b    
    paraset = map(cvxopt.matrix, (P, q, bigG, h, D, d))
    return qp(*paraset)['x']
#============================================================
def kalman_predict(b, V, Phi, S):
    r"""
    This fucntion is the predicted step of Kalman Filter
    
    .. math::
      :nowrap:

      \begin{eqnarray*}
      \beta_{t+1|t} &=& \Phi  \beta_{t|t}\\
      V_{t+1|t} &=& \Phi  V_{t|t} \Phi^T + S
      \end{eqnarray*}


    :param b: :math:`\beta_{t|t}`
    :type b: scipy.matrix<float>
    :param V: :math:`V_{t|t}`
    :type V: scipy.matrix<float>
    :param Phi: :math:`\Phi`
    :type Phi: scipy.matrix<float>
    :param S: :math:`S`
    :type S: scipy.matrix<float>
    :return: :math:`(\beta_{t+1|t}, V_{t+1|t})`
    :rtype: Tuple of scipy.matrix<float>


    """
    (b, V, Phi, S) = map(scipy.matrix, (b, V, Phi, S))
    b = Phi * b
    V = Phi * V * Phi.T + S
    return b, V

def kalman_upd(beta,
               V,
               y,
               X,
               s,
               S,
               switch = 0,
               D = None,
               d = None,
               G = None,
               a = None,
               b = None):
    r"""
    This is the update step of kalman filter. 

    .. math::
       :nowrap:

       \begin{eqnarray*}
       e_t &=& y_t -  X_t \beta_{t|t-1} \\
       K_t &=&  V_{t|t-1} X_t^T (\sigma + X_t V_{t|t-1} X_t )^{-1}\\
       \beta_{t|t} &=& \beta_{t|t-1} + K_t e_t\\
       V_{t|t} &=& (I - K_t X_t^T) V_{t|t-1}\\
       \end{eqnarray*}


    
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
        paraset = map(cvxopt.matrix, (P, q, bigG, h, D, d))
        beta = qp(*paraset)['x']
    temp = K*X
    V = (scipy.identity(temp.shape[0]) - temp) * V
    return (beta, V, e, K)

def kalman_smoother(b,
                    V,
                    Phi,
                    y,
                    X,
                    sigma,
                    Sigma,
                    switch = 0,
                    D = None,
                    d = None,
                    G = None,
                    a = None,
                    c = None):
    r"""
    .. math::
       :nowrap:

       \begin{eqnarray*}
       K_t &= V_{t|t} \Phi ({V_{t+1|t} })^{-1}\\
       \beta_{t|T} &= \beta_{t|t} + K_t [ \beta_{t+1|T} - \Phi \beta_{t+1|t}]
       \end{eqnarray*}
  
    """

    t, n = scipy.shape(X)
    betaP = scipy.empty(scipy.shape(X))
    betaU = scipy.empty(scipy.shape(X))
    varP = []
    varU = []
    
#    n = len(b)
    if D is None:
        D = scipy.ones((1, n))
    if d is None:
        d = scipy.matrix(1.)
    if G is None:
        G = scipy.identity(n)
    if a is None:
        a = scipy.zeros((n, 1))
    if c is None:
        c = scipy.ones((n, 1))
    (b, V) = kalman_predict(b, V, Phi, Sigma)
    for i in xrange(len(X)):
        betaP[i] = scipy.array(b).T
        varP.append(V)
        (b, V, e, K) = kalman_upd(b,
                                  V,
                                  y[i],
                                  X[i],
                                  sigma,
                                  Sigma,
                                  switch,
                                  D,
                                  d,
                                  G,
                                  a,
                                  c)
        betaU[i] = scipy.array(b).T
        varU.append(V)
        
        (b, V) = kalman_predict(b, V, Phi, Sigma)
    ################filtering part done#################
    varP = varP[:-1]
    betaP = betaP[:-1]

    for i in xrange(len(varP)):
        varP[i] = varU[i+1] * scipy.matrix(varP[i]).I
    del varU

    sbeta = scipy.empty(scipy.shape(X))

    sbeta[t-1] = betaU[t-1]
    for i in xrange(2, t):
        sbeta[t-i] = (scipy.matrix(betaU[t-i]).T + varP[t-i] 
                      * scipy.matrix(betaU[t-i+1] - betaP[t-i]).T).T
    return sbeta


def kalman_filter(b,
                  V,
                  Phi,
                  y,
                  X,
                  sigma,
                  Sigma,
                  switch = 0,
                  D = None,
                  d = None,
                  G = None,
                  a = None,
                  c = None):
    r"""
    
    .. math::
       :nowrap:

       \begin{eqnarray*}
       \beta_{t|t-1} = \Phi \: \beta_{t-1|t-1}\\
       V_{t|t-1} = \Phi  V_{t-1|t-1} \Phi ^T + \Sigma \\
       e_t = y_t -  X_t \beta_{t|t-1}\\
       K_t =  V_{t|t-1} X_t^T (\sigma + X_t V_{t|t-1} X_t )^{-1}\\
       \beta_{t|t} = \beta_{t|t-1} + K_t e_t\\
       V_{t|t} = (I - K_t X_t^T) V_{t|t-1}\\
       \end{eqnarray*}

    """

    n = scipy.shape(X)[1]
    beta = scipy.empty(scipy.shape(X))
    n = len(b)
    if D is None:
        D = scipy.ones((1, n))
    if d is None:
        d = scipy.matrix(1.)
    if G is None:
        G = scipy.identity(n)
    if a is None:
        a = scipy.zeros((n, 1))
    if c is None:
        c = scipy.ones((n, 1))
#        import code; code.interact(local=locals())
    (b, V) = kalman_predict(b, V, Phi, Sigma)
    for i in xrange(len(X)):
        beta[i] = scipy.array(b).T
        (b, V, e, K) = kalman_upd(b,
                                V,
                                y[i],
                                X[i],
                                sigma,
                                Sigma,
                                switch,
                                D,
                                d,
                                G,
                                a,
                                c)
        (b, V) = kalman_predict(b, V, Phi, Sigma)
    return beta

def constrainedflexibleleastsquare(X,
                                   y,
                                   lamb,
                                   W1,
                                   W2,
                                   Phi,
                                   D,
                                   d,
                                   smallG,
                                   a,
                                   b):
    r"""
    
    .. math::
       :nowrap:

       \begin{eqnarray*}
       \arg_{\beta_t,\forall t}\min\sum_{t = 1}^m (y_t - X^T_t \beta_t)^T W_1 (y_t - X^T_t \beta_t) +& \sum_{t=1}^{m-1}\lambda (\beta_{t+1} - \Phi \beta_t)^T W_2 (\beta_{t+1} - \Phi \beta_t)\\
       G \beta_t & \geq g, \quad \forall t\\
       D \beta_t & = d, \quad \forall t
       \end{eqnarray*}

    """
    t, n = scipy.shape(X)
    P = scipy.empty( (t * n, t * n))
    G = scipy.empty( (2*n, n))
    G[0:n, 0:n] = smallG
    G[n: 2*n, 0:n] = -smallG
    g = scipy.empty( (2*n, 1))
    g[0:n] = b
    g[n:2*n] = -a
####P#####
    for i in xrange(t):
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
        
    for i in xrange(t):
        bigG[i*gr:(i+1)*gr, i*gc:(i+1)*gc] = G

    bigg = scipy.empty((gr*t, 1))
    for i in xrange(t):
        bigg[i*gr:(i+1)*gr] = g
                
    dr, dc = scipy.shape(D)
    A = scipy.empty((t* dr, t* dc))
    for i in xrange(t):
        A[i*dr: (i+1) * dr, i*dc:(i+1)*dc] = D
        b = scipy.empty((t, 1))
    for i in xrange(t):
        b[i:(i+1)] = d

    paraset = map(cvxopt.matrix, (P, q, bigG, bigg, A, b))
    beta =  mat(qp(*paraset)['x']).reshape(t, n).tolist()
    return beta
