#from regression import Regression, ToBeImplemented
from ecKalmanFilter import ECKalmanFilter
import csv, scipy, cvxopt
from timeSeriesFrame import TimeSeriesFrame, StylusReader
from libregression import kalman_filter
from icRegression import ICRegression
from datetime import date

from numpy import multiply as mlt
from numpy import mat
from cvxopt.solvers import qp

DEBUG = 0
    
class ICFlexibleLeastSquare(ICRegression):
    """This is a KalmanFilter Class subclassed from Regression"""
    intercept = True
    def __init__(self,
                 respond = None,
                 regressors = None,
                 intercept = False,
                 lamb = 1., 
                 W1 = None,
                 W2 = None,
                 Phi = None,
                 D = None,
                 d = scipy.matrix(1.00),
                 G = None,
                 a = None,
                 b = None):
        """Input: paras where they are expected to be tuple or dictionary"""
        
        ICRegression.__init__(self,
                              respond,
                              regressors,
                              intercept,
                              D,
                              d,
                              G,
                              a,
                              b)
        if W1 is not None:
            self.W1 = W1
        else:
            self.W1 = 1.
            
        if W2 is None:
            self.W2 = scipy.identity(self.n)
        else:
            self.W2 = W2

        if Phi is None:
            self.Phi = scipy.identity(self.n)
        else:
            self.Phi = Phi


        self.lamb = lamb

    def train(self):
        """
        This fucntion will start the estimation. This is separated from addData.
        """
        P = scipy.empty( (self.t * self.n, self.t * self.n))

        beta = scipy.empty((self.t,self.n))
        Phi = self.Phi
        y = self.respond.data
        X = self.regressors.data
        D = self.D
        d = self.d
        smallG = self.G
        a = self.a
        c = self.b
        W1 = self.W1
        W2 = self.W2
        n = self.n
        t = self.t
        G = scipy.empty( (2*n, n))
        G[0:n, 0:n] = smallG
        G[n: 2*n, 0:n] = -smallG
        g = scipy.empty( (2*n, 1))
        g[0:n] = c
        g[n:2*n] = -a
        lamb = self.lamb
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

        bigg = scipy.empty((gr*t,1))
        for i in xrange(t):
            bigg[i*gr:(i+1)*gr] = g
                
        dr, dc = scipy.shape(D)
        A = scipy.empty((t* dr,t* dc))
        for i in xrange(t):
            A[i*dr: (i+1) * dr, i*dc:(i+1)*dc] = D
            b = scipy.empty((t, 1))
        for i in xrange(t):
            b[i:(i+1)] = d

        paraset = map(cvxopt.matrix, (P, q, bigG, bigg, A, b))
        beta =  mat(qp(*paraset)['x']).reshape(t,n).tolist()
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self
    
def main():
#    try:
    intercept = False
    stock_data = list(csv.reader(open("sine_wave.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressors = stock[:,1:]
    initBeta = scipy.matrix([0.528744, 0.471256]).T
    Sigma = scipy.matrix([[0.123873, -0.12387], [-0.12387,0.123873]])
    obj = ICFlexibleLeastSquare(respond, regressors, intercept, 1.)
                                
    obj.train()
#    print obj.getEstimate()
#    print obj.getEstimate(date(2001,1,1))
#    print obj.predict()
#    print obj.predict(date(2001,1,1))
#    obj.est.toCSV("default2.csv")
    print obj.R2()
    obj.est.plot()

if __name__ == "__main__":
    main()

