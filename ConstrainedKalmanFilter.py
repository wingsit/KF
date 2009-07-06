##<<<<<<< HEAD:ConstrainedKalmanFilter.py
from Regression import Regression, ToBeImplemented
import csv,numpy, scipy
from timeSeriesFrame import *
from copy import deepcopy
from cvxopt import *
from cvxopt.solvers import qp

DEBUG = 0
kappa = 1000

    
class ConstrainedKalmanFilter(Regression):
    """This is a KalmanFilter Class subclassed from Regression"""
    intercept = True
    def __init__(self, Sigma, sigma, initBeta = None, initVariance = None, Phi = None, G = None, D = None, d = scipy.matrix(1.00),   **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        self.initBeta = initBeta
        self.initVariance = initVariance
        self.Phi = Phi
        self.paras = args.get("paras")
        self.Sigma = Sigma
        self.sigma = sigma
        self.D = D
        self.G = G
        self.d = d
        pass

    def train(self):
        """This fucntion will start the estimation. This is separated from addData."""
        t,n = self.regressors.size()
        if self.intercept:
            n = n+1
        if self.initBeta == None:
            if self.intercept:
                self.initBeta = scipy.ones((n,1))/float(n-1)
                self.initBeta[0] = 0
            else:
                self.initBeta = scipy.ones((n,1))/float(n)
        if self.initVariance == None:
            self.initVariance = scipy.identity(n)*kappa
        if self.Phi == None:
            self.Phi = scipy.identity(n)
        if self.D == None:
            self.D = scipy.ones((1,n))
            if self.intercept:
                self.D[0,0] = 0
        if self.G == None:
            temp = matrix(scipy.identity(n))
            if self.intercept: temp[0,0] = 0
            self.G = matrix([-temp, temp])
            
        def _kalmanfilter(beta, V, X, y):
            shape = scipy.shape
            Phi = self.Phi
            if self.intercept:
                X = X.tolist()[0]
                X.insert(0,1.0)
                X = scipy.matrix(X)
            pbeta = Phi * beta
            if DEBUG:
                print shape(Phi)
                print shape(V)
                print shape(self.Sigma)
            pV = Phi * V * Phi.T + self.Sigma
            solvers.options['show_progress'] = False

            if DEBUG:
                print "X: ",X
                print "pbeta: ", pbeta
                print "2 * scipy.matrix(pV).I", 2 * scipy.matrix(pV).I
                print "-2 * scipy.matrix(pV).T.I * pbeta", -2 * scipy.matrix(pV).T.I * pbeta
                print "self.G", self.G
                print "self.h", self.h
                print "self.D", self.D
                print "self.d", self.d
            pbeta =  scipy.matrix(qp( matrix(2 * scipy.matrix(pV).I), matrix(-2 * scipy.matrix(pV).T.I * pbeta), matrix(self.G), matrix(self.h), matrix(self.D), matrix(self.d))['x'] )
            e = y - X * pbeta
            K = pV * X.T *( self.sigma + X * pV * X.T).I
            ubeta = pbeta +  K * e
            if DEBUG:
                print shape(K)
                print shape(X)
                print shape(pV)
#                print "K: ", Key
                print "X: ", X
                print "K*X: ", K*X                
            uV = (scipy.identity(n) - K* X) * pV
            return (pbeta, pV, ubeta, uV)

        self.pbetaList =[]
        self.ubetaList =[]
        self.ubetaList.append(self.initBeta)
        self.pVList = []
        self.uVList = []
        self.uVList.append(self.initVariance)

        for i in zip(self.regressors.rowIterator(), self.respond.rowIterator()):
            if DEBUG:
                print i[0].data
                print i[1].data
            estimate = _kalmanfilter(scipy.matrix(self.ubetaList[-1]), self.uVList[-1], i[0].data,i[1].data)
            self.pbetaList.append(estimate[0])
            self.pVList.append(estimate[1])
            self.ubetaList.append(estimate[2])
            self.uVList.append(estimate[3])

            

        betaMatrix = scipy.zeros((t,n))
        if self.intercept:
            n = n+1
        for index, value in enumerate(self.pbetaList[1:]):
            betaMatrix[index, :] = value.T
        cheader = self.regressors.cheader
        cheader.insert(0, "Alpha")
        self.est = TimeSeriesFrame(betaMatrix, self.regressors.rheader, cheader)
        pass

    def isConstraintable(self):
        """Boolean function to see if contraints can be imposed to the model. Default is True"""
        return True
    
def main():
#    try:
    obj = ConstrainedKalmanFilter( scipy.identity(7)*kappa, 0.001)
    stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressor = stock[:,1:]
    zeros = numpy.zeros((7,1))
    ones = numpy.ones((7,1))

#    print respond, regressor
    obj.intercept = False

    obj.setConstraints(zeros,ones)
    obj.addData(respond,regressor)
    obj.train()
    obj.getEstimate().toCSV()
#    except:
#        from print_exc_plus import print_exc_plus
#        print_exc_plus()
if __name__ == "__main__":
    main()












    
##=======
##from Regression import Regression, ToBeImplemented
##import csv,numpy, scipy
##from timeSeriesFrame import *
##from copy import deepcopy
##
##DEBUG = 0
##kappa = 1000
##
##    
##class ConstrainedKalmanFilter(Regression):
##    """This is a KalmanFilter Class subclassed from Regression"""
##    intercept = True
##    def __init__(self, Sigma, sigma, initBeta = None, initVariance = None, Phi = None, G = None, D = None, d = None,   **args):
##        """Input: paras where they are expected to be tuple or dictionary"""
##        self.initBeta = initBeta
##        self.initVariance = initVariance
##        self.Phi = Phi
##        self.paras = args.get("paras")
##        self.Sigma = Sigma
##        self.sigma = sigma
##        self.D = D
##        self.G = G
##        self.d = d
##        pass
##
##    def train(self):
##        """This fucntion will start the estimation. This is separated from addData."""
##        t,n = self.regressors.size()
##        if self.intercept:
##            n = n+1
##        if self.initBeta == None:
##            if self.intercept:
##                self.initBeta = scipy.ones((n,1))/float(n-1)
##                self.initBeta[0] = 0
##            else:
##                self.initBeta = scipy.ones((n,1))/float(n)
##        if self.initVariance == None:
##                self.initVariance = scipy.identity(n)*kappa
##        if self.Phi == None:
##                self.Phi = scipy.identity(n)
##                
##        def _kalmanfilter(beta, V, X, y):
##            shape = scipy.shape
##            Phi = self.Phi
##            if self.intercept:
##                X = X.tolist()[0]
##                X.insert(0,1.0)
##                X = scipy.matrix(X)
##            pbeta = Phi * beta
##            if DEBUG:
##                print shape(Phi)
##                print shape(V)
##                print shape(self.Sigma)
##            pV = Phi * V * Phi.T + self.Sigma
##            solvers.options['show_progress'] = False
##
##            if DEBUG:
##                print "X: ",X
##                print "pbeta: ", pbeta
##                print "2 * scipy.matrix(pV).I", 2 * scipy.matrix(pV).I
##                print "-2 * scipy.matrix(pV).T.I * pbeta", -2 * scipy.matrix(pV).T.I * pbeta
##                print "self.G", self.G
##                print "self.h", self.h
##                print "self.D", self.D
##                print "self.d", self.d
##            pbeta =  scipy.matrix(qp( matrix(2 * scipy.matrix(pV).I), matrix(-2 * scipy.matrix(pV).T.I * pbeta), self.G, self.h, matrix(self.D), matrix(self.d))['x'] )
##            e = y - X * pbeta
##            K = pV * X.T *( self.sigma + X * pV * X.T).I
##            ubeta = pbeta +  K * e
##            if DEBUG:
##                print shape(K)
##                print shape(X)
##                print shape(pV)
###                print "K: ", Key
##                print "X: ", X
##                print "K*X: ", K*X                
##            uV = (scipy.identity(n) - K* X) * pV
##            return (pbeta, pV, ubeta, uV)
##
####        betaList = []
####        betaList.append(self.initBeta)
####        betaTemp = self.initBeta
####        vTemp = self.initVariance
##
##        self.pbetaList =[]
##        self.ubetaList =[]
##        self.ubetaList.append(self.initBeta)
##        self.pVList = []
##        self.uVList = []
##        self.uVList.append(self.initVariance)
##
##        for i in zip(self.regressors.rowIterator(), self.respond.rowIterator()):
##            if DEBUG:
##                print i[0].data
##                print i[1].data
##            estimate = _kalmanfilter(scipy.matrix(self.ubetaList[-1]), self.uVList[-1], i[0].data,i[1].data)
##            self.pbetaList.append(estimate[0])
##            self.pVList.append(estimate[1])
##            self.ubetaList.append(estimate[2])
##            self.uVList.append(estimate[3])
##
####            betaList.append(estimate[0])
####            betaTemp = estimate[2]
####            vTemp = estimate[3]
##        betaMatrix = scipy.zeros((t,n))
##        for index, value in enumerate(betaList[1:]):
##            betaMatrix[index, :] = value.T
##        del betaList
##        cheader = self.regressors.cheader
##        cheader.insert(0, "Alpha")
##        self.est = TimeSeriesFrame(betaMatrix, self.regressors.rheader, cheader)
##        pass
##
##    def isConstraintable(self):
##        """This Kalman Filter support constraints."""
##        return True
##    
##def main():
##    try: 
##        obj = ConstrainedKalmanFilter( scipy.identity(8)*kappa, 0.001)
##        obj.intercept = True       
##        stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
##        stock = StylusReader(stock_data)
##        del stock_data
##        respond = stock[:,0]
##        regressor = stock[:,1:]
##        zeros = numpy.zeros((7,1))
##        ones = numpy.ones((7,1))
##
###        obj.withIntercept(False)
##        obj.setConstraints(zeros,ones)
##        obj.addData(respond,regressor)
##        obj.train()
##        obj.getEstimate().toCSV()
##        print obj.R2()
##    except:
##        from print_exc_plus import print_exc_plus
##        print_exc_plus()
##if __name__ == "__main__":
##    main()
