from Regression import Regression, ToBeImplemented
import csv,numpy, scipy
from timeSeriesFrame import *
from copy import deepcopy

DEBUG = 1

class UnconstrainableError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return  repr(self.value)

    
class KalmanFilter(Regression):
    """This is a KalmanFilter Class subclassed from Regression"""
    intercept = True
    def __init__(self, Sigma, sigma, initBeta = None, initVariance = None, Phi = None,  **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        self.initBeta = initBeta
        self.initVariance = initVariance
        self.Phi = Phi
        self.paras = args.get("paras")
        self.Sigma = Sigma
        self.sigma = sigma
        pass

    def setConstraints(self, a, b):
        raise UnconstrainableError("This model does not accept constraint")

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
                self.initVariance = scipy.identity(n)
        if self.Phi == None:
                self.Phi = scipy.identity(n)
        def _kalmanfilter(beta, V, x, y):
            Phi = self.Phi
            pbeta = Phi * beta
            if DEBUG:
                print Phi.shape()
                print V.shape()
                print self.Sigma.shape()
            pV = Phi * V * V.T + self.Sigma
            e = y - X * pbeta
            K = pV * X.T *( 1./self.sigma + X * pV * X.T).I
            ubeta = pbeta +  K * e
            uV = (I - K * X.T) * pV
            return (beta, uV)

        betaList = []
        betaList.append(self.initBeta)
        vList = []
        vList.append(self.initVariance)
        for i in zip(self.regressors.rowIterator(), self.respond.rowIterator()):
            print i[0].data
            print i[1].data
            result = _kalmanfilter(betaList[-1], vList[-1], i[0].data,i[1].data)
            print result[0], result[1]
        pass

    def isConstraintable(self):
        """Boolean function to see if contraints can be imposed to the model. Default is True"""
        return False
    
def main():
    obj = KalmanFilter( numpy.ones((7,7)), 10)
    stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
    #stock = TSF()
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressor = stock[:,1:]

#    print respond, regressor
#    obj.setConstraints(zeros,ones)
    obj.addData(respond,regressor)
    try:
        obj.train()
    except:
        from print_exc_plus import print_exc_plus
        print_exc_plus()

    
if __name__ == "__main__":
    main()
