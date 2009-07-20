from regression import Regression, ToBeImplemented
import csv,numpy, scipy
from timeSeriesFrame import *
from copy import deepcopy

DEBUG = 0
kappa = 1000
    
class KalmanFilter(Regression):
    """This is a KalmanFilter Class subclassed from Regression"""
    intercept = True
    def __init__(self, respond = None, regressors = None, intercept = False,
                 Sigma = None, sigma = None, initBeta = None, initVariance = None, Phi = None,  **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        Regression.__init__(self,respond, regressors, intercept, **args)
        if (not initBeta) and self.intercept:
            self.initBeta = scipy.ones((self.n,1))/float(self.n-1)
            self.initBeta[0] = 0
        elif initBeta and self.intercept:
            self.initBeta = scipy.ones((n,1))/float(n)
        elif (not initBeta) and (not self.intercept):
            self.initBeta = initBeta
        else:
            self.initBeta = scipy.zero((self.n, 1))
            self.initBeta[1:] = initBeta

        if initVariance and self.intercept:
            self.initVariance = scipy.zero((self.n, self.n))
            self.initVariance[1:,1:] = initVariance
        elif initVariance and (not self.intercept):
            self.initVariance = initVariance
        else:
            self.initVariance = scipy.identity(self.n)*kappa
        if not Phi:
            self.Phi = scipy.identity(self.n)
        else:
            self.Phi = Phi
        self.paras = args.get("paras")
        self.Sigma = Sigma
        self.sigma = sigma

    def train(self):
        """This fucntion will start the estimation. This is separated from addData."""        
        pass

    
def main():
    try:
        intercept = False
        stock_data = list(csv.reader(open("simulated_portfolio.csv", "rb")))
        stock = StylusReader(stock_data)
        del stock_data
        respond = stock[:,0]
        regressors = stock[:,1:]
        obj = KalmanFilter(respond, regressors, intercept, scipy.identity(2)*kappa, 0.001)

        obj.train()
        print obj.getEstimate()
        print obj.getEstimate(date(2001,1,1))
        print obj.predict()
        print obj.predict(date(2001,1,1))
        obj.est.toCSV("default2.csv")
        print obj.R2()
    except:
        from print_exc_plus import print_exc_plus
        print_exc_plus()

if __name__ == "__main__":
    main()

