"""
This module contains ordinary kalman filter classes
"""

from regression import Regression
import csv, scipy
from timeSeriesFrame import TimeSeriesFrame, StylusReader
from libregression import kalman_filter

DEBUG = 0
KAPPA = 1./1000.0
    
class KalmanFilter(Regression):
    """
    This is a KalmanFilter Class subclassed from Regression
    """
    intercept = True
    def __init__(self,
                 respond = None,
                 regressors = None,
                 intercept = False,
                 Sigma = None,
                 sigma = None,
                 initBeta = None,
                 initVariance = None,
                 Phi = None,
                 **args):
        """
        :param respond: Dependent time series
        :type respond: TimeSeriesFrame<double>
        :param regressors: Independent time serieses
        :type regressors: TimeSeriesFrame<double>
        :param intercept: include/exclude intercept in the regression
        :type intercept: boolean
        """
        Regression.__init__(self, respond, regressors, intercept, **args)
        if ( initBeta is None) and self.intercept:
            self.initBeta = scipy.ones((self.n, 1))/float(self.n-1)
            self.initBeta[0] = 0
        elif initBeta and self.intercept:
            self.initBeta = scipy.ones((n, 1))/float(n)
        elif (initBeta is None) and (not self.intercept):
            self.initBeta = scipy.ones((self.n, 1))/float(self.n)
        else:
            self.initBeta = scipy.zero((self.n, 1))
            self.initBeta[1:] = initBeta

        if initVariance and self.intercept:
            self.initVariance = scipy.zero((self.n, self.n))
            self.initVariance[1:, 1:] = initVariance
        elif initVariance and (not self.intercept):
            self.initVariance = initVariance
        else:
            self.initVariance = scipy.identity(self.n)*KAPPA

        if  Phi is None:
            self.Phi = scipy.identity(self.n)
        else:
            self.Phi = Phi
        self.paras = args.get("paras")
        self.Sigma = Sigma
        self.sigma = sigma

    def train(self):
        """
        This fucntion will start the estimation. This is separated from addData.
        """
        beta = scipy.empty((self.t, self.n))
        b = self.initBeta
        V = self.initVariance
        Phi = self.Phi
        S = self.Sigma
        s = self.sigma
        y = self.respond.data
        X = self.regressors.data
        beta =  kalman_filter(b, V, Phi, y, X, s, S)
        self.est = TimeSeriesFrame(beta, 
                                   self.regressors.rheader, 
                                   self.regressors.cheader)
        return self
    
def main():
#    try:
    intercept = False
    stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:, 0]
    regressors = stock[:, 1:]
    obj = KalmanFilter(respond, regressors, intercept, scipy.identity(7), 1.)
    obj.train()
    print obj.getEstimate().data
#    print obj.getEstimate(date(2001,1,1))
#    print obj.predict()
#    print obj.predict(date(2001,1,1))
    obj.est.toCSV("simulated_dodge_cox.csv")
#    print obj.R2()
    obj.getEstimate().plot()
#    import code; code.interact(local=locals())
#    except:
#        from print_exc_plus import print_exc_plus
        #print_exc_plus()

if __name__ == "__main__":
    main()

