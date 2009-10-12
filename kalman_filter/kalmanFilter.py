from regression import Regression, ToBeImplemented
import csv,numpy, scipy
from timeSeriesFrame import *
from copy import deepcopy
from libregression import kalman_predict, kalman_upd

DEBUG = 0
kappa = 1./1000.0
    
class KalmanFilter(Regression):
    """
    This is a KalmanFilter Class subclassed from Regression
    """
    intercept = True
    def __init__(self, respond = None, regressors = None, intercept = False,
                 Sigma = None, sigma = None, initBeta = None, initVariance = None, Phi = None,  **args):
        """
        :param respond: Dependent time series
        :type respond: TimeSeriesFrame<double>
        :param regressors: Independent time serieses
        :type regressors: TimeSeriesFrame<double>
        :param intercept: include/exclude intercept in the regression
        :type intercept: boolean
        """
        Regression.__init__(self,respond, regressors, intercept, **args)
        if ( initBeta is None) and self.intercept:
            self.initBeta = scipy.ones((self.n,1))/float(self.n-1)
            self.initBeta[0] = 0
        elif initBeta and self.intercept:
            self.initBeta = scipy.ones((n,1))/float(n)
        elif (initBeta is None) and (not self.intercept):
            self.initBeta = scipy.ones((self.n,1))/float(self.n)
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
        """
        This fucntion will start the estimation. This is separated from addData.
        """
        beta = scipy.empty((self.t,self.n))
        b = self.initBeta
        V = self.initVariance
        Phi = self.Phi
        S = self.Sigma
#        import code; code.interact(local=locals())
        (b, V) = kalman_predict(b,V,Phi, S)
        y = self.respond.data
        X = self.regressors.data
        for i, (xs, ys) in enumerate(zip(X,y)):
            (b,V, e,K) = kalman_upd(b,V, ys ,xs, self.sigma, self.Sigma)
            print "b:\n", b
            print "V:\n", V
            print "e:\n", e
            print "K:\n", K
            beta[i,:] = scipy.array(b).T
            (b, V) = kalman_predict(b,V,Phi, S)
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self
    
def main():
#    try:
    intercept = False
    stock_data = list(csv.reader(open("simulated_portfolio.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressors = stock[:,1:]
    obj = KalmanFilter(respond, regressors, intercept, scipy.identity(2)*kappa, 0.001)
    obj.train()
    print obj.getEstimate().data
#    print obj.getEstimate(date(2001,1,1))
#    print obj.predict()
#    print obj.predict(date(2001,1,1))
    #    obj.est.toCSV("simulated_portoflio.csv")
#    print obj.R2()
    obj.getEstimate().plot()
#    import code; code.interact(local=locals())
#    except:
#        from print_exc_plus import print_exc_plus
        #print_exc_plus()

if __name__ == "__main__":
    main()

