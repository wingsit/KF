"""
This module contains ordinary kalman filter classes
"""

from regression import *
import csv, scipy
from timeSeriesFrame import TimeSeriesFrame, StylusReader

try:
    from clibregression import kalman_predict, kalman_upd, kalman_filter
except ImportError:
    print "Cannot import clibregression"
    from libregression import kalman_predict, kalman_upd, kalman_filter

DEBUG = 0
KAPPA = 1./100.0
    
class KalmanFilter(Regression):
    """
    This is a Kalman filter Class subclassed from Regression
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
            self.initBeta = scipy.ones((self.n, 1))/float(self.n - 1)
            self.initBeta[0] = 0
        elif initBeta is not None and self.intercept:
            self.initBeta = scipy.ones((n, 1))/float(n)
        elif (initBeta is None) and (not self.intercept):
            self.initBeta = scipy.ones((self.n, 1))/float(self.n)
        else:
            self.initBeta = scipy.zeros((self.n, 1))
            self.initBeta = initBeta

        if initVariance and self.intercept:
            self.initVariance = scipy.zeros((self.n, self.n))
            self.initVariance[1:, 1:] = initVariance
        elif initVariance and (not self.intercept):
            self.initVariance = initVariance
        else:
            self.initVariance = scipy.zeros((self.n, self.n))

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

class KalmanSmoother(KalmanFilter):
    """
    This is a Kalman Smoother Class subclassed from Kalman Filter
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
        KalmanFilter.__init__(self, respond, regressors, intercept, Sigma, sigma, initBeta, initVariance, Phi, **args)

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
        beta =  kalman_smoother(b, V, Phi, y, X, s, S)
        self.est = TimeSeriesFrame(beta, 
                                   self.regressors.rheader, 
                                   self.regressors.cheader)
        return self

class ECKalmanFilter(KalmanFilter, ECRegression):
    """This is a KalmanFilter Class subclassed from Regression"""
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
                 D = None,
                 d = scipy.matrix(1.00),
                 **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        KalmanFilter.__init__(self,
                              respond,
                              regressors,
                              intercept,
                              Sigma,
                              sigma,
                              initBeta,
                              initVariance,
                              Phi,
                              **args)
        ECRegression.__init__(self,
                              respond,
                              regressors,
                              intercept,
                              D,
                              d,
                              **args)
    def train(self):
        """This fucntion will start the estimation. This is separated from addData."""
        beta = scipy.empty((self.t,self.n))
        b = self.initBeta
        V = self.initVariance
        Phi = self.Phi
        S = self.Sigma
        (b, V) = kalman_predict(b,V,Phi, S)
        y = self.respond.data
        X = self.regressors.data
        D = self.D
        d = self.d
        s = self.sigma
        beta =  kalman_filter(b, V, Phi, y, X, s, S, 1, D, d)

        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

class ICKalmanFilter(ECKalmanFilter, ICRegression):
    """This is a KalmanFilter Class subclassed from Regression"""
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
                 D = None,
                 d = scipy.matrix(1.00),
                 G = None,
                 a = None,
                 b = None,
                 **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        ECKalmanFilter.__init__(self,
                                respond,
                                regressors,
                                intercept,
                                Sigma,
                                sigma,
                                initBeta,
                                initVariance,
                                Phi,
                                D,
                                d,
                                **args)
        
        ICRegression.__init__(self,
                              respond,
                              regressors,
                              intercept,
                              D,
                              d,
                              G,
                              a,
                              b,
                              **args)





    def train(self):
        """
        This fucntion will start the estimation. This is separated from addData.
        """
        beta = scipy.empty((self.t,self.n))
        b = self.initBeta
        V = self.initVariance
        Phi = self.Phi
        s = self.sigma
        S = self.Sigma
        y = self.respond.data
        X = self.regressors.data
        D = self.D
        d = self.d
        G = self.G
        a = self.a
        c = self.b
        beta =  kalman_filter(b, V, Phi, y, X, s, S, 2, D, d, G, a, c)
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

																														
def main():
#    try:
    intercept = False
    stock_data = list(csv.reader(open("sine_wave.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:, 0]
    regressors = stock[:, 1:]
    initBeta = scipy.matrix([0.55, 0.45]).T
    Sigma = scipy.matrix([[0.123873, -0.12387], [-0.12387,0.123873]])
    obj = KalmanFilter(respond, regressors, intercept, Sigma*KAPPA, 0.12, initBeta = initBeta)
 
#    obj = KalmanFilter(respond, regressors, intercept, scipy.identity(7), 1.)
    obj.train()
#    print obj.getEstimate().data
#    print obj.getEstimate(date(2001,1,1))
#    print obj.predict()
#    print obj.predict(date(2001,1,1))
#    obj.est.toCSV("simulated_dodge_cox.csv")
#    print obj.R2()
    obj.getEstimate().plot()
#    import code; code.interact(local=locals())
#    except:
#        from print_exc_plus import print_exc_plus
        #print_exc_plus()

if __name__ == "__main__":
    main()

