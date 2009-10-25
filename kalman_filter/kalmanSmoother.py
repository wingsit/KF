"""
This module contains ordinary kalman filter classes
"""

from kalmanFilter import KalmanFilter
import csv, scipy
from timeSeriesFrame import TimeSeriesFrame, StylusReader

try:
    from clibregression import kalman_smoother
except ImportError:
    print "Cannot import clibregression"
    from libregression import kalman_smoother

DEBUG = 0
KAPPA = 1./100.0
    
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
#        print beta
        self.est = TimeSeriesFrame(beta, 
                                   self.regressors.rheader, 
                                   self.regressors.cheader)
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
    obj = KalmanSmoother(respond, regressors, intercept, Sigma*KAPPA, 0.12, initBeta = initBeta)
 
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

