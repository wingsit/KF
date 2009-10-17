#from regression import Regression, ToBeImplemented
from ecKalmanFilter import ECKalmanFilter
import csv,numpy, scipy
from timeSeriesFrame import *
from copy import deepcopy
from libregression import kalman_filter
from icRegression import ICRegression

DEBUG = 0
kappa = 1
    
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
        """This fucntion will start the estimation. This is separated from addData."""
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
    stock_data = list(csv.reader(open("simulated_portfolio.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressors = stock[:,1:]
    obj = ICKalmanFilter(respond, regressors, intercept, scipy.identity(2)*kappa, 0.001)
    obj.train()
    print obj.getEstimate()
    print obj.getEstimate(date(2001,1,1))
    print obj.predict()
    print obj.predict(date(2001,1,1))
    obj.est.toCSV("default2.csv")
    print obj.R2()
#    import code; code.interact(local=locals())
#    except:
#        from print_exc_plus import print_exc_plus
        #print_exc_plus()

if __name__ == "__main__":
    main()

