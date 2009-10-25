#from regression import Regression, ToBeImplemented
from ecKalmanFilter import ECKalmanFilter
import csv, scipy, cvxopt
from timeSeriesFrame import TimeSeriesFrame, StylusReader
from libregression import constrainedflexibleleastsquare
from icRegression import ICRegression
from datetime import date

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


        beta = scipy.empty((self.t,self.n))
        Phi = self.Phi
        y = self.respond.data
        X = self.regressors.data
        D = self.D
        d = self.d
        smallG = self.G
        a = self.a
        b = self.b
        W1 = self.W1
        W2 = self.W2
        lamb = self.lamb
        beta = constrainedflexibleleastsquare(X, y, lamb, W1, W2, Phi, D, d, smallG, a, b)
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

