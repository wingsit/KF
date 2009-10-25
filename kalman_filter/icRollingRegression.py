import csv,scipy
from timeSeriesFrame import *
from datetime import date
from exc import *
from copy import copy

try:
    from clibregression import icregression
except ImportError:
    print "Cannot import clibregression"
    from libregression import icregression
from windows import *
from rollingRegression import RollingRegression
from icRegression import ICRegression
from libregression import icregression

DEBUG = 0
WINDOWSIZE = 24

class ICRollingRegression(RollingRegression, ICRegression):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, respond = None, regressors = None, intercept = False,
                 D = None, d=None,
                 G = None, a = None, b = None,
                 window = WINDOWSIZE,  **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        RollingRegression.__init__(self, respond, regressors, intercept,window, **args)
        ICRegression.__init__(self, respond, regressors, intercept, D, d, G,a,b, **args)

    def train(self):
        date = self.respond.rheader
        it = enumerate(zip(headRollingWindows(date, self.window, self.window-1),
                 headRollingWindows(self.respond.data, self.window, self.window-1),
                 headRollingWindows(self.regressors.data, self.window, self.window-1)))
        beta = scipy.empty((self.t,self.n))
        for i,(d,y,X) in it:
            b = icregression(X,y,self.W, self.D, self.d, self.G, self.a, self.b, self.n)
#            print b.T
            beta[i,:] = b.T
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

    def isECConstraintable(self):
        """Boolean function to see if equality contraints can be imposed to the model. Default is True"""
        return True

    def isICConstraintable(self):
        """Boolean function to see if inequality and equality contraints can be imposed to the model. Default is True"""
        return True


def main():
    stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
    stock = StylusReader(stock_data)
    respond = stock[:,0]
    regressors = stock[:,1:]
    t,n = regressors.size()
    weight = scipy.identity(WINDOWSIZE)
    D = scipy.ones((1,7))
    d = scipy.matrix(1.0)
    intercept = False
    a = scipy.zeros((7,1))
    b = scipy.ones((7,1))
    G = scipy.identity(n)

    obj = ICRollingRegression(respond, regressors, intercept, D,d,G,a,b,weight = weight).train()
    obj.getEstimate().toCSV()
#    print obj.predict()
#    print obj.predict(date(1999,1,1))
#    print obj.error()
    print obj.R2()
    obj.est.plot()
    pass


if __name__ == "__main__":
    main()
