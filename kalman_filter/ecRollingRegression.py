import csv,scipy
from timeSeriesFrame import *
from datetime import date
from exc import *
from copy import copy

try:
    from clibregression import ecregression
except ImportError:
    print "Cannot import clibregression"
    from libregression import ecregression

from windows import *
from rollingRegression import RollingRegression
from ecRegression import ECRegression
DEBUG = 0
WINDOWSIZE = 60

class ECRollingRegression(RollingRegression, ECRegression):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, respond = None, regressors = None, intercept = False, D = None, d=None,window = WINDOWSIZE,  **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        RollingRegression.__init__(self, respond, regressors, intercept,window, **args)
        ECRegression.__init__(self, respond, regressors, intercept, D, d, **args)

    def train(self):
        date = self.respond.rheader
        it = enumerate(zip(headRollingWindows(date, self.window, self.window-1),
                 headRollingWindows(self.respond.data, self.window, self.window-1),
                 headRollingWindows(self.regressors.data, self.window, self.window-1)))
        beta = scipy.empty((self.t,self.n))
        for i,(d,y,X) in it:
            b = ecregression(X,y,self.W, self.D, self.d)
            beta[i,:] = b.T
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self
    
    def isECConstraintable(self):
        """Boolean function to see if equality contraints can be imposed to the model. Default is True"""
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
    obj = ECRollingRegression(respond, regressors, intercept, D,d,weight = weight)
    obj.train()
    print obj.getEstimate().toCSV()
#    print obj.predict()
#    print obj.predict(date(1999,1,1))
#    print obj.error()
    print obj.R2()
    pass


if __name__ == "__main__":
    main()
