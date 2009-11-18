import csv,scipy
from timeSeriesFrame import *
from datetime import date
from exc import *
from copy import copy
try:
    from clibregression import *
except ImportError:
    print "Cannot import clibregression"
    from libregression import *
from windows import *
from regression import *

DEBUG = 0
WINDOWSIZE = 24

class RollingRegression(Regression):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, respond = None, regressors = None, intercept = False, window = WINDOWSIZE,  **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        Regression.__init__(self, respond, regressors, intercept, **args)
        self.window = window
        
    def train(self):
        date = self.respond.rheader
        it = enumerate(zip(headRollingWindows(date, self.window, self.window-1),
                 headRollingWindows(self.respond.data, self.window, self.window-1),
                 headRollingWindows(self.regressors.data, self.window, self.window-1)))
        beta = scipy.empty((self.t,self.n))
        for i,(d,y,X) in it:
            b = regression(X,y,self.W)
            beta[i,:] = b.T
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

								
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
    intercept = True
    obj = RollingRegression(respond, regressors, intercept, weight = weight, window = WINDOWSIZE).train()

    obj.getEstimate().plot()
    print obj.predict()
#    print obj.predict(date(1999,1,1))
    print obj.error()
    print obj.R2()
    pass

if __name__ == "__main__":
    main()
