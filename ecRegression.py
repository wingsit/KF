import csv,numpy, scipy
from cvxopt import *
from cvxopt.solvers import qp
from timeSeriesFrame import *
from datetime import date
from regression import Regression
from exc import *
from libregression import ecregression
DEBUG = 0


class ECRegression(Regression):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, respond = None, regressors = None, intercept = False, D = None, d = None, **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        Regression.__init__(self,respond, regressors, intercept, **args)
        if d:
            self.d = d
        else:
            self.d = scipy.matrix(1.0)
        if isinstance(D,numpy.ndarray):
            self.D = D
        else:
            self.D = scipy.ones((n,1))            
            self.D[0,0] = 0
        pass

    def train(self):
        if DEBUG:
            print "X: ", self.X
            print "W: ", self.W
            print "y: ", self.y
            print "D: ", self.D
            print "d: ", self.d
##        covinv = (self.X.T * self.W * self.X).I
##        lamb = (self.D * covinv * self.D.T).I * (self.D * covinv * self.X.T * self.W * self.y - self.d)
##        beta = covinv * (self.X.T * self.W * self.y - self.D.T * lamb)
        beta = ecregression(self.X, self.y, self.W, self.D, self.d)
#        print beta
        beta =  scipy.kron(scipy.ones((self.t, 1)),beta.T )
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)


    def isECConstraintable(self): return True


def main():
    stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
    stock = StylusReader(stock_data)
    respond = stock[:,0]
    regressors = stock[:,1:]
    t,n = regressors.size()
    weight = scipy.identity(t)
    D = scipy.ones((1,7))
    d = scipy.matrix(1.0)
    intercept = False
    obj = ECRegression(respond, regressors, intercept, D,d,weight = weight)
    obj.train()
    print obj.getEstimate()
    print obj.predict()
#    print obj.predict(date(1999,1,1))
    print obj.error()
    print obj.R2()
    obj.getEstimate().toCSV()
    pass

if __name__ == "__main__":
    main()
