import csv,numpy, scipy
from cvxopt import *
from cvxopt.solvers import qp
from TimeSeriesFrame import *
from datetime import date
from regression import Regression
from exc import *
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
        beta =  self.D * (self.X.T * self.W * self.X).I * self.D.T *(self.D* self.X.T * self.W * self.y-self.d) # will optimise it one day.... but this is not too slow
        beta =  scipy.kron(scipy.ones((self.t, 1)),beta.T )
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)


    def isECConstraintable(self): return True


def main():
    stock_data = list(csv.reader(open("simulated_portfolio.csv", "rb")))
    stock = StylusReader(stock_data)
    respond = stock[:,0]
    regressors = stock[:,1:]
    t,n = regressors.size()
    weight = scipy.identity(t)
    D = scipy.ones((3,1))
    d = scipy.matrix(1.0)
    intercept = True
    obj = ECRegression(respond, regressors, intercept, D,d,weight = weight)
    obj.train()
    print obj.predict()
#    print obj.predict(date(1999,1,1))
    print obj.error()
    print obj.R2()
    pass

if __name__ == "__main__":
    main()
