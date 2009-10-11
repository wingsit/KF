import csv,numpy, scipy, cvxopt
from cvxopt.solvers import qp
from timeSeriesFrame import *
from datetime import date
from ecRegression import ECRegression
from exc import *
from libregression import icregression
DEBUG = 0

class ICRegression(ECRegression):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, respond = None, regressors = None, intercept = False, D = None, d = None, G = None, a = None, b = None, **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        ECRegression.__init__(self,respond, regressors, intercept, D, d, **args)

        if self.intercept and G != None:
            self.G = scipy.zeros((self.n, self.n))
            self.G[1:, 1:] = G
        elif self.intercept and G == None :
            self.G = scipy.identity(self.n)
            self.G[0, 0] = 0.0
        elif not self.intercept and G != None:
            self.G = G
        else:
            self.G = scipy.identity(self.n)
            
        if self.intercept:
            self.a = scipy.zeros((self.n, 1))
            self.a[1:] = a            
            self.b = scipy.zeros((self.n, 1))
            self.b[1:] = b
        else:
            if a is None:
                self.a = scipy.matrix( scipy.zeros((self.n,1)))
            else: self.a = a
            if b is None:
                self.b = scipy.matrix( scipy.ones((self.n,1)))
            else: self.b = b

    def train(self):
        if DEBUG:
            print "X: ", self.X
            print "W: ", self.W
            print "y: ", self.y
            print "D: ", self.D
            print "d: ", self.d
            print "G: ", self.G
            print "a: ", self.a
            print "b: ", self.b
        beta = icregression(self.X, self.y, self.W, self.D, self.d, self.G,
                            self.a, self.d, self.n)
#         P = 2*self.X.T * self.W * self.X
#         q = -2*self.X.T * self.W * self.y
#         bigG = scipy.empty((2*self.n, self.n))
#         h = scipy.empty((2*self.n, 1))
#         bigG[:self.n, :] = -self.G
#         bigG[self.n:, :] = self.G
#         h[:self.n, :] = -self.a
#         h[self.n:, :] = self.b

#         if DEBUG:
#             print "P: ",P
#             print "q: ",q
#             print "bigG: ",bigG
#             print "h: ",h
#             print "D: ", self.D
#             print "d: ", self.d
#         paraset = map(cvxopt.matrix , (P,q,bigG,h,self.D,self.d))
#         beta = qp(*paraset)['x']
        beta =  scipy.kron(scipy.ones((self.t, 1)),beta.T )
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

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
    a = scipy.zeros((7,1))
    b = scipy.ones((7,1))
    G = scipy.identity(n)
    obj = ICRegression(respond, regressors, intercept, D,d,G,a,b,weight = weight).train()
    print obj.getEstimate()
    print obj.predict()
#    print obj.predict(date(1999,1,1))
    print obj.error()
    print obj.R2()
    obj.getEstimate().toCSV()
    pass

if __name__ == "__main__":
    main()
