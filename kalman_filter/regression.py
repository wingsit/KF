import csv, scipy, numpy, cvxopt
from cvxopt.solvers import qp
from timeSeriesFrame import *
from datetime import date
from exc import *
from copy import copy
try:
    from clibregression import *
except ImportError:
    print "Cannot import C module"
    from libregression import *

DEBUG = 0


class Regression(object):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, respond = None, regressors = None, intercept = False, **args):
        """
        :param respond: Dependent time series
        :type respond: TimeSeriesFrame<double>
        :param regressors: Independent time serieses
        :type regressors: TimeSeriesFrame<dobule>
        :param intercept: include/exclude intercept
        :type intercept: boolean
        :param args: reserve for future developement
        """
        self.intercept = intercept
        self.respond = respond
        self.regressors = regressors
        self.respond = respond
        self.weight = args.get("weight")
        self.t, self.n = regressors.size()
        if self.intercept:
            self.regressors.data = scipy.hstack((scipy.ones((self.t,1)), self.regressors.data))
            self.regressors.cheader.insert(0,"Intercept")
            self.n = self.n + 1
        if self.weight is None:
            self.weight = scipy.identity(self.t)

        self.X, self.y, self.W = map(scipy.matrix, (self.regressors.data, self.respond.data, self.weight))
       
    def train(self):
        """
        This fucntion will estimate the weight in the regression.
        
        :return: reference to the object itself
        """
        if DEBUG:
            print "X: ", self.X
            print "y: ", self.y
            print "W: ", self.W
        beta = regression(self.X,self.y,self.W)
#        beta =  (self.X.T * self.W * self.X).I*(self.X.T * self.W * self.y) # will optimise it one day.... but this is not too slow
        beta =  scipy.kron(scipy.ones((self.t, 1)),beta.T )
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

    def getEstimate(self, date = None):
        """
        Get the estimate of the regression
        
        :param date: return the weight on a specific date
        :type date: datetime.date
        :return: Weight computed from the regression
        :rtype: scipy.matrix
        """
        if date is None: return self.est
        else: return self.est[date]

    def isECConstraintable(self):
        """
        :return: Boolean function to see if equality contraints can be imposed to the model. Default is True
        :rtype: boolean
        """
        return False

    def isICConstraintable(self):
        """
        :return: Boolean function to see if inequality and equality contraints can be imposed to the model. Default is True
        :rtype: boolean
        """
        return False

    def predict(self, time = None):
        """
        This function take the (list of) date and return prediction in a timeseriesframe
        
        :param time: the specific date of the weight
        :type time: datetime.date
        :return: TimeSeriesFrame of estimate
        :rtype: TimeSeriesFram<double>
        """
        pre = TimeSeriesFrame( scipy.multiply(self.X ,self.est.data).sum(axis = 1), self.respond.rheader, self.respond.cheader)
        if time is None: return pre
        elif isinstance(time, date): return pre[time]
        else: raise TypeError("time is not in datetime.date format")

    def error(self, time = None):
        """
        Compute and return the estimation error

        :param time: The specific date of the weight
        :type time: datetime.date
        :reutrn: TimeSeriesFrame of the estimation error
        :rtype: TimeSeriesFrame<double>
        """
        newts = copy(self.respond)
        newts.data = newts.data - self.predict().data
        if time: return newts[time]
        else: return newts

    def R2(self):
        """
        **FIX ME**
        Simple R Squared by the definition on Wikipedia
        
        :return: R squared statistics
        :rtype: double
        """
        sser = sum(i**2 for i in (self.respond.data - self.predict().data))
        sstol = sum(i**2 for i in (self.respond.data - sum(self.respond.data)/len(self.respond.data)))
        rsq = 1.0 - sser/sstol
#        print sser
#        print sstol
#        assert 0. < rsq and rsq < 1.
        return rsq
    
class ECRegression(Regression):
    def __init__(self, respond = None, regressors = None, intercept = False, D = None, d = None, **args):
        Regression.__init__(self,respond, regressors, intercept, **args)
        if d:
            self.d = d
        else:
            self.d = scipy.matrix(1.0)
        if isinstance(D,numpy.ndarray):
            self.D = scipy.matrix(D)
        else:
            self.D = scipy.matrix(scipy.ones((self.n,1)))
            self.D[0,0] = 0
        pass

    def train(self):
        """
        This fucntion will estimate the weight in the regression.
        
        :return: reference to the object itself
        """
        if DEBUG:
            print "X: ", self.X
            print "W: ", self.W
            print "y: ", self.y
            print "D: ", self.D
            print "d: ", self.d
        beta = ecregression(self.X, self.y, self.W, self.D, self.d)
        beta =  scipy.kron(scipy.ones((self.t, 1)),beta.T )
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

    def isECConstraintable(self): return True

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
        """
        This fucntion will estimate the weight in the regression.
        
        FIXME
        
        :return: reference to the object itself
        """
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
    stock_data = list(csv.reader(open("simulated_portfolio.csv", "rb")))
    stock = StylusReader(stock_data)
    respond = stock[:,0]
    regressors = stock[:,1:]
    t,n = regressors.size()
    weight = scipy.identity(t)
    intercept = True
    print respond.size()
    print regressors.size()
    obj = Regression(respond, regressors, intercept, weight = weight)
    obj.train()
    print obj.predict()
#    print obj.predict(date(1999,1,1))
    print obj.error()
    print obj.R2()
    pass

if __name__ == "__main__":
    main()
