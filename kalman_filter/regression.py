import csv,scipy
from timeSeriesFrame import *
from datetime import date
from exc import *
from copy import copy
from libregression import regression

DEBUG = 0

class Regression(object):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, respond = None, regressors = None, intercept = False, **args):
        """Input: paras where they are expected to be tuple or dictionary"""
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
        self.X, self.y, self.W = map(scipy.matrix, (self.regressors.data, self.respond.data, self.weight))
        
    def train(self):
        if DEBUG:
            print "X: ", self.X
            print "y: ", self.y
            print "W: ",self.W
        beta = regression(self.X,self.y,self.W)
#        beta =  (self.X.T * self.W * self.X).I*(self.X.T * self.W * self.y) # will optimise it one day.... but this is not too slow
        beta =  scipy.kron(scipy.ones((self.t, 1)),beta.T )
        self.est = TimeSeriesFrame(beta, self.regressors.rheader, self.regressors.cheader)
        return self

    def getEstimate(self, date = None):
        """Return the estimate of the regression"""
        if date is None: return self.est
        else: return self.est[date]

    def isECConstraintable(self):
        """Boolean function to see if equality contraints can be imposed to the model. Default is True"""
        return False

    def isICConstraintable(self):
        """Boolean function to see if inequality and equality contraints can be imposed to the model. Default is True"""
        return False

    def predict(self, time = None):
        """This function take the (list of) date and return prediction in a timeseriesframe"""
#        print self.X * self.est.data[0].T
        pre = TimeSeriesFrame( (self.X * self.est.data[0].T).sum(axis= 1), self.respond.rheader, self.respond.cheader)
        if time is None: return pre
        elif isinstance(time, date): return pre[time]
        else: raise TypeError("time is not in datetime.date format")

    def error(self, time = None):
        newts = copy(self.respond)
        newts.data = newts.data - self.predict().data
        if time: return newts[time]
        else: return newts

    def R2(self):
        """Simple R Squared by the definition on Wikipedia"""
        sser = sum(i**2 for i in (self.respond.data - self.predict().data))
        sstol = sum(i**2 for i in (self.respond.data - sum(self.respond.data)/len(self.respond.data)))
        return  1.0 - sser/sstol

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
