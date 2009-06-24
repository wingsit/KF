from cvxopt import *
from timeSeriesFrame import *
import csv,numpy
from cvxopt.solvers import qp
from datetime import date
DEBUG = 0

class ToBeImplemented(Exception):
    def __init__(self):
        print "Why dont you implement it?"

class Regression(object):
    """ This is an abstruct class for Regression Type of problem."""
    def __init__(self, weight = None, **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        self.paras = args.get("paras")
        pass

    def addData(self, respond, regressors):
        """This function will add data to the object"""
        self.respond = respond
        self.regressors = regressors        
        pass

    def setConstraints(self, a, b):
        """setConstraints for the constrained regression problem. The constrains are ignored when the regression
is not contrain-able, a <= \beta <= b"""
        self.h = matrix([matrix(0.0),-matrix(a),matrix(0.0),matrix(b)]) # set constraints
        pass

    def train(self):
        """This fucntion will start the estimation. This is separated from addData."""
        IndexSize = self.regressors.size()
        X = matrix([[matrix(1.0, (IndexSize[0],1))], [matrix(self.regressors.data) ]])
        Y = matrix(self.respond.data)
        P = 2 * X.T *X
        q = -2 * X.T * Y
        I = spmatrix(1.0, range(IndexSize[1]+1), range(IndexSize[1]+1))
        I[0,0] = 0
        G = matrix([-I,I])
        A = matrix(1.0, (1,IndexSize[1]+1))        
        A[0] = 0
        b = matrix(1.0)
        if DEBUG:
            solvers.options['show_progress'] = True
            print self.qp['x']
            print "P:", P.size
            print "q:", q.size
            print "G:", G.size
            print "h:", self.h.size
            print "A:", A.size
            print "b:", b.size
        else:
            solvers.options['show_progress'] = False
        self.qp = qp(P, q, G, self.h, A, b)
        est = qp(P, q, G, self.h, A, b)['x']
        est2 = []
        for i in xrange(self.regressors.size()[0]):
            est2.append(list(est))
        self.est = TimeSeriesFrame(matrix(est2).T, self.regressors.rheader, self.regressors.cheader)
        pass

    def getEstimate(self, date = None):
        """Return the estimate of the regression"""
        if date == None:
            return self.est
        else: raise ToBeImplemented

    def isConstraintable(self):
        """Boolean function to see if contraints can be imposed to the model. Default is True"""
        return True

    def predict(self, date):
        """This function take the (list of) date and return prediction in a timeseriesframe"""
        pass
    
    def withIntercept(self, setting = True):
        self.intercept = setting
        pass
    
def main():
    obj = Regression(data = 10, paras = (10, 100))
    stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
    #stock = TSF()
    stock = StylusReader(stock_data)
    respond = stock[:,0]
    regressor = stock[:,1:]
    zeros = numpy.zeros((7,1))
    ones = numpy.ones((7,1))

#    print respond, regressor
    obj.setConstraints(zeros,ones)
    obj.addData(respond,regressor)
    obj.train()
    print obj.getEstimate(None)
    try:
        print obj.getEstimate(Date(2001,1,1))
    except:
        pass
    print stock


    
if __name__ == "__main__":
    main()
