from cvxopt import *
from cvxopt.solvers import qp

from timeSeriesFrame import *
import csv,numpy
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
        if self.intercept:
            self.h = matrix([matrix(0.0),-matrix(a),matrix(0.0),matrix(b)]) # set constraints
        else:
            self.h = matrix([-matrix(a),matrix(b)]) # set constraints

    def train(self):
        """This fucntion will start the estimation. This is separated from addData."""
        IndexSize = self.regressors.size()
        cheader = self.regressors.cheader
        if self.intercept:
            X = matrix([[matrix(1.0, (IndexSize[0],1))], [matrix(self.regressors.data) ]])
            I = spmatrix(1.0, range(IndexSize[1]+1), range(IndexSize[1]+1))
            I[0,0] = 0
            A = matrix(1.0, (1,IndexSize[1]+1))        
            A[0] = 0
            cheader.insert(0,'Alpha')
        else:
            X = matrix(self.regressors.data)
            I = spmatrix(1.0, range(IndexSize[1]), range(IndexSize[1]))
            A = matrix(1.0, (1,IndexSize[1]))        
        Y = matrix(self.respond.data) 
        P = 2 * X.T *X
        q = -2 * X.T * Y
        G = matrix([-I,I])
        b = matrix(1.0)
        if DEBUG:
            solvers.options['show_progress'] = True
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
        self.est = TimeSeriesFrame(matrix(est2).T, self.regressors.rheader, cheader)
        pass


    def getEstimate(self, date = None):
        """Return the estimate of the regression"""
        if date == None: return self.est
        else: return self.est[date]

    def isConstraintable(self):
        """Boolean function to see if contraints can be imposed to the model. Default is True"""
        return True

    def predict(self, date = None):
        """This function take the (list of) date and return prediction in a timeseriesframe"""
        if date == None:
            if not self.intercept:
                data =  scipy.array(self.getEstimate(date).data)*scipy.array(self.regressors.data)
                data = scipy.matrix(data.sum(axis = 1)).T
                return TimeSeriesFrame(data, self.respond.rheader, self.respond.cheader)
            else:
                size = list(self.regressors.size())
                size = size[0]+1, 1
                size = tuple(size)
                ones = numpy.array(scipy.ones(size))
                data = numpy.array(self.regressors.data)
                data = scipy.matrix((numpy.insert(data, [0], [1], axis = 1) * numpy.array(self.getEstimate().data)).sum(axis = 1)).T
                return TimeSeriesFrame(data, self.respond.rheader, self.respond.cheader)
        else:
            if self.intercept:
                data = numpy.array(self.regressors[date].data)
                data = numpy.insert(data, 0,1)
                beta = numpy.array(self.getEstimate(date).data)
                return TimeSeriesFrame(scipy.matrix((beta*data).sum(axis = 1)), date, self.respond.cheader)
            else:
                data = numpy.array(self.regressors[date].data)
                beta = numpy.array(self.getEstimate(date).data)
                return TimeSeriesFrame(scipy.matrix((beta*data).sum(axis = 1)), date, self.respond.cheader)
    
    def withIntercept(self, setting = True):
        self.intercept = setting
        pass

    def R2(self):
        """Simple R Squared by the definition on Wikipedia"""
        sser = sum(i**2 for i in (self.respond.data - self.predict().data))
        sstol = sum(i**2 for i in (self.respond.data - sum(self.respond.data)/len(self.respond.data)))
#        print sser,sstol, 1.0 - sser/sstol
        return  1.0 - sser/sstol








def main():
    obj = Regression(data = 10, paras = (10, 100))
    stock_data = list(csv.reader(open("simulated_portfolio.csv", "rb")))
    stock = StylusReader(stock_data)
    respond = stock[:,0]
    regressor = stock[:,1:]
    zeros = numpy.zeros((2,1))
    ones = numpy.ones((2,1))

    obj.withIntercept(False)
    obj.setConstraints(zeros,ones)
    obj.addData(respond,regressor)
    obj.train()
#    print obj.getEstimate(None)
#    print obj.getEstimate(date(2001,1,1))
    obj.est.toCSV("default2.csv")
#    print obj.predict(date(2001,1,1))
#    obj.getEstimate().toCSV()
    obj.R2()






if __name__ == "__main__":
    main()
