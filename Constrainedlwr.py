import numpy, scipy, csv
import utility
from Regression import Regression
from TimeSeriesFrame import *
from cvxopt import matrix, solvers

class ConstrainedLWR(Regression):
    def __init__(self, weight = None, window = None, overlap=None):
        Regression.__init__(self, weight)
        if window <= overlap and not window == None and not overlap == None:
            raise Exception
        else:
            self.window = window
            self.overlap = overlap
        pass

    def train(self):
        pass        

    def isConstraintable(self):
        """Boolean function to see if contraints can be imposed to the model. Default is True"""
        return True

def main():
    obj = ConstrainedLWR()
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
##    print obj.getEstimate(None)
##    print obj.getEstimate(date(2001,1,1))
##    obj.est.toCSV("default2.csv")
##    print obj.predict(date(2001,1,1))
##    obj.getEstimate().toCSV()
##    obj.R2()



if __name__ == "__main__":
    main()
