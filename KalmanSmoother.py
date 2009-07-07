from KalmanFilter import *
from Regression import Regression, ToBeImplemented
import csv,numpy, scipy
from timeSeriesFrame import *


DEBUG = 0
kappa = 1000

class UnconstrainableError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return  repr(self.value)


class KalmanSmoother(Regression):    
    def __init__(self, Sigma, sigma, initBeta = None, initVariance = None, Phi = None,  **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        self.initBeta = initBeta
        self.initVariance = initVariance
        self.Phi = Phi
        self.paras = args.get("paras")
        self.Sigma = Sigma
        self.sigma = sigma
        pass

    def setConstraints(self, a, b):
        raise UnconstrainableError("This model does not accept constraint")

    def train(self):
        kf = KalmanFilter(self.Sigma, self.sigma, self.initBeta, self.initVariance, self.Phi )
        kf.addData(self.respond, self.regressors)
        kf.intercept = self.intercept
        kf.train()
        sbetaList = [kf.ubetaList[-1]]
        sVList = [kf.uVList[-1]]
        T = len(kf.pbetaList)
        for t in reversed(xrange(0, T-1)):
            K = kf.uVList[t] * kf.Phi * scipy.matrix(kf.pVList[t-1]).I
            sV = kf.uVList[t] + K* ( sVList[0]-kf.pVList[t-1]) * K.T
            sVList.insert(0, sV)
            sbeta = kf.ubetaList[t] + K * ( sbetaList[0] - kf.Phi* kf.pbetaList[t-1])
            sbetaList.insert(0,sbeta)
        print len(sbetaList)
        t,n = self.regressors.size()
        if self.intercept:
            n = n+1
        betaMatrix = scipy.zeros((t,n))
        for index, value in enumerate(sbetaList):
            betaMatrix[index, :] = value.T
        cheader = self.regressors.cheader
        if self.intercept:
            cheader.insert(0, "Alpha")
        self.est = TimeSeriesFrame(betaMatrix, self.regressors.rheader, cheader)

        
    
def main():
    try:
        intercept = True
        if intercept:
            obj = KalmanSmoother( scipy.identity(3)*kappa, 0.001)
        else:
            obj = KalmanSmoother( scipy.identity(2)*kappa, 0.001)
        stock_data = list(csv.reader(open("simulated_portfolio.csv", "rb")))
        stock = StylusReader(stock_data)
        del stock_data
        respond = stock[:,0]
        regressor = stock[:,1:]
        obj.addData(respond,regressor)
        obj.intercept = intercept
        obj.train()
        print obj.getEstimate()
        print obj.getEstimate(date(2001,1,1))
        print obj.predict()
    #        print obj.predict(date(2001,1,1))
        obj.est.toCSV()
        print obj.R2()
    except:
        from print_exc_plus import print_exc_plus
        print_exc_plus()

if __name__ == "__main__":
    main()


    
