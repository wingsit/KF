from Regression import Regression, ToBeImplemented
import csv,numpy, scipy
from timeSeriesFrame import *
from copy import deepcopy

DEBUG = 0
kappa = 1000
    
class KalmanFilter(Regression):
    """This is a KalmanFilter Class subclassed from Regression"""
    intercept = True
    def __init__(self, respond = None, regressors = None, intercept = False,
                 Sigma, sigma, initBeta = None, initVariance = None, Phi = None,  **args):
        """Input: paras where they are expected to be tuple or dictionary"""
        Regression(self).__init__(respond, regressor, intercept, **args)
        if (not initBeta) and self.intercept:
            self.initBeta = scipy.ones((self.n,1))/float(self.n-1)
            self.initBeta[0] = 0
        elif initBeta and self.intercept:
            self.initBeta = scipy.ones((n,1))/float(n)
        elif (not initBeta) and (not self.intercept):
            self.initBeta = initBeta
        else:
            self.initBeta = scipy.zero((self.n, 1))
            self.initBeta[1:] = initBeta

        if initVariance and self.intercept:
            self.initVariance = scipy.zero((self.n, self.n))
            self.initVariance[1:,1:] = initVariance
        elif initVariance and (not self.intercept):
            self.initVariance = initVariance
        else:
            self.initVariance = scipy.identity(n)*kappa
        if not Phi:
            self.Phi = scipy.identity(self.n)
        else:
            self.Phi = Phi
        self.paras = args.get("paras")
        self.Sigma = Sigma
        self.sigma = sigma

    def train(self):
        """This fucntion will start the estimation. This is separated from addData."""
        
        def _kalmanfilter(beta, V, X, y):
            shape = scipy.shape
            Phi = self.Phi
            pbeta = Phi * beta
            pV = Phi * V * Phi.T + self.Sigma
            e = y - X * pbeta
            K = pV * X.T *( self.sigma + X * pV * X.T).I
            ubeta = pbeta +  K * e
            uV = (scipy.identity(n) - K* X) * pV
            return (pbeta, pV, ubeta, uV)

#         betaList = []
#         betaList.append(self.initBeta)
#         betaTemp = self.initBeta
#         vTemp = self.initVariance
        
        self.pbetaList =[]
        self.ubetaList =[]
        self.ubetaList.append(self.initBeta)
        self.pVList = []
        self.uVList = []
        self.uVList.append(self.initVariance)


        for i in zip(self.regressors.rowIterator(), self.respond.rowIterator()):
            if DEBUG:
                print i[0].data
                print i[1].data
            estimate = _kalmanfilter(scipy.matrix(self.ubetaList[-1]), self.uVList[-1], i[0].data,i[1].data)
            self.pbetaList.append(estimate[0])
            self.pVList.append(estimate[1])
            self.ubetaList.append(estimate[2])
            self.uVList.append(estimate[3])

            

        betaMatrix = scipy.zeros((t,n))
        for index, value in enumerate(self.pbetaList):
            betaMatrix[index, :] = value.T
        cheader = self.regressors.cheader
        if self.intercept:
            cheader.insert(0, "Alpha")
        self.est = TimeSeriesFrame(betaMatrix, self.regressors.rheader, cheader)


    def isConstraintable(self):
        """Boolean function to see if contraints can be imposed to the model. Default is True"""
        return False
    
def main():
    try:
        intercept = False
        if intercept:
            obj = KalmanFilter( scipy.identity(3)*kappa, 0.001)
        else:
            obj = KalmanFilter( scipy.identity(2)*kappa, 0.001)
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
        print obj.predict(date(2001,1,1))
        obj.est.toCSV("default2.csv")
        print obj.R2()
    except:
        from print_exc_plus import print_exc_plus
        print_exc_plus()

if __name__ == "__main__":
    main()

