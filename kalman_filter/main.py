from timeSeriesFrame import TimeSeriesFrame, StylusReader
from regression import *
from rollingRegression import *
from icRollingRegression import ICRollingRegression
from ecRollingRegression import ECRollingRegression
from ecKalmanFilter import ECKalmanFilter
from kalmanFilter import KalmanFilter
from icKalmanFilter import ICKalmanFilter

def main():
    intercept = False
    stock_data = list(csv.reader(open("sine_wave.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressors = stock[:,1:]
    t, n= regressors.size()
#    obj = Regression(respond, regressors, intercept).train().getEstimate().plot()

    D = scipy.ones((1,n))
    d = scipy.matrix(1.0)
#    obj = ECRegression(respond, regressors, intercept, D,d).train().getEstimate().plot()

    a = scipy.zeros((n,1))
    b = scipy.ones((n,1))
    G = scipy.identity(n)
#    obj = ICRegression(respond, regressors, intercept, D,d,G,a,b).train().getEstimate().plot()


    windowsize = 24
##    obj = RollingRegression(respond,
##                            regressors,
##                            intercept,
##                            weight = scipy.identity(WINDOWSIZE),
##                            window = WINDOWSIZE).train().getEstimate().plot()

##    obj = ECRollingRegression(respond, regressors, intercept, D,d,weight = scipy.identity(WINDOWSIZE),
##                              window = WINDOWSIZE).train().getEstimate().plot()
    
##    obj = ICRollingRegression(respond, regressors, intercept,
##                              D,d,G,a,b,
##                              weight = scipy.identity(WINDOWSIZE),window = WINDOWSIZE).train().getEstimate().plot()


    
    initBeta = scipy.matrix([0.528744, 0.471256]).T
    Sigma = scipy.matrix([[0.123873, -0.12387], [-0.12387,0.123873]])
#    obj = KalmanFilter(respond, regressors, intercept, Sigma, 0.12, initBeta = initBeta).train().getEstimate().plot()
#    obj = ECKalmanFilter(respond, regressors, intercept, Sigma, 0.12, eta = initBeta).train().getEstimate().plot()


#    obj = ICKalmanFilter(respond, regressors, intercept, Sigma, 0.12, initBeta = initBeta).train().getEstimate().plot()
#    obj = ICKalmanFilter(respond, regressors, intercept, Sigma, 0.12, initBeta = initBeta).train()
#    print obj.R2()
if __name__ == "__main__":
    main()
