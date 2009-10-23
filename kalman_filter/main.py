from regression import *
from timeSeriesFrame import TimeSeriesFrame, StylusReader
from ecKalmanFilter import ECKalmanFilter
from kalmanFilter import KalmanFilter
from icKalmanFilter import ICKalmanFilter
from rollingRegression import *

def main():
    intercept = False
    stock_data = list(csv.reader(open("sine_wave.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressors = stock[:,1:]
    t, n= regressors.size()
    obj = Regression(respond, regressors, intercept).train().getEstimate().plot()

    D = scipy.ones((1,n))
    d = scipy.matrix(1.0)
    obj = ECRegression(respond, regressors, intercept, D,d).train().getEstimate().plot()

    a = scipy.zeros((n,1))
    b = scipy.ones((n,1))
    G = scipy.identity(n)
    obj = ICRegression(respond, regressors, intercept, D,d,G,a,b).train().getEstimate().plot()

    windowsize = 24
    obj = RollingRegression(respond, regressors, intercept, weight = weight, window = WINDOWSIZE).train().getEstimate().plot()




    
    initBeta = scipy.matrix([0.528744, 0.471256]).T
    Sigma = scipy.matrix([[0.123873, -0.12387], [-0.12387,0.123873]])
    obj = ICKalmanFilter(respond, regressors, intercept, Sigma, 0.12, initBeta = initBeta).train().getEstimate().plot()

if __name__ == "__main__":
    main()
