from timeSeriesFrame import TimeSeriesFrame, StylusReader
from regression import *
from rollingRegression import *
from icRollingRegression import ICRollingRegression
from ecRollingRegression import ECRollingRegression
from ecKalmanFilter import ECKalmanFilter
from kalmanFilter import KalmanFilter
from kalmanSmoother import KalmanSmoother
from icKalmanFilter import ICKalmanFilter
from icFlexibleLeastSquare import ICFlexibleLeastSquare

def main():
    intercept = False
    stock_data = list(csv.reader(open("sine_wave.csv", "rb")))
    stock = StylusReader(stock_data)
    del stock_data
    respond = stock[:,0]
    regressors = stock[:,1:]
    t, n= regressors.size()
    reg = Regression(respond, regressors, intercept).train()
<<<<<<< HEAD:kalman_filter/main.py
=======


>>>>>>> c019e4eb5965fc1206a78b047cc87cd6c7ac93c5:kalman_filter/main.py

    D = scipy.ones((1,n))
    d = scipy.matrix(1.0)
    ecreg = ECRegression(respond, regressors, intercept, D,d).train()


    a = scipy.zeros((n,1))
    b = scipy.ones((n,1))
    G = scipy.identity(n)
    icreg = ICRegression(respond, regressors, intercept, D,d,G,a,b).train()



    windowsize = 24
    rreg = RollingRegression(respond,
                            regressors,
                            intercept,
                            weight = scipy.identity(WINDOWSIZE),
                            window = WINDOWSIZE).train()


    ecrreg = ECRollingRegression(respond, regressors, intercept, D,d,weight = scipy.identity(WINDOWSIZE), window = WINDOWSIZE).train()

    
    icrreg = ICRollingRegression(respond, regressors, intercept,D,d,G,a,b, weight = scipy.identity(WINDOWSIZE),window = WINDOWSIZE).train()



    
    initBeta = scipy.matrix([0.528744, 0.471256]).T
    Sigma = scipy.matrix([[0.123873, -0.12387], [-0.12387,0.123873]])
    kalman = KalmanFilter(respond, regressors, intercept, Sigma, 0.12, initBeta = initBeta).train()
<<<<<<< HEAD:kalman_filter/main.py
    kalmans = KalmanSmoother(respond, regressors, intercept, Sigma, 0.12, initBeta = initBeta).train()
=======
>>>>>>> c019e4eb5965fc1206a78b047cc87cd6c7ac93c5:kalman_filter/main.py

    
    eckalman = ECKalmanFilter(respond, regressors, intercept, Sigma, 0.12, eta = initBeta).train()


    ickalman = ICKalmanFilter(respond, regressors, intercept, Sigma, 0.12, initBeta = initBeta).train()
    icfls = ICFlexibleLeastSquare(respond, regressors, intercept, 1.).train()
<<<<<<< HEAD:kalman_filter/main.py

    all_model = [reg, ecreg, icreg, rreg, ecrreg, icrreg, kalman, kalmans, eckalman, ickalman, icfls]
=======
>>>>>>> c019e4eb5965fc1206a78b047cc87cd6c7ac93c5:kalman_filter/main.py

    print reg.R2()
    print ecreg.R2()
    print icreg.R2()
    print rreg.R2()
    print ecrreg.R2()
    print icrreg.R2()
    print kalman.R2()
    print eckalman.R2()
    print ickalman.R2()
    print icfls.R2()

<<<<<<< HEAD:kalman_filter/main.py
    for i in all_model:
        print i
        print "\t", i.R2()
#        i.getEstimate().plot()
#    raw_input()
=======
>>>>>>> c019e4eb5965fc1206a78b047cc87cd6c7ac93c5:kalman_filter/main.py
if __name__ == "__main__":
    main()
