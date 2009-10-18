import csv, scipy, sys
import numpy, libregression, clibregression
#from libregression import *
data = scipy.matrix(map(lambda x: map(float, x), csv.reader(open("test_dodge_cox.csv", "rb"))), dtype = numpy.float)
print data

y = data[:,0]
X = data[:,1:]
n=7
beta = scipy.empty(scipy.shape(X), dtype = numpy.float)
b = scipy.ones((n,1), dtype = numpy.float)/float(n)
V = scipy.identity(n, dtype = numpy.float)
Phi = scipy.identity(n, dtype = numpy.float)
S = scipy.identity(n, dtype = numpy.float)
sigma = scipy.matrix([1.0], dtype = numpy.float)
Sigma = scipy.identity(n, dtype = numpy.float)
D = scipy.ones((1,n), dtype = numpy.float)
d = scipy.matrix(1., dtype = numpy.float)
G = scipy.identity(n, dtype = numpy.float)
a = scipy.zeros((n,1), dtype = numpy.float)
c = scipy.ones((n,1), dtype = numpy.float)
#        import code; code.interact(local=locals())

beta =  clibregression.kalman_filter(b, V, Phi,  y, X, sigma, Sigma, 1 ,D , d, G, a, c)

#(b, V) = kalman_predict(b,V,Phi, S)

# for i in xrange(len(X)):
#     beta[i] = scipy.array(b).T
#     (b,V, e,K) = kalman_upd(b,V, y[i] ,X[i], sigma, Sigma, 2, D, d,G,a,c)
#     (b, V) = kalman_predict(b,V,Phi, S)


# for i, (xs, ys) in enumerate(zip(X,y)):
#     beta[i,:] = scipy.array(b).T
#     (b,V, e,K) = kalman_upd(b,V, ys ,xs, sigma, Sigma, 2, D, d,G,a,c)
# #    print "b:\n", b
# #    print "V:\n", V
# #    print "e:\n", e
# #    print "K:\n", K
# #    beta[i,:] = scipy.array(b).T
#     (b, V) = kalman_predict(b,V,Phi, S)

print beta

csv.writer(open("test_dodge_cox_weight.csv", "w"), dialect='excel').writerows(beta)
