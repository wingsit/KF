import csv, scipy, sys
from libregression import *
data = scipy.matrix(map(lambda x: map(float, x), csv.reader(open("test_dodge_cox.csv", "rb"))))
print data

y = data[:,0]
X = data[:,1:]
n=7
beta = scipy.empty(scipy.shape(X))
b = scipy.ones((n,1))/float(n)
V = scipy.identity(n)
Phi = scipy.identity(n)
S = scipy.identity(n)
sigma = scipy.matrix([1.0])
Sigma = scipy.identity(n)
D = scipy.ones((1,n))
d = scipy.matrix(1.)
G = scipy.identity(n)
a = scipy.zeros((n,1))
c = scipy.ones((n,1))
#        import code; code.interact(local=locals())

beta =  kalman_filter(b, V, Phi,  y, X, sigma, Sigma, 1 ,D , d, G, a, c)

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
