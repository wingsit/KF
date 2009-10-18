import csv, scipy, cvxopt
from numpy import multiply as mlt
from numpy import mat
#from libregression import *
data = scipy.matrix(map(lambda x: map(float, x), csv.reader(open("test_dodge_cox.csv", "rb"))))

y = data[:, 0]
X = data[:, 1:]
(t,n) = scipy.shape(X)
W1 = 1.
W2 = scipy.identity(n)
P = scipy.empty((t*n, t*n))
lamb = 1.
Phi = scipy.identity(n)
D = scipy.ones((1,n))
d = scipy.matrix(1.)
smallG = scipy.identity(n)
a = scipy.zeros((n,1))
c = scipy.ones((n,1))

G = scipy.empty( (2*n, n))
G[0:n, 0:n] = smallG
G[n: 2*n, 0:n] = -smallG
g = scipy.empty( (2*n, 1))
g[0:n] = c
g[n:2*n] = -a
####P#####
for i in xrange(t):
    if i == 0:
        p = 2* W1* mlt(X[i].T, X[i]) + lamb * Phi.T * W2 * Phi
    elif i == t-1:
        p = 2* W1* mlt(X[i].T, X[i])
    else:
        p = 2* W1* mlt(X[i].T, X[i]) + lamb * (Phi.T * W2 * Phi + W2)
    if i < t-1:
        P[(i)*n:(i+1)*n, (i+1)*n:(i+2)*n] = -2 * lamb * Phi.T  *W2
        P[(i+1)*n:(i+2)*n, (i)*n:(i+1)*n] = -2 * lamb * W2 * Phi
    P[i*n:i*n+n, i*n:i*n+n] = p.copy()

##q##
q = scipy.empty((t*n, 1))
for i in xrange(t):
    q[i*n:(i+1)*n] = -2 * X[i].T * W1 * y[i]
#q = (-2 * W1 * y)
##bigG##
gr, gc = scipy.shape(G)
bigG = scipy.empty((gr*t, gc*t))

for i in xrange(t):
    bigG[i*gr:(i+1)*gr, i*gc:(i+1)*gc] = G

bigg = scipy.empty((gr*t,1))
for i in xrange(t):
    bigg[i*gr:(i+1)*gr] = g

dr, dc = scipy.shape(D)
A = scipy.empty((t* dr,t* dc))
for i in xrange(t):
    A[i*dr: (i+1) * dr, i*dc:(i+1)*dc] = D
b = scipy.empty((t, 1))
for i in xrange(t):
    b[i:(i+1)] = d



for i in (P, q, bigG, g, A, b):
    print scipy.shape(i)

paraset = map(cvxopt.matrix, (P, q, bigG, bigg, A, b))
beta =  mat(cvxopt.qp(*paraset)['x']).reshape(t,n).tolist()

csv.writer(open("output_fls.csv", "w")).writerows(beta)
