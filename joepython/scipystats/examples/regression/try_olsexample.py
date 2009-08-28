
import numpy as np
from olsexample import ols

def generate_data(nobs):
    x = np.random.randn(nobs,2)
    btrue = np.array([[5,1,2]]).T
    y = np.dot(x,btrue[1:,:]) + btrue[0,:] + 0.5 * np.random.randn(nobs,1)
    return y,x

y,x = generate_data(15)

est = ols(y,x)   # initialize and estimate with ols, constant added by default
print 'ols estimate'
print est.b
print np.array([[5,1,2]])

ynew,xnew = generate_data(3)
ypred = est.predict(xnew)

print '    ytrue        ypred        error' 
print np.c_[ynew, ypred, ynew - ypred]



# or direct way
'''
>>> from scipy import linalg
>>> b,resid,rank,sigma = linalg.lstsq(np.c_[np.ones((x.shape[0],1)),x],y)
>>> b
array([[ 5.47073574],
       [ 0.6575267 ],
       [ 2.09241884]])
>>> xnewwc=np.c_[np.ones((xnew.shape[0],1)),xnew]
>>> ypred = np.dot(xnewwc,b)   # prediction with ols estimate of parameters b
>>> print np.c_[ynew, ypred, ynew - ypred]
[[ 8.23128832  8.69250962 -0.46122129]
 [ 9.14636291  9.66243911 -0.51607621]
 [-0.10198498 -0.27382934  0.17184436]]
 '''
