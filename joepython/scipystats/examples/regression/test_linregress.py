
import numpy as np
from scipy import stats
from numpy.testing import assert_array_almost_equal

def test_linregress():
    '''compared with multivariate ols with pinv'''
    x = np.arange(11)
    y = np.arange(5,16)
    y[[(1),(-2)]] -= 1
    y[[(0),(-1)]] += 1

    res = (1.0, 5.0, 0.98229948625750, 7.45259691e-008, 0.063564172616372733)
    assert_array_almost_equal(stats.linregress(x,y),res,decimal=14)

from olsexample import ols
resultslinreg = []
resultsols = []
result2 = []
noise_scale = 0.5#1e-6#0.01
for i in range(1000):
    x = np.random.randn(20,2)
    y = 1.0 + (x*[-1.0, noise_scale]).sum(axis=1)
    res = ols(y,x[:,0])
    result2.append([np.corrcoef((res.yest,y))[0,1]])
    resultsols.append([res.b[1], res.b[0], np.sqrt(res.R2), res.p[1], res.se[1], np.corrcoef((x[:,0],y))[0,1]])
    temp = list(stats.linregress(x[:,0],y))+[np.corrcoef((x[:,0],y))[0,1]]
    temp[2] = np.abs(temp[2])  # convert signed to positive R
    resultslinreg.append(temp)

reslr_arr = np.array(resultslinreg)
reso_arr = np.array(resultsols)
res2_arr = np.array(result2)
print np.max(np.abs(reslr_arr - reso_arr),0)
print np.mean(np.abs(reslr_arr - reso_arr),0)
print reslr_arr.mean(axis=0)
print reso_arr.mean(axis=0)
print res2_arr.mean(axis=0)

'''
Notes
-----
coefficient of determination R is signed in linregress
difference between ols and linregress in degenerate boundary cases
when noise is zero, some cases:
* perfect correlation: y is affine function of x
* y is constant, var(y) = 0
* x is constant, degenerate perfect collinearity with constant
'''
