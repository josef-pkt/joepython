'''
taken from scipy.optimize test suite
adjusted to compare with ols results
conclusion: cov_x if optimize.leastsq is missing scale factor sigma**2 (SSE)
'''


from numpy.testing import assert_array_almost_equal

from scipy import optimize, linalg
#from scipy.optimize import leastsq
from numpy import array, zeros, float64, dot, log, exp, inf, sin, cos
import numpy as np
from scipy.optimize.tnc import RCSTRINGS, MSG_NONE
import numpy.random
from math import pow

from olsexample import ols as clols

class TestLeastSq(object):
    def __init__(self):
        x = np.linspace(0, 10, 40)
        a,b,c = 3.1, 42, -304.2
        self.x = x
        self.abc = a,b,c
        y_true = a*x**2 + b*x + c
        self.y_meas = y_true + 100.01*np.random.standard_normal( y_true.shape )
        #self.xx = np.c_[x**2,x,np.ones(x.shape[0])]
        self.xx = np.c_[x**2,x,2*x,np.ones(x.shape[0])] #perfect multicollinearity

        self.nobs = self.y_meas.shape[0]                     # number of observations
        self.ncoef = self.xx.shape[1]                    # number of coef.
        self.df_e = self.nobs - self.ncoef              # degrees of freedom, error 
        self.df_r = self.ncoef - 1

    def residuals(self, p, y, x):
#        a,b,c = p
#        err = y-(a*x**2 + b*x + c)
#        print self.xx.shape
#        print p.shape
        err = y - np.dot(self.xx,p)
        return err

    def test_basic(self):
        p0 = np.zeros(self.ncoef)
        params_fit, ier = optimize.leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x))
        assert ier in (1,2,3,4), 'solution not found (ier=%d)'%ier
        assert_array_almost_equal( params_fit, self.abc, decimal=2) # low precision due to random

    def test_full_output(self):
        p0 = np.zeros(self.ncoef)
        full_output = optimize.leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert ier in (1,2,3,4), 'solution not found: %s'%mesg

        return params_fit, cov_x, infodict, mesg, ier

    def test_input_untouched(self):
        p0 = np.zeros(self.ncoef,dtype=float64)
        p0_copy = array(p0, copy=True)
        full_output = optimize.leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert ier in (1,2,3,4), 'solution not found: %s'%mesg
        assert_array_equal(p0, p0_copy)

    def ols(self):
        p_ols,resid,rank,sigma = linalg.lstsq(self.xx, self.y_meas[:,np.newaxis])
        self.p_ols = p_ols.T
        
        #self.e = self.residuals(tuple(self.p_ols.flat), self.y_meas, self.x)
        self.e = self.residuals(self.p_ols.ravel(), self.y_meas, self.x)
        self.sse = np.dot(self.e,self.e)/self.df_e         # SSE
        self.inv_xx = linalg.pinv(dot(self.xx.T,self.xx))
        self.cov_ols = self.sse*self.inv_xx
        self.se = np.sqrt(np.diagonal(self.cov_ols))
        xy = np.dot(self.xx.T,self.y_meas)
        self.b_ols = dot(self.inv_xx,xy)
        return sigma

    def olsclass(self):
        x_varnm = ['x1','x2','x3','x4']
        k = self.xx.shape[1]
        m = clols(self.y_meas,self.xx,y_varnm = 'y',x_varnm = x_varnm[:k-1],addconst = False)
        m.summary()
        print 'cov'
        print m.sse*m.inv_xx
                 

        


tls = TestLeastSq()
tls.test_basic
res_full = tls.test_full_output()
tls.olsclass()
tls.ols()
print 'OLS results'
print tls.p_ols
print tls.b_ols
print tls.se
print 'cov'
print tls.cov_ols

print 'optimize_ls results'
print 'beta estimate'
print res_full[0]
print 'cov'
print res_full[1]
print 'cov*sse'
print tls.sse*res_full[1]
#print res_full

print 
