'''Example for non-linear least-squares estimation

and comparison with quadratic function linear-in parameter ols
'''
#import matplotlib.pylab as plt
from scipy import optimize, linalg
#from scipy.optimize import leastsq
from numpy import array, zeros, float64, dot, log, exp, inf, sin, cos
import numpy as np
from scipy.optimize.tnc import RCSTRINGS, MSG_NONE
import numpy.random
from math import pow



def yfun(p,x):
    a,b,c,d = p[:4]
    return np.exp(a*(x-d)) + b*x + c

class TestLeastSq(object):
    def __init__(self, yfun):
        x = np.linspace(0, 20, 500)
        a,b,c,d = -0.5, 42, -304.2,15  #3.1, 42, -304.2
        self.x = x
        self.abc = a,b,c
        self.y_true = np.exp(a*(x-d)) + b*x + c  # a*x**2 + b*x + c
        self.y_meas = self.y_true + 100.01*np.random.standard_normal( self.y_true.shape )
        #self.xx = np.c_[x**2,x,np.ones(x.shape[0])]
        self.xx = np.c_[x**2,x,np.ones(x.shape[0])] #perfect multicollinearity

        self.nobs = self.y_meas.shape[0]                     # number of observations
        self.ncoef = 4 #self.xx.shape[1]                    # number of coef.
        self.df_e = self.nobs - self.ncoef              # degrees of freedom, error
        self.df_r = self.ncoef - 1
        #self.yfn = object.__getattribute__(self, 'yfun')
        self.yfn = yfun

    def residuals(self, p, y, x):
#        a,b,c = p
#        err = y-(a*x**2 + b*x + c)
#        print self.xx.shape
#        print p.shape
        #err = y - np.dot(self.xx,p)
        err = y - self.yfn(p, x)
        return err

    def sumsquares(self, p, y, x):
        err = y - self.yfn(p, x)
        return dot(err,err)




    def test_full_output(self):
        p0 = np.zeros(self.ncoef)
        #p0 = np.array([-0.5, 42, -304.2, 15])
        p0 = np.array([-1, 15, -300, 20])
        # requirements for convergence p0[0]<0.5, p0[3]>15
        full_output = optimize.leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert ier in (1,2,3,4), 'solution not found: %s'%mesg

        self.p_est = params_fit

        self.e = self.residuals(self.p_est.ravel(), self.y_meas, self.x)
        self.sse = np.dot(self.e,self.e)/self.df_e         # SSE


        if not cov_x is None:
            self.cov_x = cov_x
            print self.cov_x
            #self.inv_xx = linalg.pinv(dot(self.xx.T,self.xx))
            self.cov_nls = self.sse*self.cov_x
            self.se = np.sqrt(np.diagonal(self.cov_nls))
            self.t = self.p_est / self.se                       # coef. t-statistics
            #self.se = self.sse * np.sqrt(np.diagonal(self.cov_x))
        else:
            print 'no cov_x returned'
        return params_fit, cov_x, infodict, mesg, ier

    def ols(self):
        p_ls,resid,rank,sigma = linalg.lstsq(self.xx, self.y_meas[:,np.newaxis])
        self.p_ols = p_ls.T

        #self.e = self.residuals(tuple(self.p_ols.flat), self.y_meas, self.x)
        #self.e = self.residuals(self.p_ols.ravel(), self.y_meas, self.x)
        self.yhat = np.dot(self.xx,self.p_ols.T)
        self.e = self.y_meas - self.yhat[:,0]   # yhat is 2D
        self.sse = np.dot(self.e.T,self.e)/self.df_e         # SSE
        #self.sse = np.sum(self.e*self.e,0)/self.df_e   
        self.inv_xx = linalg.pinv(dot(self.xx.T,self.xx))
        self.cov_ols = self.sse*self.inv_xx
        self.se = np.sqrt(np.diagonal(self.cov_ols))
        self.t = self.p_ols / self.se                       # coef. t-statistics
        xy = np.dot(self.xx.T,self.y_meas)
        self.b_ols = dot(self.inv_xx,xy)
        return sigma

    def plot_results(self,plot=None):
        if plot is None:
            import matplotlib.pylab as plot
        plot.figure()
        plot.title('Non-Linear Regression Example')
        plot.plot(nls_res.x,nls_res.y_true,'g.--')
        plot.plot(nls_res.x,nls_res.y_meas,'k.')
        plot.plot(nls_res.x,nls_res.y_meas-nls_res.e,'r.-')
        plot.legend(['original','plus noise', 'regression'], loc='lower right')
        #add quadratic ols
        self.ols()   # overwrites results
        plot.plot(nls_res.x,nls_res.yhat,'b.-')
        plot.figure()
        plot.title('Non-Linear Regression Residuals - quadratic approximation')
        plot.plot(nls_res.x,nls_res.e,'bo')

if __name__ == '__main__':
    nls_res = TestLeastSq(yfun)
    nls_res.test_full_output()
    print nls_res.p_est
    print nls_res.se
    print nls_res.t

    #matplotlib ploting
    import matplotlib.pylab as plt
    #nls_res.plot_results()#plt=plt)
    
    #plt.show()
