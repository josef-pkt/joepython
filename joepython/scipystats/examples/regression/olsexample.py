'''
From: scipy cookbook
'''

from __future__ import division
from scipy import linalg, stats
#from scipy import c_, ones, dot, stats, diff
#from scipy.linalg import pinv, solve, det
#from numpy import log, pi, sqrt, square, diagonal
import numpy.random as mtrand #import randn, seed
from numpy import dot
import numpy as np
import time

class ols:
    """
    Author: Vincent Nijs (+ ?)

    Email: v-nijs at kellogg.northwestern.edu

    Last Modified: Mon Jan 15 17:56:17 CST 2007

    Dependencies: See import statement at the top of this file

    Doc: Class for multi-variate regression using OLS

    For usage examples of other class methods see the class tests at the bottom of this file. To see the class in action
    simply run this file using 'python ols.py'. This will generate some simulated data and run various analyses. If you have rpy installed
    the same model will also be estimated by R for confirmation.

    Input:
        y = dependent variable
        y_varnm = string with the variable label for y
        x = independent variables, note that a constant is added by default
        x_varnm = string or list of variable labels for the independent variables

    Output:
        There are no values returned by the class. Summary provides printed output.
        All other measures can be accessed as follows:

        Step 1: Create an OLS instance by passing data to the class

            m = ols(y,x,y_varnm = 'y',x_varnm = ['x1','x2','x3','x4'])

        Step 2: Get specific metrics

            To print the coefficients:
                >>> print m.b
            To print the coefficients p-values:
                >>> print m.p

    """

    def __init__(self,y,x,y_varnm = 'y',x_varnm = '',addconst = True):
        """
        Initializing the ols class.
        """
        self.y = y
        if addconst:
            self.x = np.c_[np.ones(x.shape[0]),x]
        else:
            self.x = x
        self.y_varnm = y_varnm
        if not isinstance(x_varnm,list):
            self.x_varnm = ['const'] + list(x_varnm)
        else:
            self.x_varnm = ['const'] + x_varnm

        # Estimate model using OLS
        self.estimate()

    def estimate(self):

        # estimating coefficients, and basic stats
        self.inv_xx = linalg.pinv(dot(self.x.T,self.x)) # use Moore-Penrose pseudoinverse
        xy = dot(self.x.T,self.y)
        self.b = dot(self.inv_xx,xy)                    # estimate coefficients

        self.nobs = self.y.shape[0]                     # number of observations
        self.ncoef = self.x.shape[1]                    # number of coef.
        self.df_e = self.nobs - self.ncoef              # degrees of freedom, error
        self.df_r = self.ncoef - 1                      # degrees of freedom, regression

        self.e = self.y - dot(self.x,self.b)            # residuals
        self.sse = dot(self.e,self.e)/self.df_e         # SSE
        self.se = np.sqrt(np.diagonal(self.sse*self.inv_xx))  # coef. standard errors
        self.t = self.b / self.se                       # coef. t-statistics
        self.p = (1-stats.t.cdf(np.abs(self.t), self.df_e)) * 2    # coef. p-values

        self.R2 = 1 - self.e.var()/self.y.var()         # model R-squared
        self.R2adj = 1-(1-self.R2)*((self.nobs-1)/(self.nobs-self.ncoef))   # adjusted R-square

        self.F = (self.R2/self.df_r) / ((1-self.R2)/self.df_e)  # model F-statistic
        self.Fpv = 1-stats.f.cdf(self.F, self.df_r, self.df_e)  # F-statistic p-value

    def dw(self):
        """
        Calculates the Durbin-Waston statistic
        """
        de = np.diff(self.e,1)
        dw = dot(de,de) / dot(self.e,self.e);

        return dw

    def omni(self):
        """
        Omnibus test for normality
        """
        return stats.normaltest(self.e)

    def JB(self):
        """
        Calculate residual skewness, kurtosis, and do the JB test for normality
        """

        # Calculate residual skewness and kurtosis
        skew = stats.skew(self.e)
        kurtosis = 3 + stats.kurtosis(self.e)

        # Calculate the Jarque-Bera test for normality
        JB = (self.nobs/6) * (np.square(skew) + (1/4)*np.square(kurtosis-3))
        JBpv = 1-stats.chi2.cdf(JB,2);

        return JB, JBpv, skew, kurtosis

    def ll(self):
        """
        Calculate model log-likelihood and two information criteria
        """

        # Model log-likelihood, AIC, and BIC criterion values
        ll = -(self.nobs*1/2)*(1+np.log(2*np.pi)) - (self.nobs/2)*np.log(dot(self.e,self.e)/self.nobs)
        aic = -2*ll/self.nobs + (2*self.ncoef/self.nobs)
        bic = -2*ll/self.nobs + (self.ncoef*np.log(self.nobs))/self.nobs

        return ll, aic, bic

    def summary(self):
        """
        Printing model output to screen
        """

        # local time & date
        t = time.localtime()

        # extra stats
        ll, aic, bic = self.ll()
        JB, JBpv, skew, kurtosis = self.JB()
        omni, omnipv = self.omni()

        # printing output to screen
        print '\n=============================================================================='
        print "Dependent Variable: " + self.y_varnm
        print "Method: Least Squares"
        print "Date: ", time.strftime("%a, %d %b %Y",t)
        print "Time: ", time.strftime("%H:%M:%S",t)
        print '# obs:               %5.0f' % self.nobs
        print '# variables:     %5.0f' % self.ncoef
        print '=============================================================================='
        print 'variable     coefficient     std. Error      t-statistic     prob.'
        print '=============================================================================='
        for i in range(len(self.x_varnm)):
            print '''% -5s          % -5.6f     % -5.6f     % -5.6f     % -5.6f''' % tuple([self.x_varnm[i],self.b[i],self.se[i],self.t[i],self.p[i]])
        print '=============================================================================='
        print 'Models stats                         Residual stats'
        print '=============================================================================='
        print 'R-squared            % -5.6f         Durbin-Watson stat  % -5.6f' % tuple([self.R2, self.dw()])
        print 'Adjusted R-squared   % -5.6f         Omnibus stat        % -5.6f' % tuple([self.R2adj, omni])
        print 'F-statistic          % -5.6f         Prob(Omnibus stat)  % -5.6f' % tuple([self.F, omnipv])
        print 'Prob (F-statistic)   % -5.6f			JB stat             % -5.6f' % tuple([self.Fpv, JB])
        print 'Log likelihood       % -5.6f			Prob(JB)            % -5.6f' % tuple([ll, JBpv])
        print 'AIC criterion        % -5.6f         Skew                % -5.6f' % tuple([aic, skew])
        print 'BIC criterion        % -5.6f         Kurtosis            % -5.6f' % tuple([bic, kurtosis])
        print '=============================================================================='

if __name__ == '__main__':
    xxsingular = False#True
    x = np.linspace(0, 15, 40)
    a,b,c = 3.1, 42, -304.2
    y_true = a*x**2 + b*x + c
    y_meas = y_true + 100.01*np.random.standard_normal( y_true.shape )
    if xxsingular:
        xx = np.c_[x**2,x,2*x,np.ones(x.shape[0])]
    else:
        xx = np.c_[x**2,x,np.ones(x.shape[0])]
    
    x_varnm = ['x1','x2','x3','x4']
    
    k = xx.shape[1]
    m = ols(y_meas,xx,y_varnm = 'y',x_varnm = x_varnm[:k-1],addconst = False)
    m.summary()
    
    #matplotlib ploting
    import matplotlib.pylab as plt
    plt.title('Linear Regression Example')
    plt.plot(x,y_true,'g.--')
    plt.plot(x,y_meas,'k.')
    plt.plot(x,y_meas-m.e,'r.-')
    plt.legend(['original','plus noise', 'regression'], loc='lower right')
    plt.show()
