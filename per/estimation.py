


import scipy
#from scipy import stats
from scipy.integrate import quad
from scipy.linalg import pinv2
from scipy.misc import comb, derivative
from scipy import special
from scipy import optimize
import inspect

#which imports are necessary?
from numpy import alltrue, where, arange, put, putmask, \
     ravel, take, ones, sum, shape, product, repeat, reshape, \
     zeros, floor, logical_and, log, sqrt, exp, arctanh, tan, sin, arcsin, \
     arctan, tanh, ndarray, cos, cosh, sinh, newaxis, array, log1p, expm1
from numpy import atleast_1d, polyval, angle, ceil, place, extract, \
     any, argsort, argmax, vectorize, r_, asarray, nan, inf, pi, isnan, isinf, \
     power
import numpy
import numpy as np
import numpy.random as mtrand
from numpy import flatnonzero as nonzero
from scipy.special import gammaln as gamln
from copy import copy



import numdifftools
from plotbackend import plotbackend

import distributions
from distributions import rv_frozen, norm, chi2, valarray, findcross


# internal class to profile parameters of a given distribution
class Profile(object):
    ''' Profile Log- likelihood or Product Spacing-function.
            which can be used for constructing confidence interval for
            either phat(i), probability or quantile.
    Call
    -----
      Lp = Profile(fit_dist,**kwds)

    Parameters
    ----------
    fit_dist : FitDistribution object with ML or MPS estimated distribution parameters.

    **kwds : named arguments with keys
          i          - Integer defining which distribution parameter to
                         profile, i.e. which parameter to keep fixed
                         (default index to first non-fixed parameter)
          pmin, pmax - Interval for either the parameter, phat(i), prb, or x,
                        used in the optimization of the profile function (default
                        is based on the 100*(1-alpha)% confidence interval
                        computed using the delta method.)
          N          - Max number of points used in Lp (default 100)
          x          - Quantile (return value)
          logSF      - log survival probability,i.e., SF = Prob(X>x;phat)
          link       - function connecting the quantile (x) and the
                         survival probability (SF) with the fixed distribution
                         parameter, i.e.: self.par[i] = link(x,logSF,self.par,i),
                         where logSF = log(Prob(X>x;phat)).
                         This means that if:
                          1) x is not None then x is profiled
                          2) logSF is not None then logSF is profiled
                          3) x and logSF both are None then self.par[i] is profiled (default)
          alpha       - confidence coefficent (default 0.05)
    Returns
    -------
    Lp : Profile log-likelihood function with parameters phat given
               the data, phat(i), probability (prb) and quantile (x) (if given), i.e.,
                 Lp = max(log(f(phat|data,phat(i)))),
               or
                 Lp = max(log(f(phat|data,phat(i),x,prb)))
    Member methods
      plot()
      get_CI()

    Member variables
      fit_dist - fitted data object.
      data - profile function values
      args - profile function arguments
      alpha - confidence coefficient
      Lmax - Maximum value of profile function
      alpha_cross_level -

    PROFILE is a utility function for making inferences either on a particular
    component of the vector phat or the quantile, x, or the probability, SF.
    This is usually more accurate than using the delta method assuming
    asymptotic normality of the ML estimator or the MPS estimator.


    Examples
    --------
    #MLE and better CI for phat.par[0]
    >>> import numpy as np
    >>> R = weibull_min.rvs(1,size=100);
    >>> phat = weibull_min.fit(R,1,1,par_fix=[np.nan,0.,np.nan])
    >>> Lp = Profile(phat,i=0)
    >>> Lp.plot()
    >>> Lp.get_CI(alpha=0.1)
    >>> SF = 1./990
    >>> x = phat.isf(SF)

    # CI for x
    >>> Lx = phat.profile(i=1,x=x,link=phat.dist.link)
    >>> Lx.plot()
    >>> Lx.get_CI(alpha=0.2)

    # CI for logSF=log(SF)
    >>> Lpr = phat.profile(i=1,logSF=log(SF),link = phat.dist.link)


    '''
    def __init__(self, fit_dist, **kwds):
        self.fit_dist = fit_dist
        self.data = None
        self.args = None
        self.title = 'Profile log'
        self.xlabel = ''
        self.ylabel = ''
        self.i_fixed, self.N, self.alpha, self.pmin,self.pmax,self.x,self.logSF,self.link = map(kwds.get,
                            ['i','N','alpha','pmin','pmax','x','logSF','link'],
                            [0,100,0.05,None,None,None,None,None])

        self.ylabel = '%g%s CI' % (100*(1.0-self.alpha), '%')
        if fit_dist.method.startswith('ml'):
            self.title = self.title + 'likelihood'
            Lmax = fit_dist.LLmax
        elif fit_dist.method.startswith('mps'):
            self.title = self.title + ' product spacing'
            Lmax = fit_dist.LPSmax
        else:
            raise ValueError("PROFILE is only valid for ML- or MPS- estimators")
        if fit_dist.par_fix==None:
            isnotfixed = valarray(fit_dist.par.shape,True)
        else:
            isnotfixed = 1-numpy.isfinite(fit_dist.par_fix)

        self.i_notfixed = nonzero(isnotfixed)

        self.i_fixed = atleast_1d(self.i_fixed)

        if 1-isnotfixed[self.i_fixed]:
            raise ValueError("Index i must be equal to an index to one of the free parameters.")

        isfree = isnotfixed
        isfree[self.i_fixed] = False
        self.i_free = nonzero(isfree)

        self.Lmax = Lmax
        self.alpha_Lrange = 0.5*chi2.isf(self.alpha,1)
        self.alpha_cross_level = Lmax - self.alpha_Lrange
        lowLevel = self.alpha_cross_level-self.alpha_Lrange/7.0

        ## Check that par are actually at the optimum
        phatv = fit_dist.par.copy()
        self._par = phatv.copy()
        phatfree = phatv[self.i_free].copy()


        ## Set up variable to profile and _local_link function

        self.profile_x = not self.x==None
        self.profile_logSF = not (self.logSF==None or self.profile_x)
        self.profile_par = not (self.profile_x or self.profile_logSF)

        if self.link==None:
            self.link = self.fit_dist.dist.link
        if self.profile_par:
            self._local_link = lambda fix_par, par : fix_par
            self.xlabel = 'phat(%d)'% self.i_fixed
            p_opt = self._par[self.i_fixed]
        elif self.profile_x:
            self.logSF = log(fit_dist.sf(self.x))
            self._local_link = lambda fix_par, par : self.link(fix_par,self.logSF,par,self.i_fixed)
            self.xlabel = 'x'
            p_opt = self.x
        elif self.profile_logSF:
            p_opt = self.logSF
            self.x = fit_dist.isf(exp(p_opt))
            self._local_link = lambda fix_par, par : self.link(self.x,fix_par,par,self.i_fixed)
            self.xlabel= 'log(R)'
        else:
            raise ValueError("You must supply a non-empty quantile (x) or probability (logSF) in order to profile it!")

        self.xlabel = self.xlabel + ' (' + fit_dist.dist.name + ')'

        pvec = self._get_pvec(p_opt)


        mylogfun = self._nlogfun
        self.data = numpy.empty_like(pvec)
        self.data[:] = nan
        k1 = (pvec>=p_opt).argmax()
        for ix in xrange(k1,-1,-1):
            phatfree = optimize.fmin(mylogfun,phatfree,args =(pvec[ix],) ,disp=0)
            self.data[ix] = -mylogfun(phatfree,pvec[ix])
            if self.data[ix]<self.alpha_cross_level:
                pvec[:ix] = nan
                break

        phatfree = phatv[self.i_free].copy()
        for ix in xrange(k1+1,pvec.size):
            phatfree = optimize.fmin(mylogfun,phatfree,args =(pvec[ix],) ,disp=0)
            self.data[ix] = -mylogfun(phatfree,pvec[ix])
            if self.data[ix]<self.alpha_cross_level:
                pvec[ix+1:] = nan
                break

        # prettify result
        ix = nonzero(numpy.isfinite(pvec))
        self.data = self.data[ix]
        self.args = pvec[ix]
        cond =self.data==-numpy.inf
        if any(cond):
            ind, = cond.nonzero()
            self.data.put(ind, numpy.finfo(float).min/2.0)
            ind1 = numpy.where(ind==0,ind,ind-1)
            cl = self.alpha_cross_level-self.alpha_Lrange/2.0
            t0 = ecross(self.args,self.data,ind1,cl)

            self.data.put(ind,cl)
            self.args.put(ind,t0)


    def _get_pvec(self,p_opt):
        ''' return proper interval for the variable to profile
        '''

        linspace = numpy.linspace
        if self.pmin==None or self.pmax==None:

            if self.profile_par:
                pvar = self.fit_dist.par_cov[self.i_fixed,:][:,self.i_fixed]
            else:
                i_notfixed = self.i_notfixed
                phatv = self._par

                if self.profile_x:
                    gradfun = numdifftools.Gradient(self._myinvfun)
                else:
                    gradfun = numdifftools.Gradient(self._myprbfun)
                drl = gradfun(phatv[self.i_notfixed])

                pcov = self.fit_dist.par_cov[i_notfixed,:][:,i_notfixed]
                pvar = sum(numpy.dot(drl,pcov)*drl)

            p_crit = norm.isf(self.alpha/2.0)*sqrt(numpy.ravel(pvar))*1.5
            if self.pmin==None:
                self.pmin = p_opt-5.0*p_crit
            if self.pmax==None:
                self.pmax = p_opt+5.0*p_crit

            N4 = numpy.floor(self.N/4.0)

            pvec1 = linspace(self.pmin,p_opt-p_crit,N4+1)
            pvec2 = linspace(p_opt-p_crit,p_opt+p_crit,self.N-2*N4)
            pvec3 = linspace(p_opt+p_crit,self.pmax,N4+1)
            pvec = numpy.unique(numpy.hstack((pvec1,p_opt,pvec2,pvec3)))

        else:
            pvec = linspace(self.pmin,self.pmax,self.N)
        return pvec
    def  _myinvfun(self,phatnotfixed):
        mphat = self._par.copy()
        mphat[self.i_notfixed] = phatnotfixed;
        prb = exp(self.logSF)
        return self.fit_dist.dist.isf(prb,*mphat);

    def _myprbfun(phatnotfixed):
        mphat = self._par.copy()
        mphat[self.i_notfixed] = phatnotfixed;
        return self.fit_dist.dist.sf(self.x,*mphat);


    def _nlogfun(self,free_par,fix_par):
        ''' Return negative of loglike or logps function

           free_par - vector of free parameters
           fix_par  - fixed parameter, i.e., either quantile (return level),
                      probability (return period) or distribution parameter

        '''
        par = self._par
        par[self.i_free] = free_par
        # _local_link: connects fixed quantile or probability with fixed distribution parameter
        par[self.i_fixed] = self._local_link(fix_par,par)
        return self.fit_dist.fitfun(par)

    def get_CI(self,alpha=0.05):
        '''Return confidence interval
        '''
        if alpha<self.alpha:
            raise ValueError('Unable to return CI with alpha less than %g' % self.alpha)

        cross_level = self.Lmax - 0.5*chi2.isf(alpha,1)
        ind = findcross(self.data,cross_level)
        N = len(ind)
        if N==0:
            #Warning('upper bound for XXX is larger'
            #Warning('lower bound for XXX is smaller'
            CI = (self.pmin,self.pmax)
        elif N==1:
            x0 = ecross(self.args,self.data,ind,cross_level)
            isUpcrossing = self.data[ind]>self.data[ind+1]
            if isUpcrossing:
                CI = (x0,self.pmax)
                #Warning('upper bound for XXX is larger'
            else:
                CI = (self.pmin,x0)
                #Warning('lower bound for XXX is smaller'

        elif N==2:
            CI = ecross(self.args,self.data,ind,cross_level)
        else:
            # Warning('Number of crossings too large!')
            CI = ecross(self.args,self.data,ind[[0,-1]],cross_level)
        return CI

    def plot(self):
        ''' Plot profile function with 100(1-alpha)% CI
        '''
        plotbackend.plot(self.args,self.data,
            self.args[[0,-1]],[self.Lmax,]*2,'r',
            self.args[[0,-1]],[self.alpha_cross_level,]*2,'r')
        plotbackend.title(self.title)
        plotbackend.ylabel(self.ylabel)
        plotbackend.xlabel(self.xlabel)

# internal class to fit given distribution to data
class FitDistribution(rv_frozen):
    def __init__(self, dist, data, *args, **kwds):
        extradoc = '''

    RV.plotfitsumry() - Plot various diagnostic plots to asses quality of fit.
    RV.plotecdf()     - Plot Empirical and fitted Cumulative Distribution Function
    RV.plotesf()      - Plot Empirical and fitted Survival Function
    RV.plotepdf()     - Plot Empirical and fitted Probability Distribution Function
    RV.plotresq()     - Displays a residual quantile plot.
    RV.plotresprb()   - Displays a residual probability plot.

    RV.profile()      - Return Profile Log- likelihood or Product Spacing-function.

    Member variables:
        data - data used in fitting
        alpha - confidence coefficient
        method - method used
        LLmax  - loglikelihood function evaluated using par
        LPSmax - log product spacing function evaluated using par
        pvalue - p-value for the fit
        search - True if search for distribution parameters (default)
        copydata - True if copy input data (default)

        par     - parameters (fixed and fitted)
        par_cov - covariance of parameters
        par_fix - fixed parameters
        par_lower - lower (1-alpha)% confidence bound for the parameters
        par_upper - upper (1-alpha)% confidence bound for the parameters

        '''
        if rv_frozen.__doc__ != None:
            self.__doc__ = rv_frozen.__doc__ + extradoc
        else:
            self.__doc__ = extradoc
        self.dist = dist
        numargs = dist.numargs

        self.method, self.alpha, self.par_fix, self.search, self.copydata = \
                     map(kwds.get,['method','alpha','par_fix','search',
                                   'copydata'],['ml',0.05,None,True,True])
        self.data = ravel(data)
        if self.copydata:
            self.data = self.data.copy()
        self.data.sort()
        if self.method.lower()[:].startswith('mps'):
            self._fitfun = dist.nlogps
        else:
            self._fitfun = dist.nnlf

        allfixed  = False
        isfinite = numpy.isfinite
        somefixed = (self.par_fix !=None) and any(isfinite(self.par_fix))

        if somefixed:
            fitfun = self._fxfitfun
            self.par_fix = tuple(self.par_fix)
            allfixed = all(isfinite(self.par_fix))
            self.par = atleast_1d(self.par_fix)
            self.i_notfixed = nonzero(1-isfinite(self.par))
            self.i_fixed  = nonzero(isfinite(self.par))
            if len(self.par) != numargs+2:
                raise ValueError, "Wrong number of input arguments."
            if len(args)!=len(self.i_notfixed):
                raise ValueError("Length of args must equal number of non-fixed parameters given in par_fix! (%d) " % len(self.i_notfixed))
            x0 = atleast_1d(args)
        else:
            fitfun = self.fitfun
            loc0, scale0 = map(kwds.get, ['loc', 'scale'])
            args, loc0, scale0 = dist.fix_loc_scale(args, loc0, scale0)
            Narg = len(args)
            if Narg != numargs:
                if Narg > numargs:
                    raise ValueError, "Too many input arguments."
                else:
                    args += (1.0,)*(numargs-Narg)
            # location and scale are at the end
            x0 = args + (loc0, scale0)
            x0 = atleast_1d(x0)

        numpar = len(x0)
        if self.search and not allfixed:
            #args=(self.data,),
            par = optimize.fmin(fitfun,x0,disp=0)
            if not somefixed:
                self.par = par
        elif  (not allfixed) and somefixed:
            self.par[self.i_notfixed] = x0
        else:
            self.par = x0

        np = numargs+2   #this shadows np for numpy

        self.par_upper = None
        self.par_lower = None
        self.par_cov = zeros((np,np))
        self.LLmax = -dist.nnlf(self.par,self.data)
        self.LPSmax = -dist.nlogps(self.par,self.data) 
        self.pvalue = dist.pvalue(self.par,self.data,unknown_numpar=numpar)
        H = numpy.asmatrix(dist.hessian_nnlf(self.par,self.data))
        self.H = H
        try:
            if allfixed:
                pass
            elif somefixed:
                pcov = -pinv2(H[self.i_notfixed,:][...,self.i_notfixed])
                for row,ix in enumerate(list(self.i_notfixed)):
                    self.par_cov[ix,self.i_notfixed] = pcov[row,:]

            else:
                self.par_cov = -pinv2(H)
        except:
            self.par_cov[:,:] = nan

        pvar = numpy.diag(self.par_cov)
        zcrit = -norm.ppf(self.alpha/2.0)
        self.par_lower = self.par-zcrit*sqrt(pvar)
        self.par_upper = self.par+zcrit*sqrt(pvar)

    def fitfun(self,phat):
        return self._fitfun(phat,self.data)

    def _fxfitfun(self,phat10):
        self.par[self.i_notfixed] = phat10
        return self._fitfun(self.par,self.data)


    def profile(self,**kwds):
        ''' Profile Log- likelihood or Log Product Spacing- function,
            which can be used for constructing confidence interval for
            either phat(i), probability or quantile.

        CALL:  Lp = RV.profile(**kwds)


       RV = object with ML or MPS estimated distribution parameters.
       Parameters
       ----------
       **kwds : named arguments with keys:
          i          - Integer defining which distribution parameter to
                         profile, i.e. which parameter to keep fixed
                         (default index to first non-fixed parameter)
          pmin, pmax - Interval for either the parameter, phat(i), prb, or x,
                        used in the optimization of the profile function (default
                        is based on the 100*(1-alpha)% confidence interval
                        computed using the delta method.)
          N          - Max number of points used in Lp (default 100)
          x          - Quantile (return value)
          logSF       - log survival probability,i.e., R = Prob(X>x;phat)
          link       - function connecting the quantile (x) and the
                         survival probability (R) with the fixed distribution
                         parameter, i.e.: self.par[i] = link(x,logSF,self.par,i),
                         where logSF = log(Prob(X>x;phat)).
                         This means that if:
                          1) x is not None then x is profiled
                          2) logSF is not None then logSF is profiled
                          3) x and logSF both are None then self.par[i] is profiled (default)
          alpha       - confidence coefficent (default 0.05)
       Returns
       --------
         Lp = Profile log-likelihood function with parameters phat given
               the data, phat(i), probability (prb) and quantile (x) (if given), i.e.,
                 Lp = max(log(f(phat|data,phat(i)))),
               or
                 Lp = max(log(f(phat|data,phat(i),x,prb)))

          PROFILE is a utility function for making inferences either on a particular
          component of the vector phat or the quantile, x, or the probability, R.
          This is usually more accurate than using the delta method assuming
          asymptotic normality of the ML estimator or the MPS estimator.


          Examples
          --------
          # MLE and better CI for phat.par[0]
          >>> R = weibull_min.rvs(1,size=100);
          >>> phat = weibull_min.fit(R)
          >>> Lp = phat.profile(i=0)
          >>> Lp.plot()
          >>> Lp.get_CI(alpha=0.1)
          >>> R = 1./990
          >>> x = phat.isf(R)

          # CI for x
          >>> Lx = phat.profile(i=1,x=x,link=phat.dist.link)
          >>> Lx.plot()
          >>> Lx.get_CI(alpha=0.2)

          # CI for logSF=log(SF)
          >>> Lpr = phat.profile(i=1,logSF=log(R),link = phat.dist.link)

          See also
          --------
          Profile
        '''
        if not self.par_fix==None:
            i1 = kwds.setdefault('i',(1-numpy.isfinite(self.par_fix)).argmax())

        return Profile(self,**kwds)



    def plotfitsumry(self):
        ''' Plot various diagnostic plots to asses the quality of the fit.

        PLOTFITSUMRY displays probability plot, density plot, residual quantile
        plot and residual probability plot.
        The purpose of these plots is to graphically assess whether the data
        could come from the fitted distribution. If so the empirical- CDF and PDF
        should follow the model and the residual plots will be linear. Other
        distribution types will introduce curvature in the residual plots.
        '''
        plotbackend.subplot(2,2,1)
        #self.plotecdf()
        self.plotesf()
        plotbackend.subplot(2,2,2)
        self.plotepdf()
        plotbackend.subplot(2,2,3)
        self.plotresprb()
        plotbackend.subplot(2,2,4)
        self.plotresq()
        fixstr = ''
        if not self.par_fix==None:
            numfix = len(self.i_fixed)
            if numfix>0:
                format = '%d,'*numfix
                format = format[:-1]
                format1 = '%g,'*numfix
                format1 = format1[:-1]
                phatistr = format % tuple(self.i_fixed)
                phatvstr = format1 % tuple(self.par[self.i_fixed])
                fixstr = 'Fixed: phat[%s] = %s ' % (phatistr,phatvstr)


        infostr = 'Fit method: %s, Fit p-value: %2.2f %s' % (self.method,self.pvalue,fixstr)
        try:
            plotbackend.figtext(0.05,0.01,infostr)
        except:
            pass

    def plotesf(self):
        '''  Plot Empirical and fitted Survival Function

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution.
        If so the empirical CDF should resemble the model CDF.
        Other distribution types will introduce deviations in the plot.
        '''
        n = len(self.data)
        SF = (arange(n,0,-1))/n
        plotbackend.semilogy(self.data,SF,'b.',self.data,self.sf(self.data),'r-')
        #plotbackend.plot(self.data,SF,'b.',self.data,self.sf(self.data),'r-')

        plotbackend.xlabel('x');
        plotbackend.ylabel('F(x) (%s)' % self.dist.name)
        plotbackend.title('Empirical SF plot')

    def plotecdf(self):
        '''  Plot Empirical and fitted Cumulative Distribution Function

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution.
        If so the empirical CDF should resemble the model CDF.
        Other distribution types will introduce deviations in the plot.
        '''
        n = len(self.data)
        F = (arange(1,n+1))/n
        plotbackend.plot(self.data,F,'b.',self.data,self.cdf(self.data),'r-')


        plotbackend.xlabel('x');
        plotbackend.ylabel('F(x) (%s)' % self.dist.name)
        plotbackend.title('Empirical CDF plot')

    def plotepdf(self):
        '''Plot Empirical and fitted Probability Density Function

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution.
        If so the histogram should resemble the model density.
        Other distribution types will introduce deviations in the plot.
        '''

        bin,limits = numpy.histogram(self.data,normed=True,new=True)
        limits.shape = (-1,1)
        xx = limits.repeat(3,axis=1)
        xx.shape = (-1,)
        xx = xx[1:-1]
        bin.shape = (-1,1)
        yy = bin.repeat(3,axis=1)
        #yy[0,0] = 0.0 # pdf
        yy[:,0] = 0.0 # histogram
        yy.shape = (-1,)
        yy = numpy.hstack((yy,0.0))

        #plotbackend.hist(self.data,normed=True,fill=False)
        plotbackend.plot(self.data,self.pdf(self.data),'r-',xx,yy,'b-')

        plotbackend.xlabel('x');
        plotbackend.ylabel('f(x) (%s)' % self.dist.name)
        plotbackend.title('Density plot')


    def plotresq(self):
        '''PLOTRESQ displays a residual quantile plot.

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution. If so the
        plot will be linear. Other distribution types will introduce
        curvature in the plot.
        '''
        n=len(self.data)
        eprob = (arange(1,n+1)-0.5)/n
        y = self.ppf(eprob)
        y1 = self.data[[0,-1]]
        plotbackend.plot(self.data,y,'b.',y1,y1,'r-')

        plotbackend.xlabel('Empirical')
        plotbackend.ylabel('Model (%s)' % self.dist.name)
        plotbackend.title('Residual Quantile Plot');
        plotbackend.axis('tight')
        plotbackend.axis('equal')


    def plotresprb(self):
        ''' PLOTRESPRB displays a residual probability plot.

        The purpose of the plot is to graphically assess whether
        the data could come from the fitted distribution. If so the
        plot will be linear. Other distribution types will introduce curvature in the plot.
        '''
        n = len(self.data);
        #ecdf = (0.5:n-0.5)/n;
        ecdf = arange(1,n+1)/(n+1)
        mcdf = self.cdf(self.data)
        p1 = [0,1]
        plotbackend.plot(ecdf,mcdf,'b.',p1,p1,'r-')


        plotbackend.xlabel('Empirical')
        plotbackend.ylabel('Model (%s)' % self.dist.name)
        plotbackend.title('Residual Probability Plot');
        plotbackend.axis([0, 1, 0, 1])
        plotbackend.axis('equal')

if __name__ == '__main__':

#I get an error with this example
##    R = distributions.weibull_min.rvs(1,size=100);
##    phat = distributions.weibull_min.fit(R)
##    #FitDistribution(self, data, *args, **kwds)
##    phat_d = FitDistribution(distributions.weibull_min,R, method='ml')
##    Lp = phat.profile(i=0)
##    Lp.plot()
##    Lp.get_CI(alpha=0.1)
##    R = 1./990
##    x = phat.isf(R)

##t-distribution example raises exception
##    R = distributions.t.rvs(20,size=(100,1));
##    phat = distributions.t.fit(R,20)
##    #FitDistribution(self, data, *args, **kwds)
##    #phat_d = FitDistribution(distributions.t,R, method='ml')
##    Lp = phat.profile(i=0)
##    Lp.plot()
##    Lp.get_CI(alpha=0.1)
##    R = 1./990
##    x = phat.isf(R)


    #this works
    plotbackend.figure()
    R = distributions.weibull_min.rvs(1, size=100);
    phat = distributions.weibull_min.fit(R,1,1,par_fix=[nan,0,nan])
    Lp = phat.profile(i=0)
    Lp.plot()
    
    
    #this works exactly the same as the previous call to fit
    plotbackend.figure()
    #R = distributions.weibull_min.rvs(1, size=100);
    phat_d = FitDistribution(distributions.weibull_min,R,1,1,par_fix=[nan,0,nan])
    Lp = phat_d.profile(i=0)
    Lp.plot()
    #plotbackend.show()

    ##Example: genpareto
    #this works with fit method but incorrect (?) with FitDistribution
    #plotecdf and plotresprb look wrong, different from fit method call
    #are there some hidden name/scope collisions?
    np.random.seed(828709827)
    r = distributions.genpareto.rvs(0.00001,size=100)
    #pht = distributions.genpareto.fit(r,1,par_fix=[0,0,nan])
    pht = FitDistribution(distributions.genpareto,r,1,par_fix=[0,0,nan])
    lp = pht.profile()

    print pht.stats()
    print pht.pvalue
    plotbackend.figure()
    pht.plotecdf()
    plotbackend.figure()
    pht.plotepdf()
    plotbackend.figure()
    pht.plotresq()
    plotbackend.figure()
    pht.plotresprb()
    plotbackend.figure()
    pht.plotfitsumry()
    plotbackend.figure()
    lp.plot()
    #plotbackend.show()
