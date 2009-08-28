''' A class for the distribution of a non-linear monotonic transformation of a continuous random variable

simplest usage:
example: create log-gamma distribution, i.e. y = log(x),
            where x is gamma distributed (also available in scipy.stats)
    loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp)

example: what is the distribution of the discount factor y=1/(1+x)
            where interest rate x is normally distributed with N(mux,stdx**2)')?
            (just to come up with a story that implies a nice transformation)
    invnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, a=-np.inf)

This class does not work well for distributions with difficult shapes,
    e.g. 1/x where x is standard normal, because of the singularity and jump at zero.

Note: I'm working from my version of scipy.stats.distribution.
      But this script runs under scipy 0.6.0 (checked with numpy: 1.2.0rc2 and python 2.4)

This is not yet thoroughly tested, polished or optimized

TODO:
  * numargs handling is not yet working properly, numargs needs to be specified (default = 0 or 1)
  * feeding args and kwargs to underlying distribution is untested and incomplete
  * distinguish args and kwargs for the transformed and the underlying distribution
    - currently all args and no kwargs are transmitted to underlying distribution
    - loc and scale only work for transformed, but not for underlying distribution
    - possible to separate args for transformation and underlying distribution parameters

  * add _rvs as method, will be faster in many cases
     
'''

from scipy import integrate # for scipy 0.6.0

from scipy import stats, info
from scipy.stats import distributions
import numpy as np

def get_u_argskwargs(**kwargs):
        u_kwargs = dict((k.replace('u_','',1),v) for k,v in kwargs.items()
                        if k.startswith('u_'))
        u_args = u_kwargs.pop('u_args',None)
        return u_args, u_kwargs 

class Transf_gen(distributions.rv_continuous):
    #a class for non-linear monotonic transformation of a continuous random variable
    def __init__(self, kls, func, funcinv, *args, **kwargs):
        #print args
        #print kwargs
        
        self.func = func
        self.funcinv = funcinv
        #explicit for self.__dict__.update(kwargs)
        #need to set numargs because inspection does not work
        self.numargs = kwargs.pop('numargs', 0) 
        print self.numargs
        name = kwargs.pop('name','transfdist')
        longname = kwargs.pop('longname','Non-linear transformed distribution')
        extradoc = kwargs.pop('extradoc',None)
        a = kwargs.pop('a', 0)
        self.decr = kwargs.pop('decr', False)
            #defines whether it is a decreasing (True)
            #       or increasing (False) monotonic transformation

        
        self.u_args, self.u_kwargs = get_u_argskwargs(**kwargs)
        self.kls = kls   #(self.u_args, self.u_kwargs)
                         # possible to freeze the underlying distribution
        
        super(Transf_gen,self).__init__(a=a, name = name, 
                                longname = longname, extradoc = extradoc)
        
    def _cdf(self,x,*args, **kwargs):
        print args
        if not self.decr:
            return self.kls._cdf(self.funcinv(x),*args, **kwargs)
            #note scipy _cdf only take *args not *kwargs
        else:
            return 1.0 - self.kls._cdf(self.funcinv(x),*args, **kwargs)
    def _ppf(self, q, *args, **kwargs):
        if not self.decr:
            return self.func(self.kls._ppf(q,*args, **kwargs))
        else:
            return self.func(self.kls._ppf(1-q,*args, **kwargs))

       
def inverse(x):
    return np.divide(1.0,x)

mux, stdx = 0.05, 0.1
mux, stdx = 9.0, 1.0
def inversew(x):
    return 1.0/(1+mux+x*stdx)
def inversew_inv(x):
    return (1.0/x - 1.0 - mux)/stdx #.np.divide(1.0,x)-10

def identit(x):
    return x





print '\nResults for invnormal with regularization y=1/(1+x), x is N(0.05, 0.1*0.1)'
print   '-----------------------------------------------------------------------'
invnormalg = Transf_gen(stats.norm, inversew, inversew_inv, decr=True, a=-np.inf,
                numargs = 0, name = 'discf', longname = 'normal-based discount factor',
                extradoc = '\ndistribution of discount factor y=1/(1+x)) with x N(0.05,0.1**2)')
                        #u_loc=l, u_scale=s)

l,s = 0.0, 0.1
print
#print invnormalg.__doc__
print
print 'cdf for [0.95,1.0,1.1]:', invnormalg.cdf([0.95,1.0,1.1],loc=l, scale=s)
print 'pdf for [0.95,1.0,1.1]:', invnormalg.pdf([0.95,1.0,1.1],loc=l, scale=s)

print 'rvs:', invnormalg.rvs(loc=l, scale=s,size=5)
print 'stats: ', invnormalg.stats(loc=l, scale=s)
print 'stats kurtosis, skew: ', invnormalg.stats(moments='ks')
print 'median:', invnormalg.ppf(0.5)
rvs = invnormalg.rvs(loc=l, scale=s,size=10000)
print 'sample stats:              ', rvs.mean(), rvs.var()
rvs = inversew(stats.norm.rvs(loc=l, scale=s,size=10000))
print 'std norm sample stats:    ', rvs.mean(), rvs.var()
rvs = inversew(stats.norm.rvs(size=100000))*s + l
print 'transf. norm sample stats: ', rvs.mean(), rvs.var()


print
print 'Results for lognormal'
print '---------------------'

lognormalg = Transf_gen(stats.norm, np.exp, np.log, 
                numargs = 2, a=0, name = 'lnnorm', longname = 'Exp transformed normal',
                extradoc = '\ndistribution of y = exp(x), with x standard normal')
print 'cdf for [2.0,2.5,3.0,3.5]:      ',lognormalg.cdf([2.0,2.5,3.0,3.5])
print 'scipy cdf for [2.0,2.5,3.0,3.5]:', stats.lognorm.cdf([2.0,2.5,3.0,3.5],1)
print 'pdf for [2.0,2.5,3.0,3.5]:     ', lognormalg.pdf([2.0,2.5,3.0,3.5])
print 'scipy pdf for [2.0,2.5,3.0,3.5]:', stats.lognorm.pdf([2.0,2.5,3.0,3.5],1)
print 'stats:      ', lognormalg.stats()
print 'scipy stats:', stats.lognorm.stats(1)
print 'rvs:', lognormalg.rvs(size=5)
#print info(lognormalg)

##
##
##print '\nResults for idnormal'
##print   '--------------------'
##idnormalg = Transf_gen(stats.norm, identit, identit, a=-np.inf, name = 'normal')
##print idnormalg.cdf(1,loc=100,scale=10)
##print stats.norm.cdf(1,loc=100,scale=10)
##print idnormalg.stats(loc=100,scale=10)
##print stats.norm.stats(loc=100,scale=10)
##print idnormalg.rvs(loc=100,scale=10,size=5)
##rvs = idnormalg.rvs(loc=100,scale=10,size=10000)
##print rvs.mean(), rvs.var()
##
##
##
print '\nResults for expgamma'
print   '--------------------'
loggammaexpg = Transf_gen(stats.gamma, np.log, np.exp, numargs = 1)
print loggammaexpg._cdf(1,10)
print stats.loggamma.cdf(1,10)
print loggammaexpg._cdf(2,15)
print stats.loggamma.cdf(2,15)

print 'cdf for [2.0,2.5,3.0,3.5]:      ', loggammaexpg._cdf([2.0,2.5,3.0,3.5],10)
print 'scipy cdf for [2.0,2.5,3.0,3.5]:', stats.loggamma.cdf([2.0,2.5,3.0,3.5],10)
print 'pdf for [2.0,2.5,3.0,3.5]:     ', loggammaexpg.pdf([2.0,2.5,3.0,3.5],10) # not in scipy 0.6.0
#print loggammaexpg._pdf(2.0,10) # ok in scipy 0.6.0
print 'scipy pdf for [2.0,2.5,3.0,3.5]:', stats.loggamma.pdf([2.0,2.5,3.0,3.5],10)
##
##
##
##print '\n\n the rest is not so good\n\n'
##
##
##print '\nResults for invnormal'
##print   '--------------------'
##invnormalg2 = Transf_gen(stats.norm, inverse, inverse, decr=True, a=-np.inf,
##                        longname = 'inverse normal')
##print invnormalg2.cdf(1,loc=10)
##print stats.invnorm.cdf(1,1,loc=10)
##print invnormalg2.stats(loc=10)
##print stats.invnorm.stats(1,loc=10)
##print invnormalg2.rvs(loc=10,size=5)
##
##
##
##print '\nResults for invexpon'
##print   '--------------------'
##invexpong = Transf_gen(stats.expon, inverse, inverse, a=0.0, name = 'Log transformed normal general')
##print invexpong.cdf(1)
###print stats.invnorm.cdf(1,1)
##print invexpong.stats()
###print stats.invnorm.stats(1)
##print invexpong.rvs(size=5)
##
##print '\nResults for inv gamma'
##print   '---------------------'
##invgammag = Transf_gen(stats.gamma, inverse, inverse, a=0.0, name = 'Log transformed normal general')
##
##
##print invgammag._cdf(1,5) # .cdf  #not in scipy 0.6.0
##print stats.invgamma.cdf(1,1)
###print invgammag.stats(1) #not in scipy 0.6.0
##print stats.invgamma.stats(1)
###print invgammag.rvs(1,size=5) #not in scipy 0.6.0
##print invgammag._rvs(1)
##
##
##
##
##print '\nResults for loglaplace'
##print   '----------------------'
###no idea what the relationship is
##loglaplaceg = Transf_gen(stats.laplace, np.log, np.exp)
##print loglaplaceg._cdf(2)
##print stats.loglaplace.cdf(2,10)
##loglaplaceexpg = Transf_gen(stats.laplace, np.exp, np.log)
##print loglaplaceexpg._cdf(2)
##
##
##
##
###this are the results, that I get:
##results = \
##'''
##Results for invnormal with regularization y=1/(1+x), x is N(0, 0.1*0.1)
##-----------------------------------------------------------------------
##
##normal-based discount factor continuous random variable.
##
##    Continuous random variables are defined from a standard form chosen
##    for simplicity of representation.  The standard form may require
##    some shape parameters to complete its specification.  The distributions
##    also take optional location and scale parameters using loc= and scale=
##    keywords (defaults: loc=0, scale=1)
##
##    These shape, scale, and location parameters can be passed to any of the
##    methods of the RV object such as the following:
##
##    discf.rvs(loc=0,scale=1)
##        - random variates
##
##    discf.pdf(x,loc=0,scale=1)
##        - probability density function
##
##    discf.cdf(x,loc=0,scale=1)
##        - cumulative density function
##
##    discf.sf(x,loc=0,scale=1)
##        - survival function (1-cdf --- sometimes more accurate)
##
##    discf.ppf(q,loc=0,scale=1)
##        - percent point function (inverse of cdf --- percentiles)
##
##    discf.isf(q,loc=0,scale=1)
##        - inverse survival function (inverse of sf)
##
##    discf.stats(loc=0,scale=1,moments='mv')
##        - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')
##
##    discf.entropy(loc=0,scale=1)
##        - (differential) entropy of the RV.
##
##    Alternatively, the object may be called (as a function) to fix
##       the shape, location, and scale parameters returning a
##       "frozen" continuous RV object:
##
##    myrv = discf(loc=0,scale=1)
##        - frozen RV object with the same methods but holding the
##            given shape, location, and scale fixed
##    
##distribution of discount factor y=1/(1+x)) with x N(0,0.1**2)
##
##cdf for [0.95,1.0,1.1]: [ 0.48950273  0.69146246  0.92059586]
##pdf for [0.95,1.0,1.1]: [ 4.41888273  3.52065327  1.22171744]
##rvs: [ 0.83915166  1.05916973  0.92895054  0.97653308  0.88997135]
##stats: (array(0.9612657841613822), array(0.0088754849141799985))
##stats kurtosis, skew:  (array(0.61482268025485831), array(0.78380821248995103))
##median: 0.952380952381
##sample stats:               0.962079869848 0.00884562090518
##std norm sample stats:     0.961089776974 0.00874620747261
##transf. norm sample stats:  0.961224369048 0.00883332533467
##
##Results for lognormal
##---------------------
##cdf for [2.0,2.5,3.0,3.5]:       [ 0.7558914   0.82024279  0.86403139  0.89485401]
##scipy cdf for [2.0,2.5,3.0,3.5]: [ 0.7558914   0.82024279  0.86403139  0.89485401]
##pdf for [2.0,2.5,3.0,3.5]:      [ 0.15687402  0.10487107  0.07272826  0.05200533]
##scipy pdf for [2.0,2.5,3.0,3.5]: [ 0.15687402  0.10487107  0.07272826  0.05200533]
##stats:       (array(1.6487212707089971), array(4.6707742108633177))
##scipy stats: (array(1.6487212707001282), array(4.670774270471604))
##rvs: [ 0.45054343  0.17302502  0.15131355  0.74181241  0.30077315]
##Exp transformed normal continuous random variable.
##
##    Continuous random variables are defined from a standard form chosen
##    for simplicity of representation.  The standard form may require
##    some shape parameters to complete its specification.  The distributions
##    also take optional location and scale parameters using loc= and scale=
##    keywords (defaults: loc=0, scale=1)
##
##    These shape, scale, and location parameters can be passed to any of the
##    methods of the RV object such as the following:
##
##    lnnorm.rvs(loc=0,scale=1)
##        - random variates
##
##    lnnorm.pdf(x,loc=0,scale=1)
##        - probability density function
##
##    lnnorm.cdf(x,loc=0,scale=1)
##        - cumulative density function
##
##    lnnorm.sf(x,loc=0,scale=1)
##        - survival function (1-cdf --- sometimes more accurate)
##
##    lnnorm.ppf(q,loc=0,scale=1)
##        - percent point function (inverse of cdf --- percentiles)
##
##    lnnorm.isf(q,loc=0,scale=1)
##        - inverse survival function (inverse of sf)
##
##    lnnorm.stats(loc=0,scale=1,moments='mv')
##        - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')
##
##    lnnorm.entropy(loc=0,scale=1)
##        - (differential) entropy of the RV.
##
##    Alternatively, the object may be called (as a function) to fix
##       the shape, location, and scale parameters returning a
##       "frozen" continuous RV object:
##
##    myrv = lnnorm(loc=0,scale=1)
##        - frozen RV object with the same methods but holding the
##            given shape, location, and scale fixed
##    
##distribution of y = exp(x), with x standard normal
##None
##
##Results for idnormal
##--------------------
##2.08137521949e-023
##2.08137521949e-023
##(array(100.0), array(99.999999999476046))
##(array(100.0), array(100.0))
##[ 108.95135206  105.97192526   96.56950463  103.55316046   95.69133148]
##99.9529496989 102.460238876
##
##Results for expgamma
##--------------------
##0.00052773949978
##0.00052773949978
##0.00906529986325
##0.00906529986325
##cdf for [2.0,2.5,3.0,3.5]:       [ 0.21103986  0.77318778  0.99524757  0.99999926]
##scipy cdf for [2.0,2.5,3.0,3.5]: [ 0.21103986  0.77318778  0.99524757  0.99999926]
##pdf for [2.0,2.5,3.0,3.5]:      [  8.26228773e-01   1.01580211e+00   5.57228823e-02   1.81420198e-05]
##scipy pdf for [2.0,2.5,3.0,3.5]: [  8.26228773e-01   1.01580211e+00   5.57228823e-02   1.81420259e-05]
##
##
## the rest is not so good
## '''
