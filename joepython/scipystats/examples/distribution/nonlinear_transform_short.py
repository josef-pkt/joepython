'''univariate distribution of a non-linear monotonic transformation of a
random variable

'''
from scipy import stats
from scipy.stats import distributions
import numpy as np

class ExpTransf_gen(distributions.rv_continuous):
    def __init__(self, kls, *args, **kwargs):
        #print args
        #print kwargs
        #explicit for self.__dict__.update(kwargs)
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'numargs' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0
        super(ExpTransf_gen,self).__init__(a=0, name = 'Log transformed distribution')
        self.kls = kls
    def _cdf(self,x,*args):
        #print args
        return self.kls._cdf(np.log(x),*args)
    def _ppf(self, q, *args):
        return np.exp(self.kls._ppf(q,*args))

class LogTransf_gen(distributions.rv_continuous):
    def __init__(self, kls, *args, **kwargs):
        #explicit for self.__dict__.update(kwargs)
        if 'numargs' in kwargs:
            self.numargs = kwargs['numargs']
        else:
            self.numargs = 1
        if 'numargs' in kwargs:
            name = kwargs['name']
        else:
            name = 'Log transformed distribution'
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0

        super(LogTransf_gen,self).__init__(a=a, name = name)
        self.kls = kls
        
    def _cdf(self,x,*args):
        #print args
        return self.kls._cdf(np.exp(x),*args)
    def _ppf(self, q, *args):
        return np.log(self.kls._ppf(q,*args))
        

##lognormal = ExpTransf(a=0.0, xa=-10.0, name = 'Log transformed normal')
##print lognormal.cdf(1)
##print stats.lognorm.cdf(1,1)
##print lognormal.stats()
##print stats.lognorm.stats(1)
##print lognormal.rvs(size=10)

print 'Results for lognormal'
lognormalg = ExpTransf_gen(stats.norm, a=0, name = 'Log transformed normal general')
print lognormalg.cdf(1)
print stats.lognorm.cdf(1,1)
print lognormalg.stats()
print stats.lognorm.stats(1)
print lognormalg.rvs(size=5)

##print 'Results for loggamma'
##loggammag = ExpTransf_gen(stats.gamma)
##print loggammag._cdf(1,10)
##print stats.loggamma.cdf(1,10)

print 'Results for expgamma'
loggammaexpg = LogTransf_gen(stats.gamma)
print loggammaexpg._cdf(1,10)
print stats.loggamma.cdf(1,10)
print loggammaexpg._cdf(2,15)
print stats.loggamma.cdf(2,15)


# this requires change in scipy.stats.distribution
#print loggammaexpg.cdf(1,10)

print 'Results for loglaplace'
loglaplaceg = LogTransf_gen(stats.laplace)
print loglaplaceg._cdf(2,10)
print stats.loglaplace.cdf(2,10)
loglaplaceexpg = ExpTransf_gen(stats.laplace)
print loglaplaceexpg._cdf(2,10)
