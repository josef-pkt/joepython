
import numpy as np
from scipy import stats, special, optimize
from scipy.stats import distributions


rvsg = stats.gamma.rvs(2.5,scale=20,size=1000)
print rvsg.min()
print stats.gamma.fit(rvsg)

class gamma_gen(distributions.rv_continuous):
    def _rvs(self, a):
        return mtrand.standard_gamma(a, self._size)
    def _pdf(self, x, a):
        return x**(a-1)*np.exp(-x)/special.gamma(a)
    def _cdf(self, x, a):
        return special.gammainc(a, x)
    def _ppf(self, q, a):
        return special.gammaincinv(a,q)
    def _stats(self, a):
        return a, a, 2.0/np.sqrt(a), 6.0/a
    def _entropy(self, a):
        return special.psi(a)*(1-a) + 1 + special.gammaln(a)

    def _nnlf(self, x, *args):
        return -np.sum(np.log(self._pdf(x, *args)),axis=0)

    def nnlf_floc(self, theta, x):
        # - sum (log pdf(x, theta),axis=0)
        #   where theta are the parameters (including loc and scale)
        #
        try:
            #loc = theta[-2]
            loc = 0
            scale = theta[-1]
            args = tuple(theta[:-1])
        except IndexError:
            raise ValueError, "Not enough input arguments."
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = np.array((x-loc) / scale)
        cond0 = (x <= self.a) | (x >= self.b)
        if (np.any(cond0)):
            return inf
        else:
            N = len(x)
            return self._nnlf(x, *args) + N*np.log(scale)

    def fit_floc(self, data, *args, **kwds):
        loc0, scale0 = map(kwds.get, ['loc', 'scale'],[0.0, 1.0])
        Narg = len(args)
        if Narg != self.numargs:
            if Narg > self.numargs:
                raise ValueError, "Too many input arguments."
            else:
                args += (1.0,)*(self.numargs-Narg)
        # location and scale are at the end
        x0 = args + (scale0,)
        #print x0
        return optimize.fmin(self.nnlf_floc,x0,args=(np.ravel(data),),disp=0)

    
gamma = gamma_gen(a=0.0,name='gamma',longname='A gamma',
                  shapes='a',extradoc="""

Gamma distribution

For a = integer, this is the Erlang distribution, and for a=1 it is the
exponential distribution.

gamma.pdf(x,a) = x**(a-1)*exp(-x)/gamma(a)
for x >= 0, a > 0.
"""
                  )

#rvsg = stats.gamma.rvs(2.5,scale=20,size=10)

print gamma.fit(rvsg)
print gamma.fit_floc(rvsg)

result = []

niter = 1000
ssize = 100
for ii in range(niter):
    rvsg = stats.gamma.rvs(2.5,scale=20,size=ssize)
    result.append(np.hstack([gamma.fit_floc(rvsg), stats.gamma.fit(rvsg)]))

ptrue = np.array([2.5,20.0,2.5,0.0,20.0])
resarr = np.array(result)
print ' sample size = %d,  number of iterations = %d' % (niter, ssize)
print '         with fixed location        with estimated location'
print '          shape       scale       shape       location    scale'
bias = np.mean((resarr - ptrue), axis=0)
errvar = np.var((resarr - ptrue), axis=0)
maxabs = np.max(np.abs(resarr - ptrue), axis=0)
mad = np.mean(np.abs(resarr - ptrue), axis=0)
mse = np.mean((resarr - ptrue)**2, axis=0)
print 'bias  ', bias
print 'errvar', errvar
print 'mse   ', mse
print 'maxabs', maxabs
print 'mad   ', mad
