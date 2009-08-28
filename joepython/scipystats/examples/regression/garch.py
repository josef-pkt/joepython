''' toy implementation of GARCH

'''

import numpy as np
from scipy import optimize

def garch_nlogl(param, u):
    '''loglikelihood for garch(1,1)
    reparameterized to force positive variance
    '''
    omega, alpha, beta = param**2
    n = len(u)
    v = np.ones(n)
    for i in range(1,n):
        #v[i] = omega + alpha*u[i-1] + beta*np.abs(v[i-1])
        v[i] = omega + alpha*u[i-1]**2 + beta*v[i-1]
    nloglike = 0.5*((np.log(v) +  u**2 / (v)).sum() + n/np.log(2*np.pi))
    return nloglike

def egarch_nlogl(param, u):
    omega, alpha, beta = param
    v = np.ones(len(u))
    for i in range(1,len(u)):
        #v[i] = omega + alpha*u[i-1] + beta*np.abs(v[i-1])
        v[i] = np.exp(omega + alpha*u[i-1]**2 + beta*np.log(v[i-1]))
    #nloglike = -(-np.log(v) -  u**2 / (1e-10 +np.abs(v))).sum()
    nloglike = -(-np.log(np.sqrt(v)) -  u**2 / 2.0 / (1e-6+v)).sum()
    return nloglike

def mygarch_nlogl(param, u):
    '''loglikelihood for garch(1,1)
    reparameterized to force positive variance
    '''
    omega, alpha, beta = param
    n=len(u)
    v = np.ones(len(u))
    for i in range(1,len(u)):
        #v[i] = omega + alpha*u[i-1] + beta*np.abs(v[i-1])
        v[i] = np.sqrt(np.abs(omega + alpha*u[i-1]**2 + beta*v[i-1]**2))
    nloglike = -0.5*(-np.log(np.sqrt(2*np.pi)*v) -  u**2 / 2.0 / (v**2)).sum()
    return nloglike

u = np.random.randn(500)
u[1:] = u[1:]*np.abs(u[:-1])


result = optimize.fmin(garch_nlogl, np.array([1.0, 0.0, 0.0]), args=(u,), full_output=1)
#result2 = optimize.fmin_bfgs(garch_nlogl, [1.0, 0.0, 0.0], args=(u,), maxiter=2)
#result2 = optimize.fmin_powell(garch_nlogl, np.array([1.0, 0.0, 0.0]), args=(u,), full_output=1)
result2 = optimize.fmin_powell(garch_nlogl, result[0], args=(u,), full_output=1)
result2a = optimize.fmin_powell(mygarch_nlogl, result[0], args=(u,), full_output=1)
#result3 = optimize.fmin_powell(egarch_nlogl, np.array([1.0, 0.0, 0.0]), args=(u,), full_output=1)
print result
print result2
print result2a
#print result3
