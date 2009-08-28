'''Estimate and Simulate a multinomial logit model

Initially, I looked at jplv, but I didn't like mlogit and multilogit.
This is programmed from scratch using the more general model

P(y=j) = exp(x*beta + z*gamma_j)/den  for j = 1, ... nk
p(y=0) = 1/den
where den = sum_{j=1}^{nk} exp(x*beta + z*gamma_j)

This should work if either x or z are empty. A category specific constant
can be included as column of ones in z.

For large sample, nobs = 5000, the estimates are close to the true parameters.

todo
----
* refactor: class structure and methods
* remove duplicate calculations, that I used for verification
* summary, result statistics including confusion matrix
* add a check whether model is identified, how?
* add analytical gradient and maybe Hessian
* run Monte Carlo to check estimator
* add example as test
* possible extension: wrapper for stepwise inclusion of regressors (?)
* check for extensibility to panel data

References
----------
Greene
Judge, Griffiths, Hill (?), Luetkepoll and Lee


'''

import numpy as np
from numpy import linalg as npla
from scipy import stats, optimize


ey2 = np.empty((0,0))
ey = np.array([])

def mlogit(y, x=None, z=None, beta=None, gamma=None):
    '''does currently all the calculation for the multinomial logit model

    parameters: beta (nx,1), gamma (nz,nk)

    '''
    nobs = y.shape[0]
    #assume for now y is in 0,1,2,...,nk-1  i.e. range(nk)
    nk = y.max() + 1
    if x is None or x.size == 0:
        nx = 0
    else:
        nx = x.shape[1]
    if z is None or z.size == 0:
        nz = 0
    else:
        nz = z.shape[1]
    #param beta (nx,1) , gamma: nz,nk
##    #not here:
##    if beta is None:
##        beta = np.zeros((nx, 1))
##    if gamma is None:
##        gamma = np.zeros((nz, nk))
    
    # use loop for now
##    score = np.zeros((nobs,nk))
##    #print score.shape
##    for i in range(1,nk):
##        score[:,i:i+1] = np.dot(x,beta) + np.dot(z,gamma[:,i-1:i])
    #same in one matrix command
    score2 = np.zeros((nobs,nk))
    score2[:,1:] = np.dot(x,beta) + np.dot(z,gamma)
    #print np.max(score-score2)
    probs = np.exp(score2)
    #den = 1 + np.sum(probs[:,1:],1)[:,np.newaxis]
    den = np.sum(probs,1)[:,np.newaxis]
    #probs[:,:1] = 1 + np.sum(probs[:,1:],1)[:,np.newaxis]
    probs = probs/den
    # For the latest version of the examples I get a shape mismatch:
#    loglikecontr1 = 1#np.log(pr[np.arange(nobs),y[:,0]])[:,np.newaxis]  #select column by realization of cat
    loglikecontr2 = score2[np.arange(nobs),y[:,0]][:,np.newaxis] - np.log(den)
    #return score, probs, loglikecontr1, loglikecontr2, -loglikecontr2.sum()
    return score2, probs, loglikecontr2, -loglikecontr2.sum()


#generate some test examples
nobs = 1000
nx = 2
nk = 3
nz = 2
y = np.mod(np.arange(nobs),nk)[:,np.newaxis]
#x = np.ones((nobs,1))*np.arange(1,nx+1)[np.newaxis,:]
x = y * np.arange(1,nx+1)[np.newaxis,:]
z = -2 + np.ones((nobs,1))*np.arange(1,nz+1)[np.newaxis,:]
x = np.random.randn(nobs, nx)
z = np.random.randn(nobs, nz)
beta = 1.05*np.ones((nx,1))
gamma = 1.01*np.ones((nz,nk-1))*np.arange(1,nk)[np.newaxis,:] #*np.arange
beta = np.arange(nx)[:,np.newaxis] -1
gamma = 0.5+np.eye(2)
sc, pr, ll2, loglike = mlogit(y, x, z, beta, gamma)
nz=1
z2 = np.ones((nobs,1))
gamma2 = 0.01*np.ones((nz,nk-1))
             
sc2, pr2, ll22, loglike2 = mlogit(y, ey, z2, ey, gamma2)

# generate a sample, (nobs,1), based on probabilities given by pr, (nobs,nk).
rvsunif = np.random.rand(nobs,1)
yrvs = (rvsunif<np.cumsum(pr,axis=1)).argmax(1)[:,np.newaxis]
y = yrvs

def mlogitloglike(param, y, x, z):
    '''wrapper to get negative loglikelihood to feed to optimization procedure
    assumes param is one dimensional, reshapes gamma to 2d

    does not yet allow x or y to be empty or none
    '''
    
    
    if x.size > 0:
        nx = x.shape[1]
        beta = param[:nx][:,np.newaxis]
    else:
        nx = 0
        beta = ey
    if z.size > 0:
        nz = z.shape[1]
        gamma = param[nx:].reshape(nz,-1)
    else:
        nz = 0
        gamma = ey
    #print beta.shape
    #print gamma.shape
    sc, pr, ll2, loglike = mlogit(y, x, z, beta, gamma)
    return loglike

param = np.hstack((beta.ravel(),gamma.ravel()))
loglikea = mlogitloglike(param, y,x,z)

param0 = param*0.9#1.05
param0 = param0[:nx+nz*nk+1]
param0 = np.zeros(nx+nz*nk+1)
res1 = optimize.fmin_bfgs(mlogitloglike, param0, args=(y,x,z), full_output=1)
res2 = optimize.fmin(mlogitloglike, param0, args=(y,x,z))

print 'results'
print 'true ', param
print 'fmin ', res2
print 'bfgs ', res1[0]
print 'fmerr', res2-param
print 'bferr', res1[0]-param
print 'bfstd', np.sqrt(np.diag(res1[3]))
print 'bf_t ', (res1[0]-param)/np.sqrt(np.diag(res1[3]))
# bf_t: is there a /nobs missing? I guess not

res2 = optimize.fmin(mlogitloglike, np.zeros(2), args=(y,x,ey))

#this call has wrong parameter length but doesn't complain:
optimize.fmin(mlogitloglike, np.zeros((2,2)), args=(y,x,ey))
#this looks good:
optimize.fmin(mlogitloglike, np.zeros((2,2)), args=(y,ey,z))
#this looks ok:
optimize.fmin(mlogitloglike, np.zeros(2), args=(y,x,ey))
