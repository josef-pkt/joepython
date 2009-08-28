

import numpy as np
from numpy import linalg as npla
from scipy import stats, optimize




def mlogit(y, x=None, z=None, beta=None, gamma=None):
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
    #param beta (nx,1) , gamma: nx,nk
    
    # use loop for now
    score = np.zeros((nobs,nk))
    #print score.shape
    for i in range(1,nk):
        score[:,i:i+1] = np.dot(x,beta) + np.dot(z,gamma[:,i-1:i])
    score2 = np.zeros((nobs,nk))
    score2[:,1:] = np.dot(x,beta) + np.dot(z,gamma)
    #print np.max(score-score2)
    probs = np.exp(score)
    #den = 1 + np.sum(probs[:,1:],1)[:,np.newaxis]
    den = np.sum(probs,1)[:,np.newaxis]
    #probs[:,:1] = 1 + np.sum(probs[:,1:],1)[:,np.newaxis]
    probs = probs/den
    loglikecontr1 = 1#np.log(pr[np.arange(nobs),y[:,0]])[:,np.newaxis]  #select column by realization of cat
    loglikecontr2 = score[np.arange(nobs),y[:,0]][:,np.newaxis] - np.log(den)
    return score, probs, loglikecontr1, loglikecontr2, -loglikecontr2.sum()

nobs = 200
nx = 2
nk = 3
nz = 2
y = np.mod(np.arange(nobs),nk)[:,np.newaxis]
#x = np.ones((nobs,1))*np.arange(1,nx+1)[np.newaxis,:]
x = y * np.arange(1,nx+1)[np.newaxis,:]
z = -2 + np.ones((nobs,1))*np.arange(1,nz+1)[np.newaxis,:]
x = np.random.randn(nobs, nx)
z = np.random.randn(nobs, nz)
beta = 0.05*np.ones((nx,1))
gamma = 0.01*np.ones((nz,nk-1))*np.arange(1,nk)[np.newaxis,:] #*np.arange
sc, pr, ll1, ll2, loglike = mlogit(y, x, z, beta, gamma)
nz=1
z2 = np.ones((nobs,1))
gamma2 = 0.01*np.ones((nz,nk-1))
             
sc2, pr2, ll12, ll22, loglike2 = mlogit(y, ey, z2, ey, gamma2)

rvsunif = np.random.rand(nobs,1)
yrvs = (rvsunif<np.cumsum(pr,axis=1)).argmax(1)[:,np.newaxis]
y = yrvs

def mlogitloglike(param,x,y,z):
    nx = x.shape[1]
    nz = z.shape[1]
    beta = param[:nx][:,np.newaxis]
    gamma = param[nx:].reshape(nz,-1)
    #print beta.shape
    #print gamma.shape
    sc, pr, ll1, ll2, loglike = mlogit(y, x, z, beta, gamma)
    return loglike

param = np.hstack((beta.ravel(),gamma.ravel()))
loglikea = mlogitloglike(param,x,y,z)

param0 = param*1.05

res1 = optimize.fmin_bfgs(mlogitloglike, param0, args = (x,y,z))
res2 = optimize.fmin(mlogitloglike, param0, args = (x,y,z))

