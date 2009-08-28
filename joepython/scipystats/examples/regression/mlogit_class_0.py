'''Estimate and Simulate a multinomial logit model

Initially, I looked at jplv, but I didn't like mlogit and multilogit.
This is programmed from scratch using the more general model

P(y=j) = exp(x*beta + z*gamma_j)/den  for j = 1, ... nk
p(y=0) = 1/den
where den = sum_{j=1}^{nk} exp(x*beta + z*gamma_j)

This should work if either x or z are empty. A category specific constant
can be included as column of ones in z.

Results
-------
* Estimated loglikelhood is close to the loglikelihood of the data generating model.
* t-statistics for deviation from true parameters are sometimes large
* For large sample, nobs = 5000, the estimates are close to the true parameters.
* If only the category specific constant is included then the loglikelihood is
  very close to the analytical loglikelihood. In this case bfgs return negative
  variance in the inverse Hessian, which needs a correction or alternative estimate

todo
----
* refactor: class structure and methods
* remove duplicate calculations, that I used for verification
* summary, result statistics including confusion matrix,
  missing statistics when using fmin
* add a check whether model is identified, how?
* add analytical gradient and maybe Hessian
* run Monte Carlo to check estimator
* add examples as test (added z-constant only)
* possible extension: wrapper for stepwise inclusion of regressors (?)
* check for extensibility to panel data

Note:
-----
scipy.optimize is missing a maxlik optimizer, i.e. uses jacobian, objective function
returns likelihood contributions instead of sum. Check with matlab and gauss.

References
----------
Greene
Judge, Griffiths, Hill (?), Luetkepoll and Lee

Author: Josef Perktold
License: New BSD
'''

import numpy as np
from numpy import linalg as npla
from scipy import stats, optimize


ey2 = np.empty((0,0))
ey = np.array([])   #Caution: uses np.dot(ey,ey) == 0

class MultiNomialLogit(object):
    def __init__(self, x=None, z=None):
        nobs = 0
        #nk = y.max() + 1
        if x is None or x.size == 0:
            nx = 0
            x = ey
        else:
            nobs, nx = x.shape
        if z is None or z.size == 0:
            nz = 0
            z = ey
        else:
            nobsz, nz = z.shape
            if nobs>0 and nobsz!=nobs:
                raise ValueError, 'x and z have unequal number of observations'
            else:
                nobs = nobsz
        self.nobs = nobs
        self.nx = nx
        self.nz = nz
        self.x = x
        self.z = z
    
    def mlogit(self, y, x=None, z=None, beta=None, gamma=None):
        '''does currently all the calculation for the multinomial logit model

        parameters: beta (nx,1), gamma (nz,nk)

        self contained function: does not use side information, no side effects
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

##        print 'beta.shape, gamma.shape', beta.shape, gamma.shape
##        print 'y.shape, x.shape, z.shape', y.shape, x.shape, z.shape
        score = np.zeros((nobs,nk))
##        print 'score.shape', score.shape
##        print '(np.dot(x,beta) + np.dot(z,gamma)).shape', (np.dot(x,beta) + np.dot(z,gamma)).shape
        score[:,1:] = np.dot(x,beta) + np.dot(z,gamma)
        probs = np.exp(score)
        den = np.sum(probs,1)[:,np.newaxis]
        probs = probs/den
        loglikecontr = score[np.arange(nobs),y[:,0]][:,np.newaxis] - np.log(den)
        
        return score, probs, loglikecontr, -loglikecontr.sum()

    def estimate(self, y, beta0=None, gamma0=None, optimizer='fmin'):
        '''
        
        Notes:
        * fmin needs good starting values or increase maximum number of function
          evaluations
        * bfgs reports loss in precision

        
        '''
        # check cases with empty x or z for beta0, gamma0

        self.optimizer = optimizer
        self.y = y
        self.nk = y.max() + 1
        if beta0 is None:
            beta0 = np.zeros((self.nx, 1))
        if gamma0 is None:
            gamma0 = np.zeros((self.nz, self.nk-1))
            
        param0 = np.hstack((beta0.ravel(),gamma0.ravel()))
        if optimizer == 'fmin':
            self.res = optimize.fmin(self.mlogitloglike, param0, args=(y,self.x,self.z))
            paramest = self.res
        else:
            self.res = optimize.fmin_bfgs(self.mlogitloglike, param0, args=(y,self.x,self.z),
                                  full_output=1)
            paramest = self.res[0]

        # calculate statistics
        print self.nz
        if self.nx > 0:
            self.betaest = paramest[:self.nx][:,np.newaxis]
        else:
            self.betaest = ey
        if self.nz > 0:
##            print 'I m here',self.nz
##            print (paramest[nx:]).reshape(nz,-1)
##            print (paramest[nx:])[np.newaxis,:].reshape(self.nz,-1)
            self.gammaest = paramest[self.nx:].reshape(self.nz,-1)
##            if self.gammaest.shape[0]<2:
##                raise
        else:
            self.gammaest = ey
        return self.res


    def mlogitloglike(self, param, y, x, z):
        '''wrapper to get negative loglikelihood to feed to optimization procedure
        assumes param is one dimensional, reshapes gamma to 2d

        does not yet allow x or y to be empty or none
        
        is self contained function: does not use side information except call to self.mlogit,
            no side effects
        
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
        sc, pr, ll2, loglike = self.mlogit(y, x, z, beta, gamma)
        return loglike

    def rvs(self, x=None, z=None, beta=None, gamma=None):
        if x is None and z is None:
            x = self.x
            z = self.z
            nobs = self.nobs
        elif z is None:
            nobs = x.shape[0]
        else:
            nobs = z.shape[0]
            
        if beta is None and gamma is None:
            beta = self.betaest
            gamma = self.gammaest
        nk = gamma.shape[1] + 1
        #arbitrary initialization, with nk distinct values
        y = np.mod(np.arange(nobs),nk)[:,np.newaxis]
        sc, pr, ll2, loglike = self.mlogit(y, x, z, beta, gamma)
        rvsunif = np.random.rand(nobs,1)
        yrvs = (rvsunif<np.cumsum(pr,axis=1)).argmax(1)[:,np.newaxis]
        sc, pr, ll2, loglike = self.mlogit(yrvs, x, z, beta, gamma)
        return yrvs, pr, loglike

    def summary(self, param=None, loglike_rvs=None):

        sc, pr, ll2, loglike = self.mlogit(y, x=self.x, z=self.z,
                                      beta=self.betaest, gamma=self.gammaest)
        self.yest = np.argmax(pr, axis=1)[:,np.newaxis]
        self.pr = pr
        if self.optimizer == 'fmin':
            if param is None:
                param = np.zeros(self.res)
            print 'results'
            print 'true ', param
            print 'bfgs ', self.res
            print 'fmerr', self.res-param
            # replace this with full output of optimizer if availabl
            self.nloglik = self.mlogitloglike(self.res, self.y, self.x, self.z)
        elif self.optimizer == 'bfgs':
            if param is None:
                param = np.zeros(self.res[0])
            print 'results'
            print 'true ', param
            print 'bfgs ', self.res[0]
            print 'bferr', self.res[0]-param
            print 'bfstd', np.sqrt(np.diag(self.res[3]))
            print 'bf_t ', (self.res[0]-param)/np.sqrt(np.diag(self.res[3]))
            print 'true, est. negloglike ', loglike_rvs, self.res[1]
            self.nloglik = self.res[1]
        else:
            print 'estimator not yet defined'

        # basic specification testing;
        dummyvar = (self.y == np.arange(nk)).astype(int)
        self.dummyvar = dummyvar
        p = np.sum(dummyvar,axis=0)/float(self.nobs)
        self.freq = p
        lnlr = self.nobs*np.sum(p*np.log(p)) # restricted log-likelihood: intercepts only
        self.lnlr = lnlr
        self.lratio = -2*(lnlr + self.nloglik) # note: positive vs negative loglikelihood
        self.rsqr = 1 - (-self.nloglik / lnlr) # McFadden pseudo-R^2
        print 'lnlr', lnlr
        print 'lratio', self.lratio
        print 'rsqr', self.rsqr

        self.dummyvarhat = (self.yest == np.arange(self.nk)).astype(int)
        conf = np.dot(dummyvar.T, self.dummyvarhat)
        self.conf = conf
        self.confcol = conf / conf.sum(0,dtype=float)
        self.confrow = conf / (conf.sum(1,dtype=float)[:,np.newaxis])

        # describtive statistics for categories, groups
        if self.x.size > 0:
            m1 = np.dot(self.x.T,dummyvar)/dummyvar.sum(0,float) # category/group mean
            mv1 = np.dot(dummyvar,m1.T) # category/group means as array in shape of x
            mdevmg = self.x - mv1  # deviation from category/group mean
            #assert np.all(np.abs(np.dot(mdevmg.T,dummyvar))<1e-14)
                        
        


#generate some test examples
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
beta = 1.05*np.ones((nx,1))
gamma = 1.01*np.ones((nz,nk-1))*np.arange(1,nk)[np.newaxis,:] #*np.arange
beta = np.arange(nx)[:,np.newaxis] -1
gamma = 0.5+np.eye(2)
#sc, pr, ll2, loglike = mlogit(y, x, z, beta, gamma)
sc, pr, ll2, loglike = MultiNomialLogit(x,z).mlogit(y, x, z, beta, gamma)
nz=1
z2 = np.ones((nobs,1))
gamma2 = 0.01*np.ones((nz,nk-1))
             
#sc2, pr2, ll22, loglike2 = mlogit(y, ey, z2, ey, gamma2)

# generate a sample, (nobs,1), based on probabilities given by pr, (nobs,nk).
rvsunif = np.random.rand(nobs,1)
yrvs = (rvsunif<np.cumsum(pr,axis=1)).argmax(1)[:,np.newaxis]
y = yrvs

y_rvs, pr_rvs, loglike_rvs = MultiNomialLogit(x,z).rvs(beta=beta, gamma=gamma)

param = np.hstack((beta.ravel(),gamma.ravel()))
#loglikea = mlogitloglike(param, y,x,z)

param0 = param*0.9#1.05
param0 = param0[:nx+nz*nk+1]
param0 = np.zeros(nx+nz*nk+1)
#res1 = optimize.fmin_bfgs(mlogitloglike, param0, args=(y,x,z), full_output=1)
#res2 = optimize.fmin(mlogitloglike, param0, args=(y,x,z))
mnl = MultiNomialLogit(x,z)
print 'estimating with fmin'
res2 = mnl.estimate(y, beta0=beta*0.9, gamma0=gamma*0.9, optimizer='fmin')
print 'estimating with bfgs'
res1 = mnl.estimate(y, optimizer='bfgs')

print 'results'
print 'true ', param
print 'fmin ', res2
print 'bfgs ', res1[0]
print 'fmerr', res2-param
print 'bferr', res1[0]-param
print 'bfstd', np.sqrt(np.diag(res1[3]))
print 'bf_t ', (res1[0]-param)/np.sqrt(np.diag(res1[3]))
# bf_t: is there a /nobs missing? I guess not

##res2 = optimize.fmin(mlogitloglike, np.zeros(2), args=(y,x,ey))
##
###this call has wrong parameter length but doesn't complain:
##optimize.fmin(mlogitloglike, np.zeros((2,2)), args=(y,x,ey))
###this looks good:
##optimize.fmin(mlogitloglike, np.zeros((2,2)), args=(y,ey,z))
###this looks ok:
##optimize.fmin(mlogitloglike, np.zeros(2), args=(y,x,ey))

'''Example: given x,z simulate model and then estimate
'''
mnl2 = MultiNomialLogit(x,z)
y_rvs, pr_rvs, loglike_rvs = mnl2.rvs(beta=beta, gamma=gamma)
# use new instance to avoid potential contamination
mnl3 = MultiNomialLogit(x,z)
res3 = mnl3.estimate(y_rvs, beta0=beta*0.1, gamma0=gamma*0.1, optimizer='bfgs')
##print 'results'
##print 'true ', param
##print 'bfgs ', res3[0]
##print 'bferr', res3[0]-param
##print 'bfstd', np.sqrt(np.diag(res3[3]))
##print 'bf_t ', (res3[0]-param)/np.sqrt(np.diag(res3[3]))
##print 'true, est. negloglike ', loglike_rvs, res3[1]
mnl3.summary(param, loglike_rvs)


'''Example: constant in z only: simulate model and then estimate
'''
x = ey
beta = ey
z = np.ones((nobs,1))
gamma = np.array([[-0.5, 1]])
y_rvs, pr_rvs, loglike_rvs = MultiNomialLogit(x,z).rvs(beta=beta, gamma=gamma)
# use new instance to avoid potential contamination
mnl4 = MultiNomialLogit(x,z)
res4 = mnl4.estimate(y_rvs, beta0=beta*0.1, gamma0=gamma*0.1, optimizer='bfgs')
mnl4.summary(gamma.ravel(), loglike_rvs)
###note: this reproduces the direct estimator for constant in z only, e.g.
##true, est. negloglike  1001.26967064 1000.02755653
##lnlr -1000.02755652
##lratio -2.11114183912e-008
##rsqr -1.05553343843e-011


''' result dummy vars

>>> np.sum(yrvs == np.arange(nk),axis=0)
array([70, 70, 60])
>>> np.sum(yrvs == 0,axis=0)
array([70])
>>> np.sum(yrvs == 1,axis=0)
array([70])
>>> np.sum(yrvs == 2,axis=0)
array([60])
>>> yrvs[:10,:] == np.arange(nk)
array([[ True, False, False],
       [False, False,  True],
       [ True, False, False],
       [False,  True, False],
       [False,  True, False],
       [False,  True, False],
       [ True, False, False],
       [False,  True, False],
       [ True, False, False],
       [False, False,  True]], dtype=bool)
>>> dummyvar = (yrvs == np.arange(nk)).astype(int)
>>> dummyvar[:10,:]
array([[1, 0, 0],
       [0, 0, 1],
       [1, 0, 0],
       [0, 1, 0],
       [0, 1, 0],
       [0, 1, 0],
       [1, 0, 0],
       [0, 1, 0],
       [1, 0, 0],
       [0, 0, 1]])
'''


