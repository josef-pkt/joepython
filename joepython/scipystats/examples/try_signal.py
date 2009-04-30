'''ARMA estimation: using signal.lfilter and optimize.leastsquares

seems to work
'''

import numpy as np
from scipy import signal, optimize



#r,p,k = signal.residue([1.0, 0],[1, -0.5])

'''
>>> signal.lfilter([1, -0.5],[1.0],np.ones(10))
array([ 1. ,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5])
>>> signal.lfilter([1, -0.5],[1.0],np.ones(10),zi=np.array([ -0.5]))
(array([ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]), array([-0.5]))
>>> signal.lfiltic([1, -0.5],[1.0] ,[0.5,00.5], x=[1.0,1.0])
array([-0.5])

>>> xe = np.zeros(10)
>>> xe[3] = 1
>>> signal.lfilter([1.0],[1, -0.5],xe)
array([ 0.      ,  0.      ,  0.      ,  1.      ,  0.5     ,  0.25    ,
        0.125   ,  0.0625  ,  0.03125 ,  0.015625])
'''

print signal.lfilter([1, -0.5],[1.0],np.ones(10))


rv = np.random.randn(11)
x = rv[1:] + 0.8*rv[-1:]

y = signal.lfilter([1, -0.8],[1.0],rv)

print signal.lfilter([1, -0.5],[1.0],x)
print signal.lfilter([1.0],[1, -0.8],x)

# Simulate AR(1)
#--------------
# ar * y = ma * eta
ar = [1, -0.8]
ma = [1.0]

# generate AR data
eta = 0.1 * np.random.randn(1000)
yar1 = signal.lfilter(ar, ma, eta)

etahat = signal.lfilter(ma, ar, y)
np.all(etahat == eta)

# find error for given filter on data
print 'AR(2)'
for rho in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.8, 0.81, 0.9]:
    etahatr = signal.lfilter(ma, [1, --rho], yar1)
    print rho,np.sum(etahatr*etahatr)
print 'AR(2)'
for rho2 in np.linspace(-0.4,0.4,9):
    etahatr = signal.lfilter(ma, [1, -0.8, -rho2], yar1)
    print rho2,np.sum(etahatr*etahatr)    

def errfn(rho):
    etahatr = signal.lfilter(ma, [1, -rho], yar1)
    #print rho,np.sum(etahatr*etahatr)
    return etahatr

def errssfn(rho):
    etahatr = signal.lfilter(ma, [1, -rho], yar1)
    return np.sum(etahatr*etahatr)


resultls = optimize.leastsq(errfn,[0.5])
print 'LS ARMA(1,0)', resultls

resultfmin = optimize.fmin(errssfn, 0.5)
print 'fminLS ARMA(1,0)', resultfmin


# Simulate MA(1)
#--------------
# ar * y = ma * eta
ar = [1.0]
ma = [1, 0.8]

# generate MA data
eta = 0.1 * np.random.randn(1000)
y = signal.lfilter(ar, ma, eta)

etahat = signal.lfilter(ma, ar, y)
np.all(etahat == eta)

# find error for given filter on data
print 'MA(1)'
for rho in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.8, 0.81, 0.9]:
    etahatr = signal.lfilter( [1, rho], ar, y)
    print rho,np.sum(etahatr*etahatr)

print 'MA(2)'
for rho2 in np.linspace(-0.4, 0.4, 9):
    etahatr = signal.lfilter([1, 0.8, rho2], ar, y)
    print rho2,np.sum(etahatr*etahatr)

def errfn(rho):
    etahatr = signal.lfilter([1, rho], ar, y)
    #print rho,np.sum(etahatr*etahatr)
    return etahatr

def errssfn(rho):
    etahatr = signal.lfilter([1, rho], ar, y)
    return np.sum(etahatr*etahatr)


resultls = optimize.leastsq(errfn,[0.5])
print 'LS ARMA(0,1)', resultls

resultfmin = optimize.fmin(errssfn, 0.5)
print 'fminLS ARMA(0,1)', resultfmin
    



# Simulate ARMA(1)
#-----------------
# ar * y = ma * eta
ar = [1.0, -0.8]
ma = [1.0,  0.5]

# generate ARMA data
eta = 0.1 * np.random.randn(100)
y = signal.lfilter(ar, ma, eta)

etahat = signal.lfilter(ma, ar, y)
np.all(etahat == eta)

# find error for given filter on data
##print 'MA(1)'
##for rho in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.79, 0.8, 0.81, 0.9]:
##    etahatr = signal.lfilter( [1, rhoe], [1, -rhoy], y)
##    print rho,np.sum(etahatr*etahatr)
##
##print 'MA(2)'
##for rho2 in np.linspace(-0.4, 0.4, 9):
##    etahatr = signal.lfilter([1, 0.8, rho2], ar, y)
##    print rho2,np.sum(etahatr*etahatr)

def errfn(rho):
    rhoy, rhoe = rho
    etahatr = signal.lfilter([1, rhoe], [1, -rhoy], y)
    #print rho,np.sum(etahatr*etahatr)
    return etahatr

def errssfn(rho):
    rhoy, rhoe = rho
    etahatr = signal.lfilter([1, rhoe], [1, -rhoy], y)
    return np.sum(etahatr*etahatr)


resultls = optimize.leastsq(errfn,[0.5, 0.5])
print 'LS ARMA(1,1)', resultls

resultfmin = optimize.fmin(errssfn, [0.5, 0.5])
print 'fminLS ARMA(1,1)', resultfmin


class ARIMA(object):
    def __init__(self):
        pass
    def estimate(self,x,p,q, rhoy0=None, rhoe0=None):
        def errfn( rho):
            #rhoy, rhoe = rho
            rhoy = np.concatenate(([1], -rho[:p]))
            rhoe = np.concatenate(([1],  rho[p:]))
            etahatr = signal.lfilter(rhoe, -rhoy, x)
            #print rho,np.sum(etahatr*etahatr)
            return etahatr
        
        if rhoy0 is None:
            rhoy0 = 0.5 * np.ones(p)
        if rhoe0 is None:
            rhoe0 = 0.5 * np.ones(q)
        rh, cov_x, infodict, mesg, ier = \
           optimize.leastsq(errfn, np.r_[rhoy0, rhoe0],full_output=True)
        self.rh = rh
        self.rhoy = np.concatenate(([1], -rh[:p]))
        self.rhoe = np.concatenate(([1],  rh[p:])) #rh[-q:])) doesnt work for q=0
        
        return rh, cov_x, infodict, mesg, ier
        
    def errfn(self, rho=None):
        #rhoy, rhoe = rho
        if not rho is None:
            rhoy = np.concatenate(([1], -rho[:p]))
            rhoe = np.concatenate(([1],  rho[p:]))
        else:
            rhoy = self.rhoy
            rhoe = self.rhoe         
        etahatr = signal.lfilter(rhoe, -rhoy, y)
        #print rho,np.sum(etahatr*etahatr)
        return etahatr

    def generate_sample(self,ar,ma,std,nsample):
        eta = std * np.random.randn(nsample)
        return signal.lfilter(ar, ma, eta)
        

arest = ARIMA()
rhohat, cov_x, infodict, mesg, ier = arest.estimate(yar1,1,1)
print rhohat
print cov_x

ar = [1.0, -0.8]
ma = [1.0,  0.5]
y2 = arest.generate_sample(ar,ma,0.1,100)
rhohat2, cov_x2, infodict, mesg, ier = arest.estimate(y2,1,1)
print rhohat2.shape
print cov_x2
err2 = arest.errfn()
print np.var(err2)

arest3 = ARIMA()
nsample = 1000
ar = [1.0, -0.8, -0.4]
ma = [1.0,  0.5,  0.2]
y3 = arest3.generate_sample(ar,ma,0.1,nsample)
rhohat3, cov_x3, infodict, mesg, ier = arest3.estimate(y3,1,2)
print rhohat3
print cov_x3
err3 = arest.errfn()
print np.var(err3)
print arest3.rhoy
print arest3.rhoe

