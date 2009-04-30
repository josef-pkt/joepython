'''fit a gaussian or gaussian mixture to a histogram using optimize least squares

From: original fit and error functions are from a question in scipy-user list
      http://projects.scipy.org/pipermail/scipy-user/2009-January/019479.html

added data generation and plot
'''


import numpy as np
from scipy import optimize, stats
import matplotlib.pylab as plt


#fit gaussian
fitfunc = lambda p, x: (p[0]**2)*np.exp(-(x-p[1])**2/(2*p[2]**2))  # Target function

errfunc = lambda p, x, y: fitfunc(p,x) - y          # Distance to the target function
doublegauss = lambda q,x: (q[0]**2)*np.exp(-(x-q[1])**2/(2*q[2]**2)) + \
                                (q[3]**2)*np.exp(-(x-q[4])**2/(2*q[5]**2))
doublegausserr = lambda q,x,y: doublegauss(q,x) - y




def gethist(y, bins=30):
    hist, histb = np.histogram(y,bins=bins, normed = True)
    #bincenters
    histb = histb[:-1] + np.diff(histb)/2.0
    hista = np.vstack([hist, histb])
    return hista


nobs = 500
bins = 20

#x = -3.0 * np.ones(500) #np.linspace(-5,5)
y = stats.norm.rvs(loc=-3, size = nobs)
hista = gethist(y, bins=bins)

# find parameters and estimates of single gaussian

p0 = [10.0,-2,0.5] # initial guess
p1,success  = optimize.leastsq(errfunc, p0[:], args = (hista[1],hista[0]))
errors_sq = errfunc(p1,hista[1],hista[0])**2
yest1 = fitfunc(p1,hista[1])
plt.figure()
plt.hist(y, bins=bins)
plt.figure()
#plt.plot(hista[1],hista[0],'o',hista[1],yest1,'.-')
x = np.linspace(hista[1,0],hista[1,-1],100)
yest1a = fitfunc(p1,x)
plt.plot(hista[1],hista[0],'o',x,yest1a,'-')


y1 = stats.norm.rvs(loc=-2, size = nobs*0.6)
y2 = stats.norm.rvs(loc=2, size = nobs*0.4)
y = np.hstack([y1,y2])
hista = gethist(y, bins=bins)

# find parameters and estimates of gaussian mixture
q0 = [10.0, -3, 0.5, 5, 3, 0.5] # initial guess
q1,success  = optimize.leastsq(doublegausserr, q0[:], args = (hista[1],hista[0]))
errors_sq = doublegausserr(q1,hista[1],hista[0])**2
yest2 = doublegauss(q1,hista[1])

plt.figure()
plt.hist(y)
plt.figure()
#plt.plot(hista[1],hista[0],'o',hista[1],yest2)
x = np.linspace(hista[1,0],hista[1,-1],100)
yest2a = doublegauss(q1,x)
plt.plot(hista[1],hista[0],'o',x,yest2a,'-')
