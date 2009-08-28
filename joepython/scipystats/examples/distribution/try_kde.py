'''looking at a bivariate empirical distribution created with scipy.stats.kde.

The empirical distribution is generated from the kernel density estimated empirical
density from an inital normal sample.

Then, we graph marginal and conditional densities of the empirical distribution
and the marginal frequency histograms of a random sample drawn from the
empirical distribution.
'''
import numpy as np
from scipy import stats
import matplotlib.pylab as plt

design = '' #'indep'
n_basesample = 1000
if design == 'indep':
    x2n = np.random.randn(n_basesample,2)
else:    
    z = 1.5 * np.random.randn(n_basesample,1)    
    x2n = z + np.random.randn(n_basesample,2)
    sstd = np.sqrt(1.5**2 + 1)

##initialize empirical density estimator based on initial sample
##Note: gaussian_kde expects row variables
gkde2=stats.gaussian_kde(x2n.T)

ind = np.linspace(-5,5)
yneg, ypos = (-1, 1)


##Approximation to marginal density

##We need to use 2D box integration since conditional integration on only
##one dimension is not available.

##for xcond in [yneg, 0, ypos]:
##    print gkde2.integrate_box((xcond-1e-6,-100),(xcond+1e-6,100))/(1e-6*2)
##for ycond in [yneg, 0, ypos]:
##    print gkde2.integrate_box((-100,ycond-1e-6),(100,ycond+1e-6))/(1e-6*2)

def marginal_density_x(x):
    '''calculate marginal probability of x'''
    x = np.array(x)
    xprob = np.zeros(x.size)
    for i,xv in enumerate(x.flat):
        xprob[i] = gkde2.integrate_box((xv-1e-6,-100),(xv+1e-6,100))/(1e-6*2)
    return xprob.reshape(x.shape)

def marginal_density_y(y):
    '''calculate marginal probability of  y'''
    y = np.array(y)
    yprob = np.zeros(y.size)
    for i,yv in enumerate(y.flat):
        yprob[i] = gkde2.integrate_box((-100, yv-1e-6),(100, yv+1e-6))/(1e-6*2)
    return yprob.reshape(y.shape) 

def conditional_density_xgy(x,ycond):
    '''calculate conditional probability of x given y'''
    x = np.array(x)
    margy = gkde2.integrate_box((-100,ycond-1e-6),(100,ycond+1e-6))/(1e-6*2)
    return gkde2.evaluate(np.c_[x, ycond*np.ones(x.shape)].T)/margy

xcoords = np.linspace(-2,2,11)
print 'marginal densities'
print 'marginal_density_x(0)', marginal_density_x(0)
print 'marginal densities of x at', xcoords
print marginal_density_x(xcoords)
print 'standard normal densities of x at', xcoords
print stats.norm.pdf(xcoords, scale=sstd)
print 'marginal densities of y at', xcoords
print marginal_density_y(xcoords)

print 'conditional_density_x(0,0)', conditional_density_xgy(0,0)
print 'conditional densities given y=0 of x at', xcoords
print conditional_density_xgy(xcoords,0)


c0 = gkde2.evaluate(np.c_[ind, 0*np.ones(ind.shape)].T)
cneg = gkde2.evaluate(np.c_[ind, yneg*np.ones(ind.shape)].T)
cpos = gkde2.evaluate(np.c_[ind, ypos*np.ones(ind.shape)].T)    

plt.figure()
plt.plot(ind, cneg, label='y=%s'%yneg)
plt.plot(ind, c0, label='y=0')
plt.plot(ind, cpos, label='y=%s'%ypos)
plt.legend()
plt.title('Shape of Conditional Distribution (without normalizing factor)')


plt.figure()
plt.plot(ind, conditional_density_xgy(ind,yneg), label='y=%s'%yneg)
plt.plot(ind, conditional_density_xgy(ind,0), label='y=0')
plt.plot(ind, conditional_density_xgy(ind,ypos), label='y=%s'%ypos)
plt.legend()
plt.title('Conditional Distribution of x given y')




## contour plot of joint distribution

#ind = np.linspace(-5,5)
x,y=np.meshgrid(ind,ind)
z=gkde2.evaluate(np.c_[x.flat,y.flat].T)
z2 = z.reshape(x.shape)
plt.figure()
plt.contour(x, y, z2)
plt.title('Joint Distribution')


##plotting the marginal distribution of a random sample

xs,ys=gkde2.resample(500)

plt.figure()
H, xedges, yedges = np.histogram2d(xs, ys, bins=20, normed=True)
plt.bar(xedges[:-1],H.sum(axis=0),np.diff(xedges)[0])
plt.title('Marginal Sample Frequence of x')
plt.figure()
plt.bar(yedges[:-1],H.sum(axis=1),np.diff(yedges)[0])
plt.title('Marginal Sample Frequence of y')
#plt.show()
