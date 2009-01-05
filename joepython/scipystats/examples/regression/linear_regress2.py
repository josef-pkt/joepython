from scipy import stats
import numpy as np
from matplotlib import pylab as plt 

##Example: Simple Linear Regression
##---------------------------------

## This is a very simple example of using stats.linregress for linear regression

#Sample data creation

#number of points 
n = 100
t = np.linspace(0,10,n)
#parameters
a = 1.8; b = 0.75
x = a + b* t
#add some noise
xn = x + np.random.randn(n)


#Linear regression using stats.linregress
(b_s,a_s,r,tt,stderr) = stats.linregress(t,xn)
print('Linear regression using stats.linregress')
print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f,'
      'std error= %.3f, t-statistic=%.2f, Rsquare=%.2f' % (a,b,a_s,b_s,stderr,tt,r))

#estimated points on regression line
xr = a_s + b_s * t

#matplotlib ploting
plt.title('Linear Regression Example')
plt.plot(t,x,'g.--')
plt.plot(t,xn,'k.')
plt.plot(t,xr,'r.-')
plt.legend(['original','plus noise', 'regression'], loc='lower right')

plt.show()


