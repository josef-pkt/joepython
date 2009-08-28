
'''using scipy signal and numpy correlate to calculate some time series
statistics

see also timeseries scikits (movstat is partially inspired by it)
(added 2009-08-29:
timeseries moving stats are in c, autocorrelation similar to here
I thought I saw moving stats somewhere in python, maybe not)


TODO:

moving statistics
* filters don't handle boundary conditions nicely (correctly ?)
  e.g. minimum order filter uses 0 for out of bounds value
  -> append and prepend with last resp. first value
* enhance for nd arrays, with axis = 0

VAR, VARMA
Note: filter has smallest lag at end of array and largest lag at beginning,
    be careful for asymmetric lags coefficients
    check this again if it is consistently used


Note: Equivalence for 1D signals
>>> np.all(signal.correlate(x,[1,1,1],'valid')==np.correlate(x,[1,1,1]))
True
>>> np.all(ndimage.filters.correlate(x,[1,1,1], origin = -1)[:-3+1]==np.correlate(x,[1,1,1]))
True

# multidimensional, but, it looks like it uses common filter across time series, no VAR
ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)
ndimage.filters.correlate(x,[1,1,1],origin = 1))
ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), \
origin = 1)

>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[1,1,1],[0,0,0]]), origin = 1)[0]==\
ndimage.filters.correlate(x,[1,1,1],origin = 1))
True
>>> np.all(ndimage.filters.correlate(np.vstack([x,x]),np.array([[0.5,0.5,0.5],[0.5,0.5,0.5]]), \
origin = 1)[0]==ndimage.filters.correlate(x,[1,1,1],origin = 1))
'''


import numpy as np
from scipy import signal
import matplotlib.pylab as plt
from numpy.testing import assert_array_equal




def expandarr(x,k):
    #make it work for 2D or nD with axis
    return np.r_[np.ones(k)*x[0],x,np.ones(k)*x[-1]]

def mov_order(x, order = 'med', windsize=3, lag='lagged'):

    #if windsize is even raise ValueError
    if lag == 'lagged':
        lead = windsize//2
    elif lag == 'centered':
        lead = 0
    elif lag == 'leading':
        lead = -windsize//2 +1
    else:
        raise ValueError
    if np.isfinite(order) == True: #if np.isnumber(order):
        ord = order   # note: ord is a builtin function
    elif order == 'med':
        ord = (windsize - 1)/2
    elif order == 'min':
        ord = 0
    elif order == 'max':
        ord = windsize - 1
    else:
        raise ValueError

    #return signal.order_filter(x,np.ones(windsize),ord)[:-lead]
    xext = expandarr(x, windsize)
    #np.r_[np.ones(windsize)*x[0],x,np.ones(windsize)*x[-1]]
    return signal.order_filter(xext,np.ones(windsize),ord)[windsize-lead:-(windsize+lead)]

def check_movorder():
    x = np.arange(1,10)
    xo = mov_order(x, order='max')
    assert_array_equal(xo, x)
    x = np.arange(10,1,-1)
    xo = mov_order(x, order='min')
    assert_array_equal(xo, x)
    assert_array_equal(mov_order(x, order='min', lag='centered')[:-1], x[1:])

    tt = np.linspace(0,2*np.pi,15)
    x = np.sin(tt) + 1
    xo = mov_order(x, order='max')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max lagged')
    xo = mov_order(x, order='max', lag='centered')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max centered')
    xo = mov_order(x, order='max', lag='leading')
    plt.figure()
    plt.plot(tt,x,'.-',tt,xo,'.-')
    plt.title('moving max leading')

# identity filter
##>>> signal.order_filter(x,np.ones(1),0)
##array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
# median filter
##signal.medfilt(np.sin(x), kernel_size=3)
##>>> plt.figure()
##<matplotlib.figure.Figure object at 0x069BBB50>
##>>> x=np.linspace(0,3,100);plt.plot(x,np.sin(x),x,signal.medfilt(np.sin(x), kernel_size=3))


def movmeanvar(x):
    m = np.correlate(xm,np.array([1,1,1])/3.0,'full')
    v = np.correlate(xm*xm,np.array([1,1,1])/3.0,'full') - m**2
#>>> np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')-np.correlate(xm*xm,np.array([1,1,1])/3.0,'valid')**2
    return m,v

def movmoment(x,k):
    '''non-central moment'''
    return np.correlate(xm**k,np.array([1,1,1])/3.0,'full')


#None of the acovf, ... are tested; starting index? orientation?
def acovf(x):
    ''' autocovariance for 1D
    '''
    n = len(x)
    xo = x - x.mean();
    xi = np.ones(n);
    d = np.correlate(xi,xi,'full')
    return ( np.correlate(xo,xo,'full')/d )[n-1:]

def ccovf(x,y):
    ''' crosscovariance for 1D
    '''
    n = len(x)
    xo = x - x.mean();
    yo = y - y.mean();
    xi = np.ones(10);
    d = np.correlate(xi,xi,'full')
    return ( np.correlate(xo,yo,'full')/d )[n-1:]

def acf(x):
    avf = acovf(x)
    return avf/avf[0]

def ccf(x,y):
    cvf = ccovf(x,y)
    return cvf/np.std(x)/np.std(y)


#x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,[1],'full')
#x=0.5**np.arange(3);np.correlate(x,x,'same')
##>>> x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')
##
##>>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> xo
##xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> x=np.ones(10);xo=x-x.mean();a=np.correlate(xo,xo,'full')
##>>> xo=np.ones(10);d=np.correlate(xo,xo,'full')
##>>> d
##array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,   9.,
##         8.,   7.,   6.,   5.,   4.,   3.,   2.,   1.])


##def ccovf():
##    pass
##    #x=0.5**np.arange(10);xm=x-x.mean();a=np.correlate(xm,xo,'full')




def VAR(x,B, const=0):
    ''' multivariate linear filter

    Parameters
    ----------
    x: (TxK) array
        columns are variables, rows are observations for time period
    B: (PxKxK) array
        b_t-1 is bottom "row", b_t-P is top "row" when printing
        B(:,:,0) is lag polynomial matrix for variable 1
        B(:,:,k) is lag polynomial matrix for variable k
        B(p,:,k) is pth lag for variable k
        B[p,:,:].T corresponds to A_p in Wikipedia
    const: float or array (not tested)
        constant added to autoregression

    Returns
    -------
    xhat: (TxK) array
        filtered, predicted values of x array

    Notes
    -----
    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) }  for all i = 0,K-1, for all t=p..T

    xhat does not include the forecasting observation, xhat(T+1),
    xhat is 1 row shorter than signal.correlate

    References
    ----------
    http://en.wikipedia.org/wiki/Vector_Autoregression
    http://en.wikipedia.org/wiki/General_matrix_notation_of_a_VAR(p)
    '''
    p = B.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    for t in range(p,T): #[p+2]:#
##        print p,T
##        print x[t-p:t,:,np.newaxis].shape
##        print B.shape
        #print x[t-p:t,:,np.newaxis]
        xhat[t,:] = const + (x[t-p:t,:,np.newaxis]*B).sum(axis=1).sum(axis=0)
    return xhat

T = 20
K = 2
P = 3
#x = np.arange(10).reshape(5,2)
x = np.column_stack([np.arange(T)]*K)
B = np.ones((P,K,K))
#B[:,:,1] = 2
B[:,:,1] = [[0,0],[0,0],[0,1]]
xhat = VAR(x,B)
print np.all(xhat[P:,0]==np.correlate(x[:-1,0],np.ones(P))*2)
#print xhat


def VARMA(x,B,C, const=0):
    ''' multivariate linear filter

    x (TxK)
    B (PxKxK)

    xhat(t,i) = sum{_p}sum{_k} { x(t-P:t,:) .* B(:,:,i) } +
                sum{_q}sum{_k} { e(t-Q:t,:) .* C(:,:,i) }for all i = 0,K-1

    '''
    P = B.shape[0]
    Q = C.shape[0]
    T = x.shape[0]
    xhat = np.zeros(x.shape)
    e = np.zeros(x.shape)
    start = max(P,Q)
    for t in range(start,T): #[p+2]:#
##        print p,T
##        print x[t-p:t,:,np.newaxis].shape
##        print B.shape
        #print x[t-p:t,:,np.newaxis]
        xhat[t,:] =  const + (x[t-P:t,:,np.newaxis]*B).sum(axis=1).sum(axis=0) + \
                     (e[t-Q:t,:,np.newaxis]*C).sum(axis=1).sum(axis=0)
        e[t,:] = x[t,:] - xhat[t,:]
    return xhat, e

T = 20
K = 2
Q = 2
P = 3
const = 1
#x = np.arange(10).reshape(5,2)
x = np.column_stack([np.arange(T)]*K)
B = np.ones((P,K,K))
#B[:,:,1] = 2
B[:,:,1] = [[0,0],[0,0],[0,1]]
C = np.zeros((Q,K,K))
xhat1 = VAR(x,B, const=const)
xhat2, err2 = VARMA(x,B,C, const=const)
print np.all(xhat2 == xhat1)
print np.all(xhat2[P:,0] == np.correlate(x[:-1,0],np.ones(P))*2+const)

C[1,1,1] = 0.5
xhat3, err3 = VARMA(x,B,C)

x = np.r_[np.zeros((P,K)),x]  #prepend inital conditions
xhat4, err4 = VARMA(x,B,C)

C[1,1,1] = 1
B[:,:,1] = [[0,0],[0,0],[0,1]]
xhat5, err5 = VARMA(x,B,C)
#print err5

#in differences
#VARMA(np.diff(x,axis=0),B,C)


#Note:
# * signal correlate applies same filter to all columns if kernel.shape[1]<K
#   e.g. signal.correlate(x0,np.ones((3,1)),'valid')
# * if kernel.shape[1]==K, then `valid` produces a single column
#   -> possible to run signal.correlate K times with different filters,
#      see the following example, which replicates VAR filter
x0 = np.column_stack([np.arange(T), 2*np.arange(T)])
B[:,:,0] = np.ones((P,K))
B[:,:,1] = np.ones((P,K))
B[1,1,1] = 0
xhat0 = VAR(x0,B)
xcorr00 = signal.correlate(x0,B[:,:,0])#[:,0]
xcorr01 = signal.correlate(x0,B[:,:,1])
print np.all(signal.correlate(x0,B[:,:,0],'valid')[:-1,0]==xhat0[P:,0])
print np.all(signal.correlate(x0,B[:,:,1],'valid')[:-1,0]==xhat0[P:,1])

aav = acovf(x[:,0])
print aav[0] == np.var(x[:,0])
aac = acf(x[:,0])
