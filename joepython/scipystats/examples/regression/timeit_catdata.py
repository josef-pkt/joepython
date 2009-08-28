
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import ndimage

def groupmeanbin(factors, values):
    '''uses np.bincount, assumes factors/labels are integers
    '''
    #n = len(factors)
    ix,rind = np.unique1d(factors, return_inverse=1)
    gcount = np.bincount(rind)
    gmean = np.bincount(rind, weights=values)/ (1.0*gcount)
    return gmean


def labelmean_str(factors, values):
    # works also for string labels in ys, but requires 1D
    # from mailing list scipy-user 2009-02-11
    # check mistake: returns one element to much
    unil, unilinv = np.unique1d(factors, return_index=False, return_inverse=True)
    #labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=np.arange(len(unil)+1)))
    labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=np.arange(len(unil))))

    return labelmeans

def groupstatsbin(factors, values, ddof=0):
    '''uses np.bincount, assumes factors/labels are integers,
    create labels with unique1d return_inverse if string labels
    '''
    n = len(factors)
    ix,rind = np.unique1d(factors, return_inverse=1)
    gcount = np.bincount(rind)
    gmean = np.bincount(rind, weights=values)/ (1.0*gcount)
    meanarr = gmean[rind]
    withinvar = np.bincount(rind, weights=(values-meanarr)**2) / (1.0*gcount-ddof)
    #withinvararr = withinvar[rind]
    #return gcount, gmean , meanarr, withinvar, withinvararr

    #calculate min, max per factor
    sortind = np.lexsort((values, rind))
    fs = rind[sortind]
    vs = values[sortind]
    fsd = np.hstack((np.inf,np.diff(fs),np.inf))
    gmin = vs[fsd[:-1] != 0]
    gmax = vs[fsd[1:] != 0]
    return gmean, withinvar, gmin, gmax


def labelstats_str(factors, values):
    # works also for string labels in ys, but requires 1D
    # from mailing list scipy-user 2009-02-11
    unil, unilinv = np.unique1d(factors, return_index=False, return_inverse=True)
    labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=np.arange(len(unil))))
    labelvars = np.array(ndimage.variance(values, labels=unilinv, index=np.arange(len(unil))))
    labelmin = np.array(ndimage.minimum(values, labels=unilinv, index=np.arange(len(unil))))
    labelmax = np.array(ndimage.maximum(values, labels=unilinv, index=np.arange(len(unil))))
    return labelmeans, labelvars, labelmin, labelmax


nobs = 1000000
nfact = 1000
factors = np.random.randint(nfact,size=nobs)#.astype('S2')
values = np.random.randn(nobs)

def test_compare(nrun):
    nobs = 100
    nfact = 5
    for i in range(nrun):
        factors = 100*np.random.randint(nfact,size=nobs)#.astype('S2')
        values = 1e2 + np.random.randn(nobs)
        assert_array_almost_equal(groupstatsbin(factors, values, ddof=1), \
                       labelstats_str(factors, values), decimal=15)

if __name__ == "__main__":

    #print groupstatsbin(factors, values)
    #print labelmean_str(factors, values)

    setupstr = '''from __main__ import np, groupmeanbin, labelmean_str, \
                groupstatsbin, labelstats_str, factors, values'''

    from timeit import Timer

    n = 10
    t1 = Timer(setup=setupstr, stmt='groupmeanbin(factors, values)')
    t2 = Timer(setup=setupstr, stmt='labelmean_str(factors, values)')
    t3 = Timer(setup=setupstr, stmt='groupstatsbin(factors, values)')
    t4 = Timer(setup=setupstr, stmt='labelstats_str(factors, values)')

    print 'number of observations %d, factors %d' % (nobs, nfact)
    print 'number of runs %d' % n
    print 'np.bincount ', t1.timeit(n)/float(n)
    print 'ndimage.mean', t2.timeit(n)/float(n)
    print 'np.bincount mv', t3.timeit(n)/float(n)
    print 'ndimage     mv', t4.timeit(n)/float(n)

    test_compare(10)


##    nobs = 100
##    nfact = 5
##    for i in range(1):
##        factors = np.random.randint(nfact,size=nobs)#.astype('S2')
##        values = np.random.randn(nobs)
##        print np.array(groupstatsbin(factors, values,ddof=1)) - \
##                       np.array(labelstats_str(factors, values))#[:,:-1]
##
    nobs = 100
    nfact = 5
    factors = np.random.randint(nfact,size=nobs)#.astype('S2')
    values = np.random.randn(nobs)
    sortind = np.lexsort((values, factors))
    fs = factors[sortind]
    vs = values[sortind]
    #fsd = np.inf*np.ones(len(fs)+)
    fsd = np.hstack((np.inf,np.diff(fs),np.inf)) #fs[:-1]-fs[1:]
    gmin = vs[fsd[:-1] != 0]
    gmax = vs[fsd[1:] != 0]
    print gmin
    print ndimage.minimum(values, labels=factors, index=np.arange(5))
    print gmax
    print ndimage.maximum(values, labels=factors, index=np.arange(5))