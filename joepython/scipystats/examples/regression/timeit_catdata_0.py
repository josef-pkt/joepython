
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
    unil,unilinv = np.unique1d(factors, return_index=False, return_inverse=True)

    #labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=np.arange(len(unil)+1)))
    labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=np.arange(len(unil))))
    return labelmeans

def labelmean_dig(factors, values):
    # works also for string labels in ys, but requires 1D
    # from mailing list scipy-user 2009-02-11
    # check mistake: returns one element to much
    #unil = np.unique1d(factors, return_index=False, return_inverse=False)
    unil = np.unique1d(factors)
    unilinv = np.digitize(factors, unil).astype('int64')
    #print unilinv.shape
    #print unilinv.dtype
    #labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=np.arange(len(unil)+1)))
    labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=np.arange(len(unil))))
    return labelmeans

def labelmean_ndi(factors, values, index=None):
    # works also for string labels in ys, but requires 1D
    # from mailing list scipy-user 2009-02-11
    # check mistake: returns one element to much
    #unil,unilinv = np.unique1d(factors, return_index=False, return_inverse=True)
    unilinv = np.asanyarray(factors)

    #note: bincount uses complete list of integers, maybe range(max(factor)+1)
    if not np.issubdtype(unilinv.dtype, int):
        unil, unilinv = np.unique1d(factors, return_inverse=1)
        if index is None:
            index = np.arange(len(unil)) #unil
    elif index is None:
        index = np.arange(np.max(unilinv)+ 1)

    labelmeans = np.array(ndimage.mean(values, labels=unilinv, index=index))
    return labelmeans

def bincount2d(factors):
    # array check copied from np.histogramdd
    try:
        # Sample is an ND-array.
        N, D = factors.shape
        sample = factors
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(factors).T
        N, D = sample.shape
    tmp = np.ascontiguousarray(sample) #b/c view works on base not another view
    factarr = tmp.view([('',tmp.dtype)]*tmp.shape[-1])
    uni, rind = np.unique1d(factarr, return_inverse=1)
    return np.bincount(rind), uni.view(tmp.dtype).reshape(-1,D)
    #or return unique values as structured array:
    #return np.bincount(rind), uni


def groupstatsbin(factors, values, ddof=0, stat='mvnx', drop=True):
    '''uses np.bincount, assumes factors/labels are integers,
    create labels with unique1d return_inverse if string labels
    '''
    #n = len(factors)

    rind = np.asanyarray(factors)
    #note: bincount uses complete list of integers, maybe range(max(factor)+1)
    if not np.issubdtype(rind.dtype, int):
        ix, rind = np.unique1d(factors, return_inverse=1)
##    else:
##        rind = factors
    #print rind.shape
    res = []
    if 'c' in stat or 'm' in stat or 'v' in stat:
        gcount = np.bincount(rind)
        if drop:
            keepidx = np.nonzero(gcount)
        if 'c' in stat:
            if drop:
                res.append(gcount[keepidx])
            else:
                res.append(gcount)

    if 'm' in stat or 'v' in stat:
        #gcount = np.bincount(rind)
        gmean = np.bincount(rind, weights=values)/ (1.0*gcount)
        if np.max(np.abs(gmean)) > 1e6:
            # improve numerical precision if means are large
            # (to "cheat" on NIST anova test examples)
            meanarr = gmean[rind]
            gmean += np.bincount(rind, weights=(values-meanarr)) / (1.0*gcount)
        if 'm' in stat:
            if drop:
                res.append(gmean[keepidx])
            else:
                res.append(gmean)

    if 'v' in stat:
        meanarr = gmean[rind]
        withinvar = np.bincount(rind, weights=(values-meanarr)**2) / (1.0*gcount-ddof)
        if drop:
            res.append(withinvar[keepidx])
        else:
            res.append(withinvar)

    #withinvararr = withinvar[rind]
    #return gcount, gmean , meanarr, withinvar, withinvararr

    if 'n' in stat or 'x' in stat:
        #calculate min, max per factor
        sortind = np.lexsort((values, rind))
        fs = rind[sortind]
        vs = values[sortind]
        fsd = np.hstack((np.inf,np.diff(fs),np.inf))
    if 'n' in stat:
        #minimum
        res.append(vs[fsd[:-1] != 0])
    if 'x' in stat:
        #maximum
        res.append(vs[fsd[1:] != 0])
    return res


def labelstats_str(factors, values, stat='mvnx'):
    # works also for string labels in ys, but requires 1D
    # from mailing list scipy-user 2009-02-11
    unil, unilinv = np.unique1d(factors, return_index=False, return_inverse=True)
    res = []
    if 'm' in stat:
        labelmeans = np.array(ndimage.mean(values, labels=unilinv,
                                index=np.arange(len(unil))))
        res.append(labelmeans)
    if 'v' in stat:
        labelvars = np.array(ndimage.variance(values, labels=unilinv,
                                index=np.arange(len(unil))))
        res.append(labelvars)
    if 'n' in stat:
        labelmin = np.array(ndimage.minimum(values, labels=unilinv,
                                index=np.arange(len(unil))))
        res.append(labelmin)
    if 'x' in stat:
        labelmax = np.array(ndimage.maximum(values, labels=unilinv,
                                index=np.arange(len(unil))))
        res.append(labelmax)
    return res


nobs = 4000
nfact = 100
factors = (0+1*np.random.randint(nfact,size=nobs))#.astype('S6')#.astype(float)#copy()#
#factors = np.random.randint(nfact,size=nobs)
#factors = np.array(factors.tolist(),int)
print factors.dtype
values = np.random.randn(nobs)
indexs = 100 + 2*np.arange(nfact)

def test_compare(nrun):
    nobs = 100
    nfact = 5
    for i in range(nrun):
        #factors = (100*np.random.randint(nfact,size=nobs)).astype('S2')
        factors = (np.random.randint(nfact,size=nobs))#.astype('S2')
        values = 1e0 + 1.e1 * np.random.randn(nobs)
        m,v,n,x = groupstatsbin(factors, values, ddof=1)
        #print factors.shape, values.shape, m.shape, v.shape, n.shape, x.shape
        assert_array_almost_equal(groupstatsbin(factors, values, ddof=1), \
            labelstats_str(factors, values), decimal=10, err_msg = \
            repr(np.asarray(groupstatsbin(factors, values, ddof=1, stat='mvnx')) - \
            np.asarray(labelstats_str(factors, values))) + str(i))

if __name__ == "__main__":

    #print groupstatsbin(factors, values)
    #print labelmean_str(factors, values)

    setupstr = '''from __main__ import np, groupmeanbin, labelmean_str, \
               labelmean_dig, labelmean_ndi, groupstatsbin, labelstats_str, \
               factors, values, indexs'''

    from timeit import Timer

    n = 10
    t1 = Timer(setup=setupstr, stmt='groupmeanbin(factors, values)')
    t2 = Timer(setup=setupstr, stmt='labelmean_str(factors, values)')
    t2a = Timer(setup=setupstr, stmt='labelmean_dig(factors, values)')
    t2b = Timer(setup=setupstr+'; factors=factors.tolist()', \
                                stmt='labelmean_ndi(factors, values,indexs)')
    t2c = Timer(setup=setupstr+'; factors2=np.array(factors.tolist()); from scipy import ndimage', \
                                stmt='np.array(ndimage.mean(values, labels=factors2, index=indexs))')
    t3a = Timer(setup=setupstr, stmt="groupstatsbin(factors, values, stat='m')")
    t3 = Timer(setup=setupstr, stmt="groupstatsbin(factors, values, stat='mv')")
    t4 = Timer(setup=setupstr, stmt="labelstats_str(factors, values, stat='mv')")
    t5 = Timer(setup=setupstr, stmt="groupstatsbin(factors, values, stat='nx')")
    t6 = Timer(setup=setupstr, stmt="labelstats_str(factors, values, stat='nx')")

    print 'number of observations %d, factors %d' % (nobs, nfact)
    print 'number of runs %d' % n
    print 'np.bincount ', t1.timeit(n)/float(n)
    print 'ndimage.mean', t2.timeit(n)/float(n)
    #digitize is slow, drop it
    #print 'ndimage dig ', t2a.timeit(n)/float(n)
    print 'ndimage ndi ', t2b.timeit(n)/float(n)
    if not np.issubdtype(factors.dtype, str):
        print 'ndimage ndipl', t2c.timeit(n)/float(n)
    print 'np.bincount m', t3a.timeit(n)/float(n)
    print 'np.bincount mv', t3.timeit(n)/float(n)
    print 'ndimage     mv', t4.timeit(n)/float(n)
    print 'np.bincount nx', t5.timeit(n)/float(n)
    print 'ndimage     nx', t6.timeit(n)/float(n)

    test_compare(10)

    debug = 0
    if debug:
        nobs = 100
        nfact = 5
        for i in range(1):
            factors = 10+ np.random.randint(nfact,size=nobs)#.astype('S2')
            values = 1e6 + np.random.randn(nobs)
            print np.array(groupstatsbin(factors, values,ddof=1)) - \
                           np.array(labelstats_str(factors, values))#[:,:-1]

        sortind = np.lexsort((values, factors))
        fs = factors[sortind]
        vs = values[sortind]
        #fsd = np.inf*np.ones(len(fs)+)
        fsd = np.hstack((np.inf,np.diff(fs),np.inf)) #fs[:-1]-fs[1:]
        gmin = vs[fsd[:-1] != 0]
        gmax = vs[fsd[1:] != 0]
        print gmin
        print ndimage.minimum(values, labels=factors.tolist(), index=np.arange(15))
        print gmax
        print ndimage.maximum(values, labels=factors.tolist(), index=np.arange(15))

        m,v,n,x = groupstatsbin(factors, values,ddof=1)
        (values[factors==0]-1e6).mean() - ((values[factors==0]).mean()-1e6)
        m1=values[factors==0].mean();
        m1+(values[factors==0]-m1).mean() - (m[0])
        m1=1e6;
        m1+(values[factors==0]-m1).mean() - (m[0])
        m1=0;
        m1+(values[factors==0]-m1).mean() - (m[0])

    debug2 = 0
    if debug2:
        print bincount2d(factors)
        print np.bincount(factors)
        factors = np.random.randint(5,size=(100,3))
        print bincount2d(factors)
        nbins = 2
        nbinsp1 = nbins + 1
        digitizedarr1 = (factors - factors % nbinsp1) / nbinsp1
        digitizedarr = np.mod(factors,nbins)
        #assert np.all(digitizedarr1==digitizedarr)
        cb, bb = bincount2d(digitizedarr1)
        ch, eh = np.histogramdd(digitizedarr1,nbins)
        print cb
        print ch
        assert np.all(cb==ch.ravel())
        print bb
