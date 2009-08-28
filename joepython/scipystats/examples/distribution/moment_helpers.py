'''helper functions to convert between central, non-central moments,
skew and kurtosis and cumulants

Author: Josef Perktold

cum2mc :   convert (up to) fivecumulants to central moments
cum2mc_g : convert cumulants to central moments
mc2cum :   convert (up to) four central moments to cumulants
mnc2cum_g : convert non-central moments to cumulants

mc2mnc_g : convert central to non-central moments, uses recursive formula
mnc2mc :   convert four non-central moments to central moments
mnc2mc_g : convert non-central to central moments, uses recursive formula

mc2mvsk :  convert four central moments to mean, variance, skew, kurtosis
mvsk2m :   convert mean, variance, skew, kurtosis to central and non-central
mvsk2mc :  convert mean, variance, skew, kurtosis to central moments
mvsk2mnc : convert mean, variance, skew, kurtosis to non-central moments

Notes
------
* Functions ending with _g use recursive formulas and can convert any number
  of moments
* Functions without _g consider either exactly four moments or up to four
  (`mc2cum`) or five (`cum2mc`) moments 
* Functions are not vectorized, work only for one list of moments or cumulants.
* input parameters can be list, tuple or np.array
* output type is list if number of elements is variable or tuple if number
  of arguments is fixed (at 4)
  
'''

import numpy as np
import scipy

from numpy.testing import assert_equal

__all__ = ['cum2mc', 'cum2mc_g', 'mc2cum', 'mc2mnc_g',
           'mc2mvsk', 'mnc2cum_g', 'mnc2mc', 'mnc2mc_g', 'mvsk2m',
           'mvsk2mc', 'mvsk2mnc']

def mc2mnc_g(mc_):
    '''convert central to non-central moments, uses recursive formula
    optionally adjusts first moment to return mean

    '''
    n = len(mc_)
    mean = mc_[0]
    mc = [1] + list(mc_)    # add zero moment = 1
    mc[1] = 0  # define central mean as zero for formula
    mnc = [1, mean] # zero and first raw moments
    for nn,m in enumerate(mc[2:]):
        n=nn+2
        mnc.append(0)
        for k in range(n+1):
            mnc[n] += scipy.comb(n,k,exact=1) * mc[k] * mean**(n-k)

    return mnc[1:]


def mnc2mc_g(mnc_, wmean = True):
    '''convert non-central to central moments, uses recursive formula
    optionally adjusts first moment to return mean

    '''
    n = len(mnc_)
    mean = mnc_[0]
    mnc = [1] + list(mnc_)    # add zero moment = 1
    mu = [] #np.zeros(n+1)
    for n,m in enumerate(mnc):
        mu.append(0)
        #[scipy.comb(n-1,k,exact=1) for k in range(n)]
        for k in range(n+1):
            mu[n] += (-1)**(n-k) * scipy.comb(n,k,exact=1) * mnc[k] * mean**(n-k)
    if wmean:
        mu[1] = mean
    return mu[1:]
    
    
def cum2mc_g(kappa_):
    '''convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    References
    ----------
    Kenneth Lange: Numerical Analysis for Statisticians, page 40
    (http://books.google.ca/books?id=gm7kwttyRT0C&pg=PA40&lpg=PA40&dq=convert+cumulants+to+moments&source=web&ots=qyIaY6oaWH&sig=cShTDWl-YrWAzV7NlcMTRQV6y0A&hl=en&sa=X&oi=book_result&resnum=1&ct=result)
 
    
    '''
    mc = [1,0.0] #kappa_[0]]  #insert 0-moment and mean 
    kappa = [1] + list(kappa_)
    for nn,m in enumerate(kappa[2:]):
        n = nn+2             
        mc.append(0)
        for k in range(n-1):
            mc[n] += scipy.comb(n-1,k,exact=1) * kappa[n-k]*mc[k]

    mc[1] = kappa_[0] # insert mean as first moments by convention 
    return mc[1:]


def mnc2cum_g(mnc_):
    '''convert non-central moments to cumulants
    recursive formula produces as many cumulants as moments

    http://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    '''
    mnc = [1] + list(mnc_)
    kappa = [1]
    for nn,m in enumerate(mnc[1:]):
        n = nn+1
        kappa.append(m)
        for k in range(1,n):
            kappa[n] -= scipy.comb(n-1,k-1,exact=1) * kappa[k]*mnc[n-k]
            
    return kappa[1:]


def cum2mc(kappa_):
    '''convert (up to) five cumulants to central moments

    Reference
    ---------
    http://en.wikipedia.org/wiki/Cumulant#Cumulants_and_moments
    '''
    n = len(kappa_)
    kappa = [1] + list(kappa_)    # insert 0-moment = 1
    mu = [None]*(n+1) #np.zeros(n+1)
    mu[1] = kappa[1]
    mu[2] = kappa[2]
    if n >= 3:
        mu[3] = kappa[3]
    if n >= 4:
        mu[4] = kappa[4] + 3 * kappa[2]**2
    if n >= 5:
        mu[5] = kappa[5] + 10 * kappa[2] * kappa[3]
    if n >= 6:
        mu[6] = kappa[6] + 15 * kappa[2] * kappa[4] + 10*kappa[3]**2 + 15*kappa[2]**3
    return mu[1:]
    

def mc2cum(mu_):
    '''convert (up to) four central moments to cumulants
    The first few cumulants (kappa_n)  in terms of central moments (mu_n) are
    Source:  http://mathworld.wolfram.com/Cumulant.html
    '''
    # copied from kstat comments
    n = len(mu_)
    mu = [0] + list(mu_)    #  insert 0-moment = 1
    kappa = [None]*(n+1) #np.zeros(n+1)
    kappa[1] = mu[1]
    kappa[2] = mu[2]
    if n >= 3:
        kappa[3] = mu[3]
    if n >= 4:
        kappa[4] = mu[4] - 3 * mu[2]**2
    if n >= 5:
        kappa[5] = mu[5] - 10 * mu[2] * mu[3]
    return kappa[1:]


def mvsk2mc(args):
    '''convert mean, variance, skew, kurtosis to central moments'''
    mu,sig2,sk,kur = args
    
    cnt = [None]*4
    cnt[0] = mu
    cnt[1] = sig2 
    cnt[2] = sk * sig2**1.5
    cnt[3] = (kur+3.0) * sig2**2.0
    return tuple(cnt)

def mvsk2mnc(args):
    '''convert mean, variance, skew, kurtosis to non-central moments'''
    mc, mc2, skew, kurt = args
    mnc = mc
    mnc2 = mc2 + mc*mc
    mc3  = skew*(mc2**1.5) # 3rd central moment
    mnc3 = mc3+3*mc*mc2+mc**3 # 3rd non-central moment
    mc4  = (kurt+3.0)*(mc2**2.0) # 4th central moment
    mnc4 = mc4+4*mc*mc3+6*mc*mc*mc2+mc**4
    return (mnc, mnc2, mnc3, mnc4)

def mvsk2m(args):
    '''convert mean, variance, skew, kurtosis to central and non-central
    moments'''
    mc, mc2, skew, kurt = args
    mnc = mc
    mnc2 = mc2 + mc*mc
    mc3  = skew*(mc2**1.5) # 3rd central moment
    mnc3 = mc3+3*mc*mc2+mc**3 # 3rd non-central moment
    mc4  = (kurt+3.0)*(mc2**2.0) # 4th central moment
    mnc4 = mc4+4*mc*mc3+6*mc*mc*mc2+mc**4
    return (mc, mc2, mc3, mc4), (mnc, mnc2, mnc3, mnc4)

def mc2mvsk(args):
    '''convert central moments to mean, variance, skew, kurtosis
    '''
    mc, mc2, mc3, mc4 = args
    skew = mc3 / mc2**1.5
    kurt = mc4 / mc2**2.0 - 3.0
    return (mc, mc2, skew, kurt)

def mnc2mc(args):
    '''convert four non-central moments to central moments
    '''
    mnc, mnc2, mnc3, mnc4 = args
    mc = mnc
    mc2 = mnc2 - mnc*mnc
    mc3 = mnc3 - (3*mc*mc2+mc**3) # 3rd central moment
    mc4 = mnc4 - (4*mc*mc3+6*mc*mc*mc2+mc**4)
    return (mc, mc2, mc3, mc4)


#
# example and test functions
# --------------------------

def examples():
    print mnc2cum_g([0,1,0,0])
    print mc2cum([0,1,1,1,1])
    print mc2cum(mvsk2mc([0,1,3,0]))
    print mnc2cum_g(mvsk2mnc([0,1,3,0]))
    print mnc2mc_g([0,1,3,5])
    assert_equal(mnc2mc_g([1,1,3,5],wmean = True),mnc2mc([1,1,3,5]) )
    print mc2mnc_g([0,1,2])
    kappa = [0,1,3,0]
    print cum2mc_g(kappa)
    print cum2mc(kappa)
    mc0 = [0,1,0,3]
    #round trip test
    assert_equal(cum2mc(mc2cum(mc0)),mc0)
    print mnc2cum_g([1,1,0,3]) #should be [0.0, 1.0, 0.0, 0.0]
    kap0 = [0,1,0,5]
    assert_equal(mc2cum(cum2mc_g(kap0)),kap0)
    assert_equal(cum2mc_g(kap0), cum2mc(kap0))
    print mnc2cum_g([0.0, 1.0, 0.0, 3.0])
    assert_equal(mnc2cum_g(mc0),mc2cum(mc0))
    mc0 = [1,1,1,3];
    assert_equal(mnc2cum_g(mc2mnc_g(mc0)),mc2cum(mc0))



#from central moments to cumulants

def examples_gen():
    mcs = [[0.,1,0,3],
           [1.,1,0,3],
           [0.,1,1,3],
           [1.,1,1,3],
           [1.,1,1,4],
           [1.,2,0,3],
           [0.,2,1,3],
           [1.,0.5,0,3],
           [0.,0.5,1,3],
           [0.,1,0,3,0],
           [1.,1,0,3,1]]


    for mc0 in mcs:
        print mc0
        print '    ', mnc2cum_g(mc2mnc_g(mc0))
        if len(mc0) <= 4:
            print '    ', mc2cum(mc0)
    print 'from non-central moment'        
    for mnc0 in mcs:
        print mnc0
        print '    ', mnc2cum_g(mnc0)
        if len(mnc0) <= 4:
            print '    ', mc2cum(mnc2mc(mnc0))

def test_moment_conversion():

    ms  = [( [0.0, 1, 0, 3], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0] ),
           ( [1.0, 1, 0, 3], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, -1.0, 6.0] ),
           ( [0.0, 1, 1, 3], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0] ),
           ( [1.0, 1, 1, 3], [1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 2.0] ),
           ( [1.0, 1, 1, 4], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 3.0] ),
           ( [1.0, 2, 0, 3], [1.0, 2.0, 0.0, -9.0], [1.0, 1.0, -4.0, 9.0] ),
           ( [0.0, 2, 1, 3], [0.0, 2.0, 1.0, -9.0], [0.0, 2.0, 1.0, -9.0] ),
           ( [1.0, 0.5, 0, 3], [1.0, 0.5, 0.0, 2.25], [1.0, -0.5, 0.5, 2.25] ), #neg.variance if mnc2<mnc1
           ( [0.0, 0.5, 1, 3], [0.0, 0.5, 1.0, 2.25], [0.0, 0.5, 1.0, 2.25] ),
           ( [0.0, 1, 0, 3, 0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0] ),
           ( [1.0, 1, 0, 3, 1], [1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 0.0, -1.0, 6.0, -20.0] )]

    for mom in ms:
        # test moment -> cumulant
        assert_equal(mnc2cum_g(mc2mnc_g(mom[0])),mom[1])
        assert_equal(mnc2cum_g(mom[0]),mom[2])
        if len(mom) <= 4:
            assert_equal(mc2cum(mom[0]),mom[1])

    for mom in ms:
        # test   cumulant -> moment
        assert_equal(cum2mc_g(mom[1]),mom[0])
        assert_equal(mc2mnc_g(cum2mc_g(mom[2])),mom[0])
        if len(mom) <= 4:
            assert_equal(cum2mc(mom[1]),mom[0])
            
    for mom in ms:
        #round trip: mnc -> cum -> mc == mnc -> mc, 
        assert_equal(cum2mc_g(mnc2cum_g(mom[0])),mnc2mc_g(mom[0]))
        

    for mom in ms:
        #round trip: mc -> mnc -> mc ==  mc, 
        assert_equal(mc2mnc_g(mnc2mc_g(mom[0])), mom[0])
        
    for mom in (m for m in ms if len(m) == 4):
        #round trip: mc -> mvsk -> mc ==  mc
        assert_equal(mvsk2mc(mc2mvsk(mom[0])), mom[0])
        #round trip: mc -> mvsk -> mnc ==  mc -> mnc
        assert_equal(mvsk2mnc(mc2mvsk(mom[0])), mc2mnc(mom[0])) 

##    # printing examples for cum2mc
##    for mom in ms:
##        print mom[0]
##        print '    ', cum2mc_g(mom[1])
##        if len(mom[1]) <= 4:
##            print '    ', list(cum2mc(mom[1]))

def test_moment_conversion_types():
    # this uses globals for the location of the functions
    # it needs to be changed when test is moved
    assert np.all([isinstance(getattr(moment_helpers,f)([1.0, 1, 0, 3]),list) or
            isinstance(getattr(moment_helpers,f)(np.array([1.0, 1, 0, 3])),tuple)
            for f in __all__])
    assert np.all([isinstance(getattr(moment_helpers,f)(np.array([1.0, 1, 0, 3])),list) or
            isinstance(getattr(moment_helpers,f)(np.array([1.0, 1, 0, 3])),tuple)
            for f in __all__])
    assert np.all([isinstance(getattr(moment_helpers,f)(tuple([1.0, 1, 0, 3])),list) or
            isinstance(getattr(moment_helpers,f)(np.array([1.0, 1, 0, 3])),tuple)
            for f in __all__])

# old version using globals
#    assert np.all([isinstance(globals()[f](tuple([1.0, 1, 0, 3])),list) or
#            isinstance(globals()[f](np.array([1.0, 1, 0, 3])),tuple)
#            for f in __all__])

# print for debugging
##    print 'return types'
##    print '\n'.join([' : '.join([f,str(locals()[f]([1.0, 1, 0, 3]).__class__)])
##                     for f in __all__])
##    print 'checking np.array input'
##    print '\n'.join([' : '.join([f,str((locals()[f](np.array([1.0, 1, 0, 3])).__class__))])
##                     for f in __all__])

if __name__ == '__main__':
    test_moment_conversion_types()
    test_moment_conversion()

    # print function list and first line of doc string
##    print '\n'.join([' : '.join([f,str((locals()[f].__doc__.split('\n')[0]))])
##                     for f in moment_helpers.__all__])



