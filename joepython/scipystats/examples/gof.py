'''

Author: Josef Perktold and pymc authors

'''

import numpy as np
from numpy.linalg.linalg import LinAlgError
import numpy.linalg as nplinalg
from numpy.testing import assert_almost_equal, assert_equal, assert_array_almost_equal

from scipy import linalg, stats

#JP: note: Freeman-Tukey statistics is a goodness-of-fit test for
#    discrete data,
##    other gof tests
##    The test statistics:
##    Pearson?s Chi-Square,
##    the Kolmogorov-Smirnov test statistic for discrete data,
##    the Log-Likelihood Ratio,
##    the Freeman-Tukey and
##    the Power Divergence statistic with ?=?.


def discrepancy(observed, simulated, expected):
    """Calculates Freeman-Tukey statistics (Freeman and Tukey 1950) as
    a measure of discrepancy between observed and simulated data. This
    is a convenient method for assessing goodness-of-fit (see Brooks et al. 2000).

    D(x|\theta) = \sum_j (\sqrt{x_j} - \sqrt{e_j})^2

    :Parameters:
      observed : Iterable of observed values
      simulated : Iterable of simulated values
      expected : Iterable of expeted values

    :Returns:
      D_obs : Discrepancy of observed values
      D_sim : Discrepancy of simulated values
    
    Notes:
    ------
    this is the pymc version

    """

    D_obs = np.sum([(np.sqrt(x)-np.sqrt(e))**2 for x,e in zip(observed, expected)])
    D_sim = np.sum([(np.sqrt(s)-np.sqrt(e))**2 for s,e in zip(simulated, expected)])

    return D_obs, D_sim


def powerdiscrepancy(o, e, lambd=0.0, axis=0):
    """Calculates power discrepancy, a class of goodness-of-fit tests
    as a measure of discrepancy between observed and expected data.

    This contains several goodness-of-fit tests as special cases, see the
    describtion of lambd, the exponent of the power discrepancy. The pvalue
    is based on the asymptotic chi-square distribution of the test statistic.

    freeman_tukey:
    D(x|\theta) = \sum_j (\sqrt{x_j} - \sqrt{e_j})^2

    Parameters
    ----------
      o : Iterable of observed values
      e : Iterable of expeted values
      lambd : float or string
         * float : exponent `a` for power discrepancy
         * 'loglikeratio': a = 0
         * 'freeman_tukey': a = -0.5
         * 'pearson': a = 1
         * 'modified_loglikeratio': a = -1
         * 'cressie_read': a = 2/3

    Returns
    -------
      D_obs : Discrepancy of observed values
      pvalue : pvalue


    References
    ----------
    Cressie, Noel  and Timothy R. C. Read, Multinomial Goodness-of-Fit Tests,
        Journal of the Royal Statistical Society. Series B (Methodological),
        Vol. 46, No. 3 (1984), pp. 440-464

    Campbell B. Read: Freeman-Tukey chi-squared goodness-of-fit statistics,
        Statistics & Probability Letters 18 (1993) 271-278

    Nobuhiro Taneichi, Yuri Sekiya, Akio Suzukawa, Asymptotic Approximations
        for the Distributions of the Multinomial Goodness-of-Fit Statistics
        under Local Alternatives, Journal of Multivariate Analysis 81, 335?359 (2002)
    Steele, M. 1,2, C. Hurst 3 and J. Chaseling, Simulated Power of Discrete
        Goodness-of-Fit Tests for Likert Type Data

    Examples
    --------

    >>> observed = np.array([ 2.,  4.,  2.,  1.,  1.])
    >>> expected = np.array([ 0.2,  0.2,  0.2,  0.2,  0.2])

    for checking correct dimension with multiple series

    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd='freeman_tukey',axis=1)
    (array([[ 2.745166,  2.745166]]), array([[ 0.6013346,  0.6013346]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected,axis=1)
    (array([[ 2.77258872,  2.77258872]]), array([[ 0.59657359,  0.59657359]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=0,axis=1)
    (array([[ 2.77258872,  2.77258872]]), array([[ 0.59657359,  0.59657359]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=1,axis=1)
    (array([[ 3.,  3.]]), array([[ 0.5578254,  0.5578254]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, 10*expected, lambd=2/3.0,axis=1)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)).T, expected, lambd=2/3.0,axis=1)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))
    >>> powerdiscrepancy(np.column_stack((observed,observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  2.89714546]]), array([[ 0.57518277,  0.57518277]]))

    each random variable can have different total count/sum

    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), 10*expected, lambd=2/3.0, axis=0)
    Traceback (most recent call last):
      ...
    ValueError: observed and expected need to have the samenumber of observations, or e needs to add to 1
    >>> powerdiscrepancy(np.column_stack((2*observed,2*observed)), expected, lambd=2/3.0, axis=0)
    (array([[ 5.79429093,  5.79429093]]), array([[ 0.21504648,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((2*observed,2*observed)), 20*expected, lambd=2/3.0, axis=0)
    (array([[ 5.79429093,  5.79429093]]), array([[ 0.21504648,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), np.column_stack((10*expected,20*expected)), lambd=2/3.0, axis=0)
    (array([[ 2.89714546,  5.79429093]]), array([[ 0.57518277,  0.21504648]]))
    >>> powerdiscrepancy(np.column_stack((observed,2*observed)), np.column_stack((10*expected,20*expected)), lambd=-1, axis=0)
    (array([[ 2.77258872,  5.54517744]]), array([[ 0.59657359,  0.2357868 ]]))


    """
    o = np.array(o)
    e = np.array(e)

    if np.isfinite(lambd) == True:  # check whether lambd is a number
        a = lambd
    else:
        if   lambd == 'loglikeratio': a = 0
        elif lambd == 'freeman_tukey': a = -0.5
        elif lambd == 'pearson': a = 1
        elif lambd == 'modified_loglikeratio': a = -1
        elif lambd == 'cressie_read': a = 2/3.0
        else:
            raise ValueError, 'lambd has to be a number or one of ' + \
                    'loglikeratio, freeman_tukey, pearson, ' +\
                    'modified_loglikeratio or cressie_read'

    n = np.sum(o, axis=axis)
    nt = n
    if n.size>1:
        n = np.atleast_2d(n)
        if axis == 1:
            nt = n.T     # need both for 2d, n and nt for broadcasting
        if e.ndim == 1:
            e = np.atleast_2d(e)
            if axis == 0:
                e = e.T

    if np.all(np.sum(e, axis=axis) == n):
        p = e/(1.0*nt)
    elif np.all(np.sum(e, axis=axis) == 1):
        p = e
        e = nt * e
    else:
        raise ValueError, 'observed and expected need to have the same' \
                          'number of observations, or e needs to add to 1'
    k = o.shape[axis]
    if e.shape[axis] != k:
        raise ValueError, 'observed and expected need to have the same' \
                          'number of bins'

    # Note: taken from formulas, to simplify cancel n
    if a == 0:   # log likelihood ratio
        D_obs = 2*n * np.sum(o/(1.0*nt) * np.log(o/e), axis=axis)
    elif a == -1:  # modified log likelihood ratio
        D_obs = 2*n * np.sum(e/(1.0*nt) * np.log(e/o), axis=axis)
    else:
        D_obs = 2*n/a/(a+1) * np.sum(o/(1.0*nt) * ((o/e)**a - 1), axis=axis)

    return D_obs, stats.chi2.sf(D_obs,k-1)


def itemfreq2d(a, scores=None, axis=0):
    # JP: from scipy.stats.stats.py with changes
    """Returns a 2D array of item frequencies.

    Column 1 contains item values, column 2 to n contain their respective counts.

    Parameters
    ----------
    a : array 1D or 2D
        data, either columns or rows represent variables, see describtion
        for axis
    scores : array_like (optional)
        Contains list or array of items for which frequencies are found. If
        scores is None, then the frequency is found for those items which occur
        at least once. If scores are given, then also items that do not occur
        in the data are listed in the frequency table
    axis : 0, 1 or None
        If axis = 0, then the frequency count is calculated for each column.
        If axis = 1, then the frequency count is calculated for each row.
        If axis = None, then the frequency count is calculated for the entire
            data, (array is raveled first).

    Returns
    -------
    A 2D frequency table (col [0]=scores, col [1:n]=frequencies)

    >>> rvs = np.array([[4, 6, 3], [3, 6, 4], [4, 7, 6], [6, 1, 6]])
    >>> itemfreq2d(rvs,range(1,8),axis=1)
    array([[ 1.,  0.,  0.,  0.,  1.],
           [ 2.,  0.,  0.,  0.,  0.],
           [ 3.,  1.,  1.,  0.,  0.],
           [ 4.,  1.,  1.,  1.,  0.],
           [ 5.,  0.,  0.,  0.,  0.],
           [ 6.,  1.,  1.,  1.,  2.],
           [ 7.,  0.,  0.,  1.,  0.]])
    >>> itemfreq2d(rvs,range(1,8),axis=0)
    array([[ 1.,  0.,  1.,  0.],
           [ 2.,  0.,  0.,  0.],
           [ 3.,  1.,  0.,  1.],
           [ 4.,  2.,  0.,  1.],
           [ 5.,  0.,  0.,  0.],
           [ 6.,  1.,  2.,  2.],
           [ 7.,  0.,  1.,  0.]])
    >>> itemfreq2d(rvs,axis=1)
    array([[ 1.,  0.,  0.,  0.,  1.],
           [ 3.,  1.,  1.,  0.,  0.],
           [ 4.,  1.,  1.,  1.,  0.],
           [ 6.,  1.,  1.,  1.,  2.],
           [ 7.,  0.,  0.,  1.,  0.]])
    >>> itemfreq2d(rvs,axis=0)
    array([[ 1.,  0.,  1.,  0.],
           [ 3.,  1.,  0.,  1.],
           [ 4.,  2.,  0.,  1.],
           [ 6.,  1.,  2.,  2.],
           [ 7.,  0.,  1.,  0.]])
    >>> itemfreq2d(rvs[:,0],range(1,8))
    array([[ 1.,  0.],
           [ 2.,  0.],
           [ 3.,  1.],
           [ 4.,  2.],
           [ 5.,  0.],
           [ 6.,  1.],
           [ 7.,  0.]])
    >>> itemfreq2d(rvs[:,0])
    array([[ 3.,  1.],
           [ 4.,  2.],
           [ 6.,  1.]])
    >>> itemfreq2d(rvs,axis=None)
    array([[ 1.,  1.],
           [ 3.,  2.],
           [ 4.,  3.],
           [ 6.,  5.],
           [ 7.,  1.]])
    >>> itemfreq2d(rvs,range(1,8),axis=None)
    array([[ 1.,  1.],
           [ 2.,  0.],
           [ 3.,  2.],
           [ 4.,  3.],
           [ 5.,  0.],
           [ 6.,  5.],
           [ 7.,  1.]])
    """
    if axis is None:
        a = a.ravel()
        axis = 0
    if a.ndim == 1:
        k = 1
    elif a.ndim == 2:
        k = a.shape[1-axis]
    if scores is None:
        #scores = stats._support.unique(a.ravel())
        scores = np.unique(a.ravel())
        scores = np.sort(scores)
    freq = np.zeros((len(scores),k))
    for i in range(len(scores)):
        freq[i] = np.add.reduce(np.equal(a,scores[i]),axis=axis)
    #return np.array(stats._support.abut(scores, freq))
    return np.array(np.column_stack((scores, freq)))


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)

    #rvsunif = stats.randint.rvs(1,5,size=10)
    rvs0 = np.array([3, 2, 5, 2, 1, 2, 4, 3, 2, 1])
    # get requencies with old version
    freq = {};
    for s in range(1,6): freq[s]=0
    freq.update(stats.itemfreq(rvs0))
    # get requencies with new version
    observed = itemfreq2d(rvs0,range(1,6),axis=0)[:,1]
    expected = np.ones(5)/5.0

    print powerdiscrepancy(observed, expected, lambd=0)
    print 'the next two are identical Pearson chisquare'
    print powerdiscrepancy(observed, expected, lambd=1)
    print stats.chisquare(observed, 10*expected)
    assert_array_almost_equal(powerdiscrepancy(observed, expected, lambd=1),
                        stats.chisquare(observed, 10*expected))
    print 'the next two are identical freeman_tukey'
    print powerdiscrepancy(observed, expected, lambd='freeman_tukey')
    print discrepancy(observed, observed, 10*expected)[0]*4
    assert_array_almost_equal(powerdiscrepancy(observed, expected, lambd='freeman_tukey')[0],
                        discrepancy(observed, observed, 10*expected)[0]*4)
    powerdiscrepancy(np.column_stack((observed,observed)), 10*expected[:,np.newaxis], lambd='freeman_tukey')
    powerdiscrepancy(np.column_stack((observed,2*observed)), np.column_stack((10*expected,20*expected)), lambd=-1, axis=0)
    #print powerdiscrepancy(np.array(sorted(freq.items())), [1/4.0]*4, lambd=0)

##    print powerdiscrepancy(np.array(sorted(freq.items()))[:,1], np.asarray([1/4.0]*4), lambd=0)
##    print 'the next two are identical Pearson chisquare'
##    print powerdiscrepancy(np.array(sorted(freq.items()))[:,1], np.asarray([1/4.0]*4), lambd=1)
##    print stats.chisquare(np.array(sorted(freq.items()))[:,1], 10*np.asarray([1/4.0]*4))
##    print 'the next two are identical freeman_tukey'
##    print powerdiscrepancy(np.array(sorted(freq.items()))[:,1], np.asarray([1/4.0]*4), lambd='freeman_tukey')
##    print discrepancy(np.array(sorted(freq.items()))[:,1],np.array(sorted(freq.items()))[:,1], 10*np.asarray([1/4.0]*4) )[0]*4

