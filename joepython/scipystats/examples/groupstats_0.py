'''calculate basic statistics of a dependent variable conditional on some
explanatory variables while ignoring other explanatory variables.

works only for discrete data

see:
http://alexholcombe.wordpress.com/2009/01/26/summarizing-data-by-combinations-of-variables-with-python/
http://projects.scipy.org/pipermail/scipy-dev/2009-January/010890.html
'''

import numpy as np
from numpy.testing import assert_array_equal, assert_equal

data = np.random.randint(1,3, size=(10,5))
data[:,1] += 10
keep = [1, 4]     # index in data of explanatory variable under consideration
dv = 0            # index in data of dependent variable


# version1: using dictionary to store data rows
#----------------------------------------------

# build dictionary with unique combination as keys
#   and corresponding data as values
result = {}
for row in data:
    print row
    result.setdefault(tuple(row[ keep]),[]).append(row)

# calculate statistics for each combination (key)
stat = []
for k,v in sorted(result.iteritems()):
    y = np.asarray(v)[:,dv]
    stat.append(list(k) + [y.mean(), y.std(), y.shape[0]])

# convert result statistic to numpy arrays
statn = np.array(stat)

print "combination                mean        var         count"  
print statn
assert np.sum(statn[:,len(keep)]*statn[:,-1])/data.shape[0] \
           == data[:,dv].mean()

# version1b: using dictionary to store indices to data
#-----------------------------------------------------

# build dictionary with unique combination as keys
#   and corresponding row indices as values
result1 = {}
for index, row in enumerate(data):
    result1.setdefault(tuple(row[keep]),[]).append(index)

# calculate statistics for each combination (key)
stat1 = []
for k,v in sorted(result1.iteritems()):
    m = data[v,dv].mean()
    s = data[v,dv].std()
    stat1.append(list(k) + [m, s, len(v)])

# convert result statistic to numpy arrays
stat1n = np.array(stat1)

print "combination                mean        var         count"  
print stat1n

assert np.all(stat1n == statn)

def flat2nd(x):
    '''convert flat table to multidimensional table

    Parameters
    ----------
    x array (N,K+1)
         flat table [x1,x2,y]

    returns
    -------
    res : array
        contains variable of last column in input reformated to have
        K dimensions with rows and columns according to unique 
        

    uns: list of K 1D arrays    

    Example
    -------
    >>> mex = np.array([[ 11.,   1.,   1.],
                        [ 11.,   2.,   2.],
                        [ 12.,   1.,   3.],
                        [ 12.,   2.,   4.]])
    >>> res, unirs, uns = flat2nd(mex)
    >>> res
    array([[ 1.,  2.],
           [ 3.,  4.]])
    >>> uns
    [array([ 11.,  12.]), array([ 1.,  2.])]
    '''
    uns = []
    unirs = []
    dims = []
    #K = 
    for ii in range(x.shape[1]-1):        
        un, unir = np.unique1d(x[:,ii], return_inverse=True)
        uns.append(un)
        unirs.append(unir)
        dims.append(len(un))
    
    res = np.nan * np.ones(dims)
    res[zip(unirs)]=x[:,-1]
    return res, uns





# version2: using itertools groupby
#----------------------------------

import itertools

#sort rows, can use numpy instead
datasorted = np.array(sorted(list(data), key=lambda(x):repr(x[keep])))
#use repr in sort key to avoid numpy array comparison

stat2 = []
for k, v in itertools.groupby(datasorted, lambda(x):repr(x[keep])): 
    v2 = np.array(list(v))
    y = v2[:,dv]
    stat2.append(v2[0,keep].tolist() + [y.mean(), y.std(), y.shape[0]])

stat2n = np.array(stat2)

print "combination                mean        var         count"  
print statn
assert np.all(stat2n == statn)




def test_flat2multi():
    mex = np.array([[ 11.,   1.,   1.],
                    [ 11.,   2.,   2.],
                    [ 12.,   1.,   3.],
                    [ 12.,   2.,   4.]])
    res, uns = flat2nd(mex)
    assert_array_equal(res, np.array([[ 1.,  2.], [ 3.,  4.]]))
    assert_equal(uns, [np.array([ 11.,  12.]), np.array([ 1.,  2.])])

test_flat2multi()
    
