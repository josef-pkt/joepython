# only tested for 2D, i.e. 2 explanatory variables
# no pretty print


import numpy as np
from numpy.testing import assert_array_equal, assert_equal


def ptable(data, dv, keep, outformat='flat'):
    '''calculate basic statistics for pivot table

    Mean, standard deviation and count for a dependent variable
    conditional on some explanatory variables while ignoring other
    explanatory variables.

    This works only for discrete values of explanatory variables

    Parameters
    ----------
    data : 2D array
        assumes variables are in columns and observations in rows
    dv : int
        column index of dependent variable
    keep : array_like int
        column indices of explanatory variables
    outformat : (optional)
        * 'flat' (default) :
          return 2D array with unique values of explanatory variables
          in first columns and statistics in later columns
        * 'table'
          
    Returns
    -------
    statarr: 2D array  if outformat = 'flat'
    {uns, mmean, mstd, mcount} if outformat = 'table'
    

    '''
    # build dictionary with unique combination as keys
    #   and corresponding row indices as values
    catdata = {}
    for index, row in enumerate(data):
        catdata.setdefault(tuple(row[keep]),[]).append(index)

    # calculate statistics for each combination (key)
    stat = []
    for k,v in sorted(catdata.iteritems()):
        m = data[v,dv].mean()
        s = data[v,dv].std()
        stat.append(list(k) + [m, s, len(v)])

    # convert result statistic to numpy arrays
    statarr = np.array(stat)
    
    if outformat == 'flat':
        return statarr
    elif outformat == 'table':
        # convert flat table to multidimensional
        K = len(keep)
        mmean, uns = flat2multi(statarr[:,range(K)+[K]])
        mstd, uns = flat2multi(statarr[:,range(K)+[K+1]])
        mcount, uns = flat2multi(statarr[:,range(K)+[K+2]])
        return uns, mmean, mstd, mcount
    else:
        raise ValueError, "outformat can only be 'flat' or 'table'"
        


def flat2nd(x):
    '''convert flat table to multidimensional table

    Assumes rows on first K columns are jointly unique.
    Flat table does not need to have complete, i.e. rectangular, values
    for explanatory variables. Missing elements are filled with NaN.

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
        element i of uns is 1D array of values of the explanatory variable
        for the ith axis of `res`

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

    example with unequal dimension and not rectangular
    
    >>> mex = np.array([[ 11.,   1.,   1.],
                        [ 11.,   2.,   2.],
                        [ 12.,   1.,   3.],
                        [ 12.,   2.,   4.],
                        [ 13.,   2.,   5.],])
    >>> res, unirs, uns = flat2nd(mex)
    >>> res
    array([[  1.,   2.],
           [  3.,   4.],
           [ NaN,   5.]])
    >>> uns
    [array([ 11.,  12.,  13.]), array([ 1.,  2.])]
    '''
    uns = []
    unirs = []
    dims = []
    for ii in range(x.shape[1]-1):        
        un, unir = np.unique1d(x[:,ii], return_inverse=True)
        uns.append(un)
        unirs.append(unir)
        dims.append(len(un))
    
    res = np.nan * np.ones(dims)
    res[zip(unirs)]=x[:,-1]
    return res, uns



def test_flat2multi():
    mex = np.array([[ 11.,   1.,   1.],
                    [ 11.,   2.,   2.],
                    [ 12.,   1.,   3.],
                    [ 12.,   2.,   4.]])
    res, uns = flat2nd(mex)
    assert_array_equal(res, np.array([[ 1.,  2.], [ 3.,  4.]]))
    assert_equal(uns, [np.array([ 11.,  12.]), np.array([ 1.,  2.])])



def table2flat(res,uns):
    '''flatten a table
    works only for 2D because of meshgrid
    '''
    F = np.column_stack(map(lambda(x):np.ravel(x,order='F'),np.meshgrid(*uns)))
    return np.c_[F,res[ii2]]

def ndtable2flat(uns,*res):
    '''flatten a table
    works only for 2D because of meshgrid
    '''
    return np.column_stack([np.array(veccombo(uns))]+map(np.ravel,res)).copy()
    #return np.c_[np.array(veccombo(uns)),res.ravel()]  #[ii2]]

def veccombo(seq,y=None):
    '''create all combinations from list of lists or 1D arrays

     Given list [a1,a2,a3] where each `a` is a list or 1D array, find
     all combinations (a1_i, a2_j, a3_k) for all a1_i in a1, a2_j in a2 and
     a3_k in a3.  

    Like np.meshgrid but unrestricted dimension and each dimension is
    flattened and stacked into one array. Elements of one array are in
    columns. If original lists are not unique, then the return will
    have repeated values.


    Parameters
    ----------
    seq : iterable sequence of array_lile
        args needs to be a iterable of iterable
        typical use: list of 1D arrays or list of lists

    Returns
    -------
    res : 2D list of lists
        using np.array(res) this can be converted to a 2D array, where each
        column contains values from the corresponding list in seq.


    Example
    -------
    >>> veccombo([[11,12,13], [21,22]])
    [[11, 21], [11, 22], [12, 21], [12, 22], [13, 21], [13, 22]]
    >>> np.array(veccombo([[11,12,13],[21,22]]))
    array([[11, 21],
           [11, 22],
           [12, 21],
           [12, 22],
           [13, 21],
           [13, 22]])

    >>> np.array(veccombo([[1,2], [1,2], [1,2,3]]))
    array([[1, 1, 1],
           [1, 1, 2],
           [1, 1, 3],
           [1, 2, 1],
           [1, 2, 2],
           [1, 2, 3],
           [2, 1, 1],
           [2, 1, 2],
           [2, 1, 3],
           [2, 2, 1],
           [2, 2, 2],
           [2, 2, 3]])

    If original lists are not unique, then the values are repeated
    >>> np.array(veccombo([[1,2], [1,1,3]]))
    array([[1, 1],
           [1, 1],
           [1, 3],
           [2, 1],
           [2, 1],
           [2, 3]])
    
    '''
    res = []
    n = len(seq)
    if y is None:
        #initialize last list to list of lists
        yn = [(hasattr(ii,'__iter__') and list(ii)) or [ii] for ii in seq[-1]]
        return veccombo(seq[:-1],y=yn)
    # prepend current list
    for ii in seq[-1]:
        for jj in y:
            res.append([ii] + jj)

    if n > 2:
        return veccombo(seq[:-1],y=res[:])
    elif n == 2:
        # args needs to be an iterable of iterable(s)
        return veccombo([seq[0]],y=res[:])
    else:
        return res

def veccombo_withversions(args,y=None):
    #print args
    res = []
    n = len(args)
##    if args == []:
##        return y
    if y is None:
        #this requires last of args to be 1D np.array
        yn = [(hasattr(ii,'__iter__') and list(ii)) or [ii] for ii in args[-1]]
        #print 'yn init', yn
        return veccombo(args[:-1],y=yn) #args[-1][:,np.newaxis].tolist())
    for ii in args[-1]:
        for jj in y:
            #print [ii],jj
            res.append([ii] + jj)
##            jj.insert(0,ii)
##            print 'jjnew', jj
##            res.append(jj)
    if n > 2:
        return veccombo(args[:-1],y=res[:])
    elif n == 2:
        #print 'in n==2'
        return veccombo([args[0]],y=res[:])
    else:
        return res
    
    
            
    

if __name__ == '__main__':
    test_flat2multi()

    data = np.random.randint(1,3, size=(10,5))
    data[:,1] += 10
    keep = [1, 4]     # index in data of explanatory variable under consideration
    dv = 0            # index in data of dependent variable
    statn = ptable(data, dv, keep, outformat='flat')
    print statn
    uns0, mmean, mstd, mcount = ptable(data, dv, keep, outformat='table')
    print uns0
    print mmean
    print mstd
    print mcount
    statnr = ndtable2flat(uns0, mmean, mstd, mcount)
    statnrr = statnr[~np.isnan(statnr).any(1),:]
    assert_array_equal(statn,statnrr)
    

    mex = np.array([[ 11.,   1.,   1.],
                    [ 11.,   2.,   2.],
                    [ 12.,   1.,   3.],
                    [ 12.,   2.,   4.],
                    [ 13.,   2.,   5.],])
    res, uns = flat2nd(mex)
    print uns
    print res
    mexi = table2flat(res,uns)
    nanrows = np.isnan(mexi).any(1)
    mexir = mexi[~nanrows,:]
    assert_array_equal(mex,mexir)


    mexi2 = ndtable2flat(uns,res)
    nanrows = np.isnan(mexi2).any(1)
    mexi2r = mexi2[~nanrows,:]
    assert_array_equal(mex,mexi2r)

    

    mex = np.array([[ 11.,   1.,   1.,   1.],
                    [ 11.,   2.,   1.,   2.],
                    [ 12.,   1.,   1.,   3.],
                    [ 12.,   2.,   2.,   4.],
                    [ 13.,   2.,   2.,   5.],])
    res, uns = flat2nd(mex)
    print uns
    print res
    # calling with `uns` is list of np.arrays
    resarr = np.array(veccombo(uns))
    # calling with `uns` is list of lists
    assert_array_equal(resarr,np.array(veccombo(map(list,uns))))
    print resarr
    
    mexi3 = ndtable2flat(uns,res)
    mexi3r = mexi3[~np.isnan(mexi3).any(1),:]
    assert_array_equal(mex,mexi3r)


