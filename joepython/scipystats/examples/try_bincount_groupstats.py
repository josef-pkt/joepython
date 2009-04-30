'''
group statistics using np.bincout

'''

import numpy as np

indices = np.random.randint(3,size=20)
values = np.arange(20) #np.ones(20)

ix,rind = np.unique1d(indices, return_inverse=1)
reverse_index = np.searchsorted(ix, indices)
s2 = np.bincount(reverse_index, weights=values*values)
s1 = np.bincount(reverse_index, weights=values)

print np.all(rind==reverse_index)

def groupstatsbin(factors, values):
    n = len(factors)
    ix,rind = np.unique1d(factors, return_inverse=1)
    gcount = np.bincount(rind)
    gmean = np.bincount(rind, weights=values)/ (1.0*gcount)
    meanarr = gmean[rind]
    withinvar = np.bincount(rind, weights=(values-meanarr)**2) / (1.0*gcount)
    withinvararr = withinvar[rind]
    return gcount, gmean , meanarr, withinvar, withinvararr


gcount, gmean , meanarr, withinvar, withinvararr = groupstatsbin(indices, values)

print 'group sum and means'
print s1
print np.sum(values[indices==0])
print gmean
print np.mean(values[indices==0])
print meanarr[indices==0][:10]
print 'group sum of squares and variance'
print s2
print np.sum(values[indices==0]**2)
print withinvar
print ((values-meanarr)**2)[indices==0].mean()
print withinvararr[indices==0][:10]
