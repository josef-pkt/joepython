'''

Author: JP
'''

import numpy as np

data = np.random.randint(1,3, size=(10,5))
keep = [1, 4]     # index in data of explanatory variable under consideration
dv = 0            # index in data of dependent variable

# build dictionary with unique combination as keys
#   and corresponding data as values
result = {}
for row in data:
    print row
    result.setdefault(tuple(row[ [1, 4]]),[]).append(row)

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



import itertools, operator
stat2 = []
#sort rows, use numpy instead
datal = np.array(sorted(list(data), key=lambda(x):repr(x[[1,4]])))
for k, v in itertools.groupby(datal, lambda(x):repr(x[[1,4]])): 
    v2 = list(v)
    print k, list(v2)
    print np.array(list(v))
    y = np.array(v2)[:,dv]
    print y,y.mean(), y.std()
    z = np.array(v2)[0,keep]
    #stat2.append(list(k) + [y.mean(), y.std(), y.shape[0]])
    stat2.append(z.tolist() + [y.mean(), y.std(), y.shape[0]])
stat2.sort()
stat2n = np.array(stat2)

print "combination                mean        var         count"  
print statn
assert np.all(stat2n == statn)
