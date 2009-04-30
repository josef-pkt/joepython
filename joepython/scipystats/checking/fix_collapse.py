
'''
Examples
--------

import numpy as np
from scipy import stats

xx = np.array([[ 0.,  0.,  1.],
       [ 1.,  1.,  1.],
       [ 2.,  2.,  1.],
       [ 0.,  3.,  1.],
       [ 1.,  4.,  1.],
       [ 2.,  5.,  1.],
       [ 0.,  6.,  1.],
       [ 1.,  7.,  1.],
       [ 2.,  8.,  1.],
       [ 0.,  9.,  1.]])

>>> stats._support.collapse(xx, (0), (1,2), stderr=0, ns=0, cfcn=None)
array([[ 0. ,  4.5,  1. ],
       [ 0. ,  4.5,  1. ],
       [ 1. ,  4. ,  1. ],
       [ 1. ,  4. ,  1. ],
       [ 2. ,  5. ,  1. ],
       [ 2. ,  5. ,  1. ]])
>>> stats._support.collapse(xx, (0), (1,2), stderr=1, ns=1, cfcn=None)
array([[ 0.        ,  4.5       ,  1.93649167,  4.        ,  1.        ,
         0.        ,  4.        ],
       [ 0.        ,  4.5       ,  1.93649167,  4.        ,  1.        ,
         0.        ,  4.        ],
       [ 1.        ,  4.        ,  1.73205081,  3.        ,  1.        ,
         0.        ,  3.        ],
       [ 1.        ,  4.        ,  1.73205081,  3.        ,  1.        ,
         0.        ,  3.        ],
       [ 2.        ,  5.        ,  1.73205081,  3.        ,  1.        ,
         0.        ,  3.        ],
       [ 2.        ,  5.        ,  1.73205081,  3.        ,  1.        ,
         0.        ,  3.        ]])

'''

import numpy as np
from scipy import stats

x,y = np.mgrid[0:3,0:3]
xx = np.vstack((x.flatten(), y.flatten(), np.ones(9))).T.copy()
print stats._support.collapse(xx, (0), (1,2), stderr=0, ns=0, cfcn=None)
print stats._support.collapse(xx, (0,1), (2,), stderr=0, ns=0, cfcn=None)

x,y = np.mgrid[0:3,0:3]
y = np.minimum(y,1.0)
xx = np.vstack((x.flatten(), y.flatten(), np.ones(9))).T.copy()
print stats._support.collapse(xx, (0), (1,2), stderr=0, ns=0, cfcn=None)
print stats._support.collapse(xx, (0,1), (2,), stderr=0, ns=0, cfcn=None)
