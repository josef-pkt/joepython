#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Josef Perktold
#
# Created:     08/04/2009
# Copyright:   (c) Josef Perktold 2009
# Licence:     BSD
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#docstr = \
'''
>>> import numpy as np
>>> from scipy import stats
>>> x,y = np.mgrid[0:3,0:3]
>>> xx = np.vstack((x.flatten(), y.flatten(), np.ones(9))).T
>>> xx
array([[ 0.,  0.,  1.],
       [ 0.,  1.,  1.],
       [ 0.,  2.,  1.],
       [ 1.,  0.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  2.,  1.],
       [ 2.,  0.,  1.],
       [ 2.,  1.,  1.],
       [ 2.,  2.,  1.]])
>>> stats._support.collapse(xx, (0), (1,2), stderr=0, ns=0, cfcn=None)
array([[ 0.,  1.,  1.],
       [ 0.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 1.,  1.,  1.],
       [ 2.,  1.,  1.],
       [ 2.,  1.,  1.]])
>>> stats._support.collapse(xx[:,:2], (0), (1,), stderr=1, ns=1, cfcn=None)
array([[ 0.        ,  1.        ,  0.57735027,  3.        ],
       [ 1.        ,  1.        ,  0.57735027,  3.        ],
       [ 2.        ,  1.        ,  0.57735027,  3.        ]])
>>> np.var([0,1,2])
0.66666666666666663
>>> np.var([0,1,2],ddof=1)
1.0
>>> np.var([0,1,2],ddof=1)/3
0.33333333333333331
>>> np.sqrt(np.var([0,1,2],ddof=1)/3)
0.57735026918962573
>>> np.sqrt(np.var([0,1,2],ddof=0)/3)
0.47140452079103168

>>> varfn = lambda(x): np.var(x,axis=0)
>>> stats._support.collapse(xx[:,:2], (0), (1,), stderr=1, ns=1, cfcn=varfn)
array([[ 0.        ,  0.66666667,  0.57735027,  3.        ],
       [ 1.        ,  0.66666667,  0.57735027,  3.        ],
       [ 2.        ,  0.66666667,  0.57735027,  3.        ]])

>>> xx1=xx[xx[:,0]==0,1]
>>> np.var(xx1)
0.66666666666666663
>>> stats._support.collapse(xx[:,:2], (0), (1,), stderr=0, ns=1, cfcn=varfn)
array([[ 0.        ,  0.66666667,  3.        ],
       [ 1.        ,  0.66666667,  3.        ],
       [ 2.        ,  0.66666667,  3.        ]])
>>> stats._support.collapse(xx[:,:], (0), (1,2), stderr=0, ns=1, cfcn=varfn)
array([[ 0.        ,  0.66666667,  3.        ,  0.        ,  3.        ],
       [ 0.        ,  0.66666667,  3.        ,  0.        ,  3.        ],
       [ 1.        ,  0.66666667,  3.        ,  0.        ,  3.        ],
       [ 1.        ,  0.66666667,  3.        ,  0.        ,  3.        ],
       [ 2.        ,  0.66666667,  3.        ,  0.        ,  3.        ],
       [ 2.        ,  0.66666667,  3.        ,  0.        ,  3.        ]])
>>> stats._support.collapse(xx[:,:], (0), (1,2), stderr=0, ns=1, cfcn=None)
array([[ 0.,  1.,  3.,  1.,  3.],
       [ 0.,  1.,  3.,  1.,  3.],
       [ 1.,  1.,  3.,  1.,  3.],
       [ 1.,  1.,  3.,  1.,  3.],
       [ 2.,  1.,  3.,  1.,  3.],
       [ 2.,  1.,  3.,  1.,  3.]])
'''
import doctest
codestr = '\n'.join(line for line in doctest.script_from_examples(docstr).split('\n')
            if line[:1] != '#')
#print '\n'.join(codestr)

doctest.testmod(verbose=3)
