
import numpy as np
#from numpy import linalg as npla
from scipy import stats, optimize

'''
Working with categorical data
=============================

use of dummy variables, group statistics, within and between statistics
examples for efficient matrix algebra

dummy versions require that the number of unique groups or categories is not too large
group statistics with scipy.ndimage can handle large number of observations and groups
scipy.ndimage stats is missing count

new: np.bincount can also be used for calculating values per label

compares Anova implementation: pymvpa, stats and mine
somewhere I have certified test results for balanced anova from nist

'''

from scipy import ndimage

#problem: ndimage does not allow axis argument,
#   calculates mean or var corresponding to axis=None in np.mean, np.var
#   useless for multivariate application

def labelmeanfilter(y, x):
   # requires integer labels
   # from mailing list scipy-user 2009-02-11
   labelsunique = np.arange(np.max(y)+1)
   labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
   # returns label means for each original observation
   return labelmeans[y]

#groupcount: i.e. number of observation by group/label
#np.array(ndimage.histogram(yrvs[:,0],0,10,1,labels=yrvs[:,0],index=np.unique(yrvs[:,0])))

def labelmeanfilter_nd(y, x):
   # requires integer labels
   # from mailing list scipy-user 2009-02-11
   # adjusted for 2d x with column variables
   
   labelsunique = np.arange(np.max(y)+1)
   labmeansdata = []
   labmeans = []

   for xx in x.T:
      labelmeans = np.array(ndimage.mean(xx, labels=y, index=labelsunique))
      labmeansdata.append(labelmeans[y])
      labmeans.append(labelmeans)
   # group count:
   labelcount = np.array(ndimage.histogram(y, labelsunique[0], labelsunique[-1]+1,
                        1, labels=y, index=labelsunique))
      
   # returns array of lable/group counts and of label/group means
   #         and label/group means for each original observation
   return labelcount, np.array(labmeans), np.array(labmeansdata).T

def labelmeanfilter_str(ys, x):
    # works also for string labels in ys, but requires 1D
    # from mailing list scipy-user 2009-02-11
    unil, unilinv = np.unique1d(ys, return_index=False, return_inverse=True)
    labelmeans = np.array(ndimage.mean(x, labels=unilinv, index=np.arange(np.max(unil)+1)))
    arr3 = labelmeans[unilinv]
    return arr3

def groupstatsbin(factors, values):
    '''uses np.bincount, assumes factors/labels are integers
    '''
    n = len(factors)
    ix,rind = np.unique1d(factors, return_inverse=1)
    gcount = np.bincount(rind)
    gmean = np.bincount(rind, weights=values)/ (1.0*gcount)
    meanarr = gmean[rind]
    withinvar = np.bincount(rind, weights=(values-meanarr)**2) / (1.0*gcount)
    withinvararr = withinvar[rind]
    return gcount, gmean , meanarr, withinvar, withinvararr


def oneway_anova(labels, samples, uniquelabels):
    """`FeaturewiseDatasetMeasure` that performs a univariate ANOVA.

    F-scores are computed for each feature as the standard fraction of between
    and within group variances. Groups are defined by samples with unique
    labels.

    No statistical testing is performed, but raw F-scores are returned as a
    sensitivity map. As usual F-scores have a range of [0,inf] with greater
    values indicating higher sensitivity.

    Computes featurewise f-scores.
    Note: extracted from pymvpa for comparison by JP
    """
    # group means
    means = []
    # with group variance
    vars_ = []

    # split by groups -> [groups x [samples x features]]
    for ul in uniquelabels:
        ul_samples = samples[(labels == ul)[:,0],:]
        print ul, ul_samples.shape
        means.append(ul_samples.mean(axis=0))
        vars_.append(ul_samples.var(axis=0))

    # mean of within group variances
    mvw = np.array(vars_).mean(axis=0)
    # variance of group means
    vgm = np.array(means).var(axis=0)

    # compute f-scores (in-place to save some cycles)
    # XXX may cause problems when there are features with no variance in
    # some groups. One could deal with them here and possibly assign a
    # zero f-score to throw them out, but at least theoretically zero
    # variance is possible. Another possiblilty could be to apply
    # N.nan_to_num(), but this might hide the problem.
    # Michael therefore thinks that it is best to let the user deal with
    # it prior to any analysis.

    # for features where there is no variance between the groups,
    # we should simply leave 0 as is, and avoid that way NaNs for
    # invariance features
    vgm0 = vgm.nonzero()
    vgm[vgm0] /= mvw[vgm0]

    return vgm


def convertlabels(ys, indices=None):
    '''convert labels based on multiple variables or string labels to unique
    index labels 0,1,2,...,nk-1 where nk is the number of distinct labels
    '''
    if indices == None:
        ylabel = ys
    else:
        idx = np.array(indices)
        if idx.size > 1 and ys.ndim == 2:
            ylabel = np.array(['@%s@'%ii[:2].tostring() for ii in ys])[:,np.newaxis]
            #alternative
    ##        if ys[:,idx].dtype.kind == 'S':
    ##            ylabel = nd.array([' '.join(ii[:2]) for ii in ys])[:,np.newaxis]
        else:
            # there might be a problem here
            ylabel = ys
        
    unil, unilinv = np.unique1d(ylabel, return_index=False, return_inverse=True)
    return unilinv, np.arange(len(unil)), unil
    
def groupsstats_1d(y, x, labelsunique):
    '''use ndimage to get fast mean and variance'''
    labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
    labelvars = np.array(ndimage.var(x, labels=y, index=labelsunique))
    return labelmeans, labelvars

def cat2dummy(y, nonseq=0):
    if nonseq or (y.ndim == 2 and y.shape[1] > 1):
        ycat, uniques, unitransl =  convertlabels(y, range(y.shape[1]))
    else:
        ycat = y.copy()
        ymin = y.min()
        uniques = np.arange(ymin,y.max()+1)
    if ycat.ndim == 1:
        ycat = ycat[:,np.newaxis]
    # this builds matrix nobs*ncat
    dummy = (ycat == uniques).astype(int)
    return dummy

def groupsstats_dummy(y, x, nonseq=0):
    if x.ndim == 1:
        # use groupsstats_1d
        x = x[:,np.newaxis]
    dummy = cat2dummy(y, nonseq=nonseq)
    countgr = dummy.sum(0, dtype=float)
    meangr = np.dot(x.T,dummy)/countgr
    meandata = np.dot(dummy,meangr.T) # category/group means as array in shape of x
    xdevmeangr = x - meandata  # deviation from category/group mean
    vargr = np.dot((xdevmeangr * xdevmeangr).T, dummy) / countgr
    return meangr, vargr, xdevmeangr, countgr
    
    
    
    
    
    


def anova_1w(y, x, labelsunique):
    '''basic copied from pymvpa

    problem with small sample: when there is only one observation in a group
    then variance is zero
    '''
    # group mean and variances
    meang, varg, xdevmeangr, countg = groupsstats_dummy(y, x)
    
    # mean of within group variances
    mvw = np.array(varg).mean(axis=1)
    # variance of group means
    vgm = np.array(meang).var(axis=1)
    vgm0 = vgm.nonzero()
    vgm[vgm0] /= mvw[vgm0]
    return vgm

nobs = 200
ncat = 5
yrvs = np.random.randint(ncat, size=(nobs,2))
xrvs = np.random.randn(nobs, 3) + 0.1* yrvs.sum(1)[:,np.newaxis]
import time
t = time.time()
print anova_1w(yrvs[:,:1], xrvs, np.arange(ncat))
print time.time() - t
t = time.time()
print oneway_anova(yrvs[:,:1], xrvs, np.arange(ncat))
print time.time() - t

## two-way anova, two category variables (label, group variables)
yrvs3 = np.mod(yrvs,3)  # use only 3 categories to reduce size
meang, varg, xdevmeangr, countg = groupsstats_dummy(yrvs3, xrvs)
dummy3 = cat2dummy(yrvs3)
print 'dummy3.shape', dummy3.shape
print 'group means: meang.T'
print meang.T
print 'variance of group means divided by mean of group variances'
print meang.T.var(0)/varg.T.mean(0) 
print anova_1w(np.mod(yrvs,3), xrvs, np.arange(ncat))

ycat, uniques, unitransl =  convertlabels(yrvs3, range(yrvs3.shape[1]))
print 'ycat.shape, uniques.shape, unitransl.shape'
print ycat.shape, uniques.shape, unitransl.shape
print uniques
print unitransl
#Note: with tostring/fromstring, I need to know the dtype,
#      could use dumps/loads instead, which saves type information but is longer
uniquecats = np.array([np.fromstring(ii[1:-1], dtype=int) for ii in unitransl])
print uniquecats


xlist = [xrvs[(yrvs[:,0]==ii),:1] for ii in range(ncat)]
print stats.f_oneway(*xlist)#[0],xlist[1])

#replicating stats.f_oneway()
#TODO: vectorize correctly and make function
sstot = nobs * np.var(xrvs[:,:1])   #sstot = np.var(alldata)*bign
meang, varg, xdevmeangr, countg = groupsstats_dummy(yrvs[:,:1], xrvs[:,:1])
sswn = nobs * np.var(xdevmeangr)
ssbn0 = np.sum(countg*np.array(meang)**2) - np.sum(np.array(meang)*countg)**2/float(nobs)
ssbn1 = np.sum(countg*np.array(meang)**2) - nobs * xrvs[:,:1].mean()**2
ssbn = sstot - sswn
dfbn = ncat - 1
dfwn = nobs - ncat
msb = ssbn/float(dfbn)
msw = sswn/float(dfwn)
f = msb/msw
prob = stats.f.sf(f,dfbn,dfwn)
print f, prob

grcount, grm, grmarr = labelmeanfilter_nd(yrvs[:,0], xrvs)
print grcount
print np.sum(yrvs[:,0]==0)
print grm
