'''calculating anova and verifying with NIST test data

note:
this requires try_catdata.py to run first to have the functions of it
available in __main__

compares my implementations, stats.f_oneway and oneway_anova of pymvpa (rewritten)
'''
import numpy as np

filenameli = ['SiRstv.dat', 'SmLs01.dat', 'SmLs02.dat', 'SmLs03.dat', 'AtmWtAg.dat',
              'SmLs04.dat', 'SmLs05.dat', 'SmLs06.dat', 'SmLs07.dat', 'SmLs08.dat', 
              'SmLs09.dat']
filename = 'SmLs03.dat' #'SiRstv.dat' #'SmLs09.dat'#, 'AtmWtAg.dat' #'SmLs07.dat'
content = file(filename,'r').read().split('\n')

data = [line.split() for line in content[60:]]

certified = [line.split() for line in content[40:48] if line]
dataf = np.loadtxt(filename, skiprows=60)
y,x = dataf.T
y = y.astype(int)
caty = np.unique(y)

def getnist(filename):
    content = file(filename,'r').read().split('\n')
    data = [line.split() for line in content[60:]]
    certified = [line.split() for line in content[40:48] if line]
    dataf = np.loadtxt(filename, skiprows=60)
    y,x = dataf.T
    y = y.astype(int)
    caty = np.unique(y)
    f = float(certified[0][-1])
    R2 = float(certified[2][-1])
    resstd = float(certified[4][-1])
    dfbn = int(certified[0][-4])
    dfbn = int(certified[1][-3])
    prob = stats.f.sf(f,dfbn,dfwn)
    return y, x, np.array([f, prob, R2, resstd]), certified, caty
    


# simple slow version
xlist = []
for ii in caty:
    xlist.append(x[y==ii])#- 999900.0)

from scipy import stats
from try_catdata import oneway_anova, groupsstats_dummy, groupstatsbin, anova_1w

print stats.f_oneway(*xlist) # stats.f_oneway wants categories separately

print oneway_anova(y[:,np.newaxis], x[:,np.newaxis], np.arange(1,9))

yrvs = y[:,np.newaxis] - 1
xrvs = x[:,np.newaxis]
nobs = yrvs.shape[0] # added late without checking
#replicating stats.f_oneway()
# vectorize correctly and make function
sstot = nobs * np.var(xrvs[:,:1])   #sstot = np.var(alldata)*bign
meang, varg, xdevmeangr, countg = groupsstats_dummy(yrvs[:,:1], xrvs[:,:1])
ncat = countg.shape[0]
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
R2 = (ssbn/(sswn+ssbn)).size  #R-squared
resstd = np.sqrt(msw) #residual standard deviation
print f, prob



def anova_oneway(y, x, seq=0):
    # new version to match NIST
    # no generalization or checking of arguments, tested only for 1d 
    yrvs = y[:,np.newaxis] #- min(y)
    #subracting mean increases numerical accuracy for NIST test data sets
    xrvs = x[:,np.newaxis] - x.mean() #for 1d#- 1e12  trick for 'SmLs09.dat'

    meang, varg, xdevmeangr, countg = groupsstats_dummy(yrvs[:,:1], xrvs[:,:1])#, seq=0)
    #the following does not work as replacement
    #gcount, gmean , meanarr, withinvar, withinvararr = groupstatsbin(y, x)#, seq=0)
    sswn = np.dot(xdevmeangr.T,xdevmeangr)
    ssbn = np.dot((meang-xrvs.mean())**2, countg.T)
    nobs = yrvs.shape[0]
    ncat = meang.shape[1]
    dfbn = ncat - 1
    dfwn = nobs - ncat
    msb = ssbn/float(dfbn)
    msw = sswn/float(dfwn)
    f = msb/msw
    prob = stats.f.sf(f,dfbn,dfwn)
    R2 = (ssbn/(sswn+ssbn))  #R-squared
    resstd = np.sqrt(msw) #residual standard deviation
    #print f, prob
    def _fix2scalar(z): # return number
        if np.shape(z) == (1, 1): return z[0,0]
        else: return z
    f, prob, R2, resstd = map(_fix2scalar, (f, prob, R2, resstd))
    return f, prob, R2, resstd

print "certified", float(certified[0][-1])

for fn in filenameli:
    y, x, cert, certified, caty = getnist(fn)
    res = anova_oneway(y, x)
    print np.array(res) - cert

for fn in filenameli:
    y, x, cert, certified, caty = getnist(fn)
    xlist = [x[y==ii] for ii in caty]
    res = stats.f_oneway(*xlist)
    print np.array(res) - cert[:2]

class GroupedDataStat(object):
    pass

class DiscreteData(object):
    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.xbyy = GroupedDataStat()
        self.xbyy.meang, self.xbyy.varg, self.xbyy.xdevmeangr, self.xbyy.countg = \
                         groupsstats_dummy(y, x, seq=0)
        self.xbyy.junk
        #unfinished stopped here in the middle
    
