


def itemfreq(a):
    # fixme: I'm not sure I understand what this does. The docstring is
    # internally inconsistent.
    # comment: fortunately, this function doesn't appear to be used elsewhere
    """Returns a 2D array of item frequencies.

    Column 1 contains item values, column 2 contains their respective counts.
    Assumes a 1D array is passed.

    This version: lexigraphically sort rows,
        return row frequency in last column

    Parameters
    ----------
    a : array

    Returns
    -------
    A 2D frequency table (col [0:n-1]=scores, col n=frequencies)
    """
    scores = _support.unique(a)
    freq = zeros(len(scores))
    if a.ndim == 1:
        scores = np.sort(scores)
        for i in range(len(scores)):
            freq[i] = np.sum(a==scores[i])
    elif a.ndim == 2:
        scores = scores[np.lexsort(np.fliplr(scores).T)]
        for i in range(len(scores)):
            freq[i] = np.sum(np.all(a==scores[i],1))
    else:
        raise ValueError, "Input must be <= 2-d."

    return array(_support.abut(scores, freq))


a=stats.randint.rvs(0,3, size=(100,2))
#print a
print scipy.stats._support.unique(a)
print itemfreq(a)

a=stats.randint.rvs(0,4, size=100)
print a
print scipy.stats._support.unique(a)
print itemfreq(a)
