import numpy as np
from scipy import signal

dvec = np.array([1,2,3,4])
A1 = np.array([-dvec,[1,0,0,0],[0,1,0,0],[0,0,1,0]])
B1 = np.array([[1,0,0,0]]).T  # wrong dimension, this requires D has one column
B1 = np.eye(4)
C1 = np.array([[1,2,3,4]])
D1 = np.zeros((1,4))
print signal.ss2tf(A1,B1,C1,D1)
#same as http://en.wikipedia.org/wiki/State_space_(controls)#Canonical_realizations

signal.ss2tf(*signal.tf2ss(*signal.ss2tf(A1,B1,C1,D1)))
np.testing.assert_almost_equal(signal.ss2tf(*signal.tf2ss(*signal.ss2tf(A1,B1,C1,D1)))[0],signal.ss2tf(A1,B1,C1,D1)[0])

'''

dx_t = A x_t + B u_t
 y_t = C x_t + D u_t


>>> dvec = np.array([1,2,3,4])
>>> A = np.array([-dvec,[1,0,0,0],[0,1,0,0],[0,0,1,0]])
>>> B = np.array([[1,0,0,0]]).T  # wrong dimension, this requires D has one column
>>> B = np.eye(4)
>>> C = np.array([[1,2,3,4]])
>>> D = np.zeros((1,4))
>>> num, den = signal.ss2tf(A,B,C,D)
>>> print num
[[ 0.  1.  2.  3.  4.]]
>>> print den
[ 1.  1.  2.  3.  4.]

>>> A1,B1,C1,D1 = signal.tf2ss(*signal.ss2tf(A,B,C,D))
>>> A1
array([[-1., -2., -3., -4.],
       [ 1.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.],
       [ 0.,  0.,  1.,  0.]])
>>> B1
array([[ 1.],
       [ 0.],
       [ 0.],
       [ 0.]])
>>> C1
array([[ 1.,  2.,  3.,  4.]])
>>> D1
array([ 0.])
'''

# can only have one noise variable u_t
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dvec = np.array([1,2,3,4])
A = np.array([-dvec,[1,0,0,0],[0,1,0,0],[0,0,1,0]])
B = np.array([[1,0,0,0]]).T  # wrong dimension, this requires D has one column
B = np.eye(4)
B[2,1] = 1
C = np.array([[1,2,3,4]])
D = np.zeros((1,4))
print signal.ss2tf(A,B,C,D)


# can only have one output variable y_t
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dvec = np.array([1,2,3,4])
A = np.array([-dvec,[1,0,0,0],[0,1,0,0],[0,0,1,0]])
B = np.array([[1,0,0,0]]).T  # wrong dimension, this requires D has one column
B = np.eye(4)
B[2,1] = 1
C = np.array([[1,2,3,4],[1,0,0,0]])
D = np.zeros((2,4))
#print signal.ss2tf(A,B,C,D)
#this causes
##    type_test = A[:,0] + B[:,0] + C[0,:] + D
##ValueError: shape mismatch
#
