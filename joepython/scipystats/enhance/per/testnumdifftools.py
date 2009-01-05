""" Test functions for numdifftools module

"""
##from numpy.testing import *
##
##set_package_path()
##import numpy
##from numpy import typecodes, array
##
##restore_path()
##
##import types

import unittest
import numdifftools as nd
import numpy as np

class TestDerivative(unittest.TestCase):


    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testderivative(self):
        #derivative of exp(x), at x == 0
        dexp = nd.Derivative(np.exp)
        self.assertAlmostEqual(dexp(0),np.exp(0))
        dexp.derOrder = 2
        self.assertAlmostEqual(dexp(0),np.exp(0))

        # Evaluate the indicated (default = first)
        # derivative at multiple points
        dsin = nd.Derivative(np.sin)
        x = np.linspace(0,2.*np.pi,13)
        y = dsin(x)
        small = np.abs(y-np.cos(x))<dsin.error_estimate*10
        self.assertTrue(np.all(small))

        #Higher order derivatives (second derivative)
        # Truth: 0
        d2sin = nd.Derivative(np.sin,derOrder=2,stepFix=0.5)

        self.assertAlmostEqual(d2sin(np.pi),0.0,)

        # Higher order derivatives (up to the fourth derivative)
        # Truth: sqrt(2)/2 = 0.707106781186548
        d2sin.derOrder = 4
        y = d2sin(np.pi/4)
        small = np.abs(y-np.sqrt(2.)/2.)<d2sin.error_estimate
        self.assertTrue(small)

        # Higher order derivatives (third derivative)
        # Truth: 1
        d3cos = nd.Derivative(np.cos,derOrder=3)
        y = d3cos(np.pi/2.0)
        small = np.abs(y-1.0)< d3cos.error_estimate
        self.assertTrue(small)

        # Compute the derivative of a function using a backward difference scheme
        # And a backward scheme will only look below x0.
        dsinh = nd.Derivative(np.sinh,method='backward')
        small = np.abs(dsinh(0.0)-np.cosh(0.0))< dsinh.error_estimate
        self.assertTrue(small)

        # Although a central rule may put some samples in the wrong places, it may still succeed
        dlog = nd.Derivative(np.log,method='central')
        x = 0.001
        small = np.abs(dlog(x)-1.0/x)<dlog.error_estimate
        self.assertTrue(small)

        #But forcing the use of a one-sided rule may be smart anyway
        dlog.method = 'forward'
        small = np.abs(dlog(x)-1/x)<dlog.error_estimate
        self.assertTrue(small)

        # Control the behavior of Derivative - forward 2nd order method, with only 1 Romberg term
        # Compute the first derivative, also return the final stepsize chosen
        #[deriv,err,fdelta] = derivest(@(x) tan(x),pi,'deriv',1,'Style','for','MethodOrder',2,'RombergTerms',1)
        dtan = nd.Derivative(np.tan,derOrder=1,metOrder=2,method='forward',numTerms=1)
        y = dtan(np.pi)
        abserr = dtan.error_estimate
        self.assertTrue(np.abs(y-1)<abserr)

        dtan.finaldelta

##%% Specify the step size (default stepsize = 0.1)
##deriv = derivest(@(x) polyval(1:5,x),1,'deriv',4,'FixedStep',1)
##
##%% Provide other parameters via an anonymous function
##% At a minimizer of a function, its derivative should be
##% essentially zero. So, first, find a local minima of a
##% first kind bessel function of order nu.
##nu = 0;
##fun = @(t) besselj(nu,t);
##fplot(fun,[0,10])
##x0 = fminbnd(fun,0,10,optimset('TolX',1.e-15))
##hold on
##plot(x0,fun(x0),'ro')
##hold off
##
##deriv = derivest(fun,x0,'d',1)
##
##%% The second derivative should be positive at a minimizer.
##deriv = derivest(fun,x0,'d',2)
##
##%% Compute the numerical gradient vector of a 2-d function
##% Note: the gradient at this point should be [4 6]
##fun = @(x,y) x.^2 + y.^2;
##xy = [2 3];
##gradvec = [derivest(@(x) fun(x,xy(2)),xy(1),'d',1), ...
##           derivest(@(y) fun(xy(1),y),xy(2),'d',1)]
##
##%% Compute the numerical Laplacian function of a 2-d function
##% Note: The Laplacian of this function should be everywhere == 4
##fun = @(x,y) x.^2 + y.^2;
##xy = [2 3];
##lapval = derivest(@(x) fun(x,xy(2)),xy(1),'d',2) + ...
##           derivest(@(y) fun(xy(1),y),xy(2),'d',2)
##
##%% Compute the derivative of a function using a central difference scheme
##% Sometimes you may not want your function to be evaluated
##% above or below a given point. A 'central' difference scheme will
##% look in both directions equally.
##[deriv,err] = derivest(@(x) sinh(x),0,'Style','central')
##
##%% Compute the derivative of a function using a forward difference scheme
##% But a forward scheme will only look above x0.
##[deriv,err] = derivest(@(x) sinh(x),0,'Style','forward')
##
##%% Compute the derivative of a function using a backward difference scheme
##% And a backward scheme will only look below x0.
##[deriv,err] = derivest(@(x) sinh(x),0,'Style','backward')
##
##%% Although a central rule may put some samples in the wrong places, it may still succeed
##[d,e,del]=derivest(@(x) log(x),.001,'style','central')
##
##%% But forcing the use of a one-sided rule may be smart anyway
##[d,e,del]=derivest(@(x) log(x),.001,'style','forward')
##
##%% Control the behavior of DERIVEST - forward 2nd order method, with only 1 Romberg term
##% Compute the first derivative, also return the final stepsize chosen
##[deriv,err,fdelta] = derivest(@(x) tan(x),pi,'deriv',1,'Style','for','MethodOrder',2,'RombergTerms',1)
##
##%% Functions should be vectorized for speed, but its not always easy to do.
##[deriv,err] = derivest(@(x) x.^2,0:5,'deriv',1)
##[deriv,err] = derivest(@(x) x^2,0:5,'deriv',1,'vectorized','no')



class TestJacobian(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testjacobian(self):
        pass

class TestGradient(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testgradient(self):
        pass

class TestHessian(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testhessian(self):
        pass

class TestHessdiag(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testhessdiag(self):
        pass

class TestGlobalFunctions(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testvec2mat(self):
        pass

if __name__ == '__main__':
    unittest.main()
