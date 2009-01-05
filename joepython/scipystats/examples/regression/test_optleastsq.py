class TestLeastSq(TestCase):
    def setUp(self):
        x = np.linspace(0, 10, 40)
        a,b,c = 3.1, 42, -304.2
        self.x = x
        self.abc = a,b,c
        y_true = a*x**2 + b*x + c
        self.y_meas = y_true + 0.01*np.random.standard_normal( y_true.shape )

    def residuals(self, p, y, x):
        a,b,c = p
        err = y-(a*x**2 + b*x + c)
        return err

    def test_basic(self):
        p0 = array([0,0,0])
        params_fit, ier = leastsq(self.residuals, p0,
                                  args=(self.y_meas, self.x))
        assert ier in (1,2,3,4), 'solution not found (ier=%d)'%ier
        assert_array_almost_equal( params_fit, self.abc, decimal=2) # low precision due to random

    def test_full_output(self):
        p0 = array([0,0,0])
        full_output = leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert ier in (1,2,3,4), 'solution not found: %s'%mesg

    def test_input_untouched(self):
        p0 = array([0,0,0],dtype=float64)
        p0_copy = array(p0, copy=True)
        full_output = leastsq(self.residuals, p0,
                              args=(self.y_meas, self.x),
                              full_output=True)
        params_fit, cov_x, infodict, mesg, ier = full_output
        assert ier in (1,2,3,4), 'solution not found: %s'%mesg
        assert_array_equal(p0, p0_copy)
