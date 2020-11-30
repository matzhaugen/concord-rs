import numpy as np
from pyconcord import axpy, mult, concord


def test_concord():
	x = np.random.randn(13, 9)
	y = concord(x, 0.3)
	s11 = np.sum(x[:, 0] * x[:, 0]) / x.shape[0]
	np.testing.assert_array_almost_equal(y[0, 0], s11)


def test_axpy():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([3.0, 3.0, 3.0])
    z = axpy(3.0, x, y)
    np.testing.assert_array_almost_equal(z, np.array([6.0, 9.0, 12.0]))
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([3.0, 3.0, 3.0, 3.0])
    z = axpy(3.0, x, y)
    np.testing.assert_array_almost_equal(z, np.array([6.0, 9.0, 12.0, 15.0]))


def test_mult():
    x = np.array([1.0, 2.0, 3.0])
    mult(3.0, x)
    np.testing.assert_array_almost_equal(x, np.array([3.0, 6.0, 9.0]))