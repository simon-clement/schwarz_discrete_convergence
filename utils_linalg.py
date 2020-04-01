"""
    This module contains functions to use the banded matrix Y returned
    by the functions get_Y and get_Y_star.
"""
import numpy as np

def scal_multiply(Y, s):
    """
        Returns "Y * s", Y being a tuple of np arrays
    """
    s = float(s)
    assert type(s) == float
    ret_list = []
    for element in Y:
        ret_list += [element * s]
    return tuple(ret_list)


def add_banded(Y1, Y2):
    """
        Returns "Y1 + Y2", Y1 and Y2 being tuples of ndarrays of the same size
    """
    assert len(Y1) == len(Y2)
    ret_list = []
    for y1, y2 in zip(Y1, Y2):
        assert y1.ndim == y2.ndim
        assert y1.shape[0] == y2.shape[0]
        ret_list += [y1 + y2]
    return tuple(ret_list)

def multiply(Y, u):
    """
        Returns "Y * u"
        equivalent code :
        return (np.diag(Y[1])+np.diag(Y[0], k=-1) + np.diag(Y[2], k=1)) @ u
        Y is a tridiagonal matrix returned by get_Y or get_Y_star
    """
    assert len(Y) == 3
    assert u.ndim == Y[0].ndim == Y[1].ndim == Y[2].ndim == 1
    assert Y[1].shape[0] == Y[0].shape[0] + 1 == u.shape[0]
    assert Y[0].shape[0] == Y[2].shape[0]
    return np.concatenate(([0], Y[0] * u[:-1])) + Y[1] * u + \
        np.concatenate((Y[2] * u[1:], [0]))

def multiply_interior(Y, u):
    """
        Returns "Y * u"
        equivalent code :
        return (np.diag(Y[1])+np.diag(Y[0], k=-1) + np.diag(Y[2], k=1)) @ u
        Y is a tridiagonal matrix returned by get_Y or get_Y_star
    """
    assert len(Y) == 3
    assert u.ndim == Y[0].ndim == Y[1].ndim == Y[2].ndim == 1
    assert Y[2].shape[0] == Y[1].shape[0] == Y[0].shape[0] == u.shape[0] - 2
    return Y[0] * u[:-2] + Y[1] * u[1:-1] + Y[2] * u[2:]

def solve_linear(Y, f):
    """
        Solve the linear TRIDIAGONAL system Yu = f and returns u.
        Y can have an additional (and only one) upper diagonal
        to take into account the case with extrapolation at boundary

        This function is just a wrapper over scipy
        Y is a tuple (Y_0, Y_1, Y_2) containing respectively:
        - The left diagonal of size M-1
        - The main diagonal of size M
        - The right diagonal of size M-1
        f is an array of size M.
        f[0] is the condition on the bottom of the domain
        f[-1] is the condition on top of the domain
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        /!\ f[1:-1] should be equal to f * (hm + hmm1) !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Y is returned by the functions get_Y and get_Y_star
    """
    if len(Y) == 4:
        return solve_linear_with_ultra_right(Y, f)
    # if there is not 4 diagonals in Y, then it is a tridiagonal matrix
    # We note Y_1 the main diagonal, Y_2 the right diag, Y_0 the left diag
    Y_0, Y_1, Y_2 = Y
    assert Y_0.ndim == Y_1.ndim == Y_2.ndim == f.ndim == 1
    assert Y_1.shape[0] - 1 == Y_2.shape[0] == Y_0.shape[0] == f.shape[0] - 1

    # solve_banded function requires to put the diagonals in the following
    # form:
    Y_2 = np.concatenate(([0], Y_2))
    Y_0 = np.concatenate((Y_0, [0]))
    Y = np.vstack((Y_2, Y_1, Y_0))
    from scipy.linalg import solve_banded
    return solve_banded((1, 1), Y, f)


def solve_linear_with_ultra_right(Y, f):
    """
        Solve the linear BANDED (4 diagonals) system Yu = f and returns u.
        This function is just a wrapper over scipy
        Y is a tuple (Y_0, Y_1, Y_2, Y_3) containing respectively:
        - The left diagonal of size M-1
        - The main diagonal of size M
        - The right diagonal of size M-1
        - The ultra-right diagonal of size M-2
        f is an array of size M.
        f[0] is the condition on the bottom of the domain
        f[-1] is the condition on top of the domain
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        /!\ f[1:-1] should be equal to f * (hm + hmm1) !
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Y is returned by the functions get_Y and get_Y_star
    """
    # We note Y_1 the main diagonal, Y_2 the right diag, Y_0 the left diag
    Y_0, Y_1, Y_2, Y_3 = Y
    assert Y_0.ndim == Y_1.ndim == Y_2.ndim == Y_3.ndim == f.ndim == 1
    assert Y_1.shape[0] - 1 == Y_2.shape[0] == Y_0.shape[0] == f.shape[0] - 1
    assert Y_3.shape[0] == Y_2.shape[0] - 1

    # solve_banded function requires to put the diagonals in the following
    # form:
    Y_3 = np.concatenate(([0, 0], Y_3))
    Y_2 = np.concatenate(([0], Y_2))
    Y_0 = np.concatenate((Y_0, [0]))
    Y = np.vstack((Y_3, Y_2, Y_1, Y_0))
    from scipy.linalg import solve_banded
    return solve_banded((1, 2), Y, f)
