import numpy as np

"""
    Returns "Y * u"
    equivalent code :
    return (np.diag(Y[1])+np.diag(Y[0], k=-1) + np.diag(Y[2], k=1)) @ u
    Y is a tridiagonal matrix returned by get_Y or get_Y_star
"""
def multiply(Y, u):
    assert len(Y) == 3
    assert u.ndim == Y[0].ndim == Y[1].ndim == Y[2].ndim == 1
    assert Y[1].shape[0] == Y[0].shape[0] + 1 == u.shape[0]
    assert Y[0].shape[0] == Y[2].shape[0]
    return np.concatenate(([0], Y[0]*u[:-1])) + Y[1] * u + \
            np.concatenate((Y[2]*u[1:], [0]))


"""
    Solve the linear system Yu = f and returns u.
    This function is just a wrapper over scipy
    Y is a tuple (Y_0, Y_1, Y_2) containing respectively:
    - The left diagonal of size M-1
    - The main diagonal of size M
    - The right diagonal of size M-1
    f is an array of size M.
    f[0] is the condition on the bottom of the domain
    f[-1] is the condition on top of the domain
    /!\ f[1:-1] should be equal to f * (hm + hmm1) /!\

    Y is returned by the functions get_Y and get_Y_star
"""
def solve_linear(Y, f):
    # We note Y_1 the main diagonal, Y_2 the right diag, Y_0 the left diag
    Y_0, Y_1, Y_2 = Y
    assert Y_0.ndim == Y_1.ndim == Y_2.ndim == f.ndim == 1
    assert Y_1.shape[0] - 1 == Y_2.shape[0] == Y_0.shape[0] == f.shape[0] - 1

    # solve_banded function requires to put the diagonals in the following form:
    Y_2 = np.concatenate(([0], Y_2))
    Y_0 = np.concatenate((Y_0, [0]))
    Y = np.vstack((Y_2, Y_1, Y_0))
    from scipy.linalg import solve_banded
    return solve_banded((1, 1), Y, f)
