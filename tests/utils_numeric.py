import numpy as np
from utils_linalg import multiply, solve_linear
"""
    Performs integration in time with a theta-shema.
    u0 is the initial condition
    Y is the matrix returned by get_Y or get_Y_star, such that
    \partial t u + Yu = f
    f cannot evolve in time except its first coordinate f_x0:
    f must have the shape of u
    f_x0 must be a 1-D array
    the number of time_steps is (f_x0.shape[0] - 1)
"""


def integration(u0, Y, f_const, f_x0, dt, theta):
    f = np.copy(f_const)
    assert Y[0].ndim == Y[1].ndim == Y[2].ndim == 1
    assert f_const.ndim == f.ndim == u0.ndim == 1
    M = Y[1].shape[0]
    assert f_const.shape[0] == u0.shape[0] == M
    assert Y[0].shape[0] == Y[2].shape[0] == M - 1
    assert 0 <= theta <= 1

    u = np.empty((f_x0.shape[0], u0.shape[0]))
    u[0] = u0
    if theta != 0:
        for step in range(f_x0.shape[0] - 1):
            f[0] = (1 - theta) * f_x0[step] + theta * f_x0[step + 1]
            Y_implicit = (theta * Y[0], theta * Y[1] + np.ones(M) / dt,
                          theta * Y[2])
            Yu = multiply(Y, u[step])
            u[step + 1] = solve_linear(Y_implicit,
                                       f + u[step] / dt - (1 - theta) * Yu)
    else:  #explicit case:
        for step in range(f_x0.shape[0] - 1):
            f[0] = f_x0[step]
            Yu = multiply(Y, u[step])
            u[step + 1] = u[step] + dt * (f - Yu)
    return u
