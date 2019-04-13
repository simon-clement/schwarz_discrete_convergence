import numpy as np
from numpy import cos, sin
from numpy.random import random
from tests.utils_numeric import solve_linear
from discretizations.finite_difference import FiniteDifferences
fdifference = FiniteDifferences()
get_Y = fdifference.get_Y
get_Y_star = fdifference.get_Y_star
"""
    Test of the solver in with D!= 0, a != 0 and c!=0.
    The equation is solved analytically and compared
    to the numerical approximation.
    we choose u* = sin(dx) on [0,1]
    h is not constant
    f will be d^2 D(x)*sin(dx) - dD'(x)*cos(dx) + da*cos(dx) + c*sin(dx)
"""


def test_full_eq():
    # Our domain is [0,1]
    M = 10001
    a = 1.2
    c = 0.3
    d = 1.2
    dt = 100000.0

    x = np.linspace(0, 1, M)**3
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2]) / 2

    def D(x): return x
    def D_prime(x): return 1
    D_star = D((x[1:] + x[:-1]) / 2)

    def function_f(x): return d * d * D(x) * sin(d * x) - d * \
        D_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)
    f = np.zeros(M)
    f[1:M - 1] = function_f(x[1:M - 1]) * (hm1 + hm)
    f[-1] = d * cos(d * middle_grid_last)

    Y = get_Y_star(M_star=M, h_star=h, D_star=D_star, a=a, c=c)
    assert np.linalg.norm(sin(d * x) - solve_linear(Y, f)) < 1e-7

    Y = get_Y(M=M,
              Lambda=1.0,
              h=h,
              D=D_star,
              a=a,
              c=c,
              dt=dt,
              upper_domain=True)
    f = np.zeros(M)
    # Robin bd condition, u(0) = 0 :
    f[0] = d * cos(d * x[1] / 2) * D_star[0] - h[0] / 2 * function_f(x[1] / 2)
    f[1:M - 1] = function_f(x[1:M - 1]) * (hm1 + hm)
    f[-1] = d * cos(d * middle_grid_last)  # Neumann bd condition
    assert np.linalg.norm(sin(d * x) - solve_linear(Y, f)) < 1e-7

    # Our domain is now [-1,0]
    x = np.linspace(0, -1, M)**3
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]

    def D(x): return -x
    def D_prime(x): return -1
    D_star = D((x[1:] + x[:-1]) / 2)
    def function_f(x): return d * d * D(x) * sin(d * x) - d * \
        D_prime(x) * cos(d * x) + d * a * cos(d * x) + c * sin(d * x)

    Y = get_Y(M=M,
              Lambda=1.0,
              h=h,
              D=D_star,
              a=a,
              c=c,
              dt=dt,
              upper_domain=False)
    f = np.zeros(M)
    # Robin bd condition, u(0) = 0
    f[0] = D_star[0] * d * cos(d *
                               (x[1] / 2)) - h[0] / 2 * function_f(x[1] / 2)
    f[1:M - 1] = function_f(x[1:M - 1]) * (hm1 + hm)
    f[-1] = sin(d * x[-1])  # Dirichlet bd condition
    assert np.linalg.norm(sin(d * x) - solve_linear(Y, f)) < 2e-6
    return "ok"


"""
    Test of the solver in with a=c=0.
    The equation is solved analytically and compared
    to the numerical approximation.
    tested for D constant and u* = x, u* = x^2/2
    for D(x) = x and u* = x.
    h is not constant.
"""


def test_heat_eq():
    # Our domain is [0,1]
    M = 401
    x = np.linspace(0, 1, M)**6
    h = np.diff(x)
    hm1 = h[:-1]
    hm = h[1:]
    middle_grid_last = (x[-1] + x[-2]) / 2
    D = 1.0
    a = 0.0
    c = 0.0
    Y = get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)

    # Simplest case: f=0, u* = x
    f = np.zeros(M)
    f[1:M - 1] = 0 * (hm1 + hm)
    f[-1] = np.pi
    assert np.linalg.norm(np.pi * x - solve_linear(Y, f)) < 1e-10

    # almost simplest case: f=-1, u* = x^2/2
    f[0] = 0  # x[1:M-1]
    f[1:M - 1] = -1 * (hm1 + hm)
    f[-1] = middle_grid_last
    # real_Y = np.diag(Y[2], k=1) + np.diag(Y[1]) + np.diag(Y[0], k=-1)
    u = solve_linear(Y, f)
    assert np.linalg.norm(x * x / 2 - solve_linear(Y, f)) < 1e-10

    # now we take D = x (don't forget D[i] = D((x_i+x_{i+1})/2)
    D = (x[1:] + x[:-1]) / 2
    Y = get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)

    # almost simplest case: f=-1, D(x)=x, u* = x
    f[0] = 0  # x[1:M-1]
    f[1:M - 1] = -1 * (hm1 + hm)
    f[-1] = 1
    # real_Y = np.diag(Y[2], k=1) + np.diag(Y[1]) + np.diag(Y[0], k=-1)
    u = solve_linear(Y, f)
    assert np.linalg.norm(x - solve_linear(Y, f)) < 1e-10
    return "ok"


"""
    Test of the function get_Y:
    check the relation Y_1 = 2hc - Y_0 - Y_1
    and check explicitly coefficients in constant case
"""


def test_get_Y():

    FLOAT_TOL = 1e-6
    M = 100
    dt = 1.0
    for _ in range(1000):  # tests of multiple coeffs: \Omega_2
        a, c, h, D, Lambda = [abs(random()) + 1e-8 for _ in range(5)]

        Y_0, Y_1, Y_2 = get_Y(M=M,
                              Lambda=Lambda,
                              h=h,
                              D=D,
                              a=a,
                              c=c,
                              dt=dt,
                              upper_domain=True)
        # constant coefficients case:
        for i in range(M - 2):
            assert abs(Y_1[i + 1] - (4 * D / h + 2 * h * c)) < FLOAT_TOL
            assert abs(Y_0[i + 0] - (-2 * D / h - a)) < FLOAT_TOL
            assert abs(Y_2[i + 1] - (-2 * D / h + a)) < FLOAT_TOL

        assert abs(Y_2[0] + Y_1[0] - Lambda + h / 2 *
                   (1 / dt + c)) < FLOAT_TOL  # Robin
        assert abs(Y_0[-1] + Y_1[-1]) < FLOAT_TOL  # Neumann
        Y_1 = Y_1[1:-1]
        Y_0 = Y_0[:-1]
        Y_2 = Y_2[1:]
        assert np.linalg.norm(Y_1 - 2 * h * c + Y_0 + Y_2) < FLOAT_TOL * M

    for _ in range(1000):  # tests of multiple coeffs: \Omega_1
        a, c, h, D, Lambda = [abs(random()) + 1e-12 for _ in range(5)]
        h = -h
        Y_0, Y_1, Y_2 = get_Y(M=M,
                              Lambda=Lambda,
                              h=h,
                              D=D,
                              a=a,
                              c=c,
                              dt=dt,
                              upper_domain=False)
        # constant coefficients case:
        for i in range(M - 2):
            assert abs(Y_1[i + 1] - (4 * D / h + 2 * h * c)) < FLOAT_TOL
            assert abs(Y_0[i + 0] - (-2 * D / h - a)) < FLOAT_TOL
            assert abs(Y_2[i + 1] - (-2 * D / h + a)) < FLOAT_TOL

        assert abs(Y_2[0] + Y_1[0] - Lambda + h / 2 *
                   (1 / dt + c)) < FLOAT_TOL  # Robin
        assert abs(Y_0[-1]) < FLOAT_TOL  # Dirichlet
        assert abs(Y_1[-1] - 1) < FLOAT_TOL  # Dirichlet
        Y_1 = Y_1[1:-1]
        Y_0 = Y_0[:-1]
        Y_2 = Y_2[1:]
        assert np.linalg.norm(Y_1 - 2 * h * c + Y_0 + Y_2) < FLOAT_TOL * M

    for _ in range(1000):  # tests of multiple coeffs: whole \Omega
        a, c, h, D = [abs(random()) + 1e-12 for _ in range(4)]
        Y_0, Y_1, Y_2 = get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)
        # constant coefficients case:
        for i in range(M - 2):
            assert abs(Y_1[i + 1] - (4 * D / h + 2 * h * c)) < FLOAT_TOL
            assert abs(Y_0[i + 0] - (-2 * D / h - a)) < FLOAT_TOL
            assert abs(Y_2[i + 1] - (-2 * D / h + a)) < FLOAT_TOL

        assert abs(Y_2[0]) < FLOAT_TOL  # Dirichlet
        assert abs(Y_1[0] - 1) < FLOAT_TOL  # Dirichlet
        assert abs(Y_0[-1] + Y_1[-1]) < FLOAT_TOL  # Neumann
        Y_1 = Y_1[1:-1]
        Y_0 = Y_0[:-1]
        Y_2 = Y_2[1:]
        assert np.linalg.norm(Y_1 - 2 * h * c + Y_0 + Y_2) < FLOAT_TOL * M

    return "ok"


def launch_all_tests():
    print("Test get_Y:", test_get_Y())
    print("Test heat equation:", test_heat_eq())
    print("Test parabolic equation:", test_full_eq())
