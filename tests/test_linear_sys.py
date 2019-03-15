import numpy as np
from numpy.random import random
from cv_rate import get_Y, get_Y_star
"""
    Test of the function get_Y:
    check the relation Y_1 = 2hc - Y_0 - Y_1
    and check explicitly coefficients in constant case
"""
def test_get_Y():

    FLOAT_TOL = 1e-6
    M = 100
    for _ in range(1000): # tests of multiple coeffs: \Omega_2
        a, c, h, D, Lambda = [abs(random())+1e-8 for _ in range(5)]

        Y_0, Y_1, Y_2 = get_Y(M=M, Lambda=Lambda, h=h, D=D, a=a, c=c,
                upper_domain=True)
        # constant coefficients case:
        for i in range(M-2):
            assert abs(Y_1[i+1] - (4*D/h - 2*a + 2*h*c)) < FLOAT_TOL
            assert abs(Y_0[i+0] - (-2*D/h)) < FLOAT_TOL
            assert abs(Y_2[i+1] - (-2*D/h+2*a)) < FLOAT_TOL

        assert abs(Y_2[0] + Y_1[0] - Lambda) < FLOAT_TOL # Robin
        assert abs(Y_0[-1] + Y_1[-1]) < FLOAT_TOL # Neumann
        Y_1 = Y_1[1:-1]
        Y_0 = Y_0[:-1]
        Y_2 = Y_2[1:]
        assert np.linalg.norm(Y_1 - 2*h*c + Y_0 + Y_2) < FLOAT_TOL*M


    for _ in range(1000): # tests of multiple coeffs: \Omega_1
        a, c, h, D, Lambda = [abs(random())+1e-12 for _ in range(5)]
        Y_0, Y_1, Y_2 = get_Y(M=M, Lambda=Lambda, h=h, D=D, a=a, c=c,
                upper_domain=False)
        h = -h
        # constant coefficients case:
        for i in range(M-2):
            assert abs(Y_1[i+1] - (4*D/h - 2*a + 2*h*c)) < FLOAT_TOL
            assert abs(Y_0[i+0] - (-2*D/h+2*a)) < FLOAT_TOL
            assert abs(Y_2[i+1] - (-2*D/h)) < FLOAT_TOL

        assert abs(Y_2[0] + Y_1[0] - Lambda) < FLOAT_TOL # Robin
        assert abs(Y_0[-1]) < FLOAT_TOL # Dirichlet
        assert abs(Y_1[-1] - 1) < FLOAT_TOL # Dirichlet
        Y_1 = Y_1[1:-1]
        Y_0 = Y_0[:-1]
        Y_2 = Y_2[1:]
        assert np.linalg.norm(Y_1 - 2*h*c + Y_0 + Y_2) < FLOAT_TOL*M

    for _ in range(1000): # tests of multiple coeffs: whole \Omega
        a, c, h, D = [abs(random())+1e-12 for _ in range(4)]
        Y_0, Y_1, Y_2 = get_Y_star(M_star=M, h_star=h, D_star=D, a=a, c=c)
        # constant coefficients case:
        for i in range(M-2):
            assert abs(Y_1[i+1] - (4*D/h - 2*a + 2*h*c)) < FLOAT_TOL
            assert abs(Y_0[i+0] - (-2*D/h)) < FLOAT_TOL
            assert abs(Y_2[i+1] - (-2*D/h+2*a)) < FLOAT_TOL

        assert abs(Y_2[0]) < FLOAT_TOL # Dirichlet
        assert abs(Y_1[0] - 1) < FLOAT_TOL # Dirichlet
        assert abs(Y_0[-1] + Y_1[-1]) < FLOAT_TOL # Neumann
        Y_1 = Y_1[1:-1]
        Y_0 = Y_0[:-1]
        Y_2 = Y_2[1:]
        assert np.linalg.norm(Y_1 - 2*h*c + Y_0 + Y_2) < FLOAT_TOL*M


    return "ok"
