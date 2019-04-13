import numpy as np
from discretizations.finite_difference import FiniteDifferences
fdifference = FiniteDifferences()
get_Y = fdifference.get_Y
get_Y_star = fdifference.get_Y_star
from tests.utils_numeric import integration
"""
    Solve the equation with full system:
    \partial t u* + Y* u* = f*
    and return u* at each time step.

    f_star_0 is the value of f* in x0.
    The other values of f_star are deduced from f1 and f2

    This function should be useless in the future but it is a good example
    of how to use the integrator in time.
"""


def solve_u_time_domain(u1_init, u2_init, f_star_0, f1, f2, Lambda_1, Lambda_2,
                        D1, D2, h1, h2, a, c, dt, number_time_steps):
    assert type(float(f_star_0)) == float
    assert type(float(dt)) == float
    assert u1_init.ndim == u2_init.ndim == f1.ndim == f2.ndim == 1
    M1 = u1_init.shape[0]
    M2 = u2_init.shape[0]
    assert f1.shape[0] == M1 and f2.shape[0] == M2
    assert type(float(a)) == float
    assert type(float(c)) == float
    assert type(float(Lambda_1)) == float
    assert type(float(Lambda_2)) == float
    # Broadcasting of h and D:
    h1 = np.ones(M1 - 1) * h1
    h2 = np.ones(M2 - 1) * h2
    D1 = np.ones(M1 - 1) * D1
    D2 = np.ones(M2 - 1) * D2
    assert (h1 < 0).all()
    assert (h2 > 0).all()
    assert (D1 > 0).all()
    assert (D2 > 0).all()
    assert h1.ndim == D1.ndim == h2.ndim == D2.ndim == 1

    # Compute Y matrices:
    M_star = M1 + M2 - 1
    h_star = np.concatenate((-h1[::-1], h2))
    D_star = np.concatenate((D1[::-1], D2))
    Y_star = get_Y_star(M_star=M_star, h_star=h_star, D_star=D_star, a=a, c=c)

    f_star = np.concatenate(([f1[-1]], -f1[-2:0:-1], [f_star_0], f2[1:]))
    #we use -f1 because f1 is f*(h^1_m + h^1_{m-1}) and h^1<0
    u0_star = np.concatenate((u1_init[:0:-1], u2_init))
    f0_on_time = np.array([f1[-1] for _ in range(number_time_steps)])

    return integration(u0_star, Y_star, f_star, f0_on_time, dt, 1)


if __name__ == "__main__":
    import tests.test_linear_sys
    import tests.test_schwarz
    import tests.test_finite_volumes
    tests.test_linear_sys.launch_all_tests()
    tests.test_schwarz.launch_all_tests()
    tests.test_finite_volumes.launch_all_tests()
